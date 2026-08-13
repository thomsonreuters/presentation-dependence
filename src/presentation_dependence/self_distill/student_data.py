"""Student-side dataset views for self-distillation SFT.

Regress the student's continuous readout to the K-shot mean in
``silver_labels.jsonl``.
"""

from __future__ import annotations

import json
import hashlib
import random
import statistics
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable

from presentation_dependence.self_distill.silver_io import read_silver_jsonl


@dataclass(slots=True)
class GroupedSilverExample:
    """All teacher labels for one query/candidate set."""

    group_id: str
    query_id: str
    query_text: str
    candidate_set_id: str
    doc_ids: list[str]
    bm25_ranks: list[int]
    passage_texts: list[str]
    teacher_scores_mean: list[float]
    teacher_scores_raw: list[list[float]]
    teacher_grade_vectors: list[list[float]]
    teacher_scores_var: list[float]
    qrels: dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        n = len(self.doc_ids)
        fields = {
            "bm25_ranks": self.bm25_ranks,
            "passage_texts": self.passage_texts,
            "teacher_scores_mean": self.teacher_scores_mean,
            "teacher_scores_raw": self.teacher_scores_raw,
            "teacher_grade_vectors": self.teacher_grade_vectors,
            "teacher_scores_var": self.teacher_scores_var,
        }
        for name, values in fields.items():
            if len(values) != n:
                raise ValueError(f"{name} has {len(values)} entries but doc_ids has {n} for qid={self.query_id}")


@dataclass(slots=True)
class RegressionChunk:
    """One B-doc training example for continuous-readout MSE SFT."""

    group_id: str
    query_id: str
    query_text: str
    candidate_set_id: str
    chunk_id: int
    doc_ids: list[str]
    bm25_ranks: list[int]
    passages: list[dict[str, str]]
    target_scores: list[float]
    target_raw_vectors: list[list[float]]
    target_grade_vectors: list[list[float]]
    target_variances: list[float]
    permutation_seed: int | None = None
    inverse_permutation: list[int] | None = None
    #: Each document's within-chunk slot in canonical first-stage order, i.e.
    #: ``first_stage_index % chunk_size``. Unlike ``bm25_ranks`` this is dense,
    #: so it survives pool subsampling, and unlike the position in ``doc_ids``
    #: it is invariant to the presentation shuffle. Keyed on by the IPS slot
    #: weights (:mod:`presentation_dependence.self_distill.ips_propensity`).
    canonical_slots: list[int] | None = None


@dataclass(slots=True)
class SupervisedConsistencyChunk:
    """Two views plus doc-level labels for anchored consistency training."""

    group_id: str
    query_id: str
    query_text: str
    candidate_set_id: str
    chunk_id: int
    doc_ids: list[str]
    bm25_ranks: list[int]
    target_scores: list[float]
    target_raw_vectors: list[list[float]]
    target_grade_vectors: list[list[float]]
    target_variances: list[float]
    view_a_doc_ids: list[str]
    view_b_doc_ids: list[str]
    view_a_passages: list[dict[str, str]]
    view_b_passages: list[dict[str, str]]
    view_a_seed: int
    view_b_seed: int
    view_a_inverse_permutation: list[int]
    view_b_inverse_permutation: list[int]
    view_generator: str = "permutation"
    #: Stage-2 label-vocabulary variation. ``view_*_slot_labels[i]`` is the
    #: bracketed marker for input slot ``i+1`` in that view's doc order. ``None``
    #: (default) means numeric-by-slot markers (byte-identical to pre-Stage-2).
    view_a_slot_labels: list[str] | None = None
    view_b_slot_labels: list[str] | None = None
    view_a_id_scheme: str | None = None
    view_b_id_scheme: str | None = None

    def __post_init__(self) -> None:
        n = len(self.doc_ids)
        if n < 2:
            raise ValueError(f"supervised consistency chunk qid={self.query_id} chunk_id={self.chunk_id} has <2 docs")
        for name, values in {
            "bm25_ranks": self.bm25_ranks,
            "target_scores": self.target_scores,
            "target_raw_vectors": self.target_raw_vectors,
            "target_variances": self.target_variances,
            "view_a_doc_ids": self.view_a_doc_ids,
            "view_b_doc_ids": self.view_b_doc_ids,
            "view_a_passages": self.view_a_passages,
            "view_b_passages": self.view_b_passages,
            "view_a_inverse_permutation": self.view_a_inverse_permutation,
            "view_b_inverse_permutation": self.view_b_inverse_permutation,
        }.items():
            if len(values) != n:
                raise ValueError(f"{name} has {len(values)} entries but doc_ids has {n} for qid={self.query_id}")
        for name, labels in {
            "view_a_slot_labels": self.view_a_slot_labels,
            "view_b_slot_labels": self.view_b_slot_labels,
        }.items():
            if labels is not None and len(labels) != n:
                raise ValueError(f"{name} has {len(labels)} entries but doc_ids has {n} for qid={self.query_id}")
        canonical = set(self.doc_ids)
        if set(self.view_a_doc_ids) != canonical or set(self.view_b_doc_ids) != canonical:
            raise ValueError(f"supervised consistency views must contain the same doc set for qid={self.query_id}")


@dataclass(slots=True)
class PermutationGroup:
    group_id: str
    query_id: str
    permutation_seed: int
    doc_ids_permuted: list[str]
    inverse_permutation: list[int]
    teacher_scores_mean_permuted: list[float]


def _variance(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    return float(statistics.variance(values))


def load_fixture_index(fixture_path: str | Path) -> dict[str, dict]:
    """Index FixtureLoader JSONL by qid.

    The fixture format is ``{"qid", "query", "passages": [{"pid", "text", ...}]}``.
    BM25 rank is taken from each passage's ``rank`` field when present, else from
    its 1-indexed position in the fixture.
    """
    out: dict[str, dict] = {}
    with open(fixture_path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            rec = json.loads(line)
            if not isinstance(rec, dict):
                raise ValueError(f"fixture line {line_no} must be a JSON object")
            raw_qid = rec.get("qid")
            if raw_qid is None or not str(raw_qid):
                raise ValueError(f"fixture line {line_no} has no non-empty qid")
            qid = str(raw_qid)
            if qid in out:
                raise ValueError(f"fixture line {line_no} duplicates qid={qid}")
            passages = []
            seen_pids: set[str] = set()
            for idx, passage in enumerate(rec.get("passages") or [], start=1):
                if not isinstance(passage, dict):
                    raise ValueError(f"fixture qid={qid} passage {idx} must be a JSON object")
                raw_pid = passage.get("pid") or passage.get("doc_id") or passage.get("docid")
                if raw_pid is None or not str(raw_pid):
                    raise ValueError(f"fixture qid={qid} has passage without pid/doc_id")
                pid = str(raw_pid)
                if pid in seen_pids:
                    raise ValueError(f"fixture qid={qid} has duplicate passage id {pid}")
                seen_pids.add(pid)
                passages.append(
                    {
                        "pid": pid,
                        "text": str(passage.get("text") or ""),
                        "rank": int(passage.get("rank", idx)),
                    }
                )
            out[qid] = {
                "query": str(rec.get("query") or rec.get("text") or ""),
                "passages": passages,
            }
    return out


def load_qrels(path: str | Path | None) -> dict[str, dict[str, int]]:
    if path is None:
        return {}
    p = Path(path)
    if not p.is_file():
        return {}
    out: dict[str, dict[str, int]] = {}
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if len(parts) < 4:
                continue
            qid, _unused, doc_id, rel = parts[:4]
            out.setdefault(str(qid), {})[str(doc_id)] = int(float(rel))
    return out


def load_qids(path: str | Path | None) -> list[str]:
    """Load one qid per non-empty line, preserving file order."""
    if path is None:
        return []
    with open(path, "r", encoding="utf-8") as f:
        qids = [line.strip() for line in f if line.strip()]
    seen: set[str] = set()
    for qid in qids:
        if qid in seen:
            raise ValueError(f"duplicate qid {qid!r} in {path}")
        seen.add(qid)
    return qids


def load_grouped_silver_examples(  # noqa: C901
    *,
    silver_labels_path: str | Path,
    fixture_path: str | Path,
    candidate_set_id: str = "msmarco_seed42_bm25_top100",
    qrels_path: str | Path | None = None,
    qids_path: str | Path | None = None,
    exclude_qids_path: str | Path | None = None,
    qids: Iterable[str] | None = None,
    exclude_qids: Iterable[str] | None = None,
) -> list[GroupedSilverExample]:
    """Join silver labels with fixture text and return one group per query."""
    fixture = load_fixture_index(fixture_path)
    qrels = load_qrels(qrels_path)
    silver = read_silver_jsonl(silver_labels_path)
    include_order = list(qids) if qids is not None else load_qids(qids_path)
    exclude = {str(qid) for qid in (exclude_qids or [])}
    exclude.update(load_qids(exclude_qids_path))

    by_qid: dict[str, dict[str, object]] = {}
    for rec in silver:
        qid = str(rec.query_id)
        doc_id = str(rec.doc_id)
        if qid not in fixture:
            raise ValueError(f"silver qid={qid} missing from fixture {fixture_path}")
        entry = by_qid.setdefault(qid, {})
        if doc_id in entry:
            raise ValueError(f"duplicate silver record for qid={qid} doc_id={doc_id}")
        raw = [float(x) for x in (rec.score_raw_vector or [])]
        grade_vector = [float(x) for x in (rec.score_grade_vector or [])]
        entry[doc_id] = {
            "score": float(rec.score_continuous),
            "raw": raw,
            "grade_vector": grade_vector,
            "var": _variance(raw),
        }

    if include_order:
        qid_iter = [str(qid) for qid in include_order if str(qid) not in exclude]
        missing_fixture = [qid for qid in qid_iter if qid not in fixture]
        if missing_fixture:
            preview = ", ".join(missing_fixture[:10])
            more = "..." if len(missing_fixture) > 10 else ""
            raise ValueError(f"{len(missing_fixture)} qids missing from fixture {fixture_path}: {preview}{more}")
        missing_silver = [qid for qid in qid_iter if qid not in by_qid]
        if missing_silver:
            preview = ", ".join(missing_silver[:10])
            more = "..." if len(missing_silver) > 10 else ""
            raise ValueError(
                f"{len(missing_silver)} qids missing from silver labels {silver_labels_path}: {preview}{more}"
            )
    else:
        qid_iter = sorted(qid for qid in by_qid if qid not in exclude)

    examples: list[GroupedSilverExample] = []
    for qid in qid_iter:
        per_doc = by_qid[qid]
        fixture_rec = fixture[qid]
        doc_ids: list[str] = []
        ranks: list[int] = []
        texts: list[str] = []
        means: list[float] = []
        raws: list[list[float]] = []
        grade_vectors: list[list[float]] = []
        variances: list[float] = []

        for passage in sorted(fixture_rec["passages"], key=lambda p: int(p["rank"])):
            pid = str(passage["pid"])
            if pid not in per_doc:
                raise ValueError(f"fixture qid={qid} pid={pid} missing from silver labels")
            label = per_doc[pid]
            doc_ids.append(pid)
            ranks.append(int(passage["rank"]))
            texts.append(str(passage["text"]))
            means.append(float(label["score"]))  # type: ignore[index]
            raws.append(list(label["raw"]))  # type: ignore[index]
            grade_vectors.append(list(label["grade_vector"]))  # type: ignore[index]
            variances.append(float(label["var"]))  # type: ignore[index]
        unexpected_docs = set(per_doc) - set(doc_ids)
        if unexpected_docs:
            raise ValueError(
                f"silver qid={qid} contains {len(unexpected_docs)} docs absent "
                f"from fixture; example={min(unexpected_docs)!r}"
            )

        examples.append(
            GroupedSilverExample(
                group_id=qid,
                query_id=qid,
                query_text=str(fixture_rec["query"]),
                candidate_set_id=candidate_set_id,
                doc_ids=doc_ids,
                bm25_ranks=ranks,
                passage_texts=texts,
                teacher_scores_mean=means,
                teacher_scores_raw=raws,
                teacher_grade_vectors=grade_vectors,
                teacher_scores_var=variances,
                qrels=qrels.get(qid, {}),
            )
        )

    return examples


def inverse_permutation(order: list[int]) -> list[int]:
    inv = [0] * len(order)
    for new_idx, old_idx in enumerate(order):
        inv[old_idx] = new_idx
    return inv


def regression_chunks(
    examples: Iterable[GroupedSilverExample],
    *,
    chunk_size: int = 20,
    permutation_seed: int | None = None,
) -> list[RegressionChunk]:
    """Build B-doc chunks for supervised MSE SFT."""
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")
    chunks: list[RegressionChunk] = []
    for ex in examples:
        order = list(range(len(ex.doc_ids)))
        if permutation_seed is not None:
            random.Random(int(permutation_seed)).shuffle(order)
        inv = inverse_permutation(order) if permutation_seed is not None else None

        for chunk_id, start in enumerate(range(0, len(order), chunk_size)):
            idxs = order[start : start + chunk_size]
            chunks.append(
                RegressionChunk(
                    group_id=ex.group_id,
                    query_id=ex.query_id,
                    query_text=ex.query_text,
                    candidate_set_id=ex.candidate_set_id,
                    chunk_id=chunk_id,
                    doc_ids=[ex.doc_ids[i] for i in idxs],
                    bm25_ranks=[ex.bm25_ranks[i] for i in idxs],
                    passages=[{"pid": ex.doc_ids[i], "text": ex.passage_texts[i]} for i in idxs],
                    target_scores=[ex.teacher_scores_mean[i] for i in idxs],
                    target_raw_vectors=[ex.teacher_scores_raw[i] for i in idxs],
                    target_grade_vectors=[ex.teacher_grade_vectors[i] for i in idxs],
                    target_variances=[ex.teacher_scores_var[i] for i in idxs],
                    permutation_seed=permutation_seed,
                    inverse_permutation=inv,
                    canonical_slots=[i % chunk_size for i in idxs],
                )
            )
    return chunks


def _stable_view_seed(*, base_seed: int, epoch: int, query_id: str, chunk_id: int) -> int:
    payload = f"{base_seed}:{epoch}:{query_id}:{chunk_id}".encode("utf-8")
    return int.from_bytes(hashlib.sha1(payload).digest()[:8], "big")


def _permuted_view(
    *,
    doc_ids: list[str],
    passage_texts: list[str],
    seed: int,
) -> tuple[list[str], list[dict[str, str]], list[int]]:
    order = list(range(len(doc_ids)))
    random.Random(int(seed)).shuffle(order)
    return (
        [doc_ids[i] for i in order],
        [{"pid": doc_ids[i], "text": passage_texts[i]} for i in order],
        inverse_permutation(order),
    )


#: Salt folded into the order-shuffle seed so the label-scheme draw uses an
#: independent deterministic stream from the document-order permutation.
_LABEL_VIEW_SALT = 0x5BD1E995


def _sampled_view_slot_labels(
    *,
    n_slots: int,
    view_seed: int,
    label_scheme_pool: Sequence[str] | None,
    decorrelate_label_slots: bool,
) -> tuple[list[str] | None, str | None]:
    """Return ``(slot_labels, id_scheme)`` for one view, or ``(None, None)``.

    When ``label_scheme_pool`` is falsy the function is a no-op (numeric-by-slot
    markers, byte-identical to pre-Stage-2 training). Otherwise it samples one
    scheme from the pool with a label stream decorrelated from the order seed.
    """
    if not label_scheme_pool:
        return None, None
    from presentation_dependence.rerankers.render_variants import sample_label_view

    labels, scheme = sample_label_view(
        n_slots,
        seed=int(view_seed) ^ _LABEL_VIEW_SALT,
        pool=tuple(label_scheme_pool),  # type: ignore[arg-type]
        decorrelate=decorrelate_label_slots,
    )
    return list(labels), str(scheme)


def supervised_consistency_chunks(
    examples: Iterable[GroupedSilverExample],
    *,
    chunk_size: int = 20,
    view_seeds: tuple[int, int] = (0, 1),
    epoch: int = 0,
    view_generator: str = "permutation",
    label_scheme_pool: Sequence[str] | None = None,
    decorrelate_label_slots: bool = True,
) -> list[SupervisedConsistencyChunk]:
    """Build paired permutation or dropout views with labels preserved by doc_id.

    ``view_generator="permutation"`` is canonical OC-SFT: each forward receives
    an independently shuffled rendering. ``view_generator="dropout"`` keeps the
    prompt bytes and document order identical; the two sequential training-mode
    forwards in the loss function then differ only through model dropout masks.

    Marker variation is only supported for permutation views. Silver anchors are
    keyed by ``doc_id`` (independent of markers), so the supervised term is
    unaffected.
    """
    if chunk_size <= 1:
        raise ValueError(f"supervised consistency chunk_size must be >= 2, got {chunk_size}")
    if len(view_seeds) != 2:
        raise ValueError(f"view_seeds must contain exactly two seeds, got {view_seeds}")
    if view_generator not in {"permutation", "dropout"}:
        raise ValueError(
            f"supervised consistency view_generator must be one of {{'permutation', 'dropout'}}, got {view_generator!r}"
        )
    if view_generator == "dropout" and label_scheme_pool:
        raise ValueError("dropout views cannot also vary label markers")

    chunks: list[SupervisedConsistencyChunk] = []
    for ex in examples:
        for chunk_id, start in enumerate(range(0, len(ex.doc_ids), chunk_size)):
            stop = min(start + chunk_size, len(ex.doc_ids))
            if stop - start < 2:
                raise ValueError(f"supervised consistency chunk qid={ex.query_id} chunk_id={chunk_id} has <2 docs")
            canonical_doc_ids = ex.doc_ids[start:stop]
            canonical_texts = ex.passage_texts[start:stop]
            seed_a = _stable_view_seed(
                base_seed=int(view_seeds[0]), epoch=epoch, query_id=ex.query_id, chunk_id=chunk_id
            )
            seed_b = _stable_view_seed(
                base_seed=int(view_seeds[1]), epoch=epoch, query_id=ex.query_id, chunk_id=chunk_id
            )
            if view_generator == "permutation":
                a_doc_ids, a_passages, a_inv = _permuted_view(
                    doc_ids=canonical_doc_ids, passage_texts=canonical_texts, seed=seed_a
                )
                b_doc_ids, b_passages, b_inv = _permuted_view(
                    doc_ids=canonical_doc_ids, passage_texts=canonical_texts, seed=seed_b
                )
            else:
                identity = list(range(len(canonical_doc_ids)))
                a_doc_ids = list(canonical_doc_ids)
                b_doc_ids = list(canonical_doc_ids)
                a_passages = [
                    {"pid": doc_id, "text": text}
                    for doc_id, text in zip(canonical_doc_ids, canonical_texts, strict=True)
                ]
                b_passages = [
                    {"pid": doc_id, "text": text}
                    for doc_id, text in zip(canonical_doc_ids, canonical_texts, strict=True)
                ]
                a_inv = list(identity)
                b_inv = list(identity)
            n_slots = stop - start
            a_labels, a_scheme = _sampled_view_slot_labels(
                n_slots=n_slots,
                view_seed=seed_a,
                label_scheme_pool=label_scheme_pool,
                decorrelate_label_slots=decorrelate_label_slots,
            )
            b_labels, b_scheme = _sampled_view_slot_labels(
                n_slots=n_slots,
                view_seed=seed_b,
                label_scheme_pool=label_scheme_pool,
                decorrelate_label_slots=decorrelate_label_slots,
            )
            chunks.append(
                SupervisedConsistencyChunk(
                    group_id=ex.group_id,
                    query_id=ex.query_id,
                    query_text=ex.query_text,
                    candidate_set_id=ex.candidate_set_id,
                    chunk_id=chunk_id,
                    doc_ids=canonical_doc_ids,
                    bm25_ranks=ex.bm25_ranks[start:stop],
                    target_scores=ex.teacher_scores_mean[start:stop],
                    target_raw_vectors=ex.teacher_scores_raw[start:stop],
                    target_grade_vectors=ex.teacher_grade_vectors[start:stop],
                    target_variances=ex.teacher_scores_var[start:stop],
                    view_a_doc_ids=a_doc_ids,
                    view_b_doc_ids=b_doc_ids,
                    view_a_passages=a_passages,
                    view_b_passages=b_passages,
                    view_a_seed=seed_a,
                    view_b_seed=seed_b,
                    view_a_inverse_permutation=a_inv,
                    view_b_inverse_permutation=b_inv,
                    view_generator=view_generator,
                    view_a_slot_labels=a_labels,
                    view_b_slot_labels=b_labels,
                    view_a_id_scheme=a_scheme,
                    view_b_id_scheme=b_scheme,
                )
            )
    return chunks


def permutation_groups(
    examples: Iterable[GroupedSilverExample],
    *,
    seeds: Iterable[int],
) -> list[PermutationGroup]:
    out: list[PermutationGroup] = []
    for ex in examples:
        base = list(range(len(ex.doc_ids)))
        for seed in seeds:
            order = list(base)
            random.Random(int(seed)).shuffle(order)
            out.append(
                PermutationGroup(
                    group_id=ex.group_id,
                    query_id=ex.query_id,
                    permutation_seed=int(seed),
                    doc_ids_permuted=[ex.doc_ids[i] for i in order],
                    inverse_permutation=inverse_permutation(order),
                    teacher_scores_mean_permuted=[ex.teacher_scores_mean[i] for i in order],
                )
            )
    return out


def export_jsonl(path: str | Path, rows: Iterable[object]) -> int:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(p, "w", encoding="utf-8") as f:
        for row in rows:
            if hasattr(row, "__dataclass_fields__"):
                payload = asdict(row)
            elif isinstance(row, dict):
                payload = row
            else:
                raise TypeError(f"Cannot JSONL-export object of type {type(row).__name__}")
            f.write(json.dumps(payload, ensure_ascii=False))
            f.write("\n")
            n += 1
    return n


def export_regression_jsonl(path: str | Path, chunks: Iterable[RegressionChunk]) -> int:
    return export_jsonl(path, chunks)


def export_supervised_consistency_jsonl(path: str | Path, chunks: Iterable[SupervisedConsistencyChunk]) -> int:
    return export_jsonl(path, chunks)
