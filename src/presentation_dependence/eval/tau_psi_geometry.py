"""Infer τ-PSI@B reporting fields from experiment config + reranker.

``tau_psi_context_batch_size`` stores B for cross-run comparison. Its meaning depends on the reranker:

- Sliding generative-listwise rerankers use window width W.
- Jina scoring-head rerankers use the evaluation candidate-list length
  ``k_input``. HF ``rerank()`` may still require several model forwards; the
  run-level ``tau_psi_inference_note`` records this caveat.
- Batched-pointwise rerankers use ``docs_per_score_forward``.
- Pointwise cross-encoders use B=1. Their ``batch_size`` controls throughput and
  is recorded separately.
- RankZephyr ``block_local_rrf`` uses ``block_size``. It decodes each disjoint
  block independently and combines blocks with RRF.

Structured keys (schema ``v2``) supplement the more free-form ``tau_psi_inference_note``.

Older ``psi_metrics.json`` files may omit these keys. Missing values indicate
unknown provenance and can usually be reconstructed from the run config.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

# Hugging Face `JinaForRanking.rerank()` caps each block at `block_size`
# documents (currently 125) and may split earlier at the tokenizer length
# budget. τ-PSI uses the same rule for chunk labels.
JINA_DOC_BLOCK_CAP_DEFAULT = 125

# Default τ-PSI@B width when a batched-pointwise config omits the tuned value.
BAT_PW_DEFAULT_DOCS_FORWARD = 20

# Classes treated as batched pointwise when ``docs_per_score_forward`` is omitted from YAML.
BAT_PW_CLASS_NAMES: frozenset[str] = frozenset(
    {
        "Qwen3Reranker",
        "Granite41GradeReranker",
        "PineReranker",
    }
)

# These rerankers have dedicated geometry branches.
_BAT_PW_GEOMETRY_EXCLUDED_CLASS_NAMES: frozenset[str] = frozenset(
    {
        "IdentityReranker",
        "MxbaiPointwise",
        "JinaListwiseReranker",
        "RankZephyrReranker",
    }
)

TAU_PSI_GEOMETRY_SCHEMA = "v2"

# All keys written by :func:`compact_tau_psi_geometry` (for merge / strip).
TAU_PSI_GEOMETRY_KEYS: frozenset[str] = frozenset(
    {
        "tau_psi_geometry_schema",
        "tau_psi_context_batch_size",
        "tau_psi_inference_note",
        "tau_psi_window_size",
        "tau_psi_stride",
        "tau_psi_docs_per_score_forward",
        "tau_psi_inference_batch_size",
        "tau_psi_candidate_list_length",
        "tau_psi_upstream_doc_block_cap",
        "tau_psi_at_B_caption",
        "tau_psi_rankzephyr_protocol",
        "tau_psi_rrf_k",
    }
)


@dataclass(frozen=True)
class TauPsiGeometry:
    """Machine-readable provenance for τ-PSI headline numbers."""

    tau_psi_context_batch_size: int | None
    tau_psi_inference_note: str | None
    window_size: int | None = None
    stride: int | None = None
    docs_per_score_forward: int | None = None
    inference_batch_size: int | None = None
    candidate_list_length: int | None = None
    upstream_doc_block_cap: int | None = None
    rankzephyr_protocol: str | None = None
    rrf_k: float | None = None


def _int_positive(name: str, value: Any) -> int:
    x = int(value)
    if x <= 0:
        raise ValueError(f"{name} must be positive, got {value!r}")
    return x


def _k_input_cap(data_cfg: Mapping[str, Any]) -> int | None:
    v = data_cfg.get("k_input")
    if v is None:
        return None
    return _int_positive("data.k_input", v)


def _note_sliding_listwise(window_size: int, stride: int, model_short: str) -> str:
    overlap = stride < window_size
    ov = "yes" if overlap else "no"
    return (
        f"[sliding_generative_listwise model={model_short}] "
        f"W={window_size} S={stride} overlapping={ov} "
        f"| tau_psi aggregates rankings over full permuted candidate list"
    )


def _note_rankzephyr_block_local_rrf(block_size: int, rrf_k: float, k_input: int | None) -> str:
    kpart = f"k_input={k_input}" if k_input is not None else "k_input unset"
    return (
        f"[rankzephyr_block_local_rrf] block_size={block_size} rrf_k={rrf_k} merge=rrf {kpart} "
        "| one listwise forward per disjoint contiguous block; NO sliding-window refinement; "
        "cross-block ordering is RRF-only — NOT comparable to the published canonical protocol; "
        "this is a tier-3 block-local approximation"
    )


def _note_pointwise_CE(
    inference_batch_size: int,
    k_input: int | None,
    *,
    reranker_class_repr: str = "MxbaiPointwise",
) -> str:
    kpart = f"k_input={k_input}" if k_input is not None else "k_input unset"
    return (
        f"[pointwise_ce class={reranker_class_repr}] "
        f"docs_per_forward=1 inference_batch_size={inference_batch_size} {kpart} "
        f"| attention is one (query, passage) pair per score; tau_psi_context_batch_size=1 "
        f"is pairwise context width"
    )


def _note_batched_pointwise(
    docs_forward: int,
    k_input: int | None,
    *,
    cls_label: str,
) -> str:
    kpart = f"k_input={k_input}" if k_input is not None else "k_input unset"
    return (
        f"[batched_pointwise class={cls_label}] docs_per_score_forward={docs_forward} {kpart} "
        "| tau_psi_context_batch_size=B equals docs_per_score_forward "
        "| mean_score_variance applies when RankResult.scores_init_order is populated"
    )


def _batched_pw_forward_width(rc: dict[str, Any], cls_name: str) -> int | None:
    """Return scoring chunk width for batched-PW geometry, or None if not applicable."""
    if cls_name in _BAT_PW_GEOMETRY_EXCLUDED_CLASS_NAMES:
        return None
    if cls_name == "CapCalReranker":
        raw = rc.get("docs_per_score_forward")
        if raw is None:
            base_rc = rc.get("base_reranker") or {}
            if isinstance(base_rc, Mapping):
                raw = base_rc.get("docs_per_score_forward")
        if raw is not None:
            v = _int_positive("reranker.docs_per_score_forward", raw)
            return v if v > 1 else None
        return BAT_PW_DEFAULT_DOCS_FORWARD
    if cls_name == "ClosedModelGenBscReranker":
        scoring = rc.get("scoring") or {}
        raw = scoring.get("subset_size") or rc.get("docs_per_score_forward")
        if raw is not None:
            return _int_positive("reranker.scoring.subset_size", raw)
        return BAT_PW_DEFAULT_DOCS_FORWARD
    raw = rc.get("docs_per_score_forward")
    if raw is not None:
        v = _int_positive("reranker.docs_per_score_forward", raw)
        return v if v > 1 else None
    if cls_name in BAT_PW_CLASS_NAMES:
        return BAT_PW_DEFAULT_DOCS_FORWARD
    return None


def _note_jina_chunked(docs_forward: int, ki: int | None, block_cap: int) -> str:
    kpart = f"k_input={ki}" if ki is not None else "k_input unset"
    n_forwards = ""
    if ki is not None and docs_forward > 0:
        n_forwards = f"n_forwards≈{(ki + docs_forward - 1) // docs_forward}"
    return (
        f"[scoring_head_jina_chunked] docs_per_score_forward={docs_forward} {kpart} "
        f"upstream_doc_block_cap={block_cap} {n_forwards} "
        "| one listwise rerank() per disjoint B-doc chunk; τ-PSI@B is chunk width "
        "| cross-chunk scores merged globally (OC-SFT / expected-grade readout B=20 protocol)"
    )


def _note_jina(ki: int | None, block_cap: int) -> str:
    caveat = (
        "Upstream HF rerank() may still run multiple LM forwards per query (tokenizer "
        "length splits and/or doc-count blocks); τ-PSI@B follows k_input as evaluation "
        "list length over permutations—headline B is not identical to a single "
        "transformer forward over all k_input documents."
    )
    if ki is None:
        return (
            f"[scoring_head_jina] k_input unset upstream_doc_block_cap={block_cap} "
            "| upstream rerank() may chunk long lists + merge query embeddings per block "
            f"| {caveat}"
        )
    chunks = "1" if ki <= block_cap else "multi"
    return (
        f"[scoring_head_jina] evaluation_list_length={ki} upstream_doc_block_cap={block_cap} "
        f"chunks={chunks} merge=weighted_query_embeddings "
        "| tau_psi measures list-level stability of merged scores "
        f"| {caveat}"
    )


def infer_tau_psi_geometry(  # noqa: C901
    config: Mapping[str, Any],
    reranker: Any | None = None,
) -> TauPsiGeometry:
    rc: dict[str, Any] = dict(config.get("reranker") or {})
    dc: dict[str, Any] = dict(config.get("data") or {})

    cls_name = ""
    if reranker is not None:
        cls_name = reranker.__class__.__name__
    if not cls_name:
        cls_name = str(rc.get("class") or "").strip()

    ki = _k_input_cap(dc)

    if cls_name == "IdentityReranker":
        if ki is not None:
            note = f"[identity] evaluation_list_length={ki} synthetic_scores | not a neural rerank context"
            return TauPsiGeometry(ki, note, candidate_list_length=ki)
        return TauPsiGeometry(
            None,
            "[identity] k_input unset — list follows first-stage run per query | synthetic_scores",
            candidate_list_length=None,
        )

    if cls_name == "MxbaiPointwise":
        bs = int(rc.get("batch_size", 32))
        note = _note_pointwise_CE(bs, ki, reranker_class_repr="MxbaiPointwise")
        return TauPsiGeometry(
            1,
            note,
            docs_per_score_forward=1,
            inference_batch_size=bs,
            candidate_list_length=ki,
        )

    bp_w = _batched_pw_forward_width(rc, cls_name)
    if bp_w is not None:
        label = cls_name or str(rc.get("class") or "?")
        note = _note_batched_pointwise(bp_w, ki, cls_label=label)
        inf_bs = rc.get("batch_size")
        inference_batch_size = _int_positive("reranker.batch_size", inf_bs) if inf_bs is not None else None
        return TauPsiGeometry(
            bp_w,
            note,
            docs_per_score_forward=bp_w,
            inference_batch_size=inference_batch_size,
            candidate_list_length=ki,
        )

    if cls_name == "JinaListwiseReranker":
        block_cap = int(rc.get("doc_block_cap", JINA_DOC_BLOCK_CAP_DEFAULT))
        raw_dpf = rc.get("docs_per_score_forward")
        if reranker is not None and getattr(reranker, "docs_per_score_forward", 0) > 0:
            dpf = int(reranker.docs_per_score_forward)
        elif raw_dpf is not None and int(raw_dpf) > 0:
            dpf = _int_positive("reranker.docs_per_score_forward", raw_dpf)
        else:
            dpf = None
        if dpf is not None and dpf > 0:
            note = _note_jina_chunked(dpf, ki, block_cap)
            return TauPsiGeometry(
                dpf,
                note,
                docs_per_score_forward=dpf,
                candidate_list_length=ki,
                upstream_doc_block_cap=block_cap,
            )
        note = _note_jina(ki, block_cap)
        return TauPsiGeometry(
            ki,
            note,
            candidate_list_length=ki,
            upstream_doc_block_cap=block_cap,
        )

    if cls_name == "RankZephyrReranker":
        raw_proto = (
            str(
                (getattr(reranker, "rankzephyr_protocol", None) if reranker is not None else None)
                or rc.get("rankzephyr_protocol")
                or "sliding",
            )
            .strip()
            .lower()
        )
        if raw_proto == "block_local_rrf":
            if reranker is not None:
                bs = int(reranker.block_size)
                rk = float(reranker.rrf_k)
            else:
                w = int(rc.get("window_size", 20))
                bs = int(rc.get("block_size") or w)
                rk = float(rc.get("rrf_k", 60))
            note = _note_rankzephyr_block_local_rrf(bs, rk, ki)
            return TauPsiGeometry(
                bs,
                note,
                rankzephyr_protocol=raw_proto,
                rrf_k=rk,
                candidate_list_length=ki,
            )

    if cls_name == "RankZephyrReranker":
        if reranker is not None:
            w = int(reranker.window_size)
            s = int(reranker.stride)
        else:
            w = int(rc.get("window_size", 20))
            s = int(rc.get("stride", 10))
        return TauPsiGeometry(
            w,
            _note_sliding_listwise(w, s, "RankZephyr"),
            window_size=w,
            stride=s,
        )

    if "window_size" in rc:
        w = int(rc["window_size"])
        s = int(rc.get("stride", rc.get("step", 10)))
        label = str(rc.get("class", "windowed"))
        return TauPsiGeometry(
            w,
            _note_sliding_listwise(w, s, label),
            window_size=w,
            stride=s,
        )

    if "batch_size" in rc and "window_size" not in rc:
        bs = _int_positive("reranker.batch_size", rc["batch_size"])
        rcr = repr(rc.get("class", "?"))
        note = _note_pointwise_CE(bs, ki, reranker_class_repr=str(rc.get("class", "?")))
        return TauPsiGeometry(
            1,
            note + f"; generic_fallback class={rcr}",
            docs_per_score_forward=1,
            inference_batch_size=bs,
            candidate_list_length=ki,
        )

    if ki is not None:
        who = repr(cls_name) if cls_name else "?"
        note = (
            f"[fallback] reranker class={who} k_input={ki} "
            "| add a dedicated tau_psi_geometry branch for W/S or CE fields"
        )
        return TauPsiGeometry(ki, note, candidate_list_length=ki)

    who = repr(cls_name) if cls_name else "'?'"
    return TauPsiGeometry(
        None,
        f"[fallback] reranker class={who}; no k_input/window — cannot infer B",
        candidate_list_length=None,
    )


def compact_tau_psi_geometry(geo: TauPsiGeometry) -> dict[str, Any]:
    """Flatten :class:`TauPsiGeometry` for ``psi_metrics.json`` top-level keys."""
    out: dict[str, Any] = {
        "tau_psi_geometry_schema": TAU_PSI_GEOMETRY_SCHEMA,
        "tau_psi_context_batch_size": geo.tau_psi_context_batch_size,
        "tau_psi_inference_note": geo.tau_psi_inference_note,
    }
    if geo.window_size is not None:
        out["tau_psi_window_size"] = geo.window_size
    if geo.stride is not None:
        out["tau_psi_stride"] = geo.stride
    if geo.docs_per_score_forward is not None:
        out["tau_psi_docs_per_score_forward"] = geo.docs_per_score_forward
    if geo.inference_batch_size is not None:
        out["tau_psi_inference_batch_size"] = geo.inference_batch_size
    if geo.candidate_list_length is not None:
        out["tau_psi_candidate_list_length"] = geo.candidate_list_length
    if geo.upstream_doc_block_cap is not None:
        out["tau_psi_upstream_doc_block_cap"] = geo.upstream_doc_block_cap
    if geo.rankzephyr_protocol is not None:
        out["tau_psi_rankzephyr_protocol"] = geo.rankzephyr_protocol
    if geo.rrf_k is not None:
        out["tau_psi_rrf_k"] = geo.rrf_k
    if geo.tau_psi_context_batch_size is not None:
        out["tau_psi_at_B_caption"] = f"τ-PSI@B={geo.tau_psi_context_batch_size}"
    return out


def strip_tau_psi_geometry_fields(d: dict[str, Any]) -> None:
    """Drop all τ-PSI geometry keys in-place (before re-applying inference)."""
    for k in TAU_PSI_GEOMETRY_KEYS:
        d.pop(k, None)


def merge_tau_psi_geometry_into_psi_metrics(
    psi_metrics: Mapping[str, Any],
    config: Mapping[str, Any],
    *,
    reranker: Any | None = None,
    overwrite: bool = False,
) -> tuple[str, dict[str, Any]]:
    """Populate τ-PSI geometry keys on a loaded ``psi_metrics.json``.

    Without ``overwrite``, an existing v2 schema with both the headline scalar
    and inference note is left unchanged.
    """
    current = dict(psi_metrics)
    has_v2 = current.get("tau_psi_geometry_schema") == TAU_PSI_GEOMETRY_SCHEMA
    has_note = "tau_psi_inference_note" in current and current.get("tau_psi_inference_note") is not None
    has_b = "tau_psi_context_batch_size" in current
    if has_v2 and has_note and has_b and not overwrite:
        return "skipped", current

    geo = infer_tau_psi_geometry(config, reranker=reranker)
    strip_tau_psi_geometry_fields(current)
    current.update(compact_tau_psi_geometry(geo))
    return "merged", current
