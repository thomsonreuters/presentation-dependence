"""Deterministic silver-label derivation and cohort operations."""

from __future__ import annotations

import json
import math
import os
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class DeriveStats:
    """Receipt for one lower-K silver derivation."""

    input_path: Path
    output_path: Path
    records: int
    qids: int
    k_out: int
    start_index: int
    score_mean: float
    score_min: float
    score_max: float


@dataclass(frozen=True)
class SplitResult:
    """Receipt for a deterministic train/held-out silver split."""

    source: Path
    training_path: Path
    heldout_path: Path
    heldout_qids_path: Path
    training_qids: tuple[str, ...]
    heldout_qids: tuple[str, ...]
    input_records: int
    training_records: int
    heldout_records: int
    reused_qid_list: bool


def default_derived_path(silver_in: Path, k_out: int, start_index: int) -> Path:
    """Return the default output path for a lower-K shard."""
    stem = silver_in.stem
    if k_out == 1:
        token = f"_k1_seed{start_index}"
    else:
        token = f"_k{k_out}_seed{start_index}to{start_index + k_out - 1}"
    stem = stem.replace("_k10", token, 1) if "_k10" in stem else stem + token
    return silver_in.with_name(f"{stem}{silver_in.suffix}")


def convert_bsc_record(  # noqa: C901
    record: dict[str, Any],
    *,
    start_index: int,
    k_out: int,
    line_no: int | None = None,
) -> dict[str, Any]:
    """Derive a lower-K label from one BSC score vector."""
    location = f"line {line_no}: " if line_no is not None else ""
    if not isinstance(record, dict):
        raise ValueError(f"{location}record must be a JSON object")
    for field in ("query_id", "doc_id"):
        if field not in record or record[field] is None or not str(record[field]):
            raise ValueError(f"{location}{field} must be a non-empty identifier")
    raw = record.get("score_raw_vector")
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"{location}score_raw_vector must be a non-empty list")
    end_index = start_index + k_out
    if start_index < 0 or k_out <= 0 or end_index > len(raw):
        raise ValueError(
            f"{location}requested raw slice [{start_index}:{end_index}] is outside score_raw_vector length {len(raw)}"
        )
    try:
        selected = [float(value) for value in raw[start_index:end_index]]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{location}raw slice [{start_index}:{end_index}] contains a non-numeric score") from exc
    if any(not math.isfinite(score) for score in selected):
        raise ValueError(f"{location}raw slice [{start_index}:{end_index}] contains a non-finite score")
    declared_k = record.get("k_perms")
    if declared_k is not None:
        try:
            declared_k_int = int(declared_k)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{location}k_perms must be an integer") from exc
        if declared_k_int != len(raw):
            raise ValueError(f"{location}k_perms={declared_k_int} does not match score_raw_vector length {len(raw)}")
    output = dict(record)
    output["score_continuous"] = sum(selected) / len(selected)
    output["score_raw_vector"] = selected
    output["k_perms"] = k_out
    # The source stores only the K-averaged P(g), not one vector per
    # presentation, so it cannot be sliced consistently with the raw
    # score vector; retaining it would violate E[P(g)] == score_continuous.
    if "score_grade_vector" in output:
        output["score_grade_vector"] = None
    return output


def derive_silver_shard(
    silver_in: Path,
    output_path: Path | None = None,
    *,
    k_out: int = 1,
    start_index: int = 0,
    force: bool = False,
) -> DeriveStats:
    """Stream one BSC JSONL shard into a lower-K silver shard."""
    if start_index < 0:
        raise ValueError("start_index must be non-negative")
    if k_out <= 0:
        raise ValueError("k_out must be positive")
    if not silver_in.is_file():
        raise FileNotFoundError(f"silver input not found: {silver_in}")
    destination = output_path or default_derived_path(silver_in, k_out, start_index)
    if destination.exists() and not force:
        raise FileExistsError(f"output exists, pass force=True to overwrite: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    record_count = 0
    qids: set[str] = set()
    scores: list[float] = []
    temp_path: Path | None = None
    try:
        with (
            silver_in.open("r", encoding="utf-8") as source,
            tempfile.NamedTemporaryFile(
                "w",
                encoding="utf-8",
                dir=destination.parent,
                prefix=f".{destination.name}.",
                suffix=".tmp",
                delete=False,
            ) as output,
        ):
            temp_path = Path(output.name)
            for line_no, line in enumerate(source, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"line {line_no}: {exc}") from exc
                converted = convert_bsc_record(
                    record,
                    start_index=start_index,
                    k_out=k_out,
                    line_no=line_no,
                )
                score = float(converted["score_continuous"])
                qids.add(str(converted["query_id"]))
                scores.append(score)
                output.write(json.dumps(converted, ensure_ascii=False) + "\n")
                record_count += 1
            output.flush()
            os.fsync(output.fileno())
        if not record_count:
            raise ValueError(f"no records found in {silver_in}")
        temp_path.replace(destination)
        temp_path = None
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
    return DeriveStats(
        input_path=silver_in,
        output_path=destination,
        records=record_count,
        qids=len(qids),
        k_out=k_out,
        start_index=start_index,
        score_mean=sum(scores) / len(scores),
        score_min=min(scores),
        score_max=max(scores),
    )


def read_qids(path: Path) -> list[str]:
    """Read a unique, ordered QID file."""
    if not path.is_file():
        raise FileNotFoundError(f"qid file not found: {path}")
    qids = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    duplicates = []
    seen: set[str] = set()
    for qid in qids:
        if qid in seen and qid not in duplicates:
            duplicates.append(qid)
        seen.add(qid)
    if duplicates:
        raise ValueError(f"{path} contains duplicate qids: {', '.join(duplicates[:10])}")
    return qids


def write_qids(path: Path, qids: list[str]) -> None:
    """Write one ordered QID per line."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as output:
            temp_path = Path(output.name)
            output.write("".join(f"{qid}\n" for qid in qids))
            output.flush()
            os.fsync(output.fileno())
        temp_path.replace(path)
        temp_path = None
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


def assert_qid_prefix(
    base: list[str],
    target: list[str],
    *,
    base_name: str,
    target_name: str,
) -> None:
    """Require `base` to be an exact prefix of `target`."""
    if len(base) > len(target):
        raise ValueError(
            f"base has {len(base)} qids but target has only {len(target)} qids "
            f"({base_name} cannot be a prefix of {target_name})"
        )
    if target[: len(base)] == base:
        return
    for index, (left, right) in enumerate(zip(base, target, strict=False)):
        if left != right:
            raise ValueError(
                f"{base_name} is not a prefix of {target_name}: first mismatch "
                f"at index {index}: base={left!r}, target={right!r}"
            )
    raise ValueError(f"{base_name} is not a prefix of {target_name}")


def qid_prefix_from_sample(sample_order: Path, target_count: int) -> list[str]:
    """Take a deterministic QID prefix from a sample order."""
    if target_count <= 0:
        raise ValueError(f"target_count must be positive, got {target_count}")
    sample = read_qids(sample_order)
    if len(sample) < target_count:
        raise ValueError(f"{sample_order} contains only {len(sample)} qids; cannot materialize {target_count}")
    return sample[:target_count]


def materialize_qid_delta(
    base_path: Path,
    target_path: Path,
    output_path: Path,
    *,
    sample_order: Path | None = None,
    target_count: int | None = None,
) -> list[str]:
    """Validate nested QID prefixes and write their non-overlapping delta."""
    base = read_qids(base_path)
    if target_path.is_file():
        target = read_qids(target_path)
        if sample_order is not None and target_count is not None:
            expected = qid_prefix_from_sample(sample_order, target_count)
            if target != expected:
                raise ValueError(f"{target_path} does not match the first {target_count} qids from {sample_order}")
    else:
        if sample_order is None or target_count is None:
            raise FileNotFoundError(
                f"{target_path} does not exist; sample_order and target_count are required to materialize it"
            )
        target = qid_prefix_from_sample(sample_order, target_count)
    assert_qid_prefix(
        base,
        target,
        base_name=str(base_path),
        target_name=str(target_path),
    )
    if not target_path.is_file():
        write_qids(target_path, target)
    delta = target[len(base) :]
    write_qids(output_path, delta)
    return delta


def split_silver_cohort(
    silver_in: Path,
    *,
    n_heldout: int,
    sample_order_path: Path,
    training_qids_path: Path,
    out_dir: Path,
    out_stem: str | None = None,
    reuse_qid_list: bool = False,
    force: bool = False,
) -> SplitResult:
    """Split silver records by a deterministic suffix of the training QIDs."""
    if n_heldout <= 0:
        raise ValueError("n_heldout must be positive")
    if not silver_in.is_file():
        raise FileNotFoundError(f"silver input not found: {silver_in}")
    keep_qids, heldout_qids = _cohort_qids(sample_order_path, training_qids_path, n_heldout)
    out_dir.mkdir(parents=True, exist_ok=True)
    heldout_qid_path = out_dir / f"qids_heldout_{n_heldout}.txt"
    by_qid, input_records = _load_silver_by_qid(silver_in)
    expected_qids = set(keep_qids) | set(heldout_qids)
    actual_qids = set(by_qid)
    missing_qids = expected_qids - actual_qids
    if missing_qids:
        raise ValueError(f"{len(missing_qids)} cohort qids are absent from {silver_in}; example={min(missing_qids)!r}")
    unexpected_qids = actual_qids - expected_qids
    if unexpected_qids:
        raise ValueError(
            f"{len(unexpected_qids)} qids in {silver_in} are outside the "
            f"declared training cohort; example={min(unexpected_qids)!r}"
        )
    stem = out_stem or silver_in.stem
    training_path = out_dir / f"{stem}_train_{len(keep_qids)}.jsonl"
    heldout_path = out_dir / f"{stem}_heldout_{n_heldout}.jsonl"
    for path in (training_path, heldout_path):
        if path.exists() and not force:
            raise FileExistsError(f"{path} exists; use force to overwrite")
    reused_qid_list = _materialize_heldout_qids(
        heldout_qid_path,
        heldout_qids,
        reuse_qid_list=reuse_qid_list,
        force=force,
    )
    training_records = _write_ordered_records(training_path, keep_qids, by_qid)
    heldout_records = _write_ordered_records(heldout_path, heldout_qids, by_qid)
    return SplitResult(
        source=silver_in,
        training_path=training_path,
        heldout_path=heldout_path,
        heldout_qids_path=heldout_qid_path,
        training_qids=tuple(keep_qids),
        heldout_qids=tuple(heldout_qids),
        input_records=input_records,
        training_records=training_records,
        heldout_records=heldout_records,
        reused_qid_list=reused_qid_list,
    )


def _cohort_qids(
    sample_order_path: Path,
    training_qids_path: Path,
    n_heldout: int,
) -> tuple[list[str], list[str]]:
    order = read_qids(sample_order_path)
    training_qids = read_qids(training_qids_path)
    n_training = len(training_qids)
    if order[:n_training] != training_qids:
        raise ValueError(
            f"{sample_order_path} prefix [:{n_training}] does not match {training_qids_path}; provenance is broken"
        )
    if n_heldout >= n_training:
        raise ValueError(f"n_heldout={n_heldout} >= training size {n_training}; nothing to train on")
    heldout_qids = order[n_training - n_heldout : n_training]
    keep_qids = order[: n_training - n_heldout]
    if set(heldout_qids) & set(keep_qids):
        raise ValueError("held-out and training QID slices overlap")
    if set(heldout_qids) | set(keep_qids) != set(training_qids):
        raise ValueError("held-out and training QID slices are not exhaustive")
    return keep_qids, heldout_qids


def _materialize_heldout_qids(
    path: Path,
    heldout_qids: list[str],
    *,
    reuse_qid_list: bool,
    force: bool,
) -> bool:
    if path.exists() and not reuse_qid_list and not force:
        raise FileExistsError(f"{path} exists; use reuse_qid_list or force")
    if path.exists() and reuse_qid_list and not force:
        if read_qids(path) != heldout_qids:
            raise ValueError(f"{path} differs from the freshly-computed held-out slice")
        return True
    write_qids(path, heldout_qids)
    return False


def _load_silver_by_qid(
    path: Path,
) -> tuple[dict[str, list[dict[str, Any]]], int]:
    by_qid: dict[str, list[dict[str, Any]]] = defaultdict(list)
    count = 0
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"line {line_no}: {exc}") from exc
        if not isinstance(record, dict):
            raise ValueError(f"line {line_no}: record must be a JSON object")
        if "query_id" not in record or record["query_id"] is None:
            raise ValueError(f"line {line_no}: query_id is required")
        by_qid[str(record["query_id"])].append(record)
        count += 1
    return by_qid, count


def _write_ordered_records(
    path: Path,
    qids: list[str],
    records_by_qid: dict[str, list[dict[str, Any]]],
) -> int:
    count = 0
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as output:
            temp_path = Path(output.name)
            for qid in qids:
                for record in records_by_qid.get(qid, ()):
                    output.write(json.dumps(record, ensure_ascii=False) + "\n")
                    count += 1
            output.flush()
            os.fsync(output.fileno())
        temp_path.replace(path)
        temp_path = None
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
    return count
