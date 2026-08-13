"""Silver-label record schema + JSONL/manifest writers.

A "silver label" is a continuous teacher-side relevance score for a single
``(query, document)`` tuple. The active writer produces K-shot
batched-self-consistency (BSC) labels by averaging the per-(q, d) continuous
readout over ``K`` candidate-set permutations. The reader remains compatible
with older records carrying native-protocol identifiers and ``score_raw_native``.

On-disk layout
--------------
::

    runs/self-distill/<experiment_id>/<timestamp>/
    ├── silver/
    │   ├── manifest.json           # run-level metadata (this module)
    │   ├── silver_labels.jsonl     # one record per (q, d) (this module)
    │   ├── raw_responses/          # per-(q, d) raw model output (caller)
    │   └── logs/teacher.log        # set up by the driver
    ├── resolved_config.yaml        # frozen config + git SHA + ts
    └── teacher_run.log             # top-level driver log

Both files are append-friendly: a partial run can be resumed by checking
existing ``query_id`` / ``doc_id`` pairs in ``silver_labels.jsonl`` before
appending. The driver enforces "no partial silver-label artefacts", matching
``PsiExperimentRunner`` (refuses to publish an aggregate from a
partial run; see ``runner_setup.partial_run_error``).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Literal


TeacherProtocol = Literal[
    "k_shot_bsc",
    "native_pw",
    "native_pairwise",
    "native_lw_chunked",
]
"""Known on-disk ``teacher_protocol`` values.

Only ``k_shot_bsc`` is actively produced. Native values remain accepted so
existing JSONL artifacts can still be read and rewritten without loss.
"""


@dataclass(slots=True)
class SilverLabel:
    """One ``(query, document)`` teacher-side continuous label.

    The fields below define the on-disk schema consumed by the student-SFT
    loader.
    """

    query_id: str
    """Query identifier — same string the dataloader emits in
    ``query["qid"]``. Stringified to keep the JSON key type stable across
    pyserini int qids (TREC DL) and BEIR string qids."""

    doc_id: str
    """Document/passage identifier as it appears in the first-stage run file."""

    score_continuous: float
    """Teacher's continuous label on the student-SFT target scale."""

    teacher_model_id: str
    """HuggingFace repo id of the teacher model
    (e.g. ``Qwen/Qwen3-4B-Instruct-2507``). Pinned for reproducibility."""

    teacher_protocol: TeacherProtocol
    """Which teacher inference path produced this label. See
    :data:`TeacherProtocol`."""

    prompt_template_id: str
    """Stable identifier for the prompt template used at inference time
    (e.g. ``grade_int_v1``). Lets us version prompts independently of the
    code path."""

    timestamp: str
    """ISO-8601 UTC timestamp when this silver record was *written*. Used to
    debug stale partial files when a run resumes."""

    score_raw_vector: list[float] | None = None
    """Raw ``K`` continuous scores before averaging, when available."""

    score_grade_vector: list[float] | None = None
    """K-averaged grade probability vector ``P(g)`` when available."""

    score_raw_native: float | None = None
    """Legacy native-protocol output retained for on-disk compatibility."""

    k_perms: int | None = None
    """Number of permutations averaged, when applicable."""

    extra: dict[str, Any] = field(default_factory=dict)
    """Optional per-record extra fields (e.g. permutation seeds, raw chunk
    metadata). Kept under one explicit key so the canonical fields above
    can't drift via ad-hoc keys at the record top level."""

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-serialisable record shape."""
        d = asdict(self)
        if not d["extra"]:
            d.pop("extra")
        if d["score_raw_native"] is None:
            d.pop("score_raw_native")
        return d


def write_silver_jsonl(
    path: str | Path,
    records: Iterable[SilverLabel],
    *,
    append: bool = False,
) -> int:
    """Write silver labels to a JSONL file, one record per line.

    Parameters
    ----------
    path : str | Path
        Destination file. Parent directory is created if missing.
    records : Iterable[SilverLabel]
        Records to write. Order is preserved.
    append : bool, default ``False``
        When True, append to an existing file instead of truncating. Useful
        for resumed runs that re-emit only the queries that were missing
        from the previous attempt.

    Returns:
    -------
    int
        Number of records written.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"
    n = 0
    with open(p, mode, encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec.to_dict(), ensure_ascii=False))
            f.write("\n")
            n += 1
    return n


def read_silver_jsonl(path: str | Path) -> list[SilverLabel]:
    """Read silver labels back from a JSONL file. Mainly useful for tests
    and the partial-run resume path the driver will hook into next commit.
    """
    p = Path(path)
    out: list[SilverLabel] = []
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            extra = d.pop("extra", {}) or {}
            out.append(SilverLabel(**d, extra=extra))
    return out


def write_manifest(path: str | Path, manifest: dict[str, Any]) -> Path:
    """Write the run-level ``manifest.json``.

    The manifest wraps the record-level schema and keeps the ``runs/.../silver/``
    directory self-describing without forcing readers to scan the whole
    JSONL to learn the teacher / dataset / K. Required keys checked at
    write time:

    - ``experiment_id``
    - ``teacher_model_id``
    - ``teacher_protocol``
    - ``prompt_template_id``
    - ``k_perms``
    - ``n_queries``
    - ``n_documents`` (per-query first-stage truncation)
    - ``timestamp_utc``
    """
    required = (
        "experiment_id",
        "teacher_model_id",
        "teacher_protocol",
        "prompt_template_id",
        "n_queries",
        "n_documents",
        "timestamp_utc",
    )
    missing = [k for k in required if k not in manifest]
    if missing:
        raise ValueError(f"manifest is missing required keys: {missing}")
    if "k_perms" not in manifest:
        manifest = {**manifest, "k_perms": None}

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False, sort_keys=False)
    return p


def silver_run_dir(
    runs_root: str | Path,
    experiment_id: str,
    timestamp: str,
) -> Path:
    """Return the canonical run directory for a self-distill teacher pass.

    Layout: ``<runs_root>/self-distill/<experiment_id>/<timestamp>/``. This
    intentionally namespaces self-distill runs under their own ``self-distill``
    subdirectory rather than reusing the top-level ``runs/<exp_id>/<ts>/`` slot
    used by ExperimentManager / PsiExperimentRunner — keeps the run-listing
    tools in scripts/ from confusing teacher passes with eval runs.
    """
    return Path(runs_root) / "self-distill" / experiment_id / timestamp
