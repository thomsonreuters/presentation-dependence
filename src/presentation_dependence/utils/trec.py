"""Small helpers for TREC run files.

A TREC run file is whitespace-separated:

    <qid> Q0 <docid> <rank> <score> <tag>

`pytrec_eval.parse_run` + `pyserini.search` expect this layout exactly.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable


# Some qids contain `/`
# so `Path(per_query_dir) / qid` creates a spurious nested subdirectory.
# A reversible marker prevents nested writes and preserves exact QID round trips.
# `__SLASH__` is chosen to be visibly distinct from anything a real qid
# is expected to contain.
_QID_SLASH_MARKER = "__SLASH__"


def qid_to_dirname(qid: str) -> str:
    """Return a filesystem-safe directory name for `qid`.

    Only `/` is currently escaped. Extend this
    function and `dirname_to_qid` together if other unsafe characters occur.
    """
    return qid.replace("/", _QID_SLASH_MARKER)


def dirname_to_qid(dirname: str) -> str:
    """Return the original QID for a safe directory name.

    Must round-trip exactly so
    EvalManager can pass the reconstructed qid to `pytrec_eval` (which
    looks up the qid against qrels — qrels use the original, with `/`).
    """
    return dirname.replace(_QID_SLASH_MARKER, "/")


def read_qrels(path: str | Path) -> dict[str, dict[str, int]]:
    """Load a standard `qid 0 docid relevance` qrels file."""
    result: dict[str, dict[str, int]] = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) >= 4:
            result.setdefault(parts[0], {})[parts[2]] = int(parts[3])
    return result


def _scores_strictly_decreasing(scores: list[float | None]) -> bool:
    """Return True when every score is present and strictly decreases with rank."""
    prev: float | None = None
    for score in scores:
        if score is None:
            return False
        value = float(score)
        if prev is not None and value >= prev:
            return False
        prev = value
    return True


def write_trec_run(
    path: str | Path,
    qid: str,
    ranked_docs: Iterable[dict],
    tag: str = "presentation_dependence",
) -> None:
    """Write a single-query TREC run file.

    Scores decrease with rank so `ndcg_cut_*` and others order docs correctly
    irrespective of the `score` field's absolute magnitude. Generative-listwise
    rerankers that emit no native score use `score = N - rank_index`;
    pointwise rerankers overwrite
    with the real scalar score when it is strictly decreasing.

    When normalized pointwise grades tie (common when a model saturates at
    score_max), pytrec_eval re-sorts tied docs by docid and can permute the
    reranker's list order — use rank-decay scores instead.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    docs = list(ranked_docs)
    n = len(docs)
    raw_scores = [doc.get("score") for doc in docs]
    use_rank_scores = not _scores_strictly_decreasing(raw_scores)
    lines = []
    for rank_idx, doc in enumerate(docs):
        pid = doc["pid"]
        if use_rank_scores:
            score = float(n - rank_idx)
        else:
            score = float(raw_scores[rank_idx])  # type: ignore[arg-type]
        lines.append(f"{qid} Q0 {pid} {rank_idx + 1} {score} {tag}")

    path.write_text("\n".join(lines))


def ensure_sorted_run_file(run_path: str | Path, logger=None) -> Path:
    """Ensure a run file is sorted by (qid, rank).

    Returns a `Path` and does not mutate config.
    """
    run_path = Path(run_path)
    lines = run_path.read_text().splitlines(keepends=True)
    sorted_lines = sorted(lines, key=lambda x: (x.split()[0], int(x.split()[3])))
    if lines == sorted_lines:
        return run_path

    sorted_path = run_path.with_name(run_path.stem + "_sorted" + run_path.suffix)
    sorted_path.write_text("".join(sorted_lines))
    if logger is not None:
        logger.warning(
            "Run file %s was not sorted by (qid, rank). Sorted version written to %s",
            run_path,
            sorted_path,
        )
    return sorted_path
