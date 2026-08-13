"""Dataloaders for first-stage retrieved runs.

Run-file sorting lives in ``presentation_dependence.utils.trec`` so callers can use it
without constructing a loader.

A loader converts a TREC run into the query and passage objects consumed by a
reranker. Pyserini is the primary backend and provides prebuilt BM25 indexes for
MS MARCO, TREC DL, and BEIR, matching the stack used by Korikov et al.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from presentation_dependence.utils.pyserini_index import pyserini_import, require_prebuilt_index
from presentation_dependence.utils.setup_logging import setup_logging
from presentation_dependence.utils.trec import ensure_sorted_run_file as _ensure_sorted


def _parse_qrels_judged_qids(qrels_path: str | Path) -> set[str]:
    """Return the set of qids that have at least one row in a TREC qrels file.

    TREC format is whitespace-separated ``qid 0 pid relevance`` per line.
    We keep every qid that appears, regardless of whether any of its rows
    have ``relevance > 0``. Qids with all-zero judgments still appear in the
    judgment file and count as judged under pytrec_eval's
    ``RelevanceEvaluator``.
    """
    judged: set[str] = set()
    with open(qrels_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            judged.add(parts[0])
    return judged


class BaseLoader(ABC):
    def __init__(self, config: dict):
        self.config = config
        self.data_config = config.get("data", {})
        self.eval_config = config.get("eval", {})
        self.logger = setup_logging(self.__class__.__name__, self.config)
        self.logger.info("Initializing...")
        self.topics: Any = None
        self.index: Any = None

        self._judged_qids: set[str] | None = self._init_qrels_filter()

    def _init_qrels_filter(self) -> set[str] | None:
        """Populate the judged-qids set when ``eval.strict_qrels_filter`` is
        enabled and a qrels file is available; otherwise return None.
        """
        if not self.eval_config.get("strict_qrels_filter", True):
            self.logger.info(
                "strict_qrels_filter=false: reranker will run on all qids in the run file, even those without qrels."
            )
            return None
        qrels_path = self.eval_config.get("qrels_path")
        if not qrels_path:
            return None
        qrels_path = Path(qrels_path)
        if not qrels_path.exists():
            self.logger.warning(
                "strict_qrels_filter=true but eval.qrels_path=%s does not exist; "
                "skipping filter. Reranker will run on every qid.",
                qrels_path,
            )
            return None
        judged = _parse_qrels_judged_qids(qrels_path)
        self.logger.info(
            "Loaded %d judged qids from %s; unjudged qids will be skipped before reranking.",
            len(judged),
            qrels_path,
        )
        return judged

    def _apply_qrels_filter(self, queries: list[dict]) -> list[dict]:
        """Drop queries whose qid is not in ``self._judged_qids``.

        No-op when the filter is disabled (``self._judged_qids is None``).
        The log reports the reduction. For example, the DL20 fixture contains
        200 topics but only 54 judged qids.
        """
        if self._judged_qids is None:
            return queries
        before = len(queries)
        kept = [q for q in queries if str(q["qid"]) in self._judged_qids]
        dropped = before - len(kept)
        if dropped > 0:
            self.logger.info(
                "qrels filter: %d/%d qids kept (%d unjudged qids dropped).",
                len(kept),
                before,
                dropped,
            )
        return kept

    def ensure_sorted_run_file(self, run_path: str) -> str:
        return str(_ensure_sorted(run_path, logger=self.logger))

    @abstractmethod
    def get_qs_from_run(self, run_path: str) -> list[dict]:
        """Return `[{"qid": ..., "text": ...}, ...]` for all queries in run_path.

        Subclasses should call ``self._apply_qrels_filter(queries)`` before
        returning so the loader-level filter kicks in uniformly. See
        ``BaseLoader`` docstring for when and why.
        """

    @abstractmethod
    def get_psgs_from_run(self, run_path: str, qid: str) -> list[dict] | None:
        """Return `[{"pid": ..., "text": ...}, ...]` for the given qid."""


class PyseriniLoader(BaseLoader):
    r"""Reads queries from pyserini topics (or a local topics.tsv fallback)
    and passage text from a pyserini prebuilt index.

    Config (under `data:`):
        topics: str           -- pyserini topics name (e.g. dl19-passage).
        topics_tsv: str       -- OPTIONAL. Path to a local `qid\\tquery`
                                 TSV; takes precedence over pyserini's
                                 bundled registry when set. Used when
                                 the pyserini topic name has drifted or
                                 we've vendored topics into data/.
        index: str            -- pyserini prebuilt index name.
        run_path: str         -- first-stage run file (TREC format).
    """

    def __init__(self, config: dict):
        super().__init__(config)
        # Lazy import so the package imports on Mac without Java/pyserini set up.
        with pyserini_import():
            from pyserini.search.lucene import LuceneSearcher  # type: ignore

        self.topics_name = self.data_config.get("topics")
        self.topics_tsv = self.data_config.get("topics_tsv")
        self.index_name = self.data_config.get("index")
        self.searcher = require_prebuilt_index(LuceneSearcher.from_prebuilt_index(self.index_name), self.index_name)

    def _get_topics(self) -> dict:
        """Return `{qid: {"title": query_str}}` in pyserini-compatible shape."""
        if self.topics_tsv:
            return self._load_topics_tsv(self.topics_tsv)

        from pyserini.search._base import get_topics  # type: ignore

        return get_topics(self.topics_name)

    @staticmethod
    def _load_topics_tsv(path: str) -> dict:
        topics: dict = {}
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\n")
                if not line:
                    continue
                parts = line.split("\t", 1)
                if len(parts) != 2:
                    continue
                qid_str, query = parts
                try:
                    qid = int(qid_str)
                except ValueError:
                    qid = qid_str  # keep as-is for non-numeric BEIR qids
                topics[qid] = {"title": query}
        return topics

    def get_qs_from_run(self, run_path: str) -> list[dict]:
        topics = self._get_topics()
        query_list: list[dict] = []
        seen_qids: set = set()
        with open(run_path, "r") as f:
            for line in f:
                qid_str = line.strip().split()[0]
                # pyserini topic registries return int-keyed dicts for
                # TREC DL / MSMARCO, but BEIR tasks can be string-keyed
                # (e.g. "test-0"). Try int first, fall back to str.
                try:
                    qid_key: int | str = int(qid_str)
                except ValueError:
                    qid_key = qid_str
                if qid_key in seen_qids:
                    continue
                seen_qids.add(qid_key)
                if qid_key not in topics:
                    self.logger.warning(
                        "qid=%s not found in topics registry (%s); skipping",
                        qid_key,
                        self.topics_tsv or self.topics_name,
                    )
                    continue
                query_list.append({"qid": qid_key, "text": topics[qid_key]["title"]})
        return self._apply_qrels_filter(query_list)

    def get_psgs_from_run(self, run_path: str, qid: str) -> list[dict] | None:
        qid_str = str(qid)
        passages: list[dict] = []
        found_qid = False
        with open(run_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                line_qid, _q0, pid = parts[0], parts[1], parts[2]
                if line_qid == qid_str:
                    found_qid = True
                    doc = self.searcher.doc(pid)
                    if doc is None:
                        self.logger.warning("Missing document %s in index %s", pid, self.index_name)
                        continue
                    raw = doc.raw()
                    try:
                        parsed = json.loads(raw)
                    except json.JSONDecodeError:
                        # Some pyserini indexes return raw text, not JSON.
                        parsed = {"contents": raw}

                    if "contents" in parsed:
                        text = parsed["contents"]
                    elif "text" in parsed:
                        text = parsed["text"]
                        if "title" in parsed:
                            text = f"{parsed['title']} {text}"
                    else:
                        self.logger.error("Document %s has neither 'contents' nor 'text' field", pid)
                        return None
                    passages.append({"pid": pid, "text": text})
                elif found_qid:
                    # Run files group each qid, but their qid ordering is not
                    # uniform: numeric runs often use 1,2,...,10 while generic
                    # sorting produces 1,10,2,..., and BEIR uses string qids.
                    # Once the target block ends we can stop without making an
                    # invalid cross-type ordering comparison.
                    break
        return passages


class FixtureLoader(BaseLoader):
    """In-memory loader for tests and local debugging.

    Reads a JSONL fixture where each line represents one query and its
    candidate passages:

        {"qid": "q1", "query": "what is x?",
         "passages": [{"pid": "p1", "text": "..."}, {"pid": "p2", "text": "..."}]}

    Point the experiment config at it via:

        data:
          dataloader_class: FixtureLoader
          run_path: path/to/fixture.jsonl      # reused as the fixture path
          unsorted_run_file: true              # optional; not needed — we override ensure_sorted

    ``PyseriniLoader`` requires Java and a prebuilt MS MARCO index of about
    1.5 GB. ``FixtureLoader`` supports the ExperimentManager/EvalManager
    round-trip in CI and on machines without a JDK.
    """

    def __init__(self, config: dict):
        super().__init__(config)
        self.fixture_path = self.data_config.get("run_path") or self.data_config.get("fixture_path")
        if not self.fixture_path:
            raise ValueError(
                "FixtureLoader requires data.run_path (or data.fixture_path) to point at a JSONL fixture file."
            )
        self._data: dict[str, dict] = self._load_fixture(self.fixture_path)

    @staticmethod
    def _load_fixture(path: str) -> dict[str, dict]:
        out: dict[str, dict] = {}
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                out[str(rec["qid"])] = rec
        return out

    def ensure_sorted_run_file(self, run_path: str) -> str:
        # FixtureLoader reads JSONL rather than TREC, so sorting does not apply.
        return run_path

    def get_qs_from_run(self, run_path: str) -> list[dict]:
        del run_path
        queries = [{"qid": rec["qid"], "text": rec["query"]} for rec in self._data.values()]
        return self._apply_qrels_filter(queries)

    def get_psgs_from_run(self, run_path: str, qid: str) -> list[dict] | None:
        del run_path
        rec = self._data.get(str(qid))
        if rec is None:
            return None
        return [{"pid": str(p["pid"]), "text": p["text"]} for p in rec["passages"]]


LOADER_CLASSES: dict[str, type[BaseLoader]] = {
    "PyseriniLoader": PyseriniLoader,
    "FixtureLoader": FixtureLoader,
}
