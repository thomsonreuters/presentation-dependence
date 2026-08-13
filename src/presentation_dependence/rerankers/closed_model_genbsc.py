"""Closed-model (GPT-5.4 etc.) generated-BSC reranker baseline.

Closed-model analogue of :mod:`presentation_dependence.rerankers.qwen3_instruct_grade`.
Instead of reading continuous grade logits from a local model, this wrapper
sends each query's candidate set to an OpenAI-compatible chat endpoint in
batched mode (B = ``subset_size`` docs per prompt, integer grades in
``[score_min, score_max]``) and normalizes averaged grades to ``[0, 1]``. The
prompt and answer skeleton are the shared ones, so the row stays comparable to
the open-weight scorers beside it.

Eval protocol
------------------------------------
We follow the **open-model two-step
pattern** in one run dir: ``run_experiment`` (identity-order base →
``metrics.json``) then ``run_psi --run-dir`` (permutations → ``psi/``),
mirroring the open-weight batched-grade DL19/DL20 rows.
Korikov-style K-shot self-consistency is measured by the harness: shuffle the
candidate list K times, score each ordering once, mean-aggregate per-doc
grades. Set ``scoring.runs: 1`` so the harness, not the client, owns
list-order perturbation; at one run the client scores the identity order and
does not shuffle, whatever the seeds say. Headline nDCG@10 is **SC
K=10** from ``psi/sc_metrics.json``; base identity-order nDCG is in
``metrics.json``; τ-PSI and related robustness metrics come from ``psi/``.

Silver / labeling protocol (separate)
-------------------------------------
The GPT-5.4 silver pipeline uses ``runs: 10`` with explicit ``run_seeds``
(BSC shuffles inside the client).

Output scale convention
-----------------------
``scores_init_order`` follows the standard reranker contract: per-doc scalars
in ``[0, 1]`` (averaged generated grade ``∈ [score_min, score_max]`` divided by
``score_max``). Same ``E[grade] / score_max`` shape as
:class:`Qwen3InstructGradeReranker`, so this row is directly comparable.

This reranker calls a hosted API model, so it runs against provider credentials
(``OPENAI_API_KEY``, and ``OPENAI_BASE_URL`` for a non-default endpoint) rather
than on a GPU. Configs omit the ``execution:`` / ``vllm_settings`` blocks the
open-weight grade rerankers carry.
"""

from __future__ import annotations

import shutil
import tempfile
import time
from pathlib import Path
from typing import Any

from presentation_dependence.rerankers._rank_result import scores_to_rank_result
from presentation_dependence.rerankers.base import Passage, Query, RankResult, Reranker
from presentation_dependence.silver_data.client import SilverRequest
from presentation_dependence.silver_data.prompts import get_prompt_template
from presentation_dependence.utils.setup_logging import setup_logging


_DEFAULT_PROMPT_TEMPLATE_ID = "pathc_grade_v1"

# Defaults mirror the GPT-5.4 silver labeling config
# (configs/silver/silver-msmarco-30kx100-gpt54.yaml). Eval configs override
# with ``runs: 1`` so the harness supplies list-order diversity; silver
# configs keep ``runs: 10``.
_DEFAULT_SCORING: dict[str, Any] = {
    "subset_size": 20,
    "runs": 10,
    "run_seeds": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    "score_min": 0,
    "score_max": 3,
    # 1536 gives headroom for one grade line per document at B=20.
    "max_tokens": 1536,
}


class ClosedModelGenBscReranker(Reranker):
    """GPT-5.4 (and other closed API models) prompted as a generated-BSC scorer.

    Eval-only baseline. Emits per-doc continuous scores in ``[0, 1]`` from
    averaged generated integer grades. Separate from the open-weight grade
    rerankers because it reads generated text rather than logits at the grade
    positions, so it shares their prompt but none of their readout. See the
    module docstring for the eval and silver labeling procedures.
    """

    paradigm = "batched_pointwise"

    def __init__(self, config: dict):
        super().__init__(config)
        self.logger = setup_logging(self.__class__.__name__, config)

        rc = self.reranker_config
        self.model_id = str(rc.get("model_id") or rc.get("model_name") or "").strip()
        if not self.model_id:
            raise ValueError("reranker.model_id is required (e.g. 'gpt-5.4')")

        self.prompt_template_id = str(rc.get("prompt_template_id", _DEFAULT_PROMPT_TEMPLATE_ID)).strip()
        self.instruction = rc.get("instruction")
        self.grade_rubric_id = rc.get("grade_rubric_id")
        self.max_doc_chars = rc.get("max_doc_chars")
        # Validate the template exists now, so a typo fails loud at
        # construction rather than on the first query.
        get_prompt_template(self.prompt_template_id)

        overrides = dict(rc.get("scoring") or {})
        self.scoring = {**_DEFAULT_SCORING, **overrides}
        # `runs` and `run_seeds` are a matched pair in the defaults, so a config
        # that sets one must not inherit the other: an eval config asking for
        # `runs: 1` would otherwise arrive carrying ten default seeds and be
        # rejected as inconsistent.
        if "runs" in overrides and "run_seeds" not in overrides:
            self.scoring["run_seeds"] = None

        self.score_min = float(self.scoring.get("score_min", 0))
        self.score_max = float(self.scoring.get("score_max", 3))
        if self.score_max <= self.score_min:
            raise ValueError(f"reranker.scoring.score_max ({self.score_max}) must be > score_min ({self.score_min})")

        endpoint = rc.get("endpoint") or {}
        if not isinstance(endpoint, dict):
            raise ValueError("reranker.endpoint must be a mapping when provided")
        self.base_url = endpoint.get("base_url")

        # Raw provider replies are persisted here for provenance, then removed
        # in close().
        self._raw_dir = Path(tempfile.mkdtemp(prefix="closed_model_genbsc_"))
        self._client = self._build_client()

        self.logger.info(
            "ClosedModelGenBscReranker: model_id=%s template=%s instruction=%r max_doc_chars=%s "
            "runs=%s subset_size=%s score=[%s,%s] endpoint=%s",
            self.model_id,
            self.prompt_template_id,
            self.scoring.get("instruction"),
            self.scoring.get("max_doc_chars"),
            self.scoring.get("runs"),
            self.scoring.get("subset_size"),
            self.score_min,
            self.score_max,
            self.base_url or "default endpoint",
        )

    def _build_client(self):
        """Construct the scoring client. Split out so tests can inject a double.

        Imported here rather than at module scope to break a cycle:
        ``silver_data.openai_client`` imports ``rerankers.grade_rubrics``, which
        initializes ``rerankers/__init__`` and so the registry, which imports
        this module. At module scope the import lands back in a half-initialized
        ``openai_client``, and ``import presentation_dependence.silver_data`` before
        ``presentation_dependence.rerankers`` raises ImportError.
        """
        from presentation_dependence.silver_data.openai_client import OpenAICompatibleClient

        return OpenAICompatibleClient(
            model_id=self.model_id,
            base_url=self.base_url,
            subset_size=int(self.scoring.get("subset_size", 20)),
            runs=int(self.scoring.get("runs", 10)),
            run_seeds=self.scoring.get("run_seeds"),
            score_min=self.score_min,
            score_max=self.score_max,
            max_tokens=int(self.scoring.get("max_tokens", 1536)),
            instruction=self.instruction,
            grade_rubric_id=self.grade_rubric_id,
            max_doc_chars=self.max_doc_chars,
            impute_missing_doc_scores=True,
        )

    def rank(self, query: Query, passages: list[Passage]) -> RankResult:
        if not passages:
            return {
                "top_k_psgs": [],
                "scores_init_order": [],
                "score_raw_vector_init_order": [],
                "prompting_runtimes": [0.0],
                "paradigm": self.paradigm,
            }

        qid = str(query["qid"])
        requests = [
            SilverRequest(
                call_id=f"{qid}::{passage['pid']}",
                query_id=qid,
                doc_id=str(passage["pid"]),
                query=str(query["text"]),
                # The client renders the shared grade prompt from the document
                # text; the pre-rendered ``prompt`` is unused on this path.
                document=str(passage["text"]),
                prompt="",
                prompt_template_id=self.prompt_template_id,
                teacher_model_id=self.model_id,
                params=dict(self.scoring),
            )
            for passage in passages
        ]

        t0 = time.perf_counter()
        responses_by_doc: dict[str, Any] = {}
        for request, response in self._client.submit_query_batches(requests, self._raw_dir):
            responses_by_doc[request.doc_id] = response
        elapsed = time.perf_counter() - t0

        scores: list[float] = []
        raw_vectors: list[list[float] | None] = []
        for passage in passages:
            doc_id = str(passage["pid"])
            response = responses_by_doc.get(doc_id)
            if response is None or not getattr(response, "success", False) or response.score_raw is None:
                err = getattr(response, "error", None) if response is not None else "no response"
                # Fail loud: a missing/failed per-doc score would silently
                # degrade nDCG. ExperimentManager marks the qid failed and
                # refuses to aggregate a partial run.
                raise RuntimeError(
                    f"{self.__class__.__name__}: missing/failed score for qid={qid} doc_id={doc_id}: {err}"
                )
            try:
                avg_grade = float(response.score_raw)
            except (TypeError, ValueError) as exc:
                raise RuntimeError(
                    f"{self.__class__.__name__}: non-numeric score_raw={response.score_raw!r} "
                    f"for qid={qid} doc_id={doc_id}"
                ) from exc
            scores.append(avg_grade / self.score_max)
            raw_vectors.append(getattr(response, "score_raw_vector", None))

        result = scores_to_rank_result(scores, passages, elapsed, self.paradigm, model_name=self.__class__.__name__)
        # Per-doc K-grade vectors (one int per shuffled BSC pass), aligned to
        # the input order. Free input-order robustness diagnostic; ignored by
        # EvalManager but persisted in detailed_results.json.
        result["score_raw_vector_init_order"] = raw_vectors
        return result

    def close(self) -> None:
        """Release the client and remove the temp output dir."""
        client = getattr(self, "_client", None)
        if client is not None:
            try:
                client.shutdown()
            except Exception:  # best-effort teardown
                self.logger.warning("client shutdown raised during close()", exc_info=True)
        raw_dir = getattr(self, "_raw_dir", None)
        if raw_dir is not None:
            shutil.rmtree(raw_dir, ignore_errors=True)

    def __del__(self):  # best-effort; GC ordering is not guaranteed
        try:
            self.close()
        except Exception:
            pass
