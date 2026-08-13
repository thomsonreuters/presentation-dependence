"""CapCal-style content-free calibration wrapper.

Reimplements Lv et al., "Learning from Emptiness: De-biasing Listwise Rerankers
with Content-Agnostic Probability Calibration", ACL 2026
(arXiv:2604.10150). No upstream code is used; see THIRD_PARTY_NOTICES.md.

CapCal is a training-free, inference-time debiaser: estimate a per-slot
content-free positional prior, then subtract excess prior mass from real
outputs. Lv et al. perform this calibration in probability space:

    S_i = P_i(real) - alpha_k * (P_i(empty) - 1 / |C_k|)

For this repo's expected-grade scorers we apply the same decomposition to the
per-slot grade distribution exposed as ``grade_probabilities_init_order``:
``p_cal(g) = p_real(g) - alpha * (p_probe(g) - uniform_g)``, followed by a
minimal non-negativity projection so the exported grade vector remains a valid
probability distribution. Rerankers that emit only scalar scores can opt into a
weaker centered-score fallback, but the primary CapCal baseline should use
grade-probability outputs from a batched-PW expected-grade base.
"""

from __future__ import annotations

import math
from typing import Sequence

from presentation_dependence.rerankers._rank_result import scores_to_rank_result
from presentation_dependence.rerankers.base import Passage, Query, RankResult, Reranker

_EPS_DEFAULT = 1e-8


def _shannon_entropy(probs: Sequence[float], *, eps: float = _EPS_DEFAULT) -> float:
    if len(probs) <= 1:
        return 0.0
    total = sum(max(float(p), 0.0) for p in probs)
    if total <= 0.0:
        return 0.0
    norm = [max(float(p), 0.0) / total for p in probs]
    return -sum(p * math.log(max(p, eps)) for p in norm)


def _project_probability_vector(values: Sequence[float], *, eps: float = _EPS_DEFAULT) -> list[float]:
    clipped = [max(float(v), eps) for v in values]
    total = sum(clipped)
    if total <= 0.0 or not math.isfinite(total):
        n = len(clipped)
        return [1.0 / n] * n
    return [v / total for v in clipped]


def calibrate_grade_probabilities(
    real_probs: Sequence[float],
    prior_probs: Sequence[float],
    *,
    strength: float = 1.0,
    eps: float = _EPS_DEFAULT,
    entropy_adaptive: bool = True,
) -> list[float]:
    """Apply CapCal's probability-space prior decomposition to one slot."""
    if len(real_probs) != len(prior_probs):
        raise ValueError(f"real/prior grade vector length mismatch: {len(real_probs)} vs {len(prior_probs)}")
    if not real_probs:
        return []
    alpha = float(strength)
    if entropy_adaptive:
        alpha *= _shannon_entropy(real_probs, eps=eps)
    uniform = 1.0 / float(len(real_probs))
    adjusted = [float(p) - alpha * (float(q) - uniform) for p, q in zip(real_probs, prior_probs, strict=True)]
    return _project_probability_vector(adjusted, eps=eps)


def expected_grade_score(probs: Sequence[float]) -> float:
    """Map a grade probability vector to the repo's normalized [0, 1] score."""
    if not probs:
        return 0.0
    max_grade = max(len(probs) - 1, 1)
    return sum(float(i) * float(p) for i, p in enumerate(probs)) / float(max_grade)


class CapCalReranker(Reranker):
    """Inference-time content-free calibration around an existing reranker.

    Config shape::

        reranker:
          class: CapCalReranker
          base_reranker:
            class: Qwen3InstructGradeReranker
            ...
          docs_per_score_forward: 20        # optional; defaults from base
          capcal:
            strength: 1.0          # beta in Lv et al.
            entropy_adaptive: true
            require_grade_probabilities: true
            probe_query_texts: null  # default: use the real query, per Lv et al.
            probe_passage_text: ""

    The wrapper preserves the base reranker's paradigm and scalar contract, but
    rewrites ``scores_init_order`` and, when present, calibrated
    ``grade_probabilities_init_order`` before sorting.
    """

    paradigm = "batched_pointwise"
    supports_query_batching = False
    supports_render_variants = False
    supports_scale_variants = False

    def __init__(self, config: dict, *, base_reranker: Reranker | None = None):  # noqa: C901
        super().__init__(config)
        rc = self.reranker_config
        capcal_cfg = rc.get("capcal") or {}
        if not isinstance(capcal_cfg, dict):
            raise ValueError("reranker.capcal must be a mapping when provided")
        self.capcal_config = dict(capcal_cfg)
        self.strength = float(self.capcal_config.get("strength", 1.0))
        if self.strength < 0.0:
            raise ValueError("reranker.capcal.strength must be non-negative")
        self.eps = float(self.capcal_config.get("probability_epsilon", _EPS_DEFAULT))
        if self.eps <= 0.0:
            raise ValueError("reranker.capcal.probability_epsilon must be positive")
        self.entropy_adaptive = bool(self.capcal_config.get("entropy_adaptive", True))
        self.require_grade_probabilities = bool(self.capcal_config.get("require_grade_probabilities", True))
        self.score_fallback = str(self.capcal_config.get("score_fallback", "centered_score")).strip().lower()
        if self.score_fallback not in {"centered_score", "raise"}:
            raise ValueError("reranker.capcal.score_fallback must be 'centered_score' or 'raise'")
        self.score_min = float(self.capcal_config.get("score_min", 0.0))
        self.score_max = float(self.capcal_config.get("score_max", 1.0))
        if self.score_max <= self.score_min:
            raise ValueError("reranker.capcal.score_max must be greater than score_min")

        self.base = base_reranker if base_reranker is not None else self._instantiate_base_reranker(config)
        for attr in ("instruction", "max_doc_chars"):
            if attr in rc and hasattr(self.base, attr):
                setattr(self.base, attr, rc[attr])
        self.paradigm = getattr(self.base, "paradigm", self.paradigm)
        self.supports_query_batching = bool(getattr(self.base, "supports_query_batching", False))
        self.supports_render_variants = bool(getattr(self.base, "supports_render_variants", False))
        self.supports_scale_variants = bool(getattr(self.base, "supports_scale_variants", False))

        self.docs_per_score_forward = int(
            rc.get(
                "docs_per_score_forward",
                getattr(
                    self.base,
                    "docs_per_score_forward",
                    (rc.get("base_reranker") or {}).get("docs_per_score_forward", 20),
                ),
            )
        )
        if self.docs_per_score_forward <= 0:
            raise ValueError("reranker.docs_per_score_forward must be positive")

        probe_query_texts = self.capcal_config.get("probe_query_texts")
        if probe_query_texts is None:
            self.probe_query_texts: list[str] | None = None
        else:
            if isinstance(probe_query_texts, str):
                probe_query_texts = [probe_query_texts]
            if not probe_query_texts:
                raise ValueError("reranker.capcal.probe_query_texts must not be empty")
            self.probe_query_texts = [str(q) for q in probe_query_texts]
        self.probe_passage_text = str(self.capcal_config.get("probe_passage_text", ""))
        self._prior_cache: dict[
            tuple[int, tuple[str, ...], str, int | None, str], tuple[list[list[float]] | None, list[float]]
        ] = {}

    @property
    def instruction(self) -> str | None:
        return getattr(self.base, "instruction", None)

    @instruction.setter
    def instruction(self, value: str) -> None:
        if hasattr(self.base, "instruction"):
            self.base.instruction = value

    @property
    def max_doc_chars(self) -> int | None:
        return getattr(self.base, "max_doc_chars", None)

    @max_doc_chars.setter
    def max_doc_chars(self, value: int) -> None:
        if hasattr(self.base, "max_doc_chars"):
            self.base.max_doc_chars = int(value)

    @staticmethod
    def _instantiate_base_reranker(config: dict) -> Reranker:
        rc = config.get("reranker") or {}
        base_cfg = rc.get("base_reranker")
        if not isinstance(base_cfg, dict):
            raise ValueError("CapCalReranker requires reranker.base_reranker mapping")
        base_cls_name = base_cfg.get("class")
        if not base_cls_name:
            raise ValueError("CapCalReranker requires reranker.base_reranker.class")
        if base_cls_name == "CapCalReranker":
            raise ValueError("CapCalReranker cannot wrap another CapCalReranker")
        from presentation_dependence.rerankers.registry import get_reranker_class

        nested = dict(config)
        nested["reranker"] = dict(base_cfg)
        return get_reranker_class(str(base_cls_name))(nested)

    def _sync_base_runtime_state(self) -> None:
        if getattr(self.base, "supports_render_variants", False):
            self.base.render_variant = getattr(self, "render_variant", None)

    def set_active_adapter(self, adapter_path: str | None) -> None:
        self.active_adapter = str(adapter_path) if adapter_path else None
        self.base.set_active_adapter(adapter_path)

    def _probe_passages(self, n: int) -> list[Passage]:
        return [
            {
                "pid": f"__capcal_probe_slot_{i + 1}",
                "text": self.probe_passage_text,
            }
            for i in range(n)
        ]

    def _probe_queries_for(self, query: Query) -> list[str]:
        if self.probe_query_texts is not None:
            return self.probe_query_texts
        return [str(query["text"])]

    def _estimate_prior(  # noqa: C901
        self,
        n_slots: int,
        query: Query,
    ) -> tuple[list[list[float]] | None, list[float], list[float]]:
        probe_queries = self._probe_queries_for(query)
        prompt_instruction = str(getattr(self.base, "instruction", ""))
        prompt_max_doc_chars = getattr(self.base, "max_doc_chars", None)
        cache_key = (n_slots, tuple(probe_queries), prompt_instruction, prompt_max_doc_chars, self.probe_passage_text)
        cached = self._prior_cache.get(cache_key)
        if cached is not None:
            prior_vectors, prior_scores = cached
            return prior_vectors, prior_scores, []

        vector_sums: list[list[float]] | None = None
        score_sums = [0.0] * n_slots
        runtimes: list[float] = []
        for probe_idx, query_text in enumerate(probe_queries):
            probe_query = {"qid": f"__capcal_probe_{n_slots}_{probe_idx}", "text": query_text}
            result = self.base.rank(probe_query, self._probe_passages(n_slots))
            scores = result.get("scores_init_order")
            if scores is None or len(scores) != n_slots:
                raise ValueError(f"CapCal probe base result must populate scores_init_order with length {n_slots}")
            score_sums = [s + float(x) for s, x in zip(score_sums, scores, strict=True)]
            vectors = result.get("grade_probabilities_init_order")
            if vectors is not None:
                if len(vectors) != n_slots:
                    raise ValueError(f"CapCal probe grade_probabilities_init_order length {len(vectors)} != {n_slots}")
                if vector_sums is None:
                    vector_sums = [[0.0] * len(v) for v in vectors]
                for slot_idx, vector in enumerate(vectors):
                    if len(vector_sums[slot_idx]) != len(vector):
                        raise ValueError("CapCal probe grade vector width changed across probes")
                    vector_sums[slot_idx] = [a + float(b) for a, b in zip(vector_sums[slot_idx], vector, strict=True)]
            runtimes.extend(float(x) for x in result.get("prompting_runtimes", []))

        denom = float(len(probe_queries))
        prior_scores = [s / denom for s in score_sums]
        prior_vectors = None
        if vector_sums is not None:
            prior_vectors = []
            for slot_sum in vector_sums:
                averaged = [x / denom for x in slot_sum]
                total = sum(max(x, 0.0) for x in averaged)
                if total <= 0.0:
                    prior_vectors.append([1.0 / len(averaged)] * len(averaged))
                else:
                    prior_vectors.append([max(x, 0.0) / total for x in averaged])
        self._prior_cache[cache_key] = (prior_vectors, prior_scores)
        return prior_vectors, prior_scores, runtimes

    def _calibrate_result(self, query: Query, base_result: RankResult, passages: list[Passage]) -> RankResult:  # noqa: C901
        base_scores = base_result.get("scores_init_order")
        if base_scores is None:
            raise ValueError("CapCalReranker requires a base reranker with scores_init_order")
        if len(base_scores) != len(passages):
            raise ValueError(f"CapCal base produced {len(base_scores)} scores for {len(passages)} passages")

        base_vectors = base_result.get("grade_probabilities_init_order")
        if base_vectors is not None and len(base_vectors) != len(passages):
            raise ValueError(
                "CapCal base produced grade_probabilities_init_order length "
                f"{len(base_vectors)} for {len(passages)} passages"
            )
        calibrated_scores: list[float] = []
        calibrated_vectors: list[list[float]] | None = [] if base_vectors is not None else None
        prior_runtimes: list[float] = []

        for start in range(0, len(passages), self.docs_per_score_forward):
            end = min(start + self.docs_per_score_forward, len(passages))
            n = end - start
            prior_vectors, prior_scores, runtimes = self._estimate_prior(n, query)
            prior_runtimes.extend(runtimes)
            chunk_scores = [float(x) for x in base_scores[start:end]]
            if base_vectors is not None:
                if prior_vectors is None:
                    raise ValueError("CapCal requires probe grade probabilities when real result has them")
                for slot, real_vector in enumerate(base_vectors[start:end]):
                    calibrated = calibrate_grade_probabilities(
                        real_vector,
                        prior_vectors[slot],
                        strength=self.strength,
                        eps=self.eps,
                        entropy_adaptive=self.entropy_adaptive,
                    )
                    assert calibrated_vectors is not None
                    calibrated_vectors.append(calibrated)
                    calibrated_scores.append(expected_grade_score(calibrated))
            else:
                if self.require_grade_probabilities or self.score_fallback == "raise":
                    raise ValueError(
                        "CapCalReranker requires grade_probabilities_init_order for probability-space "
                        "calibration; set reranker.capcal.require_grade_probabilities=false "
                        "to use centered-score fallback."
                    )
                mean_prior = sum(prior_scores) / len(prior_scores)
                for slot, score in enumerate(chunk_scores):
                    adjusted = score - self.strength * (prior_scores[slot] - mean_prior)
                    calibrated_scores.append(max(self.score_min, min(self.score_max, adjusted)))

        elapsed = sum(float(x) for x in base_result.get("prompting_runtimes", [])) + sum(prior_runtimes)
        result = scores_to_rank_result(
            calibrated_scores,
            passages,
            elapsed,
            self.paradigm,
            model_name=self.__class__.__name__,
        )
        if calibrated_vectors is not None:
            result["grade_probabilities_init_order"] = calibrated_vectors
        return result

    def rank(self, query: Query, passages: list[Passage]) -> RankResult:
        if not passages:
            return {
                "top_k_psgs": [],
                "scores_init_order": [],
                "grade_probabilities_init_order": [],
                "prompting_runtimes": [0.0],
                "paradigm": self.paradigm,
            }
        self._sync_base_runtime_state()
        return self._calibrate_result(query, self.base.rank(query, passages), passages)

    def rank_query_batch(self, items: list[tuple[Query, list[Passage]]]) -> list[RankResult]:
        if not items:
            return []
        self._sync_base_runtime_state()
        if getattr(self.base, "supports_query_batching", False):
            base_results = self.base.rank_query_batch(items)
        else:
            base_results = [self.base.rank(query, passages) for query, passages in items]
        if len(base_results) != len(items):
            raise RuntimeError(f"base rank_query_batch returned {len(base_results)} results for {len(items)} items")
        return [
            self._calibrate_result(query, result, passages) for result, (query, passages) in zip(base_results, items)
        ]
