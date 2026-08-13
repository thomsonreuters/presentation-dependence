"""Abstract reranker interface.

All base models implement this interface.
Kept small. Richer surfaces per scoring family
(score-variance, rank-variance, log-prob reconstruction) belong to the
training code, not to off-the-shelf eval.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Literal, TypedDict

if TYPE_CHECKING:
    from presentation_dependence.rerankers.render_variants import RenderVariant


RerankerParadigm = Literal["pointwise", "scoring_listwise", "generative_listwise", "batched_pointwise"]


class Passage(TypedDict):
    """Input passage representation."""

    pid: str
    text: str


class RankedPassage(TypedDict, total=False):
    """Output passage after reranking. `score` is None for pure generative-
    listwise bases that don't natively emit per-doc scores (e.g. RankZephyr).
    """

    pid: str
    text: str
    score: float | None


class Query(TypedDict):
    qid: str
    text: str


class RankResult(TypedDict, total=False):
    """Return value of `Reranker.rank`. The shape `EvalManager` reads from
    `detailed_results.json`, so the downstream eval pipeline does not branch
    on scoring family.

    Fields:
        top_k_psgs: ordered best-first list of ranked passages.
        scores_init_order: Per-doc scalar scores aligned to the **input**
            passage list (same length as ``passages`` after any reranker-internal
            truncation). Required whenever the model emits native or prompted
            scores (pointwise cross-encoder, architectural scoring-head such as
            Jina, **batched pointwise** shared-context prompts). The PSI harness
            aggregates these across K permutations into ``mean_score_variance``.
            Set to ``None`` only when the model truly has no per-doc scalars
            (e.g. generative listwise text rankings); then robustness uses
            rank-variance + τ-PSI instead.
        grade_probabilities_init_order: Optional per-doc grade probability
            vectors aligned to the input passage list. Continuous grade-prompt
            rerankers populate this so the self-distill teacher can persist
            vector-loss silver labels.
        prompting_runtimes: wall-clock seconds spent in model forward passes.
            List so multi-call rerankers (sliding window) can log per-call
            timing; single-shot rerankers log a one-element list.
        paradigm: one of ``{"pointwise", "scoring_listwise", "generative_listwise",
            "batched_pointwise"}``. ``batched_pointwise`` is for LLM rerankers that
            score multiple passages per forward from shared context (native
            reranker APIs or prompted scalar outputs). Set explicitly so eval /
            τ-PSI geometry helpers stay aligned with the project taxonomy.
    """

    top_k_psgs: list[RankedPassage]
    scores_init_order: list[float] | None
    grade_probabilities_init_order: list[list[float]] | None
    prompting_runtimes: list[float]
    paradigm: RerankerParadigm


class Reranker(ABC):
    """Abstract reranker.

    Concrete subclasses live under `presentation_dependence.rerankers.*`, one file per
    base model, registered via `presentation_dependence.rerankers.registry.RERANKER_CLASSES`.

    Config contract: `__init__` receives the full experiment-config dict (same
    object written in `configs/experiments/<ID>.yaml`). The reranker reads its
    own sub-block from `config["reranker"]`. No global state outside `config`.
    """

    #: one of {"pointwise", "scoring_listwise", "generative_listwise",
    #: "batched_pointwise"}; subclasses override.
    paradigm: RerankerParadigm

    #: When True, the wrapper's :meth:`rank_query_batch` does *real* cross-query
    #: batching (one engine call covering all input queries) rather than the
    #: trivial per-query fallback below. Subclasses that override
    #: ``rank_query_batch`` with a batched implementation must also flip this
    #: flag so callers (e.g. the K-shot teacher driver) can detect the
    #: capability and choose between the per-query and batched code paths.
    supports_query_batching: bool = False

    #: When True, :class:`~presentation_dependence.eval.psi_manager.PsiExperimentRunner` may
    #: set :attr:`render_variant` for input-symmetry render probes.
    supports_render_variants: bool = False

    #: When True, the reranker can vary scoring-scale readout (``scale`` perturbation).
    supports_scale_variants: bool = False

    #: Path of the LoRA adapter currently active (None = base model). Set by
    #: :meth:`set_active_adapter`; used by shared-base eval bundles to swap the
    #: served adapter per surface from a single base model load.
    active_adapter: str | None = None

    def __init__(self, config: dict):
        self.config = config
        self.reranker_config = config.get("reranker", {})
        self.render_variant: RenderVariant | None = None

    @abstractmethod
    def rank(self, query: Query, passages: list[Passage]) -> RankResult:
        """Rerank `passages` for `query`. Must not mutate inputs."""
        raise NotImplementedError

    def set_active_adapter(self, adapter_path: str | None) -> None:
        """Select the active LoRA adapter (or the base model when ``None``).

        Family-agnostic: forwards to the underlying engine's
        ``set_active_adapter`` when it supports multi-adapter serving (the vLLM
        logit-skeleton engine does). Shared-base eval bundles
        (``bundle.run_bundle(shared_base=True)``) call this per surface so one
        base model load can serve the off-shelf base plus N adapters.

        Raises if a non-base adapter is requested but the engine can't switch
        adapters (e.g. an HF engine, a merged-checkpoint model, or an API
        reranker) — so a misconfigured bundle fails loud instead of silently
        scoring every surface with the wrong weights.
        """
        self.active_adapter = str(adapter_path) if adapter_path else None
        engine = getattr(self, "engine", None)
        setter = getattr(engine, "set_active_adapter", None)
        if setter is not None:
            setter(self.active_adapter)
        elif adapter_path:
            raise ValueError(
                f"{type(self).__name__}.set_active_adapter({adapter_path!r}): the underlying "
                f"engine ({type(engine).__name__}) does not support adapter switching. "
                f"Shared-base bundles require inference_engine='vllm' with runtime-LoRA-capable "
                f"adapters (not merged checkpoints)."
            )

    def rank_query_batch(
        self,
        items: list[tuple[Query, list[Passage]]],
    ) -> list[RankResult]:
        """Rerank a batch of ``(query, passages)`` pairs in one logical call.

        Default implementation iterates :meth:`rank` per item and preserves the
        single-query contract for any wrapper that hasn't opted in. Wrappers
        that can dispatch all queries' prompts to a single engine pass (e.g.
        the self-distill continuous-readout wrappers backed by vLLM) override
        this method and set :attr:`supports_query_batching` to True. The
        K-shot teacher driver checks the flag to decide whether cross-query
        batching is meaningful for the configured wrapper.
        """
        return [self.rank(query, passages) for query, passages in items]


def parse_lora_config(reranker: "Reranker", reranker_cfg: dict, *, engine_kind: str) -> None:
    """Parse a reranker's LoRA config onto the instance (family-agnostic).

    Sets ``reranker.lora_path`` (legacy single adapter), ``reranker.merge_lora``
    (HF-only weight merge), ``reranker.lora_adapters`` (list of adapter mount dirs
    for shared-base multi-adapter bundles) and ``reranker.active_adapter`` (base
    = None for multi-adapter, else the single ``lora_path``). Call EARLY in
    ``__init__`` (before the HF model load, which needs ``lora_path``).

    Multi-adapter (``lora_adapters``) requires the vLLM engine; the HF / merged
    paths can't runtime-swap adapters, so this raises for any other ``engine_kind``.
    """
    lora_path = reranker_cfg.get("lora_path")
    reranker.lora_path = str(lora_path) if lora_path else None
    reranker.merge_lora = bool(reranker_cfg.get("merge_lora", False))
    lora_adapters = reranker_cfg.get("lora_adapters")
    if lora_adapters is not None and not isinstance(lora_adapters, (list, tuple)):
        raise ValueError("reranker.lora_adapters must be a list of adapter paths when provided")
    reranker.lora_adapters = [str(p) for p in (lora_adapters or []) if p]
    reranker.active_adapter = None if reranker.lora_adapters else reranker.lora_path
    if reranker.lora_adapters and engine_kind != "vllm":
        raise ValueError(
            "reranker.lora_adapters (multi-adapter shared-base bundles) is only "
            f"supported with inference_engine='vllm' (got {engine_kind!r})."
        )


def apply_vllm_lora_settings(reranker: "Reranker", reranker_cfg: dict, vllm_settings: dict) -> None:
    """Wire the (already-parsed) LoRA config into ``vllm_settings``.

    Call from a vLLM reranker's ``__init__`` AFTER :func:`parse_lora_config` and
    AFTER building ``vllm_settings``, BEFORE ``make_engine("vllm", ...)``. Sets
    ``lora_adapters`` (multi-adapter) or the single ``lora_path`` plus
    ``max_lora_rank``. No-op when the reranker has no adapter.
    """
    if getattr(reranker, "lora_adapters", None):
        vllm_settings["lora_adapters"] = reranker.lora_adapters
        if "max_lora_rank" in reranker_cfg:
            vllm_settings["max_lora_rank"] = int(reranker_cfg["max_lora_rank"])
    elif getattr(reranker, "lora_path", None) is not None:
        vllm_settings["lora_path"] = reranker.lora_path
        if "max_lora_rank" in reranker_cfg:
            vllm_settings["max_lora_rank"] = int(reranker_cfg["max_lora_rank"])
