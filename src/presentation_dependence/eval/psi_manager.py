"""Permutation-sensitivity run driver.

For each query, the driver generates K presentations of the first-stage
candidate list, reranks them, and passes their aligned rankings and scores to
``presentation_dependence.eval.psi.evaluate_psi``. Owns configuration and file I/O;
``psi.py`` owns the metric calculations.

Perturbation strategies:

- ``random_shuffle`` applies Fisher-Yates shuffling with deterministic seeds.
  With K around 9 or more, long candidate lists normally cover all three
  position buckets.
- ``middle_injection`` places the first relevant document at the top, middle,
  or bottom. It adds one presentation per bucket, independent of K.

Each presentation records where the first qrels-positive document appeared in
the reranker input. These labels define the Δ-nDCG strata.

Run-dir layout:

    runs/<ID>/<ts>/
        resolved_config.yaml
        psi/
            per_query_results/<qid>/
                permutation_<k>/
                    trec_results_raw.txt       # reranker output for this seed
                    detailed_results.json      # full RankResult dict
                input_positions.json           # {k: bucket, ...}
                aligned_scores.json            # {docid: [score_perm0, ...]}
                                               # docid-aligned per-perm scores
                                               # for random_shuffle perms,
                                               # in seed-order. Always written
                                               # when the reranker emits
                                               # ``scores_init_order``. Lets the
                                               # post-hoc SC script skip the
                                               # dataloader entirely.
            psi_metrics.json                   # evaluate_psi() aggregate + protocol;
                                               # tau_psi_geometry_schema v2 +
                                               # tau_psi_inference_note + optional
                                               # tau_psi_window_size/stride, Jina
                                               # block, pointwise CE fields (τ-PSI@B)
            sc_metrics.json                    # K-shot self-consistency aggregate
                                               # (when reranker emits scalars +
                                               # qrels available + random_shuffle
                                               # in perturbations); see
                                               # ``presentation_dependence.eval.self_consistency``
            sc_per_query.json                  # per-K, per-query SC metrics

``psi_metrics.json`` is the machine-readable robustness artifact.
When ``random_shuffle`` is configured, its aggregate and
``psi_per_query.json`` use only those uniform-random presentations.
``middle_injection`` outputs remain in the per-query tree for separate
targeted bucket reductions; they are never pooled into an expectation over
presentations.
``sc_metrics.json`` evaluates self-consistency on the same K random shuffles,
averaging per-document score vectors before ranking. By Jensen's inequality,
``mean_per_perm_ndcg`` is a lower bracket for this quantity. Existing PSI
scores also support a K' ≤ K curve without additional reranker calls.

Input-symmetry render strategies (`id_relabel`, `separator`, `rubric_paraphrase`)
hold document order fixed and vary expected-grade prompt rendering via
``reranker.render_variant``; see ``render_variants.py``.

``partial_shuffle``, ``block_swap``, and ``retriever_noise`` are not yet
implemented. They belong in ``_generate_permutations``.
"""

from __future__ import annotations

import json
import logging
import random
import re
import time
from datetime import datetime
from pathlib import Path

import yaml

from presentation_dependence.eval.psi import evaluate_psi
from presentation_dependence.rerankers.render_variants import RENDER_STRATEGIES, RenderVariant, generate_render_variants
from presentation_dependence.eval.psi_artifacts import build_psi_metrics_envelope
from presentation_dependence.eval.runner_setup import (
    instantiate_dataloader,
    instantiate_reranker,
    load_qrels,
    partial_run_error,
)
from presentation_dependence.eval.runner_setup import validate_rank_result_if_enabled
from presentation_dependence.eval.runner_setup import check_resume_fingerprint, run_fingerprint
from presentation_dependence.eval.runner_setup import write_resolved_run_config
from presentation_dependence.eval.score_log import write_score_log
from presentation_dependence.eval.rank_self_consistency import compute_sc_lift, derive_rank_sc_metrics
from presentation_dependence.eval.self_consistency import derive_sc_metrics, split_headline_and_verbose
from presentation_dependence.utils.progress import ProgressTracker, progress_config
from presentation_dependence.utils.setup_logging import setup_logging
from presentation_dependence.utils.trec import qid_to_dirname, write_trec_run


# Module-level helpers below run outside any runner instance, so they report
# through the package logger rather than a per-run one.
_LOG = logging.getLogger(__name__)

_PERM_DIR_RE = re.compile(r"^permutation_(\d+)_(.+)$")


# Default SC truncations to compute when ``robustness.sc_k_subsets`` is unset.
# These values give a useful curve for the common K=10 run without enlarging
# `sc_metrics.json` unnecessarily. Values above full_K are removed at runtime.
DEFAULT_SC_K_SUBSETS: tuple[int, ...] = (1, 2, 3, 5, 10, 20)

_ORDER_PERTURBATIONS = frozenset({"random_shuffle", "middle_injection"})
_RENDER_PERTURBATIONS = frozenset(RENDER_STRATEGIES)
_KNOWN_PERTURBATIONS = _ORDER_PERTURBATIONS | _RENDER_PERTURBATIONS

PermutationUnit = tuple[list[dict], str, str, RenderVariant | None]


def _positive_int_or_none(raw: object, *, field_name: str) -> int | None:
    if raw is None:
        return None
    value = int(raw)
    if value <= 0:
        raise ValueError(f"{field_name} must be positive when set")
    return value


def _load_qids_allowlist(config: dict) -> set[str]:
    """Return qids from top-level ``qids_to_run`` and ``qids_to_run_path``."""
    out = set(map(str, config.get("qids_to_run", []) or []))
    path_s = config.get("qids_to_run_path")
    if path_s:
        path = Path(path_s)
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                qid = line.strip()
                if qid:
                    out.add(qid)
    return out


def _read_trec_ranking(trec_path: Path) -> list[str]:
    """Parse a TREC run file and return document IDs from best to worst.

    Unparseable rows are dropped, which shortens the ranking rather than
    failing. That matters on resume: this same parser decides whether a cached
    permutation counts as complete, so a truncated file would be reused as if
    it were whole. Dropped rows are logged for that reason.
    """
    rows: list[tuple[int, str]] = []
    dropped = 0
    with open(trec_path, encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if len(parts) < 6:
                if line.strip():
                    dropped += 1
                continue
            try:
                rank = int(parts[3])
            except ValueError:
                dropped += 1
                continue
            rows.append((rank, parts[2]))
    if dropped:
        _LOG.warning(
            "%s: dropped %d unparseable TREC row(s); the ranking read from it is "
            "truncated and any metric derived from it covers fewer documents.",
            trec_path,
            dropped,
        )
    rows.sort(key=lambda r: r[0])
    return [docid for _, docid in rows]


def _count_usable_perms(qdir: Path) -> int:
    """Count permutation subdirs with a non-empty ``trec_results_raw.txt``."""
    if not qdir.is_dir():
        return 0
    count = 0
    for child in qdir.iterdir():
        if not child.is_dir() or not _PERM_DIR_RE.match(child.name):
            continue
        trec_path = child / "trec_results_raw.txt"
        if trec_path.is_file() and _read_trec_ranking(trec_path):
            count += 1
    return count


def _load_qid_rankings_from_disk(qdir: Path) -> tuple[list[list[str]], list[str] | None]:
    """Return ``(rankings, bucket_labels)`` from on-disk permutation artefacts."""
    if not qdir.is_dir():
        return [], None
    perm_dirs: list[tuple[int, str, Path]] = []
    for child in qdir.iterdir():
        if not child.is_dir():
            continue
        match = _PERM_DIR_RE.match(child.name)
        if not match:
            continue
        perm_dirs.append((int(match.group(1)), match.group(2), child))
    perm_dirs.sort(key=lambda x: x[0])

    rankings: list[list[str]] = []
    for _, _label, pdir in perm_dirs:
        trec_path = pdir / "trec_results_raw.txt"
        if not trec_path.is_file():
            continue
        ranking = _read_trec_ranking(trec_path)
        if ranking:
            rankings.append(ranking)

    pos_path = qdir / "input_positions.json"
    if not pos_path.is_file():
        return rankings, None
    with open(pos_path, encoding="utf-8") as f:
        labels = json.load(f).get("labels", [])
    if len(labels) < len(rankings):
        return rankings, None
    return rankings, [str(b) for b in labels[: len(rankings)]]


def _load_qid_random_rankings_from_disk(
    qdir: Path,
) -> tuple[list[list[str]], list[str] | None]:
    """Load random-shuffle rankings and their bucket labels in seed order.

    The rank-space SC counterpart to :func:`_load_qid_random_scores_from_disk`:
    filters to ``random_s*`` permutation dirs (drops ``inject_*`` and render
    variants) so headline PSI and Borda SC see only uniform-random shots.
    """
    if not qdir.is_dir():
        return [], None
    perm_dirs: list[tuple[int, str, Path]] = []
    for child in qdir.iterdir():
        if not child.is_dir():
            continue
        match = _PERM_DIR_RE.match(child.name)
        if not match:
            continue
        perm_dirs.append((int(match.group(1)), match.group(2), child))
    perm_dirs.sort(key=lambda x: x[0])
    rankings: list[list[str]] = []
    buckets: list[str] = []
    all_buckets = _load_input_bucket_labels(qdir)
    usable_index = 0
    for _, label, pdir in perm_dirs:
        trec_path = pdir / "trec_results_raw.txt"
        if not trec_path.is_file():
            continue
        ranking = _read_trec_ranking(trec_path)
        if not ranking:
            continue
        if not label.startswith("random_s"):
            usable_index += 1
            continue
        rankings.append(ranking)
        if usable_index < len(all_buckets):
            buckets.append(all_buckets[usable_index])
        usable_index += 1
    return rankings, buckets if len(buckets) == len(rankings) else None


def _load_input_bucket_labels(qdir: Path) -> list[str]:
    """Read input-position labels, returning an empty list when absent."""
    path = qdir / "input_positions.json"
    if not path.is_file():
        return []
    with open(path, encoding="utf-8") as f:
        return [str(value) for value in json.load(f).get("labels", [])]


def _load_qid_random_scores_from_disk(qdir: Path) -> dict[str, list[float]] | None:
    """Load docid-aligned random_shuffle scores written during a prior PSI pass."""
    path = qdir / "aligned_scores.json"
    if not path.is_file():
        return None
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    scores = data.get("scores")
    if not isinstance(scores, dict) or not scores:
        return None
    return {str(doc_id): [float(v) for v in vals] for doc_id, vals in scores.items()}


# ---------------------------------------------------------------------------
# Perturbation generators.
# ---------------------------------------------------------------------------


def _bucket_of_position(pos: int, n: int) -> str:
    """Map an input position in [0, n) to one of top/middle/bottom thirds."""
    if n <= 0:
        return "middle"
    third = n / 3.0
    if pos < third:
        return "top"
    if pos < 2 * third:
        return "middle"
    return "bottom"


def _random_shuffle(
    passages: list[dict],
    seed: int,
    relevant_pid: str | None,
) -> tuple[list[dict], str]:
    """Fisher-Yates shuffle with a deterministic seed; returns the shuffled
    list plus the bucket label of the relevant doc's resulting position.
    """
    rng = random.Random(seed)
    shuffled = list(passages)
    rng.shuffle(shuffled)
    if relevant_pid is None:
        return shuffled, "middle"  # can't stratify without qrels; bucket ignored later
    pids = [p["pid"] for p in shuffled]
    try:
        pos = pids.index(str(relevant_pid))
    except ValueError:
        return shuffled, "middle"  # relevant doc not in candidate pool
    return shuffled, _bucket_of_position(pos, len(shuffled))


def _middle_injection(
    passages: list[dict],
    relevant_pid: str,
    bucket: str,
) -> list[dict]:
    """Place `relevant_pid` at the target bucket's canonical position
    (top=0, middle=n//2, bottom=n-1) keeping other docs in first-stage order.
    """
    others = [p for p in passages if str(p["pid"]) != str(relevant_pid)]
    relevant = next((p for p in passages if str(p["pid"]) == str(relevant_pid)), None)
    if relevant is None:
        return list(passages)  # nothing to inject; return input order
    n_total = len(others) + 1
    if bucket == "top":
        idx = 0
    elif bucket == "bottom":
        idx = n_total - 1
    else:
        idx = n_total // 2
    return others[:idx] + [relevant] + others[idx:]


# ---------------------------------------------------------------------------
# Driver.
# ---------------------------------------------------------------------------


class PsiExperimentRunner:
    """Drives a K-permutation robustness evaluation.

    Config extension (sibling to `eval:` block in the experiment YAML):

        robustness:
          K: 10
          seeds: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]     # len must equal K
          perturbations: [random_shuffle]            # or + middle_injection
          k_cutoff_for_ndcg: 10
    """

    def __init__(  # noqa: C901
        self,
        config_path: str | Path,
        runs_root: str | Path = "runs",
        run_dir: str | Path | None = None,
        reranker: object | None = None,
    ):
        """Initialize a K-permutation robustness evaluation.

        Parameters
        ----------
        config_path : str | Path
            YAML config file.
        runs_root : str | Path, default "runs"
            Root directory for ``<exp_id>/<ts>/`` layout. Ignored when
            ``run_dir`` is supplied.
        run_dir : str | Path | None, default None
            Reuse this directory instead of creating a timestamped run. The
            driver writes under ``<run_dir>/psi/`` and does not replace
            ``resolved_config.yaml``. The container entrypoint uses this option
            to store base nDCG and PSI output together.
        reranker : object | None, default None
            Reuse a loaded reranker, such as the instance created by
            ``ExperimentManager``. This avoids a second model download and
            weight load, measured at about three minutes for one historical 7B
            container run.
            When ``None``, load ``config.reranker``.
        """
        self.config_path = Path(config_path)
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config not found: {self.config_path}")

        with open(self.config_path, "r") as f:
            self.config: dict = yaml.safe_load(f)

        self.exp_id = self.config["id"]
        self.runs_root = Path(runs_root)

        self._reuses_run_dir = run_dir is not None
        if run_dir is not None:
            self.run_dir = Path(run_dir)
            ts = self.run_dir.name  # keep the ExperimentManager's timestamp
        else:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            self.run_dir = self.runs_root / self.exp_id / ts
        self.psi_dir = self.run_dir / "psi"
        self.results_dir = self.psi_dir / "per_query_results"
        self.results_dir.mkdir(parents=True, exist_ok=True)

        self.logger = setup_logging(
            self.__class__.__name__,
            self.config,
            output_file=str(self.run_dir / "psi_run.log"),
        )

        self.data_config = self.config.get("data", {})
        self.robust_cfg = self.config.get("robustness", {})
        self.k_input = _positive_int_or_none(self.data_config.get("k_input"), field_name="data.k_input")
        self.K = int(self.robust_cfg.get("K", 10))
        self.seeds = list(self.robust_cfg.get("seeds") or list(range(self.K)))
        if len(self.seeds) != self.K:
            raise ValueError(f"robustness.seeds ({len(self.seeds)}) must equal K ({self.K})")
        self.perturbations = list(self.robust_cfg.get("perturbations") or ["random_shuffle"])
        for p in self.perturbations:
            if p not in _KNOWN_PERTURBATIONS:
                known = ", ".join(sorted(_KNOWN_PERTURBATIONS))
                raise ValueError(f"Unknown perturbation strategy: {p!r} (known: {known})")
        self.render_strategies = [p for p in self.perturbations if p in _RENDER_PERTURBATIONS]
        self.k_cutoff = int(self.robust_cfg.get("k_cutoff_for_ndcg", 10))
        reranker_cfg = self.config.get("reranker") or {}
        self._render_n_slots = int(reranker_cfg.get("docs_per_score_forward", 20))

        # Cross-permutation batching sends up to N query/presentation pairs per
        # engine call. The default changed from 1 to 16; at K=10,
        # this normally combines ten random shuffles and three targeted
        # injections. Rerankers without rank_query_batch continue on the
        # per-presentation path. Rendering variants also run separately because
        # one batched call cannot vary reranker.render_variant. Tests on
        # Qwen3-0.6B and Granite-4.1-8B on DL19 found nDCG differences at most
        # 1e-3 and identical
        # τ-PSI/Kendall τ. Set eval.query_batch_size: 1 for the per-presentation
        # path.
        eval_cfg_for_batch = self.config.get("eval") or {}
        _qbs_raw = eval_cfg_for_batch.get("query_batch_size", 16)
        self._query_batch_size = int(_qbs_raw if _qbs_raw is not None else 16)
        if self._query_batch_size < 1:
            raise ValueError("eval.query_batch_size must be >= 1")

        # Self-consistency requires uniform random shuffles. It is skipped when
        # random_shuffle is absent or derive_self_consistency is false.
        self.derive_sc = bool(self.robust_cfg.get("derive_self_consistency", True))
        sc_subsets_raw = self.robust_cfg.get("sc_k_subsets")
        if sc_subsets_raw is None:
            self.sc_k_subsets = list(DEFAULT_SC_K_SUBSETS)
        else:
            self.sc_k_subsets = sorted({int(k) for k in sc_subsets_raw if int(k) >= 1})
        # Use eval.measures by default so SC and K=1 metrics are comparable.
        eval_cfg = self.config.get("eval") or {}
        sc_measures_raw = self.robust_cfg.get("sc_measures") or eval_cfg.get("measures") or ["ndcg_cut_10"]
        self.sc_measures = list(sc_measures_raw)

        beta_gamma_cfg = self.robust_cfg.get("beta_gamma") or {}
        self.beta_gamma_enabled = bool(beta_gamma_cfg.get("enabled", False))
        if self.beta_gamma_enabled and "random_shuffle" not in self.perturbations:
            raise ValueError("robustness.beta_gamma.enabled requires perturbations to include random_shuffle")
        self.beta_gamma_recipe = str(beta_gamma_cfg.get("recipe") or self.exp_id)
        self.beta_gamma_checkpoint = str(
            beta_gamma_cfg.get("checkpoint")
            or beta_gamma_cfg.get("checkpoint_id")
            or beta_gamma_cfg.get("checkpoint_step")
            or "unknown"
        )
        score_log_raw = beta_gamma_cfg.get("score_log_path")
        if score_log_raw:
            score_log_path = Path(str(score_log_raw))
            if not score_log_path.is_absolute():
                if score_log_path.parts[:1] == ("psi",):
                    score_log_path = self.run_dir / score_log_path
                else:
                    score_log_path = self.psi_dir / score_log_path
        else:
            score_log_path = self.psi_dir / "beta_gamma_scores.parquet"
        self.beta_gamma_score_log_path = score_log_path
        self._beta_gamma_rows: list[dict] = []

        if reranker is not None:
            self.reranker = reranker
            self.logger.info("Reusing pre-loaded reranker (%s)", type(reranker).__name__)
        else:
            self._load_reranker()
        if self.render_strategies and not getattr(self.reranker, "supports_render_variants", False):
            raise ValueError(
                f"Render perturbations {self.render_strategies} require a reranker with "
                f"supports_render_variants=True (got {type(self.reranker).__name__})"
            )
        if "scale" in self.render_strategies and not getattr(self.reranker, "supports_scale_variants", False):
            raise ValueError(
                f"Scale perturbation requires supports_scale_variants=True (got {type(self.reranker).__name__})"
            )
        self._load_dataloader()
        self._load_qrels()
        if self._reuses_run_dir:
            check_resume_fingerprint(
                self.run_dir,
                run_fingerprint(self.config, self.data_config),
                self.logger,
            )
        else:
            # Only write resolved_config.yaml when we own the run_dir.
            # Otherwise ExperimentManager has already written a canonical
            # copy (with its own git_sha + dataset_meta) and overwriting
            # would lose metadata.
            self._snapshot_config(ts)

    # Setup.

    def _load_reranker(self) -> None:
        self.reranker = instantiate_reranker(self.config, self.logger)

    def _load_dataloader(self) -> None:
        self.dataloader = instantiate_dataloader(self.config, self.data_config)

    def _load_qrels(self) -> None:
        qrels_path = self.config.get("eval", {}).get("qrels_path")
        if not qrels_path:
            self.logger.warning("No eval.qrels_path; Δ-nDCG will be None.")
            self.qrels: dict[str, dict[str, int]] = {}
            return
        self.qrels = load_qrels(qrels_path)

    def _snapshot_config(self, ts: str) -> None:
        # PSI resumes by counting usable permutation files, so the same
        # cross-configuration hazard applies here as in ExperimentManager.
        check_resume_fingerprint(
            self.run_dir,
            run_fingerprint(self.config, self.data_config),
            self.logger,
        )
        write_resolved_run_config(
            self.run_dir,
            self.config,
            config_path=self.config_path,
            ts=ts,
            data_config=self.data_config,
        )

    # Per-query presentation generation.

    def _first_relevant_pid(
        self,
        qid: str,
        passages: list[dict],
    ) -> str | None:
        """Strongest qrels-positive doc that is present in the candidate pool."""
        rels = self.qrels.get(str(qid)) or {}
        passage_pids = {str(passage["pid"]) for passage in passages}
        rel_docs = sorted(
            [(d, r) for d, r in rels.items() if r > 0 and str(d) in passage_pids],
            key=lambda x: (-x[1], x[0]),
        )
        return rel_docs[0][0] if rel_docs else None

    def _generate_permutations(
        self,
        qid: str,
        passages: list[dict],
    ) -> list[PermutationUnit]:
        """Return list of ``(passages, label, bucket, render_variant)`` units."""
        rel_pid = self._first_relevant_pid(qid, passages)
        out: list[PermutationUnit] = []

        if "random_shuffle" in self.perturbations:
            for seed in self.seeds:
                shuffled, bucket = _random_shuffle(passages, seed, rel_pid)
                out.append((shuffled, f"random_s{seed}", bucket, None))

        if "middle_injection" in self.perturbations:
            if rel_pid is None:
                self.logger.warning(
                    "qid=%s: no qrels-positive doc in candidate pool; skipping middle_injection",
                    qid,
                )
            else:
                for bucket in ("top", "middle", "bottom"):
                    injected = _middle_injection(passages, rel_pid, bucket)
                    out.append((injected, f"inject_{bucket}", bucket, None))

        for strategy in self.render_strategies:
            variants = generate_render_variants(
                strategy,
                self.K,
                self.seeds,
                n_slots=self._render_n_slots,
            )
            for variant in variants:
                out.append((list(passages), variant.label, "render", variant))

        return out

    def _batch_rank_pending_perms(
        self,
        query: dict,
        perms: list[PermutationUnit],
        qdir: Path,
    ) -> dict[int, dict]:
        """Pre-rank this query's non-cached, non-render perms in batched calls.

        Returns ``{k_idx: RankResult}`` for the presentations scored here.
        Excludes rendering variants, which require separate
        ``reranker.render_variant`` values, and presentations already stored on
        disk.

        Returns ``{}`` unless ``eval.query_batch_size > 1`` and the reranker
        advertises ``supports_query_batching``. The batched call goes
        through ``rank_query_batch``, which for the continuous-grade wrappers
        builds the same per-doc prefixes and uses the same engine scoring as
        ``rank()``; the returned scores match the per-presentation path.
        """
        if self._query_batch_size <= 1 or not getattr(self.reranker, "supports_query_batching", False):
            return {}

        pending_items: list[tuple[dict, list[dict]]] = []
        pending_kidx: list[int] = []
        for k_idx, (perm_passages, perm_label, _bucket, render_variant) in enumerate(perms):
            if render_variant is not None:
                continue
            pdir = qdir / f"permutation_{k_idx:03d}_{perm_label}"
            cached_trec = pdir / "trec_results_raw.txt"
            if cached_trec.is_file() and _read_trec_ranking(cached_trec):
                continue
            pending_items.append((query, perm_passages))
            pending_kidx.append(k_idx)

        # Batched presentations never use rendering variants. Clear stale state
        # explicitly so it cannot affect the entire batch.
        if getattr(self.reranker, "render_variant", None) is not None:
            self.reranker.render_variant = None

        precomputed: dict[int, dict] = {}
        n = self._query_batch_size
        for start in range(0, len(pending_items), n):
            chunk_items = pending_items[start : start + n]
            chunk_kidx = pending_kidx[start : start + n]
            results = self.reranker.rank_query_batch(chunk_items)
            if len(results) != len(chunk_items):
                raise RuntimeError(f"rank_query_batch returned {len(results)} results for {len(chunk_items)} items")
            for k_idx, res in zip(chunk_kidx, results, strict=True):
                precomputed[k_idx] = res
        return precomputed

    def _build_coverage(
        self,
        *,
        work_queries: list,
        skipped_qids: list[dict[str, str]],
        aggregated_rankings: dict[str, list[list[str]]],
    ) -> dict:
        """Describe how much of the intended query and presentation set the aggregate covers.

        Two things can shrink an aggregate without failing the run: a query
        dropped before scoring, and a query that contributed fewer than ``K``
        presentations because individual ones produced no output.
        """
        expected = self.K if "random_shuffle" in self.perturbations else None
        short: list[dict[str, int]] = []
        if expected is not None:
            short = [
                {"qid": qid, "presentations": len(rankings)}
                for qid, rankings in sorted(aggregated_rankings.items())
                if len(rankings) < expected
            ]
        # A query can also vanish from the aggregate without being skipped, if
        # every one of its presentations produced no output, so completeness
        # requires the counts to match rather than just an empty skip list.
        return {
            "complete": (not skipped_qids and not short and len(aggregated_rankings) == len(work_queries)),
            "queries_requested": len(work_queries),
            "queries_aggregated": len(aggregated_rankings),
            "queries_skipped": len(skipped_qids),
            "skipped": skipped_qids,
            "expected_presentations_per_query": expected,
            "queries_below_expected_presentations": short,
        }

    def _log_coverage(self, coverage: dict) -> None:
        if coverage["complete"]:
            return
        self.logger.warning(
            "Incomplete PSI: %d/%d queries aggregated, %d skipped, %d with fewer than "
            "%s presentations. Headline tau-PSI covers only the aggregated queries; "
            "see coverage in psi_metrics.json.",
            coverage["queries_aggregated"],
            coverage["queries_requested"],
            coverage["queries_skipped"],
            len(coverage["queries_below_expected_presentations"]),
            coverage["expected_presentations_per_query"],
        )

    def _planned_per_query_work(self) -> int:
        total = 0
        if "random_shuffle" in self.perturbations:
            total += self.K
        if "middle_injection" in self.perturbations:
            total += 3
        total += len(self.render_strategies) * self.K
        return total

    def _expected_perms_count(self) -> int:
        return self._planned_per_query_work()

    def _qid_results_dir(self, qid: str) -> Path:
        return self.results_dir / qid_to_dirname(qid)

    def _is_qid_complete(self, qid: str) -> bool:
        return _count_usable_perms(self._qid_results_dir(qid)) >= self._expected_perms_count()

    def _hydrate_completed_qid(
        self,
        qid: str,
        rankings_over_K: dict[str, list[list[str]]],
        positions: dict[str, list[str]],
        random_scores_over_K: dict[str, dict[str, list[float]]],
        random_rankings_over_K: dict[str, list[list[str]]],
        random_positions: dict[str, list[str]],
    ) -> bool:
        """Load a complete on-disk qid into the in-memory aggregation buffers."""
        qdir = self._qid_results_dir(qid)
        rankings, buckets = _load_qid_rankings_from_disk(qdir)
        if len(rankings) < self._expected_perms_count():
            return False
        rankings_over_K[qid] = rankings
        if buckets is not None:
            positions[qid] = buckets
        scores_random = _load_qid_random_scores_from_disk(qdir)
        if scores_random:
            random_scores_over_K[qid] = scores_random
        random_rankings, random_buckets = _load_qid_random_rankings_from_disk(qdir)
        if random_rankings:
            random_rankings_over_K[qid] = random_rankings
        if random_buckets is not None:
            random_positions[qid] = random_buckets
        return True

    # Main loop.

    def run(self) -> dict:  # noqa: C901
        self.logger.info("Starting PSI run %s (run_dir=%s)", self.exp_id, self.run_dir)

        run_path = self.data_config["run_path"]
        queries = self.dataloader.get_qs_from_run(run_path)
        self.logger.info("Loaded %d queries", len(queries))

        # The PSI-specific allowlist takes precedence, allowing the nDCG pass to
        # cover the full dataset while τ-PSI uses a fixed subset. The top-level
        # allowlist remains the fallback for older configs.
        qids_allowlist = _load_qids_allowlist(self.robust_cfg) or _load_qids_allowlist(self.config)
        work_queries = [query for query in queries if not qids_allowlist or str(query["qid"]) in qids_allowlist]
        planned_per_query = self._planned_per_query_work()

        rankings_over_K: dict[str, list[list[str]]] = {}
        scores_over_K: dict[str, dict[str, list[float]]] = {}
        # SC needs uniform-random shots only; collect those separately so
        # ``middle_injection`` permutations don't bias the mean-then-rank step.
        # Order is preserved across docids because seeds iterate deterministically.
        random_scores_over_K: dict[str, dict[str, list[float]]] = {}
        # Rank-space SC uses random-shuffle output rankings without scalar
        # scores. These rankings define the Borda K=1 to K=10 curve.
        random_rankings_over_K: dict[str, list[list[str]]] = {}
        positions: dict[str, list[str]] = {}
        # Bucket labels paired to the random-only rankings.  Targeted injection
        # shots remain on disk for their diagnostic metrics, but they are not
        # draws from the presentation distribution used by headline tau-PSI.
        random_positions: dict[str, list[str]] = {}
        failed_qids: list[str] = []
        beta_gamma_expected_docs: dict[str, set[str]] = {}
        # Queries dropped before scoring. Unlike ``failed_qids`` these do not
        # raise, so without recording them the aggregate silently covers fewer
        # queries than the protocol declares.
        skipped_qids: list[dict[str, str]] = []

        # Resume: reuse complete per-query artefacts already on disk (same
        # contract as ExperimentManager for phase-1 base scoring).
        n_reused = 0
        if not self.beta_gamma_enabled:
            for query in work_queries:
                qid = str(query["qid"])
                if self._hydrate_completed_qid(
                    qid,
                    rankings_over_K,
                    positions,
                    random_scores_over_K,
                    random_rankings_over_K,
                    random_positions,
                ):
                    n_reused += 1
        if n_reused:
            self.logger.info(
                "Reusing on-disk PSI artefacts for %d/%d queries (skip rerank)",
                n_reused,
                len(work_queries),
            )

        if self.beta_gamma_enabled:
            # Revisit every qid through the normal permutation loop. Completed
            # permutations take the cached branch, so this performs no model
            # inference while rebuilding the full beta/gamma row set.
            pending_queries = work_queries
            self.logger.info(
                "PSI resume with beta/gamma logging: replaying cached permutations for %d queries",
                len(work_queries),
            )
        else:
            pending_queries = [query for query in work_queries if not self._is_qid_complete(str(query["qid"]))]
        if n_reused:
            self.logger.info(
                "PSI resume: %d queries pending rerank (%d already complete on disk)",
                len(pending_queries),
                n_reused,
            )

        write_progress_jsonl, progress_heartbeat_s = progress_config(self.config)
        progress = ProgressTracker(
            phase="psi",
            total_work=len(pending_queries) * planned_per_query,
            total_queries=len(pending_queries),
            jsonl_path=self.psi_dir / "progress.jsonl",
            write_jsonl=write_progress_jsonl,
            heartbeat_every_s=progress_heartbeat_s,
        )

        for q_index, query in enumerate(pending_queries, start=1):
            qid = str(query["qid"])
            active_label: str | None = None
            active_work_index: int | None = None
            try:
                passages = self.dataloader.get_psgs_from_run(run_path, qid) or []
                if self.k_input is not None:
                    passages = passages[: self.k_input]
                if not passages:
                    self.logger.warning("qid=%s: empty passage list; skipping", qid)
                    skipped_qids.append({"qid": qid, "reason": "empty_passage_list"})
                    progress.skip_units(
                        count=planned_per_query,
                        label=f"qid={qid}",
                        q_index=q_index,
                        reason="empty_passage_list",
                    )
                    continue
                if self.beta_gamma_enabled:
                    beta_gamma_expected_docs[qid] = {str(passage["pid"]) for passage in passages}

                perms = self._generate_permutations(qid, passages)
                if not perms:
                    self.logger.warning("qid=%s: no permutations generated; skipping", qid)
                    skipped_qids.append({"qid": qid, "reason": "no_permutations"})
                    progress.skip_units(
                        count=planned_per_query,
                        label=f"qid={qid}",
                        q_index=q_index,
                        reason="no_permutations",
                    )
                    continue

                qdir = self.results_dir / qid_to_dirname(qid)
                qdir.mkdir(exist_ok=True)

                q_rankings: list[list[str]] = []
                q_rankings_random: list[list[str]] = []  # random_shuffle rankings only (rank-SC input)
                q_scores: dict[str, list[float]] = {}  # docid -> per-permutation score (all perms)
                q_scores_random: dict[str, list[float]] = {}  # docid -> random_shuffle scores only (SC input)
                q_positions: list[str] = []
                q_positions_random: list[str] = []
                q_random_perm_idx = 0

                # Precompute pending non-render presentations in batches. Emit
                # progress and GPU status first because the batch call blocks
                # before per-presentation completion lines are written.
                if self._query_batch_size > 1 and getattr(self.reranker, "supports_query_batching", False):
                    self.logger.info(
                        "qid=%s: batch-scoring %d permutations (%s)",
                        qid,
                        len(perms),
                        progress.format_suffix(q_index=q_index),
                    )
                    progress.maybe_heartbeat(self.logger, force=True)
                precomputed_results = self._batch_rank_pending_perms(query, perms, qdir)

                for k_idx, (perm_passages, perm_label, bucket, render_variant) in enumerate(perms):
                    pdir = qdir / f"permutation_{k_idx:03d}_{perm_label}"
                    cached_trec = pdir / "trec_results_raw.txt"
                    cached_ranking = _read_trec_ranking(cached_trec) if cached_trec.is_file() else []
                    if cached_ranking:
                        is_random_perm = perm_label.startswith("random_s")
                        q_rankings.append(cached_ranking)
                        if is_random_perm:
                            q_rankings_random.append(cached_ranking)
                            q_positions_random.append(bucket)
                        q_positions.append(bucket)
                        det_path = pdir / "detailed_results.json"
                        if det_path.is_file():
                            cached_result = json.loads(det_path.read_text(encoding="utf-8"))
                            scores_init = cached_result.get("scores_init_order")
                            if scores_init is not None:
                                if len(scores_init) != len(perm_passages):
                                    raise ValueError(
                                        "cached scores_init_order length mismatch for "
                                        f"qid={qid} perm={perm_label}: got {len(scores_init)} "
                                        f"scores for {len(perm_passages)} passages"
                                    )
                                perm_pids = [str(p["pid"]) for p in perm_passages]
                                for position, (pid, score) in enumerate(zip(perm_pids, scores_init, strict=True)):
                                    score_f = float(score)
                                    q_scores.setdefault(pid, []).append(score_f)
                                    if is_random_perm:
                                        q_scores_random.setdefault(pid, []).append(score_f)
                                        if self.beta_gamma_enabled:
                                            self._beta_gamma_rows.append(
                                                {
                                                    "checkpoint": self.beta_gamma_checkpoint,
                                                    "recipe": self.beta_gamma_recipe,
                                                    "query_id": qid,
                                                    "perm_idx": q_random_perm_idx,
                                                    "doc_id": pid,
                                                    "position": position,
                                                    "score": score_f,
                                                    "permutation_label": perm_label,
                                                    "seed": (
                                                        self.seeds[q_random_perm_idx]
                                                        if q_random_perm_idx < len(self.seeds)
                                                        else None
                                                    ),
                                                }
                                            )
                        if is_random_perm:
                            q_random_perm_idx += 1
                        self.logger.info(
                            "qid=%s perm=%s: reusing on-disk artefact (k=%d/%d)",
                            qid,
                            perm_label,
                            k_idx + 1,
                            len(perms),
                        )
                        progress.skip_units(
                            count=1,
                            label=f"{qid}/{perm_label}",
                            q_index=q_index,
                            reason="perm_resumed_from_disk",
                        )
                        continue

                    active_label = f"{qid}/{perm_label}"
                    active_work_index, suffix = progress.start_unit(
                        label=active_label,
                        q_index=q_index,
                        extra={
                            "qid": qid,
                            "perm": perm_label,
                            "k_index": k_idx + 1,
                            "k_total": len(perms),
                            "bucket": bucket,
                            "render_variant": render_variant.label if render_variant else None,
                        },
                    )
                    self.logger.info(
                        "qid=%s perm=%s (k=%d/%d %s)",
                        qid,
                        perm_label,
                        k_idx + 1,
                        len(perms),
                        suffix,
                    )
                    # With query batching, t0 excludes the earlier batch call.
                    # Per-presentation progress durations are not valid latency
                    # measurements. Use the amortized ``prompting_runtimes``
                    # from detailed_results.json.
                    t0 = time.monotonic()
                    if k_idx in precomputed_results:
                        # Already scored in a batched engine call above. Render
                        # perms are never precomputed, so render_variant is None.
                        result = precomputed_results[k_idx]
                    else:
                        if render_variant is not None:
                            self.reranker.render_variant = render_variant
                        try:
                            result = self.reranker.rank(query, perm_passages)
                        finally:
                            if render_variant is not None:
                                self.reranker.render_variant = None
                    validate_rank_result_if_enabled(self.config, result, perm_passages)
                    top = result.get("top_k_psgs") or []
                    if not top:
                        self.logger.error("qid=%s perm=%s: empty output; skipping perm", qid, perm_label)
                        progress.finish_unit(
                            label=active_label,
                            work_index=active_work_index,
                            q_index=q_index,
                            duration_s=time.monotonic() - t0,
                            success=False,
                            extra={"qid": qid, "perm": perm_label, "reason": "empty_output"},
                        )
                        active_label = None
                        active_work_index = None
                        continue

                    ranking_pids = [str(p["pid"]) for p in top]
                    q_rankings.append(ranking_pids)
                    q_positions.append(bucket)

                    is_random_perm = perm_label.startswith("random_s")
                    if is_random_perm:
                        q_rankings_random.append(ranking_pids)
                        q_positions_random.append(bucket)

                    # Per-doc scores: align to docid regardless of input order,
                    # since the doc pool is identical across permutations.
                    scores_init = result.get("scores_init_order")
                    if scores_init is None and self.beta_gamma_enabled and is_random_perm:
                        raise ValueError(
                            "beta/gamma score logging requires reranker results to include "
                            f"scores_init_order (qid={qid} perm={perm_label})"
                        )
                    if scores_init is not None:
                        if len(scores_init) != len(perm_passages):
                            raise ValueError(
                                "scores_init_order length mismatch for "
                                f"qid={qid} perm={perm_label}: got {len(scores_init)} scores "
                                f"for {len(perm_passages)} passages"
                            )
                        perm_pids = [str(p["pid"]) for p in perm_passages]
                        for position, (pid, score) in enumerate(zip(perm_pids, scores_init, strict=True)):
                            score_f = float(score)
                            q_scores.setdefault(pid, []).append(score_f)
                            if is_random_perm:
                                q_scores_random.setdefault(pid, []).append(score_f)
                                if self.beta_gamma_enabled:
                                    self._beta_gamma_rows.append(
                                        {
                                            "checkpoint": self.beta_gamma_checkpoint,
                                            "recipe": self.beta_gamma_recipe,
                                            "query_id": qid,
                                            "perm_idx": q_random_perm_idx,
                                            "doc_id": pid,
                                            "position": position,
                                            "score": score_f,
                                            "permutation_label": perm_label,
                                            "seed": (
                                                self.seeds[q_random_perm_idx]
                                                if q_random_perm_idx < len(self.seeds)
                                                else None
                                            ),
                                        }
                                    )
                    if is_random_perm:
                        q_random_perm_idx += 1

                    # Write this permutation's artefacts.
                    pdir.mkdir(exist_ok=True)
                    with open(pdir / "detailed_results.json", "w") as f:
                        json.dump(result, f, indent=2)
                    write_trec_run(
                        pdir / "trec_results_raw.txt",
                        qid=qid,
                        ranked_docs=top,
                        tag=f"presentation_dependence:psi:{self.exp_id}:{perm_label}",
                    )
                    progress.finish_unit(
                        label=active_label,
                        work_index=active_work_index,
                        q_index=q_index,
                        duration_s=time.monotonic() - t0,
                        success=True,
                        extra={"qid": qid, "perm": perm_label, "bucket": bucket},
                    )
                    active_label = None
                    active_work_index = None
                    progress.maybe_heartbeat(self.logger)

                progress.skip_units(
                    count=max(planned_per_query - len(perms), 0),
                    label=f"qid={qid}",
                    q_index=q_index,
                    reason="permutation_strategy_skipped",
                )

                if q_rankings:
                    rankings_over_K[qid] = q_rankings
                    positions[qid] = q_positions
                    if q_scores:
                        scores_over_K[qid] = q_scores
                    if q_scores_random:
                        random_scores_over_K[qid] = q_scores_random
                    if q_rankings_random:
                        random_rankings_over_K[qid] = q_rankings_random
                        random_positions[qid] = q_positions_random

                    with open(qdir / "input_positions.json", "w") as f:
                        json.dump({"labels": q_positions}, f, indent=2)
                    # Persist docid-aligned random_shuffle scores for cheap
                    # post-hoc re-aggregation (e.g. K-curve probes); keeps the
                    # SC math reproducible from on-disk artefacts alone, no
                    # dataloader needed. Only meaningful when scalars exist.
                    if q_scores_random:
                        with open(qdir / "aligned_scores.json", "w") as f:
                            json.dump(
                                {
                                    "perturbation": "random_shuffle",
                                    "seeds": list(self.seeds),
                                    "scores": q_scores_random,
                                },
                                f,
                                indent=2,
                            )

            except Exception as e:
                self.logger.error("Failed on qid=%s: %s", qid, e, exc_info=True)
                failed_qids.append(qid)
                if active_label is not None and active_work_index is not None:
                    progress.finish_unit(
                        label=active_label,
                        work_index=active_work_index,
                        q_index=q_index,
                        success=False,
                        extra={"qid": qid, "reason": type(e).__name__},
                    )

        # Aggregate and persist.

        if failed_qids:
            raise partial_run_error(failed_qids, phase="PSI reranking", aggregate_label="aggregate PSI metrics")

        self._maybe_write_beta_gamma_score_log(
            {str(query["qid"]) for query in work_queries},
            beta_gamma_expected_docs,
        )

        if not rankings_over_K:
            self.logger.warning("No queries produced permutations; psi_metrics.json will be empty.")
            metrics = evaluate_psi(rankings_over_K={})
        else:
            # ``middle_injection`` is a targeted manipulation, not a draw from
            # the presentation distribution.  When uniform shuffles exist,
            # every expectation-style PSI quantity is reduced from those random
            # shots only.  Injection outputs are intentionally retained under
            # per_query_results for separate bucket analyses.
            random_only = "random_shuffle" in self.perturbations
            primary_rankings = random_rankings_over_K if random_only else rankings_over_K
            primary_scores = random_scores_over_K if random_only else scores_over_K
            primary_positions = random_positions if random_only else positions
            metrics = evaluate_psi(
                rankings_over_K=primary_rankings,
                scores_over_K=primary_scores or None,
                qrels=self.qrels or None,
                injection_positions=primary_positions or None,
                k_cutoff=self.k_cutoff,
            )
            metrics["protocol"]["presentation_aggregation"] = (
                "random_shuffle only" if random_only else "all generated presentations (no random_shuffle configured)"
            )
            metrics["protocol"]["excluded_from_presentation_aggregation"] = (
                [p for p in self.perturbations if p != "random_shuffle"] if random_only else []
            )

        coverage = self._build_coverage(
            work_queries=work_queries,
            skipped_qids=skipped_qids,
            aggregated_rankings=(random_rankings_over_K if "random_shuffle" in self.perturbations else rankings_over_K),
        )
        self._log_coverage(coverage)

        # Keep aggregates and protocol metadata in the headline file. Full
        # per-query detail is already stored under
        # per_query_results/<qid>/*. Matches the `metrics.json` convention.
        metrics_path = self.psi_dir / "psi_metrics.json"
        compact = build_psi_metrics_envelope(
            self.config,
            metrics,
            perturbations=self.perturbations,
            K=self.K,
            seeds=self.seeds,
            reranker=self.reranker,
            exp_id=self.exp_id,
            render_strategies=self.render_strategies or None,
        )
        compact["coverage"] = coverage
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(compact, f, indent=2, ensure_ascii=False)

        # Also persist the full per-query detail once, for debugging.
        full_path = self.psi_dir / "psi_per_query.json"
        with open(full_path, "w", encoding="utf-8") as f:
            json.dump(metrics["per_query"], f, indent=2, ensure_ascii=False)

        self.logger.info("Wrote %s", metrics_path)

        # Derive score-space self-consistency when aligned random-shuffle scores
        # and qrels are available.
        self._maybe_write_sc_metrics(random_scores_over_K)
        # Rank-space SC (Borda) for generative-listwise bases that expose no
        # per-doc scalar (so the score-space path above is a no-op).
        self._maybe_write_rank_sc_metrics(
            random_rankings_over_K,
            had_scalars=bool(random_scores_over_K),
        )
        return compact

    def _index_beta_gamma_rows(self) -> dict[str, dict[int, list[dict]]]:
        """Validate row identity and index beta/gamma rows by qid/permutation."""
        seen: set[tuple[str, str, int]] = set()
        by_query_perm: dict[str, dict[int, list[dict]]] = {}
        for row in self._beta_gamma_rows:
            qid = str(row["query_id"])
            doc_id = str(row["doc_id"])
            perm_idx = int(row["perm_idx"])
            key = (qid, doc_id, perm_idx)
            if key in seen:
                raise RuntimeError(f"Duplicate beta/gamma score row: qid={qid} doc={doc_id} perm={perm_idx}")
            seen.add(key)
            if not 0 <= perm_idx < self.K:
                raise RuntimeError(f"beta/gamma perm_idx={perm_idx} is outside configured K={self.K} for qid={qid}")
            if row.get("seed") != self.seeds[perm_idx]:
                raise RuntimeError(
                    f"beta/gamma seed mismatch for qid={qid} perm={perm_idx}: "
                    f"got {row.get('seed')}, expected {self.seeds[perm_idx]}"
                )
            by_query_perm.setdefault(qid, {}).setdefault(perm_idx, []).append(row)
        return by_query_perm

    def _validate_beta_gamma_coverage(
        self,
        by_query_perm: dict[str, dict[int, list[dict]]],
        expected_qids: set[str],
        expected_docs_by_qid: dict[str, set[str]],
    ) -> None:
        """Require complete qid, permutation, document, and position coverage."""
        actual_qids = set(by_query_perm)
        if actual_qids != expected_qids:
            missing = sorted(expected_qids - actual_qids)
            extra = sorted(actual_qids - expected_qids)
            raise RuntimeError(f"beta/gamma qid coverage mismatch: missing={missing[:10]} extra={extra[:10]}")

        expected_perms = set(range(self.K))
        for qid, per_perm in by_query_perm.items():
            if set(per_perm) != expected_perms:
                raise RuntimeError(
                    f"beta/gamma permutation coverage mismatch for qid={qid}: "
                    f"got={sorted(per_perm)} expected={sorted(expected_perms)}"
                )
            expected_docs = expected_docs_by_qid.get(qid)
            if not expected_docs:
                raise RuntimeError(f"beta/gamma expected document set is missing for qid={qid}")
            for perm_idx, rows in per_perm.items():
                docs = {str(row["doc_id"]) for row in rows}
                positions = {int(row["position"]) for row in rows}
                if docs != expected_docs:
                    raise RuntimeError(f"beta/gamma document coverage mismatch for qid={qid} perm={perm_idx}")
                if positions != set(range(len(expected_docs))):
                    raise RuntimeError(
                        f"beta/gamma position coverage mismatch for qid={qid} perm={perm_idx}: got={sorted(positions)}"
                    )

    def _maybe_write_beta_gamma_score_log(
        self,
        expected_qids: set[str],
        expected_docs_by_qid: dict[str, set[str]],
    ) -> None:
        """Persist row-level random-shuffle scores for beta/gamma analysis."""
        if not self.beta_gamma_enabled:
            return
        if not self._beta_gamma_rows:
            raise RuntimeError("beta/gamma score logging was enabled, but no random-shuffle score rows were collected")
        by_query_perm = self._index_beta_gamma_rows()
        self._validate_beta_gamma_coverage(by_query_perm, expected_qids, expected_docs_by_qid)
        write_score_log(self._beta_gamma_rows, self.beta_gamma_score_log_path)
        self.logger.info(
            "Wrote beta/gamma score log: %s (%d rows)",
            self.beta_gamma_score_log_path,
            len(self._beta_gamma_rows),
        )

    def _maybe_write_sc_metrics(
        self,
        random_scores_over_K: dict[str, dict[str, list[float]]],
    ) -> None:
        """Write K-shot score-space SC metrics when inputs are available.

        The method requires enabled derivation, random shuffles, qrels, and
        aligned scalar scores. Missing inputs are logged and do not fail the
        primary PSI run.
        """
        if not self.derive_sc:
            self.logger.info("SC derivation disabled by config; skipping sc_metrics.json")
            return
        if "random_shuffle" not in self.perturbations:
            self.logger.info(
                "SC needs random_shuffle perms (got %s); skipping sc_metrics.json",
                self.perturbations,
            )
            return
        if not self.qrels:
            self.logger.info("No qrels loaded; skipping sc_metrics.json")
            return
        if not random_scores_over_K:
            self.logger.info(
                "No docid-aligned random_shuffle scores (reranker likely emits "
                "no scores_init_order); skipping sc_metrics.json"
            )
            return

        sc = derive_sc_metrics(
            scores_per_qid=random_scores_over_K,
            qrels=self.qrels,
            measures=self.sc_measures,
            k_subsets=self.sc_k_subsets,
        )
        headline, per_query_by_K = split_headline_and_verbose(sc)
        # Provenance keys mirror psi_metrics.json envelope conventions.
        headline.update(
            {
                "exp_id": self.exp_id,
                "perturbation": "random_shuffle",
                "seeds": list(self.seeds),
                "K_full": self.K,
                "measures": list(self.sc_measures),
                "n_queries_with_scores": len(random_scores_over_K),
            }
        )

        sc_path = self.psi_dir / "sc_metrics.json"
        with open(sc_path, "w", encoding="utf-8") as f:
            json.dump(headline, f, indent=2, ensure_ascii=False)
        sc_per_query_path = self.psi_dir / "sc_per_query.json"
        with open(sc_per_query_path, "w", encoding="utf-8") as f:
            json.dump(per_query_by_K, f, indent=2, ensure_ascii=False)

        # Log the first measure at each K.
        first_measure = self.sc_measures[0]
        for K, block in sorted(headline.get("by_K", {}).items()):
            m = block.get("metrics", {}).get(first_measure, {})
            self.logger.info(
                "SC K=%d %s mean=%s n=%s",
                K,
                first_measure,
                m.get("mean"),
                m.get("n"),
            )
        self.logger.info("Wrote %s", sc_path)

    def _read_base_ndcg(self) -> dict[str, float]:
        """Read BM25-order base-pass means from the sibling ``metrics.json``.

        Returns ``{measure: mean}`` for ``self.sc_measures`` present in the file
        (the ``mean_<measure>`` keys ``ExperimentManager`` writes). Empty when
        the file is absent, as in a standalone PSI run. In that case, the SC
        lift cannot be computed.
        """
        metrics_path = self.run_dir / "metrics.json"
        if not metrics_path.is_file():
            return {}
        try:
            data = json.loads(metrics_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        out: dict[str, float] = {}
        for m in self.sc_measures:
            v = data.get(f"mean_{m}")
            if isinstance(v, (int, float)):
                out[m] = float(v)
        return out

    def _maybe_write_rank_sc_metrics(
        self,
        random_rankings_over_K: dict[str, list[list[str]]],
        *,
        had_scalars: bool,
    ) -> None:
        """Write rank-space K-shot SC metrics for ranking-only rerankers.

        The method requires enabled derivation, random shuffles, qrels, no
        scalar scores, and at least one stored ranking. Missing inputs are logged
        and do not fail the primary PSI run.
        """
        if not self.derive_sc:
            self.logger.info("SC derivation disabled by config; skipping rank_sc_metrics.json")
            return
        if "random_shuffle" not in self.perturbations:
            self.logger.info(
                "Rank-SC needs random_shuffle perms (got %s); skipping rank_sc_metrics.json",
                self.perturbations,
            )
            return
        if not self.qrels:
            self.logger.info("No qrels loaded; skipping rank_sc_metrics.json")
            return
        if had_scalars:
            self.logger.info("Reranker emits per-doc scalars; using score-space SC and skipping rank_sc_metrics.json")
            return
        if not random_rankings_over_K:
            self.logger.info("No random_shuffle rankings collected; skipping rank_sc_metrics.json")
            return

        rank_sc = derive_rank_sc_metrics(
            rankings_per_qid=random_rankings_over_K,
            qrels=self.qrels,
            measures=self.sc_measures,
            k_subsets=self.sc_k_subsets,
        )
        headline, per_query_by_K = split_headline_and_verbose(rank_sc)
        base_ndcg = self._read_base_ndcg()
        primary = self.sc_measures[0]
        lift = compute_sc_lift(rank_sc, base_ndcg=base_ndcg.get(primary), measure=primary)
        headline.update(
            {
                "exp_id": self.exp_id,
                "perturbation": "random_shuffle",
                "seeds": list(self.seeds),
                "K_full": self.K,
                "measures": list(self.sc_measures),
                "n_queries_with_rankings": len(random_rankings_over_K),
                "k1_base_ndcg_by_measure": base_ndcg,
                "sc_lift": lift,
            }
        )

        rank_sc_path = self.psi_dir / "rank_sc_metrics.json"
        with open(rank_sc_path, "w", encoding="utf-8") as f:
            json.dump(headline, f, indent=2, ensure_ascii=False)
        rank_sc_pq_path = self.psi_dir / "rank_sc_per_query.json"
        with open(rank_sc_pq_path, "w", encoding="utf-8") as f:
            json.dump(per_query_by_K, f, indent=2, ensure_ascii=False)

        for K, block in sorted(headline.get("by_K", {}).items()):
            m = block.get("metrics", {}).get(primary, {})
            self.logger.info("Rank-SC K=%d %s mean=%s n=%s", K, primary, m.get("mean"), m.get("n"))
        if lift.get("sc_lift") is not None:
            self.logger.info(
                "Rank-SC lift %s = %.4f (K=%s aggregate %.4f \u2212 K=1 base %.4f)",
                primary,
                lift["sc_lift"],
                lift["k_aggregate"],
                lift["k_aggregate_ndcg"],
                lift["k1_base_ndcg"],
            )
        self.logger.info("Wrote %s", rank_sc_path)
