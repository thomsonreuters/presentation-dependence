#!/usr/bin/env python
r"""Derive K-shot score-space self-consistency from stored PSI scores.

Current PSI runs write
``psi/sc_metrics.json`` + ``psi/sc_per_query.json`` at the end of
:meth:`presentation_dependence.eval.psi_manager.PsiExperimentRunner.run`. This command
supports older runs without ``aligned_scores.json`` and recomputation with
different ``k_subsets`` or ``measures`` after changes to
:mod:`presentation_dependence.eval.self_consistency`.

Two input shapes are supported, in priority order:

- Aligned-scores path: each
  ``psi/per_query_results/<qid>/aligned_scores.json`` carries the
  docid-to-score-list mapping directly. No dataloader or GPU is needed.

- Reconstruction path: walks
  ``psi/per_query_results/<qid>/permutation_NNN_random_s<seed>/detailed_results.json``,
  re-derives the input passage order by replaying the same Fisher-Yates
  shuffle (``random.Random(seed).shuffle(<first-stage-passages>)``) using
  the dataloader from ``resolved_config.yaml``, and zips
  ``scores_init_order`` against the reconstructed pid list. This requires
  the dataloader to be instantiable (e.g. pyserini index downloaded for
  ``PyseriniLoader``, or fixture files present for ``FixtureLoader``).

Outputs use the same paths as inline PSI aggregation::

    runs/<ID>/<ts>/psi/sc_metrics.json
    runs/<ID>/<ts>/psi/sc_per_query.json

Both files carry a ``derivation`` block in the aggregate output so consumers
can tell which path produced the artefact ("inline" vs "post_hoc").

Usage::

    uv run python scripts/analyze/derive_self_consistency_from_psi.py runs/<ID>/<ts>
    uv run python scripts/analyze/derive_self_consistency_from_psi.py -e <ID>  # latest run
    uv run python scripts/analyze/derive_self_consistency_from_psi.py runs/<ID>/<ts> \\
        --measures ndcg_cut_10 ndcg_cut_5 map recip_rank \\
        --k-subsets 1 2 3 5 10 20

CLI flags map onto :func:`presentation_dependence.eval.self_consistency.derive_sc_metrics`
parameters; that function's docstring defines their semantics.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path

import yaml

from presentation_dependence.eval.runner_setup import instantiate_dataloader, load_qrels
from presentation_dependence.eval.self_consistency import derive_sc_metrics, split_headline_and_verbose
from presentation_dependence.utils.run_paths import default_runs_root, resolve_run_dir
from presentation_dependence.utils.trec import dirname_to_qid


_LOG = logging.getLogger("derive_sc")


def _read_resolved_config(run_dir: Path) -> dict:
    cfg_path = run_dir / "resolved_config.yaml"
    if not cfg_path.is_file():
        raise FileNotFoundError(f"Missing resolved_config.yaml under {run_dir}")
    with open(cfg_path, "r") as f:
        return yaml.safe_load(f)


def _aligned_scores_fast_path(psi_dir: Path) -> dict[str, dict[str, list[float]]] | None:
    """Try the modern fast path: read pre-aligned per-qid scores.

    Returns ``None`` when at least one qid is missing ``aligned_scores.json``,
    forcing the caller to fall back to reconstruction.
    """
    per_query_dir = psi_dir / "per_query_results"
    if not per_query_dir.is_dir():
        return None

    out: dict[str, dict[str, list[float]]] = {}
    n_qids = 0
    n_with_aligned = 0
    for qdir in sorted(per_query_dir.iterdir()):
        if not qdir.is_dir():
            continue
        n_qids += 1
        ap = qdir / "aligned_scores.json"
        if not ap.is_file():
            continue
        with open(ap, "r") as f:
            payload = json.load(f)
        scores = payload.get("scores")
        if not isinstance(scores, dict) or not scores:
            continue
        qid = dirname_to_qid(qdir.name)
        out[qid] = {str(k): [float(x) for x in v] for k, v in scores.items()}
        n_with_aligned += 1

    if n_qids == 0 or n_with_aligned == 0:
        return None
    if n_with_aligned < n_qids:
        _LOG.warning(
            "aligned_scores.json present for %d/%d qids — falling back to reconstruction "
            "to avoid a partial SC aggregate.",
            n_with_aligned,
            n_qids,
        )
        return None
    _LOG.info("Fast path: loaded aligned scores for %d qids", n_with_aligned)
    return out


def _reconstruct_via_dataloader(  # noqa: C901
    psi_dir: Path,
    config: dict,
    seeds: list[int],
) -> dict[str, dict[str, list[float]]]:
    """Walk per-permutation detailed_results.json files and re-align by
    replaying the Fisher-Yates shuffle for each seed.
    """
    per_query_dir = psi_dir / "per_query_results"
    if not per_query_dir.is_dir():
        raise FileNotFoundError(f"Missing {per_query_dir}")

    dataloader = instantiate_dataloader(config)
    run_path = config["data"]["run_path"]

    out: dict[str, dict[str, list[float]]] = {}
    for qdir in sorted(per_query_dir.iterdir()):
        if not qdir.is_dir():
            continue
        qid = dirname_to_qid(qdir.name)
        passages = dataloader.get_psgs_from_run(run_path, qid) or []
        if not passages:
            _LOG.warning("qid=%s: empty passage list; skipping", qid)
            continue
        first_stage_pids = [str(p["pid"]) for p in passages]

        q_aligned: dict[str, list[float]] = {}
        for seed in seeds:
            # Re-derive input pid order: same passages list + same seed → same shuffle.
            shuffled_pids = list(first_stage_pids)
            random.Random(int(seed)).shuffle(shuffled_pids)

            pdir = qdir / f"permutation_{_pad_index_for_seed(qdir, seed)}_random_s{seed}"
            if not pdir.is_dir():
                # Try a slow scan in case the index padding differs (e.g. legacy runs).
                pdir = _find_perm_dir(qdir, seed)
                if pdir is None:
                    _LOG.warning("qid=%s seed=%s: missing permutation dir; skipping seed", qid, seed)
                    continue
            with open(pdir / "detailed_results.json", "r") as f:
                result = json.load(f)
            scores_init = result.get("scores_init_order")
            if scores_init is None:
                _LOG.warning("qid=%s seed=%s: no scores_init_order; skipping seed", qid, seed)
                continue
            if len(scores_init) != len(shuffled_pids):
                raise ValueError(
                    f"qid={qid} seed={seed}: scores_init_order length "
                    f"({len(scores_init)}) != reconstructed shuffle length "
                    f"({len(shuffled_pids)})"
                )
            for pid, score in zip(shuffled_pids, scores_init, strict=True):
                q_aligned.setdefault(str(pid), []).append(float(score))
        if q_aligned:
            out[qid] = q_aligned

    _LOG.info("Reconstruction path: aligned scores for %d qids", len(out))
    return out


def _pad_index_for_seed(qdir: Path, seed: int) -> str:
    """Best-effort guess for the 3-digit zero-padded permutation prefix.

    The PSI driver names dirs ``permutation_{k_idx:03d}_<label>``, where
    ``k_idx`` is the position in the per-query perm sequence. For the default
    config (random_shuffle only with seeds=[0..K-1]), ``k_idx == seed``, so a
    ``f"{seed:03d}"`` lookup hits directly.
    """
    return f"{int(seed):03d}"


def _find_perm_dir(qdir: Path, seed: int) -> Path | None:
    """Fallback scan: any subdir whose name ends with ``random_s<seed>``."""
    suffix = f"random_s{seed}"
    for sub in qdir.iterdir():
        if sub.is_dir() and sub.name.endswith(suffix):
            return sub
    return None


def _guess_local_run_path(configured: str) -> Path | None:
    """Map a container fixture path to a likely local data/ path.

    Legacy container-artifact examples:
    --------
    /opt/ml/input/data/dl19-passage/fixture.jsonl -> data/dl19-passage/fixture.jsonl
    /opt/ml/input/data/beir-v1.0.0-webis-touche2020-test/fixture.jsonl
        -> data/beir-v1.0.0-webis-touche2020-test/fixture.jsonl
    """
    p = Path(configured)
    parts = p.parts
    if "data" in parts:
        idx = parts.index("data")
        relative = Path(*parts[idx:])
        if relative != p:
            return relative
    return None


def main() -> int:  # noqa: C901
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = parser.add_mutually_exclusive_group(required=True)
    g.add_argument("run_dir", nargs="?", default=None, help="Path to runs/<ID>/<ts>/.")
    g.add_argument("-e", "--exp-id", default=None, help="Experiment id (resolves to latest run).")
    parser.add_argument(
        "--runs-root",
        default=default_runs_root(),
        help="Root directory for run artefacts (default: SLM_RUNS_ROOT, else runs/).",
    )
    parser.add_argument(
        "--measures",
        nargs="+",
        default=None,
        help="pytrec_eval measures to compute. Defaults to eval.measures from the run config.",
    )
    parser.add_argument(
        "--k-subsets",
        nargs="+",
        type=int,
        default=None,
        help="K' truncation values to compute. Defaults to robustness.sc_k_subsets, "
        "or [1,2,3,5,10,20] when absent. Values > stored K are silently dropped.",
    )
    parser.add_argument(
        "--force-reconstruction",
        action="store_true",
        help="Skip the aligned_scores.json fast path; always re-derive via dataloader replay.",
    )
    parser.add_argument(
        "--data-run-path",
        default=None,
        help="Override ``data.run_path`` from resolved_config.yaml. Use when a "
        "an older run baked an absolute /opt/ml path that no longer "
        "resolves locally (e.g. point at data/<surface>/fixture.jsonl).",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable INFO logging.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    target = args.run_dir or args.exp_id
    run_dir = resolve_run_dir(target, runs_root=args.runs_root).resolve()
    psi_dir = run_dir / "psi"
    if not psi_dir.is_dir():
        raise FileNotFoundError(f"No psi/ subdir under {run_dir}")

    config = _read_resolved_config(run_dir)

    if args.data_run_path is not None:
        config.setdefault("data", {})["run_path"] = args.data_run_path
        _LOG.info("Overriding data.run_path -> %s", args.data_run_path)
    # Legacy artifact rewrite: /opt/ml/input/data/<surface>/... becomes
    # data/<surface>/... when the old mounted path does not resolve locally.
    # Applied to data.run_path, data.topics_tsv, and eval.qrels_path so a legacy
    # container-resolved config runs locally without manual edits.
    for sect, key in (("data", "run_path"), ("data", "topics_tsv"), ("eval", "qrels_path")):
        block = config.get(sect) or {}
        configured = block.get(key)
        if not configured or Path(configured).exists():
            continue
        local_guess = _guess_local_run_path(configured)
        if local_guess is not None and local_guess.exists():
            config.setdefault(sect, {})[key] = str(local_guess)
            _LOG.info("%s.%s %r missing locally; auto-rewriting to %s", sect, key, configured, local_guess)

    qrels_path = (config.get("eval") or {}).get("qrels_path")
    if not qrels_path:
        raise ValueError("eval.qrels_path missing in resolved_config.yaml")
    qrels = load_qrels(qrels_path)

    robust_cfg = config.get("robustness") or {}
    K = int(robust_cfg.get("K", 10))
    seeds_raw = robust_cfg.get("seeds")
    seeds = list(seeds_raw) if seeds_raw is not None else list(range(K))

    measures = (
        args.measures or robust_cfg.get("sc_measures") or (config.get("eval") or {}).get("measures") or ["ndcg_cut_10"]
    )
    k_subsets = args.k_subsets or robust_cfg.get("sc_k_subsets") or [1, 2, 3, 5, 10, 20]

    aligned: dict[str, dict[str, list[float]]] | None = None
    via = "post_hoc:aligned_scores"
    if not args.force_reconstruction:
        aligned = _aligned_scores_fast_path(psi_dir)
    if aligned is None:
        _LOG.info("Falling back to reconstruction via dataloader replay")
        aligned = _reconstruct_via_dataloader(psi_dir, config, seeds)
        via = "post_hoc:reconstruction"

    if not aligned:
        raise RuntimeError("No aligned scores recovered — check the run's perturbation set / scalars.")

    sc = derive_sc_metrics(
        scores_per_qid=aligned,
        qrels=qrels,
        measures=measures,
        k_subsets=k_subsets,
    )
    headline, per_query_by_K = split_headline_and_verbose(sc)
    headline.update(
        {
            "exp_id": config.get("id"),
            "perturbation": "random_shuffle",
            "seeds": seeds,
            "K_full": K,
            "measures": list(measures),
            "n_queries_with_scores": len(aligned),
            "derivation": {
                "path": via,
                "script": "scripts/analyze/derive_self_consistency_from_psi.py",
            },
        }
    )

    sc_path = psi_dir / "sc_metrics.json"
    with open(sc_path, "w", encoding="utf-8") as f:
        json.dump(headline, f, indent=2, ensure_ascii=False)
    sc_per_query_path = psi_dir / "sc_per_query.json"
    with open(sc_per_query_path, "w", encoding="utf-8") as f:
        json.dump(per_query_by_K, f, indent=2, ensure_ascii=False)

    print(f"[OK] wrote {sc_path}")
    print(f"[OK] wrote {sc_per_query_path}")
    first_measure = list(measures)[0]
    for K_, block in sorted(headline.get("by_K", {}).items()):
        m = block.get("metrics", {}).get(first_measure, {})
        print(f"     K={K_:>3} {first_measure} mean={m.get('mean')} n={m.get('n')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
