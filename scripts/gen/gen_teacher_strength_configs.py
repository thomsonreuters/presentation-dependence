#!/usr/bin/env python
"""Generate the teacher-strength training configs and their sweep YAML.

Each generated config mirrors a tracked student template from
``configs/self-distill/``, changing only the teacher's silver labels (train and
held-out), the id, and the output directory. Lambda is whatever the template
already carries, so the arm differs from its template in the teacher alone.

The study runs in phases, one per invocation of ``--phase``:

``tier1-wave1``
    One arm per teacher into Qwen3-4B at the ported lambda, plus frontier-teacher
    brackets that select lambda for Gemma-E4B and Granite-8B.
``tier1-wave2``
    The facet-2 teachers deferred from wave 1, once those brackets have resolved.
``tier2-facet1-sft``
    K=1 against K=10 SFT for four teachers into Qwen3-4B.
``tier2-facet2-sft``
    The same contrast into Gemma-E4B and Granite-8B, testing whether it holds
    across model families.
``tier3-density-fill-sft``
    The remaining facet-1 teachers, completing the grid.

Writes config files and one sweep YAML per phase; it never launches a run.
Dry-run is the default.

Usage:
    uv run python -m scripts.gen.gen_teacher_strength_configs --phase tier1-wave1
    uv run python -m scripts.gen.gen_teacher_strength_configs --phase tier1-wave1 --write
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
CFG_DIR = REPO / "configs/self-distill"
SWEEP_DIR = REPO / "configs/sweeps"

# Sweep ids, one per phase. Each is also the generated sweep's filename stem.
TRAIN_ANCHOR = "teacher-strength-tier1-wave1"

# student -> self-distill K=1 OC-SFT template id with a {code} slot (templates exist
# for every lambda code; the lambda is baked into each template).
STUDENT_TEMPLATE_BASE = {
    "qwen3_4b": "qwen3-4b-nonthink-k1-supervised-consistency-lambda{code}-warmup500-msmarco-30k",
    "gemma4_e4b": "gemma4-k1-supervised-consistency-lambda{code}-warmup500-msmarco-30k",
    "granite41_8b": "granite-41-8b-k1-supervised-consistency-lambda{code}-warmup500-msmarco-30k",
}

PORT_CODE = {"qwen3_4b": "500", "gemma4_e4b": "200", "granite41_8b": "500"}
BRACKET = {"gemma4_e4b": ["100", "200", "300"], "granite41_8b": ["300", "400", "500"]}
FRONTIER = "gpt54"

FACET1_TEACHERS = [
    "qwen3_1p7b",
    "qwen3_4b",
    "qwen3_8b",
    "qwen3_14b",
    "qwen3_32b",
    "gemma4_e2b",
    "gemma4_e4b",
    "gemma4_31b",
    "granite41_3b",
    "granite41_8b",
    "granite41_30b",
    "gpt54",
]
FACET2 = {"gemma4_e4b": ["qwen3_1p7b", "gemma4_31b", "gpt54"], "granite41_8b": ["qwen3_1p7b", "granite41_30b", "gpt54"]}
# Wave 2 runs the deferred facet-2 teachers once each student's bracket has
# resolved; the code below is that student's bracket-winning lambda.
WAVE2 = {"granite41_8b": ("400", ["qwen3_1p7b", "granite41_30b"]), "gemma4_e4b": ("300", ["qwen3_1p7b", "gemma4_31b"])}
WAVE2_ANCHOR = "teacher-strength-tier1-wave2"
HAVE = {("qwen3_4b", "qwen3_4b"), ("gpt54", "qwen3_4b"), ("gemma4_e4b", "gemma4_e4b"), ("granite41_8b", "granite41_8b")}
# teacher key -> silver filename stem
TEACHER_SILVER_STEM = {
    "qwen3_1p7b": "qwen3_1p7b",
    "qwen3_4b": "qwen3_4b",
    "qwen3_8b": "qwen3_8b",
    "qwen3_14b": "qwen3_14b",
    "qwen3_32b": "qwen3_32b",
    "gemma4_e2b": "gemma4_e2b",
    "gemma4_e4b": "gemma4",
    "gemma4_31b": "gemma4_31b",
    "granite41_3b": "granite41_3b",
    "granite41_8b": "granite41",
    "granite41_30b": "granite41_30b",
    "gpt54": "gpt54_generated_bsc",
}

SILVER_TRAIN_RE = re.compile(r"silver_labels_[A-Za-z0-9_]+_k1_seed0_train_29500\.jsonl")
SILVER_HELDOUT_RE = re.compile(r"silver_labels_[A-Za-z0-9_]+_k1_seed0_heldout_500\.jsonl")

# --- Tier-2 Facet-1: SFT contrast (K=1 + K=10 SFT)
TIER2_F1_TEACHERS = ["qwen3_1p7b", "qwen3_32b", "gemma4_31b", "granite41_8b"]
SFT_TEMPLATE = {  # student -> {recipe: self-distill SFT template id}
    "qwen3_4b": {
        "k1": "qwen3-4b-nonthink-sft-msmarco-30k-k1-labels",
        "k10": "qwen3-4b-nonthink-sft-msmarco-30k-k10-labels",
    },
    "gemma4_e4b": {"k1": "gemma4-sft-msmarco-30k-k1-labels", "k10": "gemma4-sft-msmarco-30k-k10-labels"},
    "granite41_8b": {
        "k1": "granite-41-8b-sft-msmarco-30k-k1-labels",
        "k10": "granite-41-8b-sft-msmarco-30k-k10-labels",
    },
}
TIER2_F1_ANCHOR = "teacher-strength-tier2-facet1-sft"
SFT_TRAIN_RE = re.compile(r"silver_labels_[A-Za-z0-9_]+_(?:k1_seed0|k10)_train_29500\.jsonl")
SFT_HELDOUT_RE = re.compile(r"silver_labels_[A-Za-z0-9_]+_(?:k1_seed0|k10)_heldout_500\.jsonl")

# --- Tier-2 Facet-2: does the SFT recipe contrast hold on Gemma-E4B and Granite-8B?
FACET2_SFT_ANCHOR = "teacher-strength-tier2-facet2-sft"

# --- Tier-3: fill the remaining teachers of the facet-1 grid.
DENSITY_FILL_TEACHERS = ["qwen3_8b", "qwen3_14b", "gemma4_e2b", "gemma4_e4b", "granite41_3b", "granite41_30b"]
DENSITY_FILL_ANCHOR = "teacher-strength-tier3-density-fill-sft"
PHASES = (
    "tier1-wave1",
    "tier1-wave2",
    "tier2-facet1-sft",
    "tier2-facet2-sft",
    "tier3-density-fill-sft",
)


def make_sft_config_text(
    teacher: str,
    student: str,
    recipe: str,
    *,
    sweep_id: str = TIER2_F1_ANCHOR,
    cell_label: str = "tier-2 facet-1 SFT cell",
) -> tuple[str, str]:
    """Mirror the student's self-distill SFT template, swapping only the teacher silver."""
    tmpl_id = SFT_TEMPLATE[student][recipe]
    text = (CFG_DIR / f"{tmpl_id}.yaml").read_text()
    stem = TEACHER_SILVER_STEM[teacher]
    suffix = "k1_seed0" if recipe == "k1" else "k10"
    if teacher == FRONTIER:  # match existing gpt54-genbsc-* frontier SFT naming
        new_id = f"gpt54-genbsc-{slug(student)}-sft-msmarco-30k-{recipe}-labels"
    else:
        new_id = f"ts-{slug(teacher)}-to-{slug(student)}-sft-msmarco-30k-{recipe}-labels"
    text = SFT_TRAIN_RE.sub(f"silver_labels_{stem}_{suffix}_train_29500.jsonl", text)
    text = SFT_HELDOUT_RE.sub(f"silver_labels_{stem}_{suffix}_heldout_500.jsonl", text)
    text = text.replace(tmpl_id, new_id)
    header = (
        f"# TEACHER-STRENGTH {cell_label} (gen_teacher_strength_configs.py).\n"
        f"# teacher={teacher} -> student={student}; {recipe.upper()} SFT (regression, no lambda);\n"
        f"# silver swapped to teacher's {suffix} BSC. sweep={sweep_id}.\n"
    )
    return new_id, header + text


def _emit_tier2_facet1_sft(args) -> None:
    jobs = []
    print(f"{'action':8} {'config id'}")
    for teacher in TIER2_F1_TEACHERS:
        for recipe in ("k1", "k10"):
            new_id, text = make_sft_config_text(teacher, "qwen3_4b", recipe)
            cfg_path = CFG_DIR / f"{new_id}.yaml"
            jobs.append(f"configs/self-distill/{new_id}.yaml")
            exists = " (exists)" if cfg_path.exists() else ""
            print(f"{'WRITE' if args.write else 'would':8} {new_id}.yaml{exists}")
            if args.write:
                cfg_path.write_text(text)
    sweep_lines = [
        f"# Teacher-strength tier 2, facet 1 ({len(jobs)} cells): K=1 and K=10 SFT for four",
        "# teachers into Qwen3-4B. Plain regression, no lambda.",
        f"id: {TIER2_F1_ANCHOR}",
        "description: >-",
        "  Teacher-strength tier 2, facet 1: the order-averaged K=10 SFT tracer against the",
        "  K=1 SFT contrast, four teachers into Qwen3-4B. Checkpoint is the held-out nDCG",
        "  argmax at eval.",
        "execution:",
        "  max_parallel: 1",
        "jobs:",
    ]
    sweep_lines += [f"  - exp_id: {j}" for j in jobs]
    sweep_path = SWEEP_DIR / f"{TIER2_F1_ANCHOR}.yaml"
    print(f"\n{'WRITE' if args.write else 'would write'} sweep: configs/sweeps/{sweep_path.name} ({len(jobs)} jobs)")
    if args.write:
        sweep_path.write_text("\n".join(sweep_lines) + "\n")
        print(f"[ok] wrote {len(jobs)} SFT configs + sweep.")
    else:
        print("[dry-run] re-run with --phase tier2-facet1-sft --write to write them.")


def _write_sft_sweep(jobs: list[str], anchor: str, title: str, desc: str, max_parallel: int, args) -> None:
    sweep_lines = [f"# {line}" for line in title.splitlines()]
    sweep_lines += [
        f"id: {anchor}",
        "description: >-",
        *[f"  {d}" for d in desc.splitlines()],
        "execution:",
        f"  max_parallel: {max_parallel}",
        "jobs:",
    ]
    sweep_lines += [f"  - exp_id: {j}" for j in jobs]
    sweep_path = SWEEP_DIR / f"{anchor}.yaml"
    print(f"\n{'WRITE' if args.write else 'would write'} sweep: configs/sweeps/{sweep_path.name} ({len(jobs)} jobs)")
    if args.write:
        sweep_path.write_text("\n".join(sweep_lines) + "\n")
        print(f"[ok] wrote {len(jobs)} configs + sweep.")
    else:
        print("[dry-run] re-run with --write to write them.")


def _emit_facet2_sft(args) -> None:
    """Tier-2 facet 2: the K=1 and K=10 SFT pair for each facet-2 teacher into Gemma-E4B and Granite-8B."""
    jobs, skipped = [], []
    print(f"{'action':8} {'config id'}")
    for student, teachers in FACET2.items():
        for teacher in teachers:
            for recipe in ("k1", "k10"):
                new_id, text = make_sft_config_text(
                    teacher,
                    student,
                    recipe,
                    sweep_id=FACET2_SFT_ANCHOR,
                    cell_label="tier-2 facet-2 SFT cell",
                )
                cfg_path = CFG_DIR / f"{new_id}.yaml"
                if cfg_path.exists():
                    skipped.append(new_id)
                    print(f"{'skip':8} {new_id}.yaml (exists)")
                    continue
                jobs.append(f"configs/self-distill/{new_id}.yaml")
                print(f"{'WRITE' if args.write else 'would':8} {new_id}.yaml")
                if args.write:
                    cfg_path.write_text(text)
    _write_sft_sweep(
        jobs,
        FACET2_SFT_ANCHOR,
        f"Teacher-strength tier 2, facet 2 ({len(jobs)} cells): K=1 and K=10 SFT for a weak,\n"
        "a strong open 30B, and a frontier teacher into Gemma-E4B and Granite-8B.\n"
        "Run after the tier-2 facet-1 cells. Gemma-E4B students need their adapter merged\n"
        "before eval; see docs/MODELS.md.",
        "Teacher-strength tier 2, facet 2: whether the SFT recipe contrast holds across model\n"
        "families. Checkpoint is the held-out nDCG argmax at eval. Self-teaching cells are\n"
        "excluded because facet 1 already covers them.",
        max_parallel=1,
        args=args,
    )
    if skipped:
        print(
            f"\n[note] {len(skipped)} pre-existing (not re-written, verify trained before eval): " + ", ".join(skipped)
        )


def _emit_density_fill_sft(args) -> None:
    """Tier-3: the K=1 and K=10 SFT pair for the six remaining facet-1 teachers into Qwen3-4B."""
    jobs = []
    print(f"{'action':8} {'config id'}")
    for teacher in DENSITY_FILL_TEACHERS:
        for recipe in ("k1", "k10"):
            new_id, text = make_sft_config_text(
                teacher,
                "qwen3_4b",
                recipe,
                sweep_id=DENSITY_FILL_ANCHOR,
                cell_label="tier-3 density-fill SFT cell",
            )
            cfg_path = CFG_DIR / f"{new_id}.yaml"
            jobs.append(f"configs/self-distill/{new_id}.yaml")
            exists = " (exists)" if cfg_path.exists() else ""
            print(f"{'WRITE' if args.write else 'would':8} {new_id}.yaml{exists}")
            if args.write:
                cfg_path.write_text(text)
    _write_sft_sweep(
        jobs,
        DENSITY_FILL_ANCHOR,
        f"Teacher-strength tier 3 ({len(jobs)} cells): K=1 and K=10 SFT for the six\n"
        "remaining facet-1 teachers into Qwen3-4B, completing the grid. Confirmatory:\n"
        "it adds density, not a new claim, so it is the first phase to cut.",
        "Teacher-strength tier 3: the SFT pair for the six remaining facet-1 teachers\n"
        "into Qwen3-4B. Checkpoint is the held-out nDCG argmax at eval.",
        max_parallel=1,
        args=args,
    )


def slug(key: str) -> str:
    return key.replace("_", "-")


def make_config_text(teacher: str, student: str, code: str) -> tuple[str, str]:
    tmpl_id = STUDENT_TEMPLATE_BASE[student].format(code=code)
    tmpl_path = CFG_DIR / f"{tmpl_id}.yaml"
    text = tmpl_path.read_text()
    stem = TEACHER_SILVER_STEM[teacher]
    new_id = f"ts-{slug(teacher)}-to-{slug(student)}-k1-oc-sft-l{code}-warmup500-msmarco-30k"
    text = SILVER_TRAIN_RE.sub(f"silver_labels_{stem}_k1_seed0_train_29500.jsonl", text)
    text = SILVER_HELDOUT_RE.sub(f"silver_labels_{stem}_k1_seed0_heldout_500.jsonl", text)
    text = text.replace(tmpl_id, new_id)  # rewrites id: and output_dir:
    header = (
        f"# TEACHER-STRENGTH Tier-1 cell (generated by gen_teacher_strength_configs.py).\n"
        f"# teacher={teacher} -> student={student}; silver swapped to teacher's K=1 BSC.\n"
        f"# lambda candidate selected by the phase-specific port/bracket protocol. sweep={TRAIN_ANCHOR}.\n"
    )
    return new_id, header + text


def _emit_wave2(args) -> None:
    """Tier-1 wave 2: the four deferred facet-2 cells at each student's bracket-winning lambda."""
    jobs = []
    print(f"{'action':8} {'config id'}")
    for student, (code, teachers) in WAVE2.items():
        for teacher in teachers:
            new_id, text = make_config_text(teacher, student, code)
            cfg_path = CFG_DIR / f"{new_id}.yaml"
            jobs.append(f"configs/self-distill/{new_id}.yaml")
            exists = " (exists)" if cfg_path.exists() else ""
            print(f"{'WRITE' if args.write else 'would':8} {new_id}.yaml{exists}")
            if args.write:
                cfg_path.write_text(text)
    sweep_lines = [
        f"# Teacher-strength tier 1, wave 2 ({len(jobs)} cells): the deferred facet-2 teachers",
        "# at each student's bracket-winning lambda (Granite-8B 400, Gemma-E4B 300).",
        f"id: {WAVE2_ANCHOR}",
        "description: >-",
        "  Teacher-strength tier 1, wave 2: the four deferred facet-2 cells at the lambda the",
        "  wave-1 bracket selected.",
        "execution:",
        "  max_parallel: 1",
        "jobs:",
    ]
    sweep_lines += [f"  - exp_id: {j}" for j in jobs]
    sweep_text = "\n".join(sweep_lines) + "\n"
    sweep_path = SWEEP_DIR / f"{WAVE2_ANCHOR}.yaml"
    print(f"\n{'WRITE' if args.write else 'would write'} sweep: configs/sweeps/{sweep_path.name} ({len(jobs)} jobs)")
    if args.write:
        sweep_path.write_text(sweep_text)
        print(f"[ok] wrote {len(jobs)} wave-2 configs + sweep.")
    else:
        print("\n--- wave-2 sweep YAML preview ---")
        print(sweep_text)
        print("[dry-run] no files written. Re-run with --phase tier1-wave2 --write to write them.")


def _emit_wave1(args: argparse.Namespace) -> None:  # noqa: C901
    """Write the Tier-1 wave-1 single arms and frontier brackets."""
    # Build WAVE-1 cell list: (teacher, student, code, kind).
    wave1: list[tuple[str, str, str, str]] = []
    # Facet-1: single ported-lambda arm per NEW teacher -> Qwen3-4B.
    for t in FACET1_TEACHERS:
        if (t, "qwen3_4b") in HAVE:
            continue
        wave1.append((t, "qwen3_4b", PORT_CODE["qwen3_4b"], "single"))
    # Facet-2: bracket on the frontier teacher (GPT-5.4) per uncertain student.
    for student, codes in BRACKET.items():
        for code in codes:
            wave1.append((FRONTIER, student, code, "bracket"))
    # Deferred WAVE-2: the non-frontier teachers of the bracketed students; lambda
    # is the bracket winner (unknown now) -> not written until the bracket resolves.
    wave2: list[tuple[str, str]] = []
    for student, teachers in FACET2.items():
        for t in teachers:
            if t == FRONTIER or (t, student) in HAVE:
                continue
            wave2.append((t, student))

    jobs = []
    print(f"{'action':8} {'kind':8} {'config id'}")
    for teacher, student, code, kind in wave1:
        new_id, text = make_config_text(teacher, student, code)
        cfg_path = CFG_DIR / f"{new_id}.yaml"
        jobs.append(f"configs/self-distill/{new_id}.yaml")
        action = "WRITE" if args.write else "would"
        exists = " (exists)" if cfg_path.exists() else ""
        print(f"{action:8} {kind:8} {new_id}.yaml{exists}")
        if args.write:
            cfg_path.write_text(text)

    sweep_lines = [
        f"# Teacher-strength tier 1, wave 1 ({len(jobs)} runs): "
        "10 single-lambda arms into Qwen3-4B, plus 2 frontier brackets (Gemma-E4B 1/2/3, Granite-8B 3/4/5).",
        f"id: {TRAIN_ANCHOR}",
        "description: >-",
        "  Teacher-strength tier 1, wave 1: the Qwen3-4B headline at the ported lambda=5 as a",
        "  single-arm curve screen, plus frontier-teacher brackets that select lambda for",
        "  Gemma-E4B and Granite-8B. Pick checkpoints with",
        "  scripts/analyze/screen_lambda_from_curve.py and scripts/select_lambda.py.",
        "execution:",
        "  max_parallel: 1",
        "jobs:",
    ]
    for j in jobs:
        sweep_lines.append(f"  - exp_id: {j}")
    sweep_text = "\n".join(sweep_lines) + "\n"
    sweep_path = SWEEP_DIR / f"{TRAIN_ANCHOR}.yaml"

    print(f"\n{'WRITE' if args.write else 'would write'} sweep: configs/sweeps/{sweep_path.name} ({len(jobs)} jobs)")
    if args.write:
        sweep_path.write_text(sweep_text)
        print(f"[ok] wrote {len(jobs)} configs + sweep.")
    else:
        print("\n--- wave-1 sweep YAML preview ---")
        print(sweep_text)

    print("Deferred to wave 2 (write once select_lambda has chosen each bracket's lambda):")
    for t, student in wave2:
        print(f"  {slug(t)}-to-{slug(student)} @ lambda=<bracket winner for {student}>")
    if not args.write:
        print("\n[dry-run] no files written. Re-run with --write to write them.")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse exactly one explicit teacher-strength phase."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--phase",
        action="append",
        choices=PHASES,
        required=True,
        help="materialization phase; specify exactly once",
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help="actually create files (default: dry-run)",
    )
    args = parser.parse_args(argv)
    if len(args.phase) != 1:
        parser.error("--phase must be specified exactly once")
    args.phase = args.phase[0]
    return args


def main(argv: list[str] | None = None) -> None:
    """Generate exactly one explicitly selected teacher-strength phase."""
    args = parse_args(argv)
    emitters = {
        "tier1-wave1": _emit_wave1,
        "tier1-wave2": _emit_wave2,
        "tier2-facet1-sft": _emit_tier2_facet1_sft,
        "tier2-facet2-sft": _emit_facet2_sft,
        "tier3-density-fill-sft": _emit_density_fill_sft,
    }
    emitters[args.phase](args)


if __name__ == "__main__":
    main()
