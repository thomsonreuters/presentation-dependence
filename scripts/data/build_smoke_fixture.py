#!/usr/bin/env python
"""Build the synthetic smoke dataset used by `configs/experiments/_smoke-fixture.yaml`.

This is the only eval surface that needs no network, no Java/Pyserini index, no
model download, and no GPU, so it is the first command a new checkout can run.
Everything is generated from a fixed seed, so repeated builds are byte-identical
and the smoke config has a stable expected nDCG@10.

The passages are synthetic but shaped like a real reranking surface (one on-topic
query, a handful of graded relevant passages, and lexically similar distractors),
so the fixture also works as a local target for a real reranker via
``--override reranker.class=...``.

Writes into ``data/_smoke/`` (gitignored, like every other `data/` slug):

    fixture.jsonl       FixtureLoader input: one query + its candidates per line
    qrels.txt           TREC qrels, graded 0/1/2
    dataset_meta.yaml   first-stage provenance, per the `data/` convention
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path
from typing import Any

import yaml


LOGGER = logging.getLogger("build_smoke_fixture")

SEED = 20260806
DATASET_SLUG = "_smoke"

# (topic, relevant-passage bodies, distractor bodies). The first relevant body is
# graded 2, the rest 1.
TOPICS: list[tuple[str, str, list[str], list[str]]] = [
    (
        "how do tides work",
        "what causes ocean tides",
        [
            "Ocean tides are driven by the gravitational pull of the moon and, to a "
            "lesser extent, the sun, which deforms the ocean surface into two bulges.",
            "Because the earth rotates beneath the tidal bulges, most coastlines see "
            "two high tides and two low tides in roughly twenty-five hours.",
            "Spring tides occur when the sun and moon align so their gravitational "
            "contributions add together, producing the largest tidal range.",
        ],
        [
            "Tide pools shelter anemones and small crabs, and are a popular subject for coastal nature photography.",
            "The tidal power station at La Rance generates electricity from the flow of water through a barrage.",
            "Laundry detergent brands have competed on stain removal claims since the middle of the twentieth century.",
            "Surfers track swell period and wind direction to predict which beaches will produce clean waves.",
            "A tide table is printed in most coastal newspapers alongside the weather forecast and shipping notices.",
        ],
    ),
    (
        "why is the sky blue",
        "why the daytime sky appears blue",
        [
            "Sunlight scatters off air molecules, and Rayleigh scattering is far "
            "stronger at short wavelengths, so blue light is redirected across the sky.",
            "The sun looks red at sunset because the light travels a long path "
            "through the atmosphere and most of the blue has been scattered away.",
            "Violet is scattered even more strongly than blue, but the eye is less "
            "sensitive to violet and sunlight contains less of it.",
        ],
        [
            "Blue pigment was historically expensive, which is why ultramarine was "
            "reserved for the most important figures in a painting.",
            "Cyanometers were early instruments built to measure the blueness of the sky at different altitudes.",
            "The blue whale is the largest animal known to have existed, reaching lengths of about thirty metres.",
            "Clouds appear white because their droplets are large enough to scatter "
            "all visible wavelengths about equally.",
            "Blueprint reproduction used a light-sensitive iron compound to copy engineering drawings.",
        ],
    ),
    (
        "what is a black hole event horizon",
        "definition of a black hole event horizon",
        [
            "The event horizon is the boundary around a black hole beyond which no "
            "path, not even one followed by light, leads back out.",
            "For a non-rotating black hole the horizon sits at the Schwarzschild "
            "radius, which is proportional to the mass.",
            "An outside observer sees infalling material appear to slow and dim as "
            "it approaches the horizon, redshifted toward invisibility.",
        ],
        [
            "The first image of a black hole shadow was published by the Event Horizon Telescope collaboration.",
            "Accretion disks around black holes radiate strongly in X-rays as infalling gas is compressed and heated.",
            "Horizon scanning is a planning technique for identifying emerging risks in an industry.",
            "Neutron stars are supported against collapse by neutron degeneracy "
            "pressure rather than by an event horizon.",
            "Science fiction often depicts wormholes as shortcuts between distant regions of space.",
        ],
    ),
    (
        "how does yeast make bread rise",
        "how yeast leavens bread dough",
        [
            "Yeast ferments the sugars in dough and releases carbon dioxide, which "
            "inflates gas bubbles trapped by the gluten network.",
            "Gluten strands developed by kneading give the dough enough elasticity "
            "to hold the carbon dioxide instead of letting it escape.",
            "Oven spring happens when the trapped gas expands rapidly in the heat "
            "before the crust sets and fixes the loaf's volume.",
        ],
        [
            "Sourdough starters host both wild yeasts and lactic acid bacteria, which is why the crumb tastes tangy.",
            "Nutritional yeast is deactivated and sold as a savoury flake rather than as a leavening agent.",
            "Bread knives use a serrated edge so they saw through crust without compressing the crumb.",
            "Baking powder leavens by a chemical reaction between an acid and a "
            "base, with no living organism involved.",
            "Brewing yeast strains are selected for alcohol tolerance and "
            "flocculation rather than for dough performance.",
        ],
    ),
    (
        "what causes seasons on earth",
        "what causes the earth's seasons",
        [
            "Seasons are caused by the tilt of the earth's rotation axis, which "
            "changes how directly sunlight strikes each hemisphere over the year.",
            "When a hemisphere is tilted toward the sun it receives more concentrated "
            "sunlight and longer days, and so warms.",
            "The earth's distance from the sun varies only slightly and is not the "
            "reason the seasons alternate between hemispheres.",
        ],
        [
            "Seasonal influenza vaccines are reformulated each year to track the circulating strains.",
            "Deciduous trees shed their leaves as day length shortens and chlorophyll production stops.",
            "The astronomical calendar fixes the solstices and equinoxes, while "
            "meteorological seasons start on the first of the month.",
            "Season tickets let supporters attend every home fixture for a single up-front payment.",
            "Migratory birds navigate using a combination of the sun, stars, and the earth's magnetic field.",
        ],
    ),
    (
        "how do vaccines produce immunity",
        "how vaccines create immune memory",
        [
            "A vaccine presents a harmless version or fragment of a pathogen so the "
            "immune system learns to recognise it without causing the disease.",
            "The response generates memory B and T cells that persist, so a later "
            "real exposure is cleared faster and more strongly.",
            "Booster doses re-expose the immune system to the antigen and raise "
            "antibody levels that decline over time.",
        ],
        [
            "Cold chain logistics keep temperature-sensitive medical products within "
            "a validated range during transport.",
            "Antibiotics act against bacteria and have no effect on viral infections such as the common cold.",
            "Herd immunity thresholds are estimated from how transmissible a pathogen is in a given population.",
            "Clinical trials are usually run in sequential phases that test safety before efficacy.",
            "Microneedle patches are being studied as an alternative delivery route to conventional injection.",
        ],
    ),
]


def _records(rng: random.Random) -> tuple[list[dict[str, Any]], list[str]]:
    """Return the fixture records and their TREC qrels lines."""
    records: list[dict[str, Any]] = []
    qrels: list[str] = []

    for query_index, (query, title, relevant, distractors) in enumerate(TOPICS, start=1):
        qid = f"s{query_index}"
        passages: list[dict[str, str]] = []
        grades: dict[str, int] = {}

        for rank, body in enumerate(relevant):
            pid = f"{qid}-rel{rank}"
            passages.append({"pid": pid, "text": f"{title}. {body}"})
            grades[pid] = 2 if rank == 0 else 1

        for rank, body in enumerate(distractors):
            pid = f"{qid}-dis{rank}"
            passages.append({"pid": pid, "text": body})
            grades[pid] = 0

        # Shuffle into a deliberately imperfect first-stage order so the
        # identity pass-through scores well below a perfect ranking and the
        # smoke number is sensitive to a reranker actually reordering.
        rng.shuffle(passages)

        records.append({"qid": qid, "query": query, "passages": passages})
        for pid, grade in sorted(grades.items()):
            qrels.append(f"{qid} 0 {pid} {grade}")

    return records, qrels


def _dataset_meta() -> dict[str, Any]:
    return {
        "dataset": DATASET_SLUG,
        "corpus": "synthetic-smoke",
        "source": "generated by scripts/data/build_smoke_fixture.py (no external data)",
        "fetched": None,
        "first_stage": {
            "synthetic": {
                "retriever": "none",
                "variant": "synthetic-fixed-order",
                "hits": len(TOPICS[0][2]) + len(TOPICS[0][3]),
                "note": (
                    "Candidate order is a seeded shuffle, not a retrieval run. "
                    "Not comparable to any BM25 or dense first stage."
                ),
            }
        },
        "topics": {"source": "hand-written in the builder script", "n": len(TOPICS)},
        "qrels": {
            "source": "hand-assigned graded judgments in the builder script",
            "n_judged_topics": len(TOPICS),
        },
        "notes": (
            "Synthetic smoke surface for pipeline verification only. Never report "
            "a number from this dataset as a result; it measures plumbing, not "
            "retrieval quality.\n"
        ),
    }


def main() -> None:
    """Write the synthetic smoke fixture, qrels, and dataset metadata."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data") / DATASET_SLUG,
        help="Output directory (default: data/_smoke/).",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    records, qrels = _records(random.Random(SEED))

    fixture_path = out_dir / "fixture.jsonl"
    with open(fixture_path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")

    qrels_path = out_dir / "qrels.txt"
    with open(qrels_path, "w", encoding="utf-8") as f:
        f.write("\n".join(qrels) + "\n")

    meta_path = out_dir / "dataset_meta.yaml"
    with open(meta_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(_dataset_meta(), f, sort_keys=False)

    passages = sum(len(record["passages"]) for record in records)
    LOGGER.info("Wrote %d queries / %d passages to %s", len(records), passages, fixture_path)
    LOGGER.info("Wrote %d qrels rows to %s", len(qrels), qrels_path)
    LOGGER.info("Wrote %s", meta_path)
    LOGGER.info("Next: uv run python scripts/run_experiment.py -e _smoke-fixture")


if __name__ == "__main__":
    main()
