#!/usr/bin/env python
"""Build claim-verdict sidecars from authoritative source datasets.

Does not alter the BEIR fixtures. Validates that every source claim id
and claim text align with the existing reranking fixture, then writes
``verdicts.jsonl`` and the non-DISPUTED ``verdict_qids.txt`` cohort beside that
fixture.
"""

from __future__ import annotations

import argparse
import io
import json
import re
import sys
import tarfile
import urllib.request
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

SLM_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SLM_ROOT / "src"))

SCIFACT_URL = "https://scifact.s3-us-west-2.amazonaws.com/release/latest/data.tar.gz"
CLIMATE_URL = "https://huggingface.co/datasets/tdiggelm/climate_fever/resolve/main/data/test-00000-of-00001.parquet"

DATA_DIRS = {
    "scifact": SLM_ROOT / "data" / "beir-v1.0.0-scifact-test",
    "climate-fever": SLM_ROOT / "data" / "beir-v1.0.0-climate-fever-test",
}


def _download(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=180) as response:
        return response.read()


def _clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text).replace("\xa0", " ")).strip()


def _fixture_claims(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            out[str(row["qid"])] = str(row["query"])
    return out


def _scifact_rows() -> list[dict]:
    archive = tarfile.open(fileobj=io.BytesIO(_download(SCIFACT_URL)), mode="r:gz")
    source = archive.extractfile("data/claims_dev.jsonl")
    if source is None:
        raise RuntimeError("SciFact archive lacks data/claims_dev.jsonl")
    out: list[dict] = []
    for raw in source:
        row = json.loads(raw)
        labels = {evidence["label"] for groups in row.get("evidence", {}).values() for evidence in groups}
        if not labels:
            label = "NEI"
        elif labels == {"SUPPORT"}:
            label = "SUPPORTED"
        elif labels == {"CONTRADICT"}:
            label = "REFUTED"
        else:
            raise ValueError(f"SciFact qid={row['id']} has mixed labels: {sorted(labels)}")
        out.append({"qid": str(row["id"]), "claim": row["claim"], "label": label})
    return out


def _climate_rows() -> list[dict]:
    import pyarrow as pa
    import pyarrow.parquet as pq

    table = pq.read_table(pa.BufferReader(_download(CLIMATE_URL)))
    labels = {0: "SUPPORTED", 1: "REFUTED", 2: "NEI", 3: "DISPUTED"}
    return [
        {
            "qid": str(row["claim_id"]),
            "claim": str(row["claim"]),
            "label": labels[int(row["claim_label"])],
        }
        for row in table.to_pylist()
    ]


def build(dataset: str) -> dict:
    """Build and validate one dataset's verdict sidecar."""
    data_dir = DATA_DIRS[dataset]
    fixture_path = data_dir / "fixture.jsonl"
    if not fixture_path.is_file():
        raise FileNotFoundError(f"missing fixture: {fixture_path}")
    source_rows = _scifact_rows() if dataset == "scifact" else _climate_rows()
    fixture = _fixture_claims(fixture_path)
    source = {row["qid"]: row for row in source_rows}
    if set(source) != set(fixture):
        raise ValueError(
            f"{dataset}: source/fixture qid mismatch "
            f"(source-only={len(set(source) - set(fixture))}, fixture-only={len(set(fixture) - set(source))})"
        )
    mismatches = [qid for qid in fixture if _clean_text(fixture[qid]) != _clean_text(source[qid]["claim"])]
    if mismatches:
        raise ValueError(f"{dataset}: {len(mismatches)} claim-text mismatches, first={mismatches[:5]}")

    output = data_dir / "verdicts.jsonl"
    with output.open("w", encoding="utf-8") as handle:
        for qid in sorted(source):
            handle.write(
                json.dumps(
                    {"qid": qid, "label": source[qid]["label"]},
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
            )
    verdict_qids = data_dir / "verdict_qids.txt"
    included_qids = sorted(qid for qid, row in source.items() if row["label"] != "DISPUTED")
    verdict_qids.write_text("".join(f"{qid}\n" for qid in included_qids), encoding="utf-8")
    meta = {
        "dataset": dataset,
        "source_url": SCIFACT_URL if dataset == "scifact" else CLIMATE_URL,
        "source_split": "claims_dev" if dataset == "scifact" else "test",
        "built_at": datetime.now(timezone.utc).isoformat(),
        "n_queries": len(source),
        "label_counts": dict(sorted(Counter(row["label"] for row in source.values()).items())),
        "fixture": str(fixture_path.relative_to(SLM_ROOT)),
        "output": str(output.relative_to(SLM_ROOT)),
        "verdict_qids": str(verdict_qids.relative_to(SLM_ROOT)),
        "n_verdict_qids": len(included_qids),
        "claim_text_validation": "exact after whitespace/NBSP normalization",
    }
    (data_dir / "verdicts.meta.json").write_text(
        json.dumps(meta, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return meta


def main() -> int:
    """Parse CLI arguments and build requested sidecars."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["all", *DATA_DIRS], default="all")
    args = parser.parse_args()
    datasets = list(DATA_DIRS) if args.dataset == "all" else [args.dataset]
    print(json.dumps([build(dataset) for dataset in datasets], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
