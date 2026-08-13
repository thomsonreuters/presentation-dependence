"""Parquet I/O for presentation-aligned per-document score logs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path


def read_score_log(path: str | Path) -> list[dict]:
    """Read a presentation score log from Parquet."""
    import pyarrow.parquet as pq

    return pq.read_table(path).to_pylist()


def write_score_log(rows: Sequence[Mapping], path: str | Path) -> None:
    """Write presentation score rows to Parquet."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist([dict(row) for row in rows])
    pq.write_table(table, output)
