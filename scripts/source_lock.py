#!/usr/bin/env python
"""Write or verify immutable reproduction source identities."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from presentation_dependence.reproduction.source_lock import (
    verify_source_lock,
    write_source_lock,
)


def main() -> int:
    """Run source-lock write or verification."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("write", "verify"))
    args = parser.parse_args()
    project_root = Path(__file__).resolve().parents[1]
    if args.command == "write":
        print(write_source_lock(project_root))
        return 0
    result = verify_source_lock(project_root)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "ok" else 2


if __name__ == "__main__":
    raise SystemExit(main())
