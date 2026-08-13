"""CLI for representative scientific analysis checks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .representative import (
    plan_representative,
    run_representative,
    validate_representative,
)


def build_parser() -> argparse.ArgumentParser:
    """Build the representative-analysis parser."""
    parser = argparse.ArgumentParser(prog="study.py representative")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("plan", "run", "validate"):
        command = commands.add_parser(name)
        command.add_argument("--family")
    return parser


def run_cli(argv: list[str], project_root: Path) -> int:
    """Run one representative-analysis command."""
    args = build_parser().parse_args(argv)
    if args.command == "plan":
        result = plan_representative(project_root, family_id=args.family)
    elif args.command == "run":
        result = run_representative(project_root, family_id=args.family)
    else:
        result = validate_representative(project_root, family_id=args.family)
    print(json.dumps(result, indent=2, sort_keys=True))
    if args.command == "validate" and result["status"] != "complete":
        return 2
    return 0
