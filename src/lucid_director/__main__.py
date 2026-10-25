"""Command-line entry point: ``python -m lucid_director`` or ``lucid-director``."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from lucid_director import __version__


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lucid-director",
        description="Run the reskinning loop: frame source -> pipeline -> display.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/stub.yaml"),
        help="YAML config file (default: %(default)s)",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=300,
        help="number of frames to run before printing a summary (default: %(default)s)",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    # Placeholder until sub-task 0.7 wires up the runtime loop.
    print(f"lucid-director {__version__}: runtime not implemented yet (Phase 0.7).")
    print(f"  config={args.config} frames={args.frames}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
