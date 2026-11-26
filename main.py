#!/usr/bin/env python3
"""Project RAKSHA CLI entrypoint."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.config import load_runtime_config
from src.pipeline import run_pipeline


def main() -> int:
    ap = argparse.ArgumentParser(description="Project RAKSHA privacy firewall")
    ap.add_argument("--input", required=True, help="Read-only input dataset directory")
    ap.add_argument("--output", required=True, help="Output directory")
    ap.add_argument("--config", default=None, help="Optional path to policy.yaml")
    args = ap.parse_args()
    cfg = load_runtime_config(Path(args.config) if args.config else None)
    return run_pipeline(Path(args.input), Path(args.output), cfg=cfg)


if __name__ == "__main__":
    sys.exit(main())
