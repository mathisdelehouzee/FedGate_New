#!/usr/bin/env python3
"""Thin wrapper for running FedGate benchmarks."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_federated_benchmark.py"


def main() -> None:
    argv = [sys.executable, str(SCRIPT), "--method", "fedgate", *sys.argv[1:]]
    raise SystemExit(__import__("subprocess").run(argv, check=False).returncode)


if __name__ == "__main__":
    main()
