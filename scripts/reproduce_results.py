#!/usr/bin/env python3
"""Reproduce the numerical records and local diagnostic outputs."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
ANALYSES = (
    "scripts/analysis/analyze_pbse_device_forward_constraint.py",
    "scripts/analysis/analyze_pbse_common_mode_contribution.py",
    "scripts/analysis/analyze_common_mode_transfer_kernel.py",
    "scripts/analysis/validate_independent_2d_common_mode.py",
    "scripts/analysis/fully_coupled_2d.py",
    "scripts/analysis/analyze_pbse_gamma_identifiability.py",
    "scripts/analysis/joint_pbse_error_model.py",
    "scripts/analysis/topology_breaking_split_pad.py",
    "scripts/analysis/analyze_bts_bst_endpoint_zero_common_mode.py",
    "scripts/analysis/adjoint_nondegeneracy.py",
)


def run(command: list[str], environment: dict[str, str]) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, env=environment, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--from-processed", action="store_true")
    arguments = parser.parse_args()
    if not arguments.from_processed:
        parser.error("select --from-processed")

    environment = os.environ.copy()
    environment.setdefault("PYTHONHASHSEED", "0")
    environment.setdefault("SOURCE_DATE_EPOCH", "1787745600")
    environment.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))
    Path(environment["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

    for relative_path in ANALYSES:
        run([sys.executable, relative_path], environment)

if __name__ == "__main__":
    main()
