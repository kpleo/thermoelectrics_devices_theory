"""Run the specified composite-device mesh and boundary-condition checks."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analysis.packaged_device import run_case


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    # Both geometries use the same material curves, total semiconductor area,
    # leg lengths, external temperatures, packaging laws and total current.
    for label, ratio in (("equal", 1.0), ("one_mode_1d", 1.04240001)):
        for refinement in (1, 2, 4):
            run_case(output=args.output_dir/f"{label}_parabola_r{refinement}.json",
                     area_ratio=ratio, refinement=refinement)
        run_case(output=args.output_dir/f"{label}_linear_r4.json",
                 area_ratio=ratio, refinement=4, mode="linear")


if __name__ == "__main__":
    main()
