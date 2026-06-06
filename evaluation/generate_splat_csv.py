#!/usr/bin/env python3
"""Generate explicit Splat CSV files for evaluation experiments."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a reproducible CSV listing selected Gaussian Splat PLY files.")
    parser.add_argument("--splat-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iterations", type=int, nargs="+", default=[1000, 2000, 3000, 5000, 8000, 12000, 18000, 30000])
    parser.add_argument("--id-prefix", default=None)
    parser.add_argument("--filename-pattern", default="{prefix}_{iteration:05d}.ply")
    parser.add_argument("--quality-label-prefix", default="iter")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    splat_dir = args.splat_dir.expanduser().resolve()
    if not splat_dir.is_dir():
        raise FileNotFoundError(f"Splat directory does not exist: {splat_dir}")

    prefix = args.id_prefix or splat_dir.name
    rows = []
    for iteration in args.iterations:
        ply_path = splat_dir / args.filename_pattern.format(prefix=prefix, iteration=iteration)
        if not ply_path.is_file():
            raise FileNotFoundError(f"Missing splat for iteration {iteration}: {ply_path}")
        rows.append(
            {
                "splat_id": f"{prefix}_{iteration:05d}",
                "ply_path": str(ply_path),
                "training_iteration": iteration,
                "quality_label": f"{args.quality_label_prefix}_{iteration}",
                "notes": "",
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["splat_id", "ply_path", "training_iteration", "quality_label", "notes"],
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
