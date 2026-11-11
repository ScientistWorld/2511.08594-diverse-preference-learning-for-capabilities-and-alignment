#!/usr/bin/env python3
"""DPO baseline: the alpha == beta special case of the SPL objective (paper Prop. 3.1)."""

from __future__ import annotations

import argparse
import subprocess
import sys


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default="checkpoints/dpo_alpha_equals_beta")
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--experiments", default=None)
    args, extra = parser.parse_known_args()
    cmd = [
        sys.executable, "method/spl.py",
        "--data-dir", args.data_dir,
        "--output-dir", args.output_dir,
        "--alpha", str(args.beta),
        "--beta", str(args.beta),
    ]
    if args.experiments:
        cmd += ["--experiments", args.experiments]
    subprocess.run(cmd + extra, check=True)


if __name__ == "__main__":
    main()
