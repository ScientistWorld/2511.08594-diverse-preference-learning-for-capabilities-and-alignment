#!/usr/bin/env python3
"""Run the DPO baseline as the alpha == beta special case of SPL."""

from __future__ import annotations

import argparse
import subprocess
import sys


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", default="data/spl_tabular_benchmark.json")
    parser.add_argument("--output-dir", default="checkpoints/dpo_tabular")
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--lr", type=float, default=0.8)
    parser.add_argument("--samples-per-context", type=int, default=128)
    parser.add_argument("--seed", type=int, default=11)
    args = parser.parse_args()

    cmd = [
        sys.executable,
        "method/train_tabular_spl.py",
        "--benchmark",
        args.benchmark,
        "--output-dir",
        args.output_dir,
        "--method-name",
        "dpo_alpha_equals_beta",
        "--alpha",
        str(args.beta),
        "--beta",
        str(args.beta),
        "--steps",
        str(args.steps),
        "--lr",
        str(args.lr),
        "--samples-per-context",
        str(args.samples_per_context),
        "--seed",
        str(args.seed),
    ]
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
