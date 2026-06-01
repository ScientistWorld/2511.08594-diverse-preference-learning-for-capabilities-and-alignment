#!/usr/bin/env python3
"""Generate two-action preference contexts for train/test split scoring.

The first context is the paper's 80/20 example so the standard evaluator can
continue reporting the core claim on the original setting. Additional contexts
provide a real held-out axis for the anti-shortcut train/test split.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_context(context_id: str, majority_preference: float, comparisons: int) -> dict:
    minority_preference = round(1.0 - majority_preference, 10)
    majority_wins = int(round(comparisons * majority_preference))
    majority_wins = min(max(majority_wins, 1), comparisons - 1)
    minority_wins = comparisons - majority_wins
    return {
        "id": context_id,
        "prompt": (
            f"A population prefers completion A with probability {majority_preference:.2f} "
            f"and completion B with probability {minority_preference:.2f}."
        ),
        "actions": [
            {
                "id": "majority",
                "text": "Majority-preferred completion.",
                "population_preference": majority_preference,
                "reference_probability": 0.5,
                "is_best_of_n_solution": False,
            },
            {
                "id": "minority",
                "text": "Minority-preferred completion.",
                "population_preference": minority_preference,
                "reference_probability": 0.5,
                "is_best_of_n_solution": True,
            },
        ],
        "comparisons": [
            {"winner": "majority", "loser": "minority", "count": majority_wins},
            {"winner": "minority", "loser": "majority", "count": minority_wins},
        ],
    }


def build_benchmark(comparisons: int = 100) -> dict:
    contexts = [
        build_context("paper_80_20_example", 0.80, comparisons),
        build_context("train_70_30_example", 0.70, comparisons),
        build_context("train_85_15_example", 0.85, comparisons),
        build_context("test_65_35_example", 0.65, comparisons),
        build_context("test_75_25_example", 0.75, comparisons),
        build_context("test_90_10_example", 0.90, comparisons),
    ]
    return {
        "name": "two_outcome_split_benchmark",
        "seed": 7,
        "pair_comparisons": comparisons,
        "description": (
            "Two-action preference contexts for SPL mode-collapse evaluation. "
            "The first context is the paper's original 80/20 example."
        ),
        "contexts": contexts,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/two_outcome_split_benchmark.json")
    parser.add_argument("--pair-comparisons", type=int, default=100)
    args = parser.parse_args()

    benchmark = build_benchmark(comparisons=args.pair_comparisons)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(benchmark, indent=2) + "\n")
    print(f"wrote {output} with {len(benchmark['contexts'])} contexts")


if __name__ == "__main__":
    main()
