#!/usr/bin/env python3
"""Generate a finite multiple-choice calibration benchmark for SPL.

Each context represents a multiple-choice item with a population distribution
over plausible answers. Pairwise comparison counts follow the same
Plackett-Luce construction used by the tabular benchmark, so DPO and SPL are
still trained with the paper's pairwise objective rather than an evaluator-side
shortcut.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _pair_count(p_i: float, p_j: float, comparisons: int) -> tuple[int, int]:
    p_i_wins = p_i / (p_i + p_j)
    wins_i = int(round(comparisons * p_i_wins))
    wins_i = min(max(wins_i, 1), comparisons - 1)
    return wins_i, comparisons - wins_i


def build_benchmark(pair_comparisons: int = 64) -> dict:
    preferences = [
        [0.52, 0.26, 0.14, 0.08],
        [0.46, 0.30, 0.16, 0.08],
        [0.58, 0.18, 0.14, 0.10],
        [0.41, 0.33, 0.17, 0.09],
        [0.64, 0.16, 0.12, 0.08],
        [0.49, 0.24, 0.18, 0.09],
        [0.56, 0.21, 0.13, 0.10],
        [0.44, 0.28, 0.20, 0.08],
        [0.60, 0.19, 0.12, 0.09],
        [0.47, 0.27, 0.15, 0.11],
    ]
    contexts = []
    for idx, prefs in enumerate(preferences):
        actions = []
        for action_idx, prob in enumerate(prefs):
            actions.append(
                {
                    "id": f"choice_{action_idx}",
                    "text": f"answer_choice_{idx}_{action_idx}",
                    "population_preference": prob,
                    "reference_probability": 0.25,
                    "is_best_of_n_solution": action_idx == 0,
                }
            )

        comparisons = []
        for i in range(len(actions)):
            for j in range(i + 1, len(actions)):
                wins_i, wins_j = _pair_count(prefs[i], prefs[j], pair_comparisons)
                comparisons.append({"winner": f"choice_{i}", "loser": f"choice_{j}", "count": wins_i})
                comparisons.append({"winner": f"choice_{j}", "loser": f"choice_{i}", "count": wins_j})

        contexts.append(
            {
                "id": f"calibration_prompt_{idx}",
                "prompt": f"Multiple-choice calibration prompt {idx}",
                "actions": actions,
                "comparisons": comparisons,
            }
        )

    return {
        "name": "calibration_multiple_choice_benchmark",
        "pair_comparisons": pair_comparisons,
        "description": "Finite multiple-choice benchmark for distributional calibration under preference learning.",
        "contexts": contexts,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/calibration_benchmark.json")
    parser.add_argument("--pair-comparisons", type=int, default=64)
    args = parser.parse_args()

    benchmark = build_benchmark(pair_comparisons=args.pair_comparisons)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(benchmark, indent=2) + "\n")
    print(f"wrote {output} with {len(benchmark['contexts'])} contexts")


if __name__ == "__main__":
    main()
