#!/usr/bin/env python3
"""Generate a finite preference-learning benchmark for SPL.

The benchmark is deliberately tabular: each prompt has a small set of possible
completions, a reference policy, and a population preference distribution.
Pairwise preferences are generated from a Plackett-Luce model so the paper's
closed-form SPL analysis applies directly.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path


def _normalize(xs: list[float]) -> list[float]:
    total = sum(xs)
    if total <= 0:
        raise ValueError("cannot normalize non-positive vector")
    return [x / total for x in xs]


def _pair_count(p_i: float, p_j: float, comparisons: int) -> tuple[int, int]:
    p_i_wins = p_i / (p_i + p_j)
    wins_i = int(round(comparisons * p_i_wins))
    wins_i = min(max(wins_i, 1), comparisons - 1)
    return wins_i, comparisons - wins_i


def build_benchmark(seed: int = 7, pair_comparisons: int = 48) -> dict:
    rng = random.Random(seed)
    contexts = []

    base_preferences = [
        [0.55, 0.20, 0.15, 0.10],
        [0.48, 0.26, 0.18, 0.08],
        [0.42, 0.30, 0.18, 0.10],
        [0.62, 0.17, 0.13, 0.08],
        [0.46, 0.24, 0.20, 0.10],
        [0.58, 0.22, 0.12, 0.08],
        [0.50, 0.21, 0.17, 0.12],
        [0.44, 0.28, 0.16, 0.12],
    ]

    # Some prompts mark a minority strategy as the one that solves a repeated
    # sampling task. This mirrors the paper's hard best-of-N setting: a collapsed
    # policy can have high one-shot majority quality while rarely exploring the
    # strategy needed to solve the instance.
    best_of_n_correct = [2, 1, 2, 3, 2, 1, 3, 1]

    for idx, prefs in enumerate(base_preferences):
        ref = _normalize([rng.uniform(0.85, 1.15) for _ in prefs])
        actions = []
        for action_idx, prob in enumerate(prefs):
            actions.append(
                {
                    "id": f"a{action_idx}",
                    "text": f"completion_{idx}_{action_idx}",
                    "population_preference": prob,
                    "reference_probability": ref[action_idx],
                    "is_best_of_n_solution": action_idx == best_of_n_correct[idx],
                }
            )

        comparisons = []
        for i in range(len(actions)):
            for j in range(i + 1, len(actions)):
                wins_i, wins_j = _pair_count(prefs[i], prefs[j], pair_comparisons)
                comparisons.append({"winner": f"a{i}", "loser": f"a{j}", "count": wins_i})
                comparisons.append({"winner": f"a{j}", "loser": f"a{i}", "count": wins_j})

        contexts.append(
            {
                "id": f"prompt_{idx}",
                "prompt": f"Finite preference prompt {idx}",
                "actions": actions,
                "comparisons": comparisons,
            }
        )

    return {
        "name": "tabular_spl_preference_benchmark",
        "seed": seed,
        "pair_comparisons": pair_comparisons,
        "description": (
            "Finite Plackett-Luce preference benchmark for testing whether SPL "
            "avoids DPO-style majority mode collapse while preserving quality."
        ),
        "contexts": contexts,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/spl_tabular_benchmark.json")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--pair-comparisons", type=int, default=48)
    args = parser.parse_args()

    benchmark = build_benchmark(seed=args.seed, pair_comparisons=args.pair_comparisons)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(benchmark, indent=2) + "\n")
    print(f"wrote {output} with {len(benchmark['contexts'])} contexts")


if __name__ == "__main__":
    main()
