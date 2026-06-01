#!/usr/bin/env python3
"""Evaluate tabular preference-learning artifacts.

This module reads policy/sample artifacts only. It intentionally avoids
solution-code imports so future agents can replace the training algorithm as
long as they write the same artifact format.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def entropy(probs: list[float]) -> float:
    return -sum(p * math.log(max(p, 1e-12)) for p in probs)


def kl_divergence(p: list[float], q: list[float]) -> float:
    return sum(pi * (math.log(max(pi, 1e-12)) - math.log(max(qi, 1e-12))) for pi, qi in zip(p, q))


def brier(policy: list[float], target: list[float]) -> float:
    return sum((pi - ti) ** 2 for pi, ti in zip(policy, target))


def expected_calibration_error(policy: list[float], target: list[float], bins: int = 10) -> float:
    # A distributional ECE analogue for finite preferences: bin actions by model
    # probability and compare predicted mass to population preference mass.
    total = 0.0
    for bin_idx in range(bins):
        lo = bin_idx / bins
        hi = (bin_idx + 1) / bins
        pred_mass = 0.0
        target_mass = 0.0
        for p, t in zip(policy, target):
            if (lo <= p < hi) or (bin_idx == bins - 1 and p == 1.0):
                pred_mass += p
                target_mass += t
        total += abs(pred_mass - target_mass)
    return total / 2.0


def load_run(run_dir: Path) -> tuple[str, dict, dict]:
    policy = json.loads((run_dir / "policy.json").read_text())
    samples_path = run_dir / "samples.json"
    samples = json.loads(samples_path.read_text()) if samples_path.exists() else {"samples": []}
    method_name = policy.get("metadata", {}).get("method_name", run_dir.name)
    return method_name, policy, samples


def evaluate_run(run_dir: Path, experiment: str) -> tuple[str, dict]:
    method_name, policy, samples = load_run(run_dir)
    context_metrics = []
    for context in policy["contexts"]:
        actions = context["actions"]
        pred = [a["policy_probability"] for a in actions]
        target = [a["population_preference"] for a in actions]
        ref = [a["reference_probability"] for a in actions]
        solution_prob = sum(
            a["policy_probability"] for a in actions if a.get("is_best_of_n_solution", False)
        )
        top_idx = max(range(len(pred)), key=lambda idx: pred[idx])
        context_metrics.append(
            {
                "entropy": entropy(pred),
                "target_entropy": entropy(target),
                "normalized_entropy": entropy(pred) / math.log(len(pred)),
                "effective_actions": math.exp(entropy(pred)),
                "preference_kl": kl_divergence(target, pred),
                "reference_cross_entropy": -sum(p * math.log(max(r, 1e-12)) for p, r in zip(pred, ref)),
                "expected_population_preference": sum(p * t for p, t in zip(pred, target)),
                "brier": brier(pred, target),
                "ece": expected_calibration_error(pred, target),
                "argmax_matches_population_mode": int(top_idx == max(range(len(target)), key=lambda idx: target[idx])),
                "best_of_n_solution_probability": solution_prob,
                "best_of_128_success_probability": 1.0 - (1.0 - solution_prob) ** 128,
            }
        )

    sample_rows = samples.get("samples", [])
    by_context = {}
    for row in sample_rows:
        by_context.setdefault(row["context_id"], []).append(row)
    empirical_success = []
    for rows in by_context.values():
        empirical_success.append(float(any(row.get("is_best_of_n_solution", False) for row in rows)))

    def avg(key: str) -> float:
        return sum(item[key] for item in context_metrics) / len(context_metrics)

    if experiment == "two_outcome_mode_collapse":
        prompt = policy["contexts"][0]
        actions = prompt["actions"]
        pred = [a["policy_probability"] for a in actions]
        target = [a["population_preference"] for a in actions]
        majority_idx = max(range(len(target)), key=lambda idx: target[idx])
        minority_idx = min(range(len(target)), key=lambda idx: target[idx])
        top_idx = max(range(len(pred)), key=lambda idx: pred[idx])
        result = {
            "minority_probability": pred[minority_idx],
            "majority_overweight": pred[majority_idx] / target[majority_idx],
            "top_choice_preserved": float(top_idx == majority_idx),
        }
        return method_name, result

    result = {
        "normalized_entropy": avg("normalized_entropy"),
        "effective_actions": avg("effective_actions"),
        "preference_kl": avg("preference_kl"),
        "brier_score": avg("brier"),
        "expected_calibration_error": avg("ece"),
        "expected_population_preference": avg("expected_population_preference"),
        "reference_cross_entropy": avg("reference_cross_entropy"),
        "argmax_population_mode_accuracy": avg("argmax_matches_population_mode"),
        "best_of_128_success_probability": avg("best_of_128_success_probability"),
        "empirical_best_of_128_success": (
            sum(empirical_success) / len(empirical_success) if empirical_success else 0.0
        ),
    }
    return method_name, result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dirs", nargs="+", help="One or more artifact directories")
    parser.add_argument("--scores-out", default="scoring/scores.json")
    parser.add_argument("--experiment", default="tabular_preference_learning")
    parser.add_argument("--merge", action="store_true", help="Merge this experiment into an existing scores file")
    args = parser.parse_args()

    results = {}
    for run_dir_text in args.run_dirs:
        method_name, result = evaluate_run(Path(run_dir_text), args.experiment)
        results[method_name] = result

    output = Path(args.scores_out)
    output.parent.mkdir(parents=True, exist_ok=True)
    if args.merge and output.exists():
        scores = json.loads(output.read_text())
        scores.setdefault("experiments", {})
    else:
        scores = {"experiments": {}}
    scores["experiments"][args.experiment] = {"results": results}
    output.write_text(json.dumps(scores, indent=2) + "\n")
    print(json.dumps(scores, indent=2))


if __name__ == "__main__":
    main()
