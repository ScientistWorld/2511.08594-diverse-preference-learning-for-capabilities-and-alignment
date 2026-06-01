#!/usr/bin/env python3
"""Visible train-slice evaluator for tabular SPL artifacts.

This script reads method artifacts only. It does not import from method/ or
regenerate training outputs, so any method that writes policy.json and optional
samples.json in the established checkpoint format can be scored.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


SLICE = "train"
TWO_OUTCOME_EXPERIMENT = "two_outcome_mode_collapse"
TABULAR_EXPERIMENT = "tabular_preference_learning"
EXPERIMENTS = (TWO_OUTCOME_EXPERIMENT, TABULAR_EXPERIMENT)

# A stratified 4/4 split over the paper reproduction's tabular evaluation
# contexts. The visible train slice has a similar mix of majority strength and
# best-of-N solution rarity to the held-out test slice.
TABULAR_CONTEXT_IDS = {"prompt_0", "prompt_1", "prompt_4", "prompt_6"}
TWO_OUTCOME_CONTEXT_IDS = {"paper_80_20_example", "train_70_30_example", "train_85_15_example"}


def entropy(probs: list[float]) -> float:
    return -sum(p * math.log(max(p, 1e-12)) for p in probs)


def kl_divergence(p: list[float], q: list[float]) -> float:
    value = sum(pi * (math.log(max(pi, 1e-12)) - math.log(max(qi, 1e-12))) for pi, qi in zip(p, q))
    return max(value, 0.0)


def brier(policy: list[float], target: list[float]) -> float:
    return sum((pi - ti) ** 2 for pi, ti in zip(policy, target))


def expected_calibration_error(policy: list[float], target: list[float], bins: int = 10) -> float:
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


def l1_error(policy: list[float], target: list[float]) -> float:
    return sum(abs(pi - ti) for pi, ti in zip(policy, target))


def validate_distribution(probs: list[float], label: str) -> None:
    if not probs:
        raise ValueError(f"{label} distribution is empty")
    if any(not math.isfinite(prob) or prob < 0.0 or prob > 1.0 for prob in probs):
        raise ValueError(f"{label} distribution contains invalid probabilities: {probs}")
    if abs(sum(probs) - 1.0) > 1e-6:
        raise ValueError(f"{label} distribution sums to {sum(probs)}, not 1.0")


def load_run(run_dir: Path) -> tuple[str, dict, dict]:
    policy_path = run_dir / "policy.json"
    if not policy_path.exists():
        raise FileNotFoundError(f"{policy_path} does not exist")
    policy = json.loads(policy_path.read_text())
    samples_path = run_dir / "samples.json"
    samples = json.loads(samples_path.read_text()) if samples_path.exists() else {"samples": []}
    method_name = policy.get("metadata", {}).get("method_name", run_dir.name)
    return method_name, policy, samples


def infer_experiment(policy: dict) -> str:
    contexts = policy.get("contexts", [])
    if contexts and all(len(context.get("actions", [])) == 2 for context in contexts):
        return TWO_OUTCOME_EXPERIMENT
    return TABULAR_EXPERIMENT


def context_ids_for(policy: dict, experiment: str) -> set[str]:
    if experiment == TWO_OUTCOME_EXPERIMENT:
        present = {context["id"] for context in policy["contexts"]}
        if len(present) == 1:
            return present
        missing = sorted(TWO_OUTCOME_CONTEXT_IDS - present)
        if missing:
            raise ValueError(f"missing {SLICE} contexts for {TWO_OUTCOME_EXPERIMENT}: {missing}")
        return set(TWO_OUTCOME_CONTEXT_IDS)
    present = {context["id"] for context in policy["contexts"]}
    missing = sorted(TABULAR_CONTEXT_IDS - present)
    if missing:
        raise ValueError(f"missing {SLICE} contexts for {TABULAR_EXPERIMENT}: {missing}")
    return set(TABULAR_CONTEXT_IDS)


def select_contexts(policy: dict, experiment: str) -> tuple[list[dict], list[str]]:
    ids = context_ids_for(policy, experiment)
    contexts = [context for context in policy["contexts"] if context["id"] in ids]
    contexts.sort(key=lambda context: context["id"])
    return contexts, [context["id"] for context in contexts]


def evaluate_run(run_dir: Path, experiment: str) -> tuple[str, dict, list[str]]:
    method_name, policy, samples = load_run(run_dir)
    actual_experiment = infer_experiment(policy)
    if actual_experiment != experiment:
        raise ValueError(f"{run_dir} contains {actual_experiment}, not {experiment}")

    contexts, selected_ids = select_contexts(policy, experiment)

    if experiment == TWO_OUTCOME_EXPERIMENT:
        context_metrics = []
        for prompt in contexts:
            actions = prompt["actions"]
            pred = [a["policy_probability"] for a in actions]
            target = [a["population_preference"] for a in actions]
            validate_distribution(pred, f"{run_dir}:{prompt['id']} policy")
            validate_distribution(target, f"{run_dir}:{prompt['id']} population")
            majority_idx = max(range(len(target)), key=lambda idx: target[idx])
            minority_idx = min(range(len(target)), key=lambda idx: target[idx])
            top_idx = max(range(len(pred)), key=lambda idx: pred[idx])
            context_metrics.append(
                {
                    "minority_probability": pred[minority_idx],
                    "majority_overweight": pred[majority_idx] / target[majority_idx],
                    "top_choice_preserved": float(top_idx == majority_idx),
                    "preference_kl": kl_divergence(target, pred),
                    "brier_score": brier(pred, target),
                    "expected_calibration_error": expected_calibration_error(pred, target),
                    "population_l1_error": l1_error(pred, target),
                }
            )

        def avg(key: str) -> float:
            return sum(item[key] for item in context_metrics) / len(context_metrics)

        result = {
            "minority_probability": avg("minority_probability"),
            "majority_overweight": avg("majority_overweight"),
            "top_choice_preserved": avg("top_choice_preserved"),
            "preference_kl": avg("preference_kl"),
            "brier_score": avg("brier_score"),
            "expected_calibration_error": avg("expected_calibration_error"),
            "population_l1_error": avg("population_l1_error"),
        }
        return method_name, result, selected_ids

    context_metrics = []
    for context in contexts:
        actions = context["actions"]
        pred = [a["policy_probability"] for a in actions]
        target = [a["population_preference"] for a in actions]
        ref = [a["reference_probability"] for a in actions]
        validate_distribution(pred, f"{run_dir}:{context['id']} policy")
        validate_distribution(target, f"{run_dir}:{context['id']} population")
        validate_distribution(ref, f"{run_dir}:{context['id']} reference")
        solution_prob = sum(a["policy_probability"] for a in actions if a.get("is_best_of_n_solution", False))
        top_idx = max(range(len(pred)), key=lambda idx: pred[idx])
        context_metrics.append(
            {
                "entropy": entropy(pred),
                "normalized_entropy": entropy(pred) / math.log(len(pred)),
                "effective_actions": math.exp(entropy(pred)),
                "preference_kl": kl_divergence(target, pred),
                "reference_cross_entropy": -sum(p * math.log(max(r, 1e-12)) for p, r in zip(pred, ref)),
                "expected_population_preference": sum(p * t for p, t in zip(pred, target)),
                "brier": brier(pred, target),
                "ece": expected_calibration_error(pred, target),
                "argmax_matches_population_mode": int(top_idx == max(range(len(target)), key=lambda idx: target[idx])),
                "best_of_128_success_probability": 1.0 - (1.0 - solution_prob) ** 128,
            }
        )

    selected_id_set = set(selected_ids)
    by_context: dict[str, list[dict]] = {}
    for row in samples.get("samples", []):
        if row.get("context_id") in selected_id_set:
            by_context.setdefault(row["context_id"], []).append(row)
    empirical_success = [
        float(any(row.get("is_best_of_n_solution", False) for row in rows)) for rows in by_context.values()
    ]

    def avg(key: str) -> float:
        return sum(item[key] for item in context_metrics) / len(context_metrics)

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
    return method_name, result, selected_ids


def discover_run_dirs() -> list[Path]:
    return sorted(path.parent for path in Path("checkpoints").glob("*/policy.json"))


def compatible_run_dirs(run_dirs: list[Path], experiment: str) -> list[Path]:
    compatible = []
    for run_dir in run_dirs:
        _method_name, policy, _samples = load_run(run_dir)
        if infer_experiment(policy) == experiment:
            compatible.append(run_dir)
    return compatible


def write_scores(scores: dict, output: Path, merge: bool) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    if merge and output.exists():
        existing = json.loads(output.read_text())
        existing.setdefault("metadata", {})["slice"] = SLICE
        existing.setdefault("experiments", {}).update(scores["experiments"])
        scores = existing
    output.write_text(json.dumps(scores, indent=2) + "\n")
    print(json.dumps(scores, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dirs", nargs="*", help="One or more artifact directories")
    parser.add_argument("--scores-out", default="scoring/scores_train.json")
    parser.add_argument("--experiment", choices=EXPERIMENTS)
    parser.add_argument("--merge", action="store_true", help="Merge these experiments into an existing scores file")
    args = parser.parse_args()

    run_dirs = [Path(path) for path in args.run_dirs] if args.run_dirs else discover_run_dirs()
    if not run_dirs:
        raise SystemExit("no run directories supplied or found under checkpoints/")

    experiments = [args.experiment] if args.experiment else list(EXPERIMENTS)
    scores = {"metadata": {"slice": SLICE}, "experiments": {}}
    for experiment in experiments:
        results = {}
        context_ids: list[str] | None = None
        for run_dir in compatible_run_dirs(run_dirs, experiment):
            method_name, result, selected_ids = evaluate_run(run_dir, experiment)
            results[method_name] = result
            context_ids = selected_ids
        if results:
            scores["experiments"][experiment] = {
                "slice": SLICE,
                "context_ids": context_ids or [],
                "results": results,
            }

    if not scores["experiments"]:
        raise SystemExit("no compatible run directories found for requested experiment(s)")
    write_scores(scores, Path(args.scores_out), args.merge)


if __name__ == "__main__":
    main()
