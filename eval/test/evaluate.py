#!/usr/bin/env python3
"""Held-out test-slice evaluator for tabular SPL artifacts.

This script reads method artifacts only. It does not import from method/ or
regenerate training outputs, so any method that writes policy.json and optional
samples.json in the established checkpoint format can be scored.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


SLICE = "test"
TWO_OUTCOME_EXPERIMENT = "two_outcome_mode_collapse"
TABULAR_EXPERIMENT = "tabular_preference_learning"
EXPERIMENTS = (TWO_OUTCOME_EXPERIMENT, TABULAR_EXPERIMENT)

# Held out from eval/train/. This is the complementary 4-context tabular slice,
# calibrated to similar majority strength and best-of-N solution rarity.
TABULAR_CONTEXT_IDS = {"prompt_2", "prompt_3", "prompt_5", "prompt_7"}


def entropy(probs: list[float]) -> float:
    return -sum(p * math.log(max(p, 1e-12)) for p in probs)


def kl_divergence(p: list[float], q: list[float]) -> float:
    return sum(pi * (math.log(max(pi, 1e-12)) - math.log(max(qi, 1e-12))) for pi, qi in zip(p, q))


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
    if len(contexts) == 1 and len(contexts[0].get("actions", [])) == 2:
        return TWO_OUTCOME_EXPERIMENT
    return TABULAR_EXPERIMENT


def context_ids_for(policy: dict, experiment: str) -> set[str]:
    if experiment == TWO_OUTCOME_EXPERIMENT:
        return {context["id"] for context in policy["contexts"]}
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
        prompt = contexts[0]
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
        return method_name, result, selected_ids

    context_metrics = []
    for context in contexts:
        actions = context["actions"]
        pred = [a["policy_probability"] for a in actions]
        target = [a["population_preference"] for a in actions]
        ref = [a["reference_probability"] for a in actions]
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


def load_train_manifest(path: Path, requested_experiment: str | None) -> dict[str, set[str]]:
    data = json.loads(path.read_text())
    experiments = data.get("experiments", data)
    wanted: dict[str, set[str]] = {}
    for experiment, payload in experiments.items():
        if requested_experiment and experiment != requested_experiment:
            continue
        if experiment not in EXPERIMENTS or not isinstance(payload, dict):
            continue
        results = payload.get("results", payload)
        wanted[experiment] = {name for name in results if not name.startswith("_")}
    return wanted


def resolve_train_methods(train_scores: Path, requested_experiment: str | None) -> dict[str, list[Path]]:
    wanted = load_train_manifest(train_scores, requested_experiment)
    if not wanted:
        raise SystemExit(f"no method entries found in {train_scores}")

    inventory: dict[tuple[str, str], list[Path]] = {}
    for run_dir in discover_run_dirs():
        method_name, policy, _samples = load_run(run_dir)
        inventory.setdefault((infer_experiment(policy), method_name), []).append(run_dir)

    resolved: dict[str, list[Path]] = {}
    missing: list[str] = []
    for experiment, methods in wanted.items():
        resolved[experiment] = []
        for method_name in sorted(methods):
            matches = sorted(inventory.get((experiment, method_name), []))
            if not matches:
                missing.append(f"{experiment}:{method_name}")
                continue
            if len(matches) > 1:
                raise SystemExit(f"ambiguous checkpoint directories for {experiment}:{method_name}: {matches}")
            resolved[experiment].append(matches[0])
    if missing:
        raise SystemExit(f"could not find checkpoint directories for train-scored methods: {missing}")
    return resolved


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
    parser.add_argument("--scores-out", default="scoring/scores_test.json")
    parser.add_argument("--experiment", choices=EXPERIMENTS)
    parser.add_argument("--methods-from-scores", help="Score the same experiment/method entries as this train score file")
    parser.add_argument("--merge", action="store_true", help="Merge these experiments into an existing scores file")
    args = parser.parse_args()

    if args.methods_from_scores:
        run_dirs_by_experiment = resolve_train_methods(Path(args.methods_from_scores), args.experiment)
        experiments = [args.experiment] if args.experiment else [exp for exp in EXPERIMENTS if exp in run_dirs_by_experiment]
    else:
        run_dirs = [Path(path) for path in args.run_dirs] if args.run_dirs else discover_run_dirs()
        if not run_dirs:
            raise SystemExit("no run directories supplied or found under checkpoints/")
        experiments = [args.experiment] if args.experiment else list(EXPERIMENTS)
        run_dirs_by_experiment = {experiment: compatible_run_dirs(run_dirs, experiment) for experiment in experiments}

    scores = {"metadata": {"slice": SLICE}, "experiments": {}}
    for experiment in experiments:
        results = {}
        context_ids: list[str] | None = None
        for run_dir in run_dirs_by_experiment.get(experiment, []):
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
