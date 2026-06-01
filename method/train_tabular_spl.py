#!/usr/bin/env python3
"""Train a tabular policy with Soft Preference Learning.

For each finite prompt, the policy is a learnable categorical distribution over
candidate completions. Training maximizes the paper's DPO-style SPL objective:

  log sigmoid(alpha * log(pi(y_w) / pi(y_l))
              - beta * log(pi_ref(y_w) / pi_ref(y_l)))

DPO is recovered by setting alpha == beta.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path


def softmax(logits: list[float]) -> list[float]:
    offset = max(logits)
    exps = [math.exp(x - offset) for x in logits]
    total = sum(exps)
    return [x / total for x in exps]


def sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def train_context(
    context: dict,
    alpha: float,
    beta: float,
    steps: int,
    lr: float,
) -> tuple[list[float], list[dict]]:
    action_ids = [a["id"] for a in context["actions"]]
    action_to_idx = {action_id: idx for idx, action_id in enumerate(action_ids)}
    ref = [a["reference_probability"] for a in context["actions"]]

    # Initialize from the reference policy, as in DPO-style finetuning from a
    # reference model.
    logits = [math.log(max(p, 1e-12)) for p in ref]
    comparisons = [
        (
            action_to_idx[item["winner"]],
            action_to_idx[item["loser"]],
            int(item["count"]),
        )
        for item in context["comparisons"]
    ]
    total_count = sum(count for _, _, count in comparisons)
    log = []

    for step in range(1, steps + 1):
        probs = softmax(logits)
        grad = [0.0 for _ in logits]
        loss = 0.0

        for winner, loser, count in comparisons:
            log_pi_ratio = math.log(probs[winner]) - math.log(probs[loser])
            log_ref_ratio = math.log(ref[winner]) - math.log(ref[loser])
            margin = alpha * log_pi_ratio - beta * log_ref_ratio
            s = sigmoid(margin)
            loss += count * (-math.log(max(s, 1e-12)))

            # d(log pi_w - log pi_l)/d logits = onehot(w) - onehot(l).
            scale = count * alpha * (s - 1.0)
            grad[winner] += scale
            grad[loser] -= scale

        for idx in range(len(logits)):
            logits[idx] -= lr * grad[idx] / total_count

        if step == 1 or step == steps or step % max(1, steps // 10) == 0:
            log.append({"step": step, "loss": loss / total_count})

    return softmax(logits), log


def sample_policy(contexts: list[dict], samples_per_context: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    samples = []
    for context in contexts:
        probs = [a["policy_probability"] for a in context["actions"]]
        cumulative = []
        running = 0.0
        for prob in probs:
            running += prob
            cumulative.append(running)
        for sample_idx in range(samples_per_context):
            draw = rng.random()
            chosen_idx = 0
            for idx, cutoff in enumerate(cumulative):
                if draw <= cutoff:
                    chosen_idx = idx
                    break
            action = context["actions"][chosen_idx]
            samples.append(
                {
                    "context_id": context["id"],
                    "sample_index": sample_idx,
                    "action_id": action["id"],
                    "is_best_of_n_solution": action["is_best_of_n_solution"],
                }
            )
    return samples


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", default="data/two_outcome_benchmark.json")
    parser.add_argument("--output-dir", default="checkpoints/spl_tabular")
    parser.add_argument("--method-name", default="spl_alpha_0.5_beta_0.1")
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--lr", type=float, default=0.8)
    parser.add_argument("--samples-per-context", type=int, default=128)
    parser.add_argument("--seed", type=int, default=11)
    args = parser.parse_args()

    benchmark = json.loads(Path(args.benchmark).read_text())
    trained_contexts = []
    train_logs = {}
    for context in benchmark["contexts"]:
        probs, train_log = train_context(context, args.alpha, args.beta, args.steps, args.lr)
        trained_context = json.loads(json.dumps(context))
        for action, prob in zip(trained_context["actions"], probs):
            action["policy_probability"] = prob
        trained_contexts.append(trained_context)
        train_logs[context["id"]] = train_log

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "method_name": args.method_name,
        "objective": "SPL DPO-style preference objective",
        "alpha": args.alpha,
        "beta": args.beta,
        "global_temperature": args.alpha / args.beta,
        "steps": args.steps,
        "lr": args.lr,
        "benchmark": args.benchmark,
    }
    policy = {"metadata": metadata, "contexts": trained_contexts}
    samples = {
        "metadata": {"samples_per_context": args.samples_per_context, "seed": args.seed},
        "samples": sample_policy(trained_contexts, args.samples_per_context, args.seed),
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    (output_dir / "policy.json").write_text(json.dumps(policy, indent=2) + "\n")
    (output_dir / "samples.json").write_text(json.dumps(samples, indent=2) + "\n")
    (output_dir / "train_log.json").write_text(json.dumps(train_logs, indent=2) + "\n")
    print(f"wrote SPL artifacts to {output_dir}")


if __name__ == "__main__":
    main()
