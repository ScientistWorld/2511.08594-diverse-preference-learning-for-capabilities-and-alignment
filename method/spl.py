#!/usr/bin/env python3
"""Soft Preference Learning (SPL) for tabular first-order chain policies.

Implements the paper's DPO-style SPL objective (Eq. 5):

    max_pi  E_{y_w > y_l ~ D} [ log sigmoid( alpha * log pi(y_w)/pi(y_l)
                                            - beta  * log pi_ref(y_w)/pi_ref(y_l) ) ]

Standard DPO is the special case alpha == beta.  The policy for each prompt is
a first-order chain: pi(y) = init[y_0] * prod_i T_i[y_{i-1}, y_i], with one
softmax-parameterised logit table per row.  A single-step categorical policy
(two-outcome and multiple-choice prompts) is the length-1 case.  Logits are
initialised at the reference policy and optimised with full-batch gradient
descent on the mean pairwise loss.

Writes checkpoints/<name>/{two_outcome,sequence,calibration}.json in the
artifact format read by eval/.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

EXPERIMENT_FILES = {
    "two_outcome_mode_collapse": ("two_outcome_contexts.json", "two_outcome.json"),
    "sequence_diversity_best_of_n": ("sequence_contexts.json", "sequence.json"),
    "calibration_multiple_choice": ("calibration_contexts.json", "calibration.json"),
}


def sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    z = math.exp(x)
    return z / (1.0 + z)


def log_sigmoid(x: float) -> float:
    if x >= 0:
        return -math.log1p(math.exp(-x))
    return x - math.log1p(math.exp(x))


def seq_entries(y, V):
    out = [(0, y[0])]
    for i in range(1, len(y)):
        out.append((1 + (i - 1) * V + y[i - 1], y[i]))
    return out


def prepare_pairs(pairs, V):
    """pairs: iterable of (y_w, y_l, count) with token-id lists. Returns sparse differences."""
    prepped = []
    for yw, yl, cnt in pairs:
        tok, row = {}, {}
        for r, b in seq_entries(yw, V):
            tok[(r, b)] = tok.get((r, b), 0) + 1
            row[r] = row.get(r, 0) + 1
        for r, b in seq_entries(yl, V):
            tok[(r, b)] = tok.get((r, b), 0) - 1
            row[r] = row.get(r, 0) - 1
        tok_l = [(r * V + b, c) for (r, b), c in tok.items() if c != 0]
        row_l = [(r, c) for r, c in row.items() if c != 0]
        if tok_l:
            prepped.append((tok_l, row_l, float(cnt)))
    return prepped


def train_chain(ref_rows, pairs, alpha, beta, steps, lr, label_smoothing=0.0, shared_token_bias=False):
    """Optimise the SPL objective over a tabular chain policy.

    ref_rows: list of reference probability rows (row 0 = initial distribution,
    then (L-1)*V transition rows).  Returns (policy_rows, loss_history).
    """
    V = len(ref_rows[0])
    R = len(ref_rows)
    ref_logp = [math.log(max(x, 1e-300)) for row in ref_rows for x in row]
    logits = ref_logp[:]
    prepped = prepare_pairs(pairs, V)
    if not prepped:
        return [list(r) for r in ref_rows], []
    dref = [sum(c * ref_logp[e] for e, c in tok_l) for tok_l, _r, _c in prepped]
    total = sum(c for _t, _r, c in prepped)
    eps = float(label_smoothing)
    use_bias = bool(shared_token_bias) and R > 1
    bias = [0.0] * V  # optional variant: one logit offset per token shared by every row
    history = []
    log_every = max(1, steps // 10)
    for step in range(1, steps + 1):
        logp = [0.0] * (R * V)
        prob = [0.0] * (R * V)
        for r in range(R):
            base = r * V
            seg = logits[base:base + V]
            if use_bias:
                seg = [x + b for x, b in zip(seg, bias)]
            mx = max(seg)
            ex = [math.exp(x - mx) for x in seg]
            s = sum(ex)
            ls = math.log(s) + mx
            for k in range(V):
                logp[base + k] = seg[k] - ls
                prob[base + k] = ex[k] / s
        G = [0.0] * (R * V)
        Rg = [0.0] * R
        loss = 0.0
        for (tok_l, row_l, cnt), dr in zip(prepped, dref):
            s_pi = 0.0
            for e, c in tok_l:
                s_pi += c * logp[e]
            marg = alpha * s_pi - beta * dr
            sg = sigmoid(marg)
            if eps:
                loss -= cnt * ((1.0 - eps) * log_sigmoid(marg) + eps * log_sigmoid(-marg))
                g = cnt * alpha * (sg - (1.0 - eps)) / total
            else:
                loss -= cnt * log_sigmoid(marg)
                g = cnt * alpha * (sg - 1.0) / total
            for e, c in tok_l:
                G[e] += g * c
            for r, c in row_l:
                Rg[r] += g * c
        bias_grad = [0.0] * V
        for r in range(R):
            rg = Rg[r]
            base = r * V
            for k in range(V):
                e = base + k
                gr = G[e] - rg * prob[e]
                logits[e] -= lr * gr
                bias_grad[k] += gr
        if use_bias:
            for k in range(V):
                bias[k] -= lr * bias_grad[k]
        if step == 1 or step == steps or step % log_every == 0:
            history.append({"step": step, "loss": loss / total})
    rows = []
    for r in range(R):
        seg = logits[r * V:(r + 1) * V]
        if use_bias:
            seg = [x + b for x, b in zip(seg, bias)]
        mx = max(seg)
        ex = [math.exp(x - mx) for x in seg]
        s = sum(ex)
        rows.append([x / s for x in ex])
    return rows, history


def decode(s: str, alphabet: str):
    return [alphabet.index(ch) for ch in s]


def run_categorical(benchmark, alpha, beta, steps, lr, label_smoothing, shared_token_bias=False):
    out, logs = {}, {}
    for ctx in benchmark["contexts"]:
        ref = ctx["reference_probability"]
        pairs = [([w], [l], c) for w, l, c in ctx["comparisons"]]
        rows, hist = train_chain([ref], pairs, alpha, beta, steps, lr, label_smoothing)
        out[ctx["id"]] = {"initial": rows[0]}
        logs[ctx["id"]] = hist[-1]["loss"] if hist else None
    return out, logs


def run_sequence(benchmark, alpha, beta, steps, lr, label_smoothing, shared_token_bias=False, progress=True):
    alphabet = benchmark["token_alphabet"]
    V = benchmark["vocab_size"]
    out, logs = {}, {}
    t0 = time.time()
    for n, ctx in enumerate(benchmark["contexts"]):
        ref_rows = [ctx["reference_initial"]] + [row for Ti in ctx["reference_transitions"] for row in Ti]
        pairs = [(decode(w, alphabet), decode(l, alphabet), 1) for w, l in ctx["pairs"]]
        rows, hist = train_chain(ref_rows, pairs, alpha, beta, steps, lr, label_smoothing, shared_token_bias)
        L = len(ctx["reference_transitions"]) + 1
        out[ctx["id"]] = {
            "initial": rows[0],
            "transitions": [rows[1 + i * V: 1 + (i + 1) * V] for i in range(L - 1)],
        }
        logs[ctx["id"]] = hist[-1]["loss"] if hist else None
        if progress:
            print(f"  sequence context {n + 1}/{len(benchmark['contexts'])} done ({time.time() - t0:.0f}s)", flush=True)
    return out, logs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--method-name", default=None)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--label-smoothing", type=float, default=0.0,
                        help="Variant knob (0 = the paper's objective).")
    parser.add_argument("--label-smoothing-sequence", type=float, default=None,
                        help="Variant knob: label smoothing for the sequence experiment only (default: --label-smoothing).")
    parser.add_argument("--shared-token-bias", action="store_true",
                        help="Variant knob: add one logit offset per token shared across rows (sequence experiment).")
    parser.add_argument("--steps-categorical", type=int, default=3000)
    parser.add_argument("--lr-categorical", type=float, default=1.0)
    parser.add_argument("--steps-sequence", type=int, default=1500)
    parser.add_argument("--lr-sequence", type=float, default=5.0)
    parser.add_argument("--experiments", default=",".join(EXPERIMENT_FILES))
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    name = args.method_name or out_dir.name
    config = {
        "method_name": name,
        "objective": "SPL DPO-style pairwise objective (paper Eq. 5); DPO when alpha == beta",
        "alpha": args.alpha,
        "beta": args.beta,
        "global_temperature": args.alpha / args.beta,
        "label_smoothing": args.label_smoothing,
        "label_smoothing_sequence": args.label_smoothing_sequence,
        "shared_token_bias": args.shared_token_bias,
        "steps_categorical": args.steps_categorical,
        "lr_categorical": args.lr_categorical,
        "steps_sequence": args.steps_sequence,
        "lr_sequence": args.lr_sequence,
    }
    wanted = [e for e in args.experiments.split(",") if e]
    final_logs = {}
    for exp in wanted:
        src, dst = EXPERIMENT_FILES[exp]
        target = out_dir / dst
        benchmark = json.loads((Path(args.data_dir) / src).read_text())
        print(f"[{name}] training {exp} on {len(benchmark['contexts'])} contexts", flush=True)
        if exp == "sequence_diversity_best_of_n":
            ls_seq = args.label_smoothing if args.label_smoothing_sequence is None else args.label_smoothing_sequence
            policies, logs = run_sequence(benchmark, args.alpha, args.beta, args.steps_sequence,
                                          args.lr_sequence, ls_seq, args.shared_token_bias)
        else:
            policies, logs = run_categorical(benchmark, args.alpha, args.beta, args.steps_categorical,
                                             args.lr_categorical, args.label_smoothing)
        target.write_text(json.dumps({"experiment": exp, "method_name": name, "contexts": policies}) + "\n")
        final_logs[exp] = logs
    (out_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    (out_dir / "train_log.json").write_text(json.dumps(final_logs) + "\n")
    print(f"[{name}] wrote artifacts to {out_dir}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
