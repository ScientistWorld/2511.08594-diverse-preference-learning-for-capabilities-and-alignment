#!/usr/bin/env python3
"""Visible train-slice evaluator.

Scores every method directory under checkpoints/ (one directory = one method)
on the visible train contexts of all three experiments.  It applies the same
metrics, gates and failure rules as the hidden test-slice evaluator, so a
method developed against this file is measured the same way on test.

Trust model
-----------
* The only thing read from a method directory is the policy probability table
  of each experiment file (``two_outcome.json``, ``sequence.json``,
  ``calibration.json``): ``contexts[<id>].initial`` (and ``.transitions`` for
  the sequence experiment).  Every other field a method writes is ignored.
* Ground truth (population preferences, reward potentials, verifier sets,
  correct-answer distributions), reference policies, gates and worst values are
  owned by this directory (``truth.json`` and the constants below).
* Every scored number is recomputed exactly (dynamic programming over the
  policy table); no samples, flags or numbers written by a method are used.
* A method whose artifact is missing, malformed, not a probability
  distribution, missing any train context, or failing a task gate in ANY
  experiment is a failed run: it receives the worst value of every metric in
  every experiment.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SLICE = "train"
TRUTH_PATH = HERE / "truth.json"
PAPER_EXAMPLE = "two_outcome_paper_80_20"  # the paper's 80/20 example (uniform reference)

TWO = "two_outcome_mode_collapse"
SEQ = "sequence_diversity_best_of_n"
CAL = "calibration_multiple_choice"
FILES = {TWO: "two_outcome.json", SEQ: "sequence.json", CAL: "calibration.json"}
BEST_OF_N = 128

# metric -> worst value (also the cap applied to valid runs)
WORST = {
    TWO: {"preference_kl": 20.0, "minority_log_ratio_error": 30.0},
    SEQ: {"best_of_128_success_hard": 0.0, "population_kl": 60.0, "coherence": 0.0},
    CAL: {"expected_calibration_error": 1.0, "brier_score": 2.0, "accuracy": 0.0},
}
LOWER_IS_BETTER = {"preference_kl", "minority_log_ratio_error", "population_kl", "expected_calibration_error", "brier_score"}

# task gates (fixed by the evaluator)
GATE_TOP_CHOICE = 0.75          # two-outcome: majority must stay on top in >= 75% of contexts
GATE_ACCURACY_SLACK = 0.02      # calibration: accuracy >= reference-policy accuracy - 0.02
SUM_TOL = 1e-4


class Invalid(Exception):
    pass


# ---------------------------------------------------------------------------
# validation helpers
# ---------------------------------------------------------------------------


def as_distribution(row, size, label):
    if not isinstance(row, list) or len(row) != size:
        raise Invalid(f"{label}: expected a list of {size} probabilities")
    out = []
    for x in row:
        if isinstance(x, bool) or not isinstance(x, (int, float)):
            raise Invalid(f"{label}: non-numeric probability {x!r}")
        x = float(x)
        if not math.isfinite(x) or x < 0.0 or x > 1.0:
            raise Invalid(f"{label}: impossible probability {x!r}")
        out.append(x)
    s = sum(out)
    if abs(s - 1.0) > SUM_TOL:
        raise Invalid(f"{label}: probabilities sum to {s!r}, not 1")
    return [x / s for x in out]


def load_contexts(method_dir: Path, experiment: str, ids):
    path = method_dir / FILES[experiment]
    if not path.is_file():
        raise Invalid(f"missing {FILES[experiment]}")
    try:
        payload = json.loads(path.read_text())
    except Exception as exc:  # noqa: BLE001
        raise Invalid(f"unreadable {FILES[experiment]}: {type(exc).__name__}")
    if not isinstance(payload, dict) or not isinstance(payload.get("contexts"), dict):
        raise Invalid(f"{FILES[experiment]}: missing 'contexts' mapping")
    ctx = payload["contexts"]
    missing = [cid for cid in ids if cid not in ctx or not isinstance(ctx[cid], dict)]
    if missing:
        raise Invalid(f"{FILES[experiment]}: missing {len(missing)} required contexts (e.g. {missing[0]})")
    return ctx


# ---------------------------------------------------------------------------
# chain utilities
# ---------------------------------------------------------------------------


def lse(xs):
    m = max(xs)
    if m == -math.inf:
        return -math.inf
    return m + math.log(sum(math.exp(x - m) for x in xs))


def chain_from_potentials(log_init, log_psi):
    V = len(log_init)
    n = len(log_psi)
    msgs = [None] * (n + 1)
    msgs[n] = [0.0] * V
    for i in range(n - 1, -1, -1):
        msgs[i] = [lse([log_psi[i][a][b] + msgs[i + 1][b] for b in range(V)]) for a in range(V)]
    T = []
    for i in range(n):
        Ti = []
        for a in range(V):
            row = [math.exp(log_psi[i][a][b] + msgs[i + 1][b] - msgs[i][a]) for b in range(V)]
            s = sum(row)
            Ti.append([x / s for x in row])
        T.append(Ti)
    a = [log_init[v] + msgs[0][v] for v in range(V)]
    z = lse(a)
    return [math.exp(x - z) for x in a], T


def marginals(init, T):
    V = len(init)
    node = [init]
    pair = []
    for Ti in T:
        prev = node[-1]
        pr = [[prev[a] * Ti[a][b] for b in range(V)] for a in range(V)]
        pair.append(pr)
        node.append([sum(pr[a][b] for a in range(V)) for b in range(V)])
    return node, pair


def xlogx(x):
    return x * math.log(x) if x > 0.0 else 0.0


def chain_entropy(init, T):
    node, _ = marginals(init, T)
    h = -sum(xlogx(x) for x in init)
    for i, Ti in enumerate(T):
        h += sum(node[i][a] * -sum(xlogx(x) for x in Ti[a]) for a in range(len(init)))
    return h


def expectation(init, T, f0, f):
    _node, pair = marginals(init, T)
    V = len(init)
    s = sum(init[v] * f0[v] for v in range(V))
    for i in range(len(T)):
        s += sum(pair[i][a][b] * f[i][a][b] for a in range(V) for b in range(V))
    return s


def cross_entropy(p_init, p_T, q_init, q_T):
    """-E_p[log q]; returns inf if q puts zero mass where p has mass."""
    _node, pair = marginals(p_init, p_T)
    V = len(p_init)
    total = 0.0
    for v in range(V):
        if p_init[v] > 0.0:
            if q_init[v] <= 0.0:
                return math.inf
            total -= p_init[v] * math.log(q_init[v])
    for i in range(len(p_T)):
        for a in range(V):
            for b in range(V):
                w = pair[i][a][b]
                if w > 0.0:
                    q = q_T[i][a][b]
                    if q <= 0.0:
                        return math.inf
                    total -= w * math.log(q)
    return total


def mask_prob(init, T, masks):
    V = len(init)
    a = [init[v] * masks[0][v] for v in range(V)]
    for i, Ti in enumerate(T):
        a = [sum(a[u] * Ti[u][v] for u in range(V)) * masks[i + 1][v] for v in range(V)]
    return sum(a)


def clip(value, metric, experiment):
    worst = WORST[experiment][metric]
    if metric in LOWER_IS_BETTER:
        return min(value, worst) if math.isfinite(value) or value > 0 else worst
    return max(value, worst)


# ---------------------------------------------------------------------------
# experiment scorers
# ---------------------------------------------------------------------------


def score_two_outcome(method_dir, truth):
    ids = sorted(truth)
    ctx = load_contexts(method_dir, TWO, ids)
    kls, lrs, tops = [], [], []
    for cid in ids:
        p = truth[cid]["population_preference"]
        pi = as_distribution(ctx[cid].get("initial"), 2, f"{TWO}:{cid}")
        maj = 0 if p[0] > p[1] else 1
        mino = 1 - maj
        kl = 0.0
        for pk, qk in zip(p, pi):
            if pk > 0.0:
                kl = math.inf if qk <= 0.0 else kl + pk * math.log(pk / qk)
        kls.append(min(kl, WORST[TWO]["preference_kl"]))
        lr = math.inf if pi[mino] <= 0.0 else abs(math.log(pi[mino] / p[mino]))
        lrs.append(min(lr, WORST[TWO]["minority_log_ratio_error"]))
        tops.append(1.0 if pi[maj] > pi[mino] else 0.0)
    n = len(ids)
    paper = {}
    if PAPER_EXAMPLE in ctx:
        try:
            pi = as_distribution(ctx[PAPER_EXAMPLE].get("initial"), 2, PAPER_EXAMPLE)
            paper = {"paper_80_20_minority_probability": pi[1], "paper_80_20_majority_overweight": pi[0] / 0.8}
        except Invalid:
            paper = {}
    res = {
        "preference_kl": sum(kls) / n,
        "minority_log_ratio_error": sum(lrs) / n,
    }
    top_rate = sum(tops) / n
    gate = None
    if top_rate < GATE_TOP_CHOICE:
        gate = f"top_choice_preserved {top_rate:.3f} < gate {GATE_TOP_CHOICE}"
    paper["top_choice_preserved"] = top_rate
    return res, paper, gate, {"preference_kl": kls, "minority_log_ratio_error": lrs, "top_choice_preserved": tops}


def score_sequence(method_dir, truth):
    V = truth["vocab_size"]
    L = truth["length"]
    contexts = truth["contexts"]
    ids = sorted(contexts)
    ctx = load_contexts(method_dir, SEQ, ids)
    kls, bons, cohs, rews, refrews, ents = [], [], [], [], [], []
    for cid in ids:
        t = contexts[cid]
        art = ctx[cid]
        init = as_distribution(art.get("initial"), V, f"{SEQ}:{cid}:initial")
        trans = art.get("transitions")
        if not isinstance(trans, list) or len(trans) != L - 1:
            raise Invalid(f"{SEQ}:{cid}: expected {L - 1} transition matrices")
        T = []
        for i, Ti in enumerate(trans):
            if not isinstance(Ti, list) or len(Ti) != V:
                raise Invalid(f"{SEQ}:{cid}: transition {i} must have {V} rows")
            T.append([as_distribution(row, V, f"{SEQ}:{cid}:T{i}[{a}]") for a, row in enumerate(Ti)])
        th0, th = t["reward_initial"], t["reward_transitions"]
        p_init, p_T = chain_from_potentials(th0, th)
        h_p = chain_entropy(p_init, p_T)
        ce = cross_entropy(p_init, p_T, init, T)
        kl = ce - h_p if math.isfinite(ce) else math.inf
        kls.append(min(max(kl, 0.0), WORST[SEQ]["population_kl"]))
        bad = set(t["incoherent_tokens"])
        coherent = [[v not in bad for v in range(V)] for _ in range(L)]
        cohs.append(mask_prob(init, T, coherent))
        succ = []
        for prob in t["hard_problems"]:
            masks = [row[:] for row in coherent]
            for j, tok in zip(prob["positions"], prob["tokens"]):
                masks[j] = [v == tok for v in range(V)]
            q = min(max(mask_prob(init, T, masks), 0.0), 1.0)
            succ.append(1.0 - (1.0 - q) ** BEST_OF_N)
        bons.append(sum(succ) / len(succ))
        rews.append(expectation(init, T, th0, th))
        refrews.append(expectation(t["reference_initial"], t["reference_transitions"], th0, th))
        ents.append(chain_entropy(init, T))
    n = len(ids)
    res = {
        "best_of_128_success_hard": sum(bons) / n,
        "population_kl": sum(kls) / n,
        "coherence": sum(cohs) / n,
    }
    reward = sum(rews) / n
    ref_reward = sum(refrews) / n
    diag = {"expected_population_reward": reward, "reference_policy_reward": ref_reward, "sequence_entropy": sum(ents) / n}
    gate = None
    if not reward >= ref_reward:
        gate = f"expected_population_reward {reward:.3f} below the reference policy's {ref_reward:.3f}"
    raw = {"best_of_128_success_hard": bons, "population_kl": kls, "coherence": cohs, "expected_population_reward": rews}
    return res, diag, gate, raw


def score_calibration(method_dir, truth):
    ids = sorted(truth)
    ctx = load_contexts(method_dir, CAL, ids)
    confs, accs, briers, ref_accs = [], [], [], []
    for cid in ids:
        p = truth[cid]["correct_answer_distribution"]
        K = len(p)
        pi = as_distribution(ctx[cid].get("initial"), K, f"{CAL}:{cid}")
        top = max(range(K), key=lambda k: (pi[k], -k))
        confs.append(pi[top])
        accs.append(p[top])
        briers.append(sum(x * x for x in pi) - 2.0 * sum(a * b for a, b in zip(pi, p)) + 1.0)
        ref = truth[cid]["reference_probability"]
        rtop = max(range(K), key=lambda k: (ref[k], -k))
        ref_accs.append(p[rtop])
    n = len(ids)
    bins = 10
    ece = 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        idx = [i for i, c in enumerate(confs) if (lo <= c < hi) or (b == bins - 1 and c == 1.0)]
        if idx:
            gap = abs(sum(confs[i] for i in idx) - sum(accs[i] for i in idx)) / len(idx)
            ece += len(idx) / n * gap
    res = {
        "expected_calibration_error": ece,
        "brier_score": sum(briers) / n,
        "accuracy": sum(accs) / n,
    }
    ref_acc = sum(ref_accs) / n
    diag = {"mean_confidence": sum(confs) / n, "reference_policy_accuracy": ref_acc}
    gate = None
    if res["accuracy"] < ref_acc - GATE_ACCURACY_SLACK:
        gate = f"accuracy {res['accuracy']:.4f} below reference-policy accuracy {ref_acc:.4f} - {GATE_ACCURACY_SLACK}"
    return res, diag, gate, {"confidence": confs, "accuracy": accs, "brier_score": briers}


SCORERS = {TWO: score_two_outcome, SEQ: score_sequence, CAL: score_calibration}


def evaluate_method(method_dir: Path, truth: dict):
    results, diags, raws, failure = {}, {}, {}, None
    for exp, scorer in SCORERS.items():
        try:
            res, diag, gate, raw = scorer(method_dir, truth[exp])
        except Invalid as exc:
            failure = failure or f"{exp}: {exc}"
            continue
        except Exception as exc:  # noqa: BLE001 - any malformed artifact is a failed run
            failure = failure or f"{exp}: {type(exc).__name__}: {exc}"
            continue
        for metric, value in res.items():
            if not isinstance(value, float) or not math.isfinite(value):
                failure = failure or f"{exp}: non-finite {metric}"
            else:
                res[metric] = clip(value, metric, exp)
        if gate:
            failure = failure or f"{exp}: task gate failed: {gate}"
        results[exp] = res
        diags[exp] = diag
        raws[exp] = raw
    if failure:
        results = {exp: dict(WORST[exp]) for exp in SCORERS}
        raws = {}
    return results, diags, raws, failure


def discover_methods(checkpoints: Path):
    if not checkpoints.is_dir():
        return []
    return sorted(p for p in checkpoints.iterdir() if p.is_dir() and not p.name.startswith("."))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("method_dirs", nargs="*", help="Optional explicit method directories (default: every checkpoints/* directory)")
    parser.add_argument("--checkpoints", default="checkpoints")
    parser.add_argument("--per-context-out", default=None,
                        help="Optional: also write per-context metric values (for paired analyses).")
    parser.add_argument("--scores-out", default="scoring/scores_train.json")
    args = parser.parse_args()

    truth = json.loads(TRUTH_PATH.read_text())
    dirs = [Path(d) for d in args.method_dirs] if args.method_dirs else discover_methods(Path(args.checkpoints))
    scores = {
        "metadata": {"slice": SLICE, "methods": [d.name for d in dirs], "best_of_n": BEST_OF_N},
        "experiments": {
            exp: {"slice": SLICE, "num_contexts": len(truth[exp]["contexts"] if exp == SEQ else truth[exp]),
                  "results": {}, "failed_methods": {}, "diagnostics": {}}
            for exp in SCORERS
        },
    }
    per_context = {exp: {} for exp in SCORERS}
    for d in dirs:
        results, diags, raws, failure = evaluate_method(d, truth)
        for exp, raw in raws.items():
            per_context[exp][d.name] = raw
        for exp in SCORERS:
            scores["experiments"][exp]["results"][d.name] = results[exp]
            if failure:
                scores["experiments"][exp]["failed_methods"][d.name] = failure
            elif exp in diags:
                scores["experiments"][exp]["diagnostics"][d.name] = diags[exp]
        status = f"FAILED ({failure})" if failure else "ok"
        print(f"[{SLICE}] {d.name}: {status}", flush=True)
    out = Path(args.scores_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(scores, indent=2) + "\n")
    if args.per_context_out:
        Path(args.per_context_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.per_context_out).write_text(json.dumps(per_context) + "\n")
    if not dirs:
        print(f"[{SLICE}] warning: no method directories found under {args.checkpoints}", file=sys.stderr)
    print(json.dumps({e: v["results"] for e, v in scores["experiments"].items()}, indent=2))


if __name__ == "__main__":
    main()
