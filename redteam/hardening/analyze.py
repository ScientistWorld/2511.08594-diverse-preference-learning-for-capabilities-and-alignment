#!/usr/bin/env python3
"""Rank-gain analysis for the hardening audit.

gain(method) = sum_exp sum_metric weight_exp * |coef| * dir * (value - value_ref) / sigma_metric

dir = +1 if higher_is_better else -1; sigma_metric = population std of the
metric over the cohort of non-failed methods.  Paired bootstrap: contexts of
each experiment are resampled jointly for the method and the reference
(sigma held fixed), giving a 95% interval for the gain difference.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
from pathlib import Path


def ece(confs, accs, bins=10):
    n = len(confs)
    total = 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        idx = [i for i, c in enumerate(confs) if (lo <= c < hi) or (b == bins - 1 and c == 1.0)]
        if idx:
            total += len(idx) / n * abs(sum(confs[i] for i in idx) - sum(accs[i] for i in idx)) / len(idx)
    return total


def metrics_from_raw(exp, raw, idx):
    if exp == "calibration_multiple_choice":
        c = [raw["confidence"][i] for i in idx]
        a = [raw["accuracy"][i] for i in idx]
        b = [raw["brier_score"][i] for i in idx]
        return {"expected_calibration_error": ece(c, a), "brier_score": sum(b) / len(b), "accuracy": sum(a) / len(a)}
    return {k: sum(v[i] for i in idx) / len(idx) for k, v in raw.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", required=True)
    ap.add_argument("--per-context", required=True)
    ap.add_argument("--train-scores", default=None)
    ap.add_argument("--reference-json", default="scoring/reference.json")
    ap.add_argument("--reference-method", default="spl_alpha_1.0_beta_0.1")
    ap.add_argument("--cohort-exclude-prefix", default="shortcut_")
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    ref = json.loads(Path(args.reference_json).read_text())["experiments"]
    scores = json.loads(Path(args.scores).read_text())["experiments"]
    per = json.loads(Path(args.per_context).read_text())
    train = json.loads(Path(args.train_scores).read_text())["experiments"] if args.train_scores else None
    R = args.reference_method
    methods = sorted(next(iter(scores.values()))["results"])
    failed = {m for e in scores.values() for m in e["failed_methods"]}

    scored = {e: {m: d for m, d in ref[e]["metrics"].items() if d["coefficient"]} for e in ref}

    def sigmas(cohort, table):
        out = {}
        for e, mets in scored.items():
            out[e] = {}
            for m in mets:
                vals = [table[e]["results"][x][m] for x in cohort]
                out[e][m] = statistics.pstdev(vals) if len(vals) > 1 else 0.0
        return out

    def gain(values, ref_values, sig):
        total, parts = 0.0, {}
        for e, mets in scored.items():
            w = ref[e]["weight"]
            parts[e] = 0.0
            for m, d in mets.items():
                s = sig[e][m]
                if not s:
                    continue
                direction = 1.0 if d["higher_is_better"] else -1.0
                c = w * abs(d["coefficient"]) * direction * (values[e][m] - ref_values[e][m]) / s
                parts[e] += c
                total += c
        return total, parts

    cohorts = {
        "honest_cohort": [m for m in methods if m not in failed and not m.startswith(args.cohort_exclude_prefix)],
        "all_successful_cohort": [m for m in methods if m not in failed],
    }
    rng = random.Random(0)
    summary = {"formula": "sum(weight*|coef|*dir*(value-ref)/pstdev_cohort)", "reference_method": R,
               "failed": {m: scores[next(iter(scores))]["failed_methods"][m] for m in sorted(failed)}, "cohorts": {}}
    table_vals = {m: {e: scores[e]["results"][m] for e in scores} for m in methods}
    for cname, cohort in cohorts.items():
        sig = sigmas(cohort, scores)
        rows = {}
        for m in methods:
            g, parts = gain(table_vals[m], table_vals[R], sig)
            row = {"gain": g, "by_experiment": parts, "failed": m in failed}
            if train and m in train[next(iter(train))]["results"]:
                tv = {e: train[e]["results"][m] for e in train}
                trv = {e: train[e]["results"][R] for e in train}
                row["train_gain_same_sigma"] = gain(tv, trv, sig)[0]
            if m not in failed and m != R and args.bootstrap:
                diffs = []
                sizes = {e: len(next(iter(per[e][R].values()))) for e in per}
                for _ in range(args.bootstrap):
                    vm, vr = {}, {}
                    for e in per:
                        idx = [rng.randrange(sizes[e]) for _ in range(sizes[e])]
                        vm[e] = metrics_from_raw(e, per[e][m], idx)
                        vr[e] = metrics_from_raw(e, per[e][R], idx)
                    diffs.append(gain(vm, vr, sig)[0])
                diffs.sort()
                row["paired_bootstrap_95ci"] = [diffs[int(0.025 * len(diffs))], diffs[int(0.975 * len(diffs)) - 1]]
                row["paired_bootstrap_frac_positive"] = sum(1 for d in diffs if d > 0) / len(diffs)
            rows[m] = row
        summary["cohorts"][cname] = {"members": cohort, "sigma": sig, "rows": rows}
    summary["raw_test_metrics"] = table_vals
    Path(args.out).write_text(json.dumps(summary, indent=2) + "\n")

    # markdown table
    lines = []
    c = summary["cohorts"]["honest_cohort"]
    keys = [(e, m) for e in scored for m in scored[e]]
    lines.append("| method | " + " | ".join(f"{m}" for _e, m in keys) + " | gain (honest σ) | 95% paired CI | gain (all σ) |")
    lines.append("|---|" + "---:|" * (len(keys) + 3))
    for m in methods:
        r = c["rows"][m]
        vals = " | ".join(f"{table_vals[m][e][k]:.4g}" for e, k in keys)
        ci = r.get("paired_bootstrap_95ci")
        ci_s = f"[{ci[0]:+.3f}, {ci[1]:+.3f}]" if ci else ("failed run" if r["failed"] else "—")
        ga = summary["cohorts"]["all_successful_cohort"]["rows"][m]["gain"]
        lines.append(f"| {m} | {vals} | {r['gain']:+.3f} | {ci_s} | {ga:+.3f} |")
    Path(args.out).with_suffix(".md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
