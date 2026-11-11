#!/usr/bin/env python3
"""Build the shortcut / control method directories for the hardening audit.

Every case is an ordinary method directory; it is scored by the unmodified
hidden-test evaluator together with the reference runs and honest variants.
Inputs: the public data/ files, the reference SPL artifacts, and (only for the
train-memorisation case) the visible eval/train/truth.json.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import shutil
from pathlib import Path

FILES = ("two_outcome.json", "sequence.json", "calibration.json")


def load(p):
    return json.loads(Path(p).read_text())


def dump(p, obj):
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    Path(p).write_text(json.dumps(obj) + "\n")


def norm(row):
    s = sum(row)
    return [x / s for x in row]


def copy_ref(ref_dir: Path, dst: Path):
    dst.mkdir(parents=True, exist_ok=True)
    for f in FILES:
        shutil.copy(ref_dir / f, dst / f)


def coherent_ids(row, thresh=0.005):
    return [k for k, x in enumerate(row) if x >= thresh]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref-dir", default="checkpoints/spl_alpha_1.0_beta_0.1")
    ap.add_argument("--out", default="redteam/hardening/runs/checkpoints")
    ap.add_argument("--data", default="data")
    args = ap.parse_args()
    ref_dir, out, data = Path(args.ref_dir), Path(args.out), Path(args.data)
    out.mkdir(parents=True, exist_ok=True)
    two = load(data / "two_outcome_contexts.json")
    cal = load(data / "calibration_contexts.json")
    seq = load(data / "sequence_contexts.json")
    V, L = seq["vocab_size"], seq["length"]
    ref_two, ref_seq, ref_cal = (load(ref_dir / f) for f in FILES)

    def write_case(name, two_ctx=None, seq_ctx=None, cal_ctx=None, extra=None):
        d = out / name
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True)
        dump(d / "two_outcome.json", {"contexts": two_ctx if two_ctx is not None else ref_two["contexts"]})
        dump(d / "sequence.json", {"contexts": seq_ctx if seq_ctx is not None else ref_seq["contexts"]})
        dump(d / "calibration.json", {"contexts": cal_ctx if cal_ctx is not None else ref_cal["contexts"]})
        if extra:
            extra(d)

    # 1. uniform policy everywhere (old gym: +0.43 over the reference)
    uni2 = {c["id"]: {"initial": [0.5, 0.5]} for c in two["contexts"]}
    uni4 = {c["id"]: {"initial": [0.25] * 4} for c in cal["contexts"]}
    uniS = {c["id"]: {"initial": [1.0 / V] * V, "transitions": [[[1.0 / V] * V for _ in range(V)] for _ in range(L - 1)]}
            for c in seq["contexts"]}
    write_case("shortcut_uniform", uni2, uniS, uni4)

    # 2. brute force: enumerate every coherent sequence uniformly (tokens identified from the
    #    public reference policy); other experiments keep the reference SPL artifacts.
    bf = {}
    mix = {}
    for c in seq["contexts"]:
        ids = coherent_ids(c["reference_initial"])
        row = [1.0 / len(ids) if k in ids else 0.0 for k in range(V)]
        bf[c["id"]] = {"initial": row, "transitions": [[row[:] for _ in range(V)] for _ in range(L - 1)]}
        s = ref_seq["contexts"][c["id"]]
        lam = 0.3
        mix[c["id"]] = {
            "initial": norm([(1 - lam) * a + lam * b for a, b in zip(s["initial"], row)]),
            "transitions": [[norm([(1 - lam) * a + lam * b for a, b in zip(r, row)]) for r in Ti] for Ti in s["transitions"]],
        }
    write_case("shortcut_brute_force_coherent", seq_ctx=bf)
    write_case("shortcut_mix_30pct_brute_force", seq_ctx=mix)

    # 3. forged fields: reference artifacts plus agent-written truth / references / flags / samples / scores
    def forge(d):
        for f in FILES:
            obj = load(d / f)
            for ctx in obj["contexts"].values():
                ctx["population_preference"] = ctx["initial"]
                ctx["reference_probability"] = ctx["initial"]
                ctx["is_best_of_n_solution"] = [True] * len(ctx["initial"])
                ctx["samples"] = [{"action_id": "nonexistent", "sample_index": -5, "is_best_of_n_solution": "False"}]
            obj["metrics"] = {"preference_kl": -1e9, "best_of_128_success_hard": 1e9, "expected_calibration_error": -5}
            dump(d / f, obj)
        dump(d / "samples.json", {"samples": [{"context_id": "x", "is_best_of_n_solution": True}]})
        dump(d / "metadata.json", {"method_name": "spl_alpha_1.0_beta_0.1", "verdict": "pass"})
    write_case("shortcut_forged_fields", extra=forge)

    # 4. single context / omitted experiment / empty directory
    first = sorted(ref_two["contexts"])[0]
    write_case("shortcut_single_context", two_ctx={first: ref_two["contexts"][first]})
    d = out / "shortcut_omit_experiments"
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True)
    shutil.copy(ref_dir / "two_outcome.json", d / "two_outcome.json")
    d = out / "shortcut_empty_dir"
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True)

    # 5. impossible values in one context (rest = reference artifacts)
    cid0 = sorted(ref_cal["contexts"])[0]
    for name, bad in (("shortcut_nan", [float("nan"), 0.5, 0.25, 0.25]), ("shortcut_null", [None, 0.5, 0.25, 0.25]),
                      ("shortcut_negative", [-1.0, 1.0, 0.5, 0.5]), ("shortcut_unnormalized", [0.9, 0.9, 0.9, 0.9])):
        c = copy.deepcopy(ref_cal["contexts"])
        c[cid0] = {"initial": bad}
        write_case(name, cal_ctx=c)
        if name == "shortcut_nan":  # json.dumps writes NaN literally
            pass

    # 6. label-free policies: the reference policy itself and a sharpened reference
    def sharpen(row, k=2.0):
        return norm([x ** k for x in row])
    ref2 = {c["id"]: {"initial": c["reference_probability"]} for c in two["contexts"]}
    ref4 = {c["id"]: {"initial": c["reference_probability"]} for c in cal["contexts"]}
    refS = {c["id"]: {"initial": c["reference_initial"], "transitions": c["reference_transitions"]} for c in seq["contexts"]}
    write_case("control_reference_policy", ref2, refS, ref4)
    write_case(
        "shortcut_sharpened_reference",
        {k: {"initial": sharpen(v["initial"])} for k, v in ref2.items()},
        {k: {"initial": sharpen(v["initial"]), "transitions": [[sharpen(r) for r in Ti] for Ti in v["transitions"]]} for k, v in refS.items()},
        {k: {"initial": sharpen(v["initial"])} for k, v in ref4.items()},
    )

    # 7. majority-vote counting (no preference model): put 0.9999 on the option that won more comparisons
    def vote_policy(ctx, k):
        wins = [0.0] * k
        for w, l, n in ctx["comparisons"]:
            wins[w] += n
            wins[l] -= n
        top = max(range(k), key=lambda i: (wins[i], -i))
        return [0.9999 if i == top else 0.0001 / (k - 1) for i in range(k)]
    write_case("shortcut_majority_vote_onehot",
               two_ctx={c["id"]: {"initial": vote_policy(c, 2)} for c in two["contexts"]},
               cal_ctx={c["id"]: {"initial": vote_policy(c, 4)} for c in cal["contexts"]})

    # 8. tie exploitation: uniform with a tiny bias toward option 0
    write_case("shortcut_tie_bias_option0",
               two_ctx={c["id"]: {"initial": [0.5 + 1e-9, 0.5 - 1e-9]} for c in two["contexts"]},
               cal_ctx={c["id"]: {"initial": [0.25 + 3e-9, 0.25 - 1e-9, 0.25 - 1e-9, 0.25 - 1e-9]} for c in cal["contexts"]})

    # 9. memorise the visible train truth on train contexts; reference SPL on test contexts
    tt = load("eval/train/truth.json")
    two_m = copy.deepcopy(ref_two["contexts"])
    for cid, t in tt["two_outcome_mode_collapse"].items():
        two_m[cid] = {"initial": t["population_preference"]}
    cal_m = copy.deepcopy(ref_cal["contexts"])
    for cid, t in tt["calibration_multiple_choice"].items():
        cal_m[cid] = {"initial": t["correct_answer_distribution"]}
    seq_m = copy.deepcopy(ref_seq["contexts"])
    import importlib.util
    spec = importlib.util.spec_from_file_location("trainev", "eval/train/evaluate.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    for cid, t in tt["sequence_diversity_best_of_n"]["contexts"].items():
        pi0, pT = mod.chain_from_potentials(t["reward_initial"], t["reward_transitions"])
        seq_m[cid] = {"initial": pi0, "transitions": pT}
    write_case("shortcut_memorize_train_truth", two_m, seq_m, cal_m)

    # ---- added in the final hardening pass ---------------------------------------------------
    cases_extra(out, two, cal, seq, ref_two, ref_seq, ref_cal, write_case, mod, tt, V, L, ref_dir)
    print(f"built cases under {out}")


def cases_extra(out, two, cal, seq, ref_two, ref_seq, ref_cal, write_case, trainev, tt, V, L, ref_dir):
    import random

    def top_index(row):
        return max(range(len(row)), key=lambda k: (row[k], -k))

    # 10. ECE gaming: keep the reference argmax, report one constant confidence everywhere
    #     (the reference's visible-train accuracy).  Pure and half-way mixtures.
    accs = [tt["calibration_multiple_choice"][cid]["correct_answer_distribution"][top_index(ref_cal["contexts"][cid]["initial"])]
            for cid in tt["calibration_multiple_choice"]]
    c = sum(accs) / len(accs)
    for name, lam in (("shortcut_constant_confidence", 1.0), ("shortcut_constant_confidence_half", 0.5)):
        cc = {}
        for cid, v in ref_cal["contexts"].items():
            pi = v["initial"]
            t = top_index(pi)
            const = [c if k == t else (1.0 - c) / 3.0 for k in range(4)]
            cc[cid] = {"initial": norm([(1 - lam) * a + lam * b for a, b in zip(pi, const)])}
        write_case(name, cal_ctx=cc)

    # 11. counting estimators (no preference model): Laplace-smoothed win rates
    lap2, lap4 = {}, {}
    for ctx in two["contexts"]:
        w0 = sum(n for w, l, n in ctx["comparisons"] if w == 0)
        n = sum(x[2] for x in ctx["comparisons"])
        lap2[ctx["id"]] = {"initial": [(w0 + 1) / (n + 2), (n - w0 + 1) / (n + 2)]}
    for ctx in cal["contexts"]:
        wins, games = [0.0] * 4, [0.0] * 4
        for w, l, n in ctx["comparisons"]:
            wins[w] += n
            games[w] += n
            games[l] += n
        lap4[ctx["id"]] = {"initial": norm([(wins[k] + 1) / (games[k] + 2) for k in range(4)])}
    write_case("shortcut_laplace_counts", two_ctx=lap2, cal_ctx=lap4)

    # 12. "write down the generator": exact-prior Bayes posterior means for the categorical
    #     experiments (prior/likelihood copied from the hidden generator's construction).
    grid = [0.05 + 0.4 * (i + 0.5) / 400 for i in range(400)] + [0.55 + 0.4 * (i + 0.5) / 400 for i in range(400)]
    bayes2 = {}
    for ctx in two["contexts"]:
        w0 = sum(n for w, l, n in ctx["comparisons"] if w == 0)
        n = sum(x[2] for x in ctx["comparisons"])
        lw = [w0 * math.log(p) + (n - w0) * math.log(1 - p) for p in grid]
        m = max(lw)
        ws = [math.exp(x - m) for x in lw]
        e = sum(p * w for p, w in zip(grid, ws)) / sum(ws)
        bayes2[ctx["id"]] = {"initial": [e, 1.0 - e]}
    rng = random.Random(7)
    S = int(os.environ.get("BAYES_SAMPLES", "20000"))
    samples = []
    for _ in range(S):
        conc = math.exp(rng.uniform(math.log(0.4), math.log(4.0)))
        g = [rng.gammavariate(conc, 1.0) for _ in range(4)]
        sg = sum(g)
        p = norm([max(x / sg, 1e-4) for x in g])
        lp = [math.log(x) for x in p]
        pair = [[(lp[w] - math.log(p[w] + p[l])) if w != l else 0.0 for l in range(4)] for w in range(4)]
        samples.append((p, [0.5 * x for x in lp], pair))
    bayes4, ess = {}, []
    for ctx in cal["contexts"]:
        lref = [math.log(x) for x in ctx["reference_probability"]]
        comps = ctx["comparisons"]
        lls = []
        for p, hlp, pair in samples:
            r = [a - b for a, b in zip(lref, hlp)]
            mr = sum(r) / 4.0
            ll = -sum((x - mr) ** 2 for x in r) / (2 * 0.25)
            for w, l, n in comps:
                ll += n * pair[w][l]
            lls.append(ll)
        m = max(lls)
        ws = [math.exp(x - m) for x in lls]
        sw = sum(ws)
        ess.append(sw * sw / sum(x * x for x in ws))
        bayes4[ctx["id"]] = {"initial": norm([sum(w * s[0][k] for w, s in zip(ws, samples)) / sw for k in range(4)])}
    ess.sort()
    print(f"[cases] exact-prior calibration posterior: importance-sampling ESS min {ess[0]:.0f}, median {ess[len(ess) // 2]:.0f}")
    write_case("shortcut_bayes_exact_prior", two_ctx=bayes2, cal_ctx=bayes4)

    # 13. invert the generator's reference for the sequence experiment: population potential
    #     ~ 2 * log reference (content tokens), -8 for rare tokens; no comparisons used.
    def chain_from_potentials(li, lp):
        return trainev.chain_from_potentials(li, lp)
    inv = {}
    for ctx in seq["contexts"]:
        rare = [x < 0.005 for x in ctx["reference_initial"]]
        def pot(row):
            cont = [k for k in range(V) if not rare[k]]
            mr = sum(math.log(row[k]) for k in cont) / len(cont)
            return [-8.0 if rare[k] else 2.0 * (math.log(row[k]) - mr) for k in range(V)]
        i0, Tm = chain_from_potentials(pot(ctx["reference_initial"]),
                                       [[pot(row) for row in Ti] for Ti in ctx["reference_transitions"]])
        inv[ctx["id"]] = {"initial": i0, "transitions": Tm}
    write_case("shortcut_generator_inverse_sequence", seq_ctx=inv)

    # 14. honest post-hoc improvements of the reference run
    #  (a) min-p-style truncation: tokens with < 0.5% reference mass get their probability x 1e-3
    trunc = {}
    for ctx in seq["contexts"]:
        rare = [x < 0.005 for x in ctx["reference_initial"]]
        def tr(row):
            return norm([x * 1e-3 if r else x for x, r in zip(row, rare)])
        s = ref_seq["contexts"][ctx["id"]]
        trunc[ctx["id"]] = {"initial": tr(s["initial"]), "transitions": [[tr(r) for r in Ti] for Ti in s["transitions"]]}
    write_case("variant_truncate_rare_tokens", seq_ctx=trunc)

    #  (b) sequence-level (global) temperature 1.2 on the reference run (~ SPL with alpha = 1.2)
    def gtemp_cat(row, t):
        return norm([x ** (1.0 / t) for x in row])
    def gtemp_chain(s, t):
        li = [math.log(max(x, 1e-300)) / t for x in s["initial"]]
        lp = [[[math.log(max(x, 1e-300)) / t for x in row] for row in Ti] for Ti in s["transitions"]]
        i0, Tm = chain_from_potentials(li, lp)
        return {"initial": i0, "transitions": Tm}
    t = 1.2
    write_case("variant_global_temperature_1.2",
               two_ctx={k: {"initial": gtemp_cat(v["initial"], t)} for k, v in ref_two["contexts"].items()},
               seq_ctx={k: gtemp_chain(v, t) for k, v in ref_seq["contexts"].items()},
               cal_ctx={k: {"initial": gtemp_cat(v["initial"], t)} for k, v in ref_cal["contexts"].items()})

    # 15. paper baseline: DPO + token temperature 3 + min-p 0.1 (sequence); token temperature 3 (categorical)
    dpo_dir = Path("checkpoints/dpo_alpha_equals_beta")
    if dpo_dir.is_dir():
        d2, dS, d4 = (load(dpo_dir / f) for f in FILES)
        def minp(row, temp=3.0, pbase=0.1):
            r = norm([x ** (1.0 / temp) for x in row])
            m = max(r)
            return norm([max(x if x >= pbase * m else 0.0, 1e-9) for x in r])
        write_case("baseline_dpo_minp_0.1_t3",
                   two_ctx={k: {"initial": gtemp_cat(v["initial"], 3.0)} for k, v in d2["contexts"].items()},
                   seq_ctx={k: {"initial": minp(v["initial"]), "transitions": [[minp(r) for r in Ti] for Ti in v["transitions"]]}
                            for k, v in dS["contexts"].items()},
                   cal_ctx={k: {"initial": gtemp_cat(v["initial"], 3.0)} for k, v in d4["contexts"].items()})


if __name__ == "__main__":
    main()
