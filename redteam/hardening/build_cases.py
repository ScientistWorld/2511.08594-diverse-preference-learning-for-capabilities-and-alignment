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
    print(f"built cases under {out}")


if __name__ == "__main__":
    main()
