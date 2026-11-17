#!/usr/bin/env python3
"""Hidden generator for the preference-learning benchmark (gym designers only).

Writes:
  data/two_outcome_contexts.json      public inputs (reference policies + pairwise comparisons)
  data/sequence_contexts.json         public inputs
  data/calibration_contexts.json      public inputs
  eval/train/truth.json               ground truth for the visible train split
  eval/test/truth.json                ground truth for the hidden test split

The population preferences, reward potentials, verifier sets and correct-answer
distributions of the test split exist only in eval/test/.  Re-running this
script with the default seed reproduces every file byte-for-byte.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import math
import random
from pathlib import Path

SEED = 81_594_2511
TOKEN_CHARS = "0123456789abcdefghijklmnopqrstuvwxyz"
SURVEY_PATH = Path(__file__).resolve().parent / "assets" / "global_opinions.csv"
SURVEY_SHA256 = "789e99ebb0eb1d0768de565ec80fab7a25bfef5baa1cccebdda75777eea86051"

# ---------------------------------------------------------------------------
# small chain utilities (exact dynamic programming on first-order chains)
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


def mask_prob(init, T, masks):
    V = len(init)
    a = [init[v] * masks[0][v] for v in range(V)]
    for i, Ti in enumerate(T):
        a = [sum(a[u] * Ti[u][v] for u in range(V)) * masks[i + 1][v] for v in range(V)]
    return sum(a)


def sample_chain(init, T, rng):
    def draw(row):
        u = rng.random()
        c = 0.0
        for k, x in enumerate(row):
            c += x
            if u <= c:
                return k
        return len(row) - 1

    y = [draw(init)]
    for Ti in T:
        y.append(draw(Ti[y[-1]]))
    return y


def sig6(x: float) -> float:
    return float(f"{x:.6g}")


def normalize6(row):
    """Round to 6 significant digits and renormalise so the stored row sums to 1."""
    r = [sig6(x) for x in row]
    s = sum(r)
    r = [x / s for x in r]
    return [float(f"{x:.9g}") for x in r]


# ---------------------------------------------------------------------------
# experiment 1: two-outcome contexts
# ---------------------------------------------------------------------------


def legacy_two_outcome(rng, n_train, n_test):
    """Advance the original RNG identically, preserving the sequence benchmark.

    These synthetic categorical targets are discarded. The scored categorical
    targets below come from empirical survey responses instead.
    """
    public, truth = [], {"train": {}, "test": {}}

    def make(cid, split, p_maj, ref0, n, majority_index, exact=False):
        p = [0.0, 0.0]
        p[majority_index] = p_maj
        p[1 - majority_index] = 1.0 - p_maj
        p = [round(x, 6) for x in p]
        if exact:
            w0 = int(round(n * p[0]))
        else:
            w0 = sum(1 for _ in range(n) if rng.random() < p[0])
        comps = []
        if w0 > 0:
            comps.append([0, 1, w0])
        if n - w0 > 0:
            comps.append([1, 0, n - w0])
        public.append(
            {
                "id": cid,
                "split": split,
                "num_actions": 2,
                "reference_probability": [round(ref0, 6), round(1.0 - ref0, 6)],
                "comparisons": comps,
            }
        )
        truth[split][cid] = {"population_preference": p, "reference_probability": [round(ref0, 6), round(1.0 - ref0, 6)]}

    # the paper's canonical 80/20 example (uniform reference, 100 comparisons)
    make("two_outcome_paper_80_20", "train", 0.8, 0.5, 100, 0, exact=True)
    for split, count in (("train", n_train), ("test", n_test)):
        for k in range(count):
            p_maj = rng.uniform(0.55, 0.95)
            ref0 = rng.uniform(0.25, 0.75)
            n = rng.randint(8, 40)
            make(f"two_outcome_{split}_{k:03d}", split, p_maj, ref0, n, rng.randrange(2))
    return public, truth


# ---------------------------------------------------------------------------
# experiment 2: sequence-level contexts (first-order chains over a vocabulary)
# ---------------------------------------------------------------------------

SEQ_CONTENT = 5
SEQ_GARBAGE = 16
SEQ_LEN = 7
SEQ_PAIRS = 2000
SEQ_GARBAGE_PENALTY = 8.0
SEQ_REF_GARBAGE_MASS = 0.01
SEQ_HARD_PROBLEMS = 6


def gen_sequence_context(rng, cid, split):
    VC, VG, L = SEQ_CONTENT, SEQ_GARBAGE, SEQ_LEN
    V = VC + VG
    # latent (unpermuted) token roles: 0..VC-1 content, VC.. garbage
    th0 = [rng.gauss(0.0, 1.0) for _ in range(VC)] + [-SEQ_GARBAGE_PENALTY] * VG
    th = [
        [
            [
                (rng.gauss(0.0, 1.0) if (a < VC and b < VC) else (-SEQ_GARBAGE_PENALTY if b >= VC else 0.0))
                for b in range(V)
            ]
            for a in range(V)
        ]
        for _ in range(L - 1)
    ]

    def ref_row(logits):
        c = [math.exp(x) for x in logits]
        s = sum(c)
        c = [x / s * (1.0 - SEQ_REF_GARBAGE_MASS) for x in c]
        g = [rng.gammavariate(0.5, 1.0) + 1e-3 for _ in range(VG)]
        sg = sum(g)
        return c + [x / sg * SEQ_REF_GARBAGE_MASS for x in g]

    r0 = ref_row([0.5 * th0[v] + rng.gauss(0.0, 0.5) for v in range(VC)])
    rT = [[ref_row([0.5 * th[i][a][b] + rng.gauss(0.0, 0.5) for b in range(VC)]) for a in range(V)] for i in range(L - 1)]

    # random relabelling of token ids (roles are not revealed by ids)
    perm = list(range(V))
    rng.shuffle(perm)  # latent token k is shown as id perm[k]
    inv = [0] * V
    for k, p in enumerate(perm):
        inv[p] = k

    def P1(vec):
        return [vec[inv[j]] for j in range(V)]

    def P2(mat):
        return [[mat[inv[a]][inv[b]] for b in range(V)] for a in range(V)]

    th0_p = P1(th0)
    th_p = [P2(m) for m in th]
    r0_p = normalize6(P1(r0))
    rT_p = [[normalize6(row) for row in P2(m)] for m in rT]
    garbage_ids = sorted(perm[k] for k in range(VC, V))
    content_ids = sorted(perm[k] for k in range(VC))

    p_init, p_T = chain_from_potentials(th0_p, th_p)

    def reward(y):
        s = th0_p[y[0]]
        for i in range(1, L):
            s += th_p[i - 1][y[i - 1]][y[i]]
        return s

    # preference data: pairs of reference samples labelled by Bradley-Terry raters
    pairs = []
    for _ in range(SEQ_PAIRS):
        y1 = sample_chain(r0_p, rT_p, rng)
        y2 = sample_chain(r0_p, rT_p, rng)
        d = reward(y1) - reward(y2)
        if rng.random() < 1.0 / (1.0 + math.exp(-d)):
            w, l = y1, y2
        else:
            w, l = y2, y1
        pairs.append(["".join(TOKEN_CHARS[t] for t in w), "".join(TOKEN_CHARS[t] for t in l)])

    # mode of the population distribution (Viterbi on potentials)
    best = th0_p[:]
    back = []
    for i in range(L - 1):
        nb, bp = [], []
        for b in range(V):
            k = max(range(V), key=lambda a: best[a] + th_p[i][a][b])
            nb.append(best[k] + th_p[i][k][b])
            bp.append(k)
        best = nb
        back.append(bp)
    mode = [max(range(V), key=lambda v: best[v])]
    for i in range(L - 2, -1, -1):
        mode.append(back[i][mode[-1]])
    mode = mode[::-1]

    # hard verifier problems: 4 fixed positions, coherent elsewhere, moderate population mass
    content_set = set(content_ids)
    problems = []
    tries = 0
    while len(problems) < SEQ_HARD_PROBLEMS:
        tries += 1
        if tries > 200000:
            raise RuntimeError("could not build hard problems")
        pos = sorted(rng.sample(range(L), 4))
        toks = [rng.choice(content_ids) for _ in pos]
        if all(mode[j] == t for j, t in zip(pos, toks)):
            continue
        if any(pp["positions"] == pos and pp["tokens"] == toks for pp in problems):
            continue
        masks = [[(v in content_set) for v in range(V)] for _ in range(L)]
        for j, t in zip(pos, toks):
            masks[j] = [v == t for v in range(V)]
        mass = mask_prob(p_init, p_T, masks)
        if 0.002 <= mass <= 0.02:
            problems.append({"positions": pos, "tokens": toks})

    public = {
        "id": cid,
        "split": split,
        "reference_initial": r0_p,
        "reference_transitions": rT_p,
        "pairs": pairs,
    }
    truth = {
        "reward_initial": [round(x, 6) for x in th0_p],
        "reward_transitions": [[[round(x, 6) for x in row] for row in m] for m in th_p],
        "incoherent_tokens": garbage_ids,
        "hard_problems": problems,
        "reference_initial": r0_p,
        "reference_transitions": rT_p,
    }
    return public, truth


def gen_sequence(rng, n_train, n_test):
    public, truth = [], {"train": {}, "test": {}}
    for split, count in (("train", n_train), ("test", n_test)):
        for k in range(count):
            cid = f"sequence_{split}_{k:03d}"
            pub, tru = gen_sequence_context(rng, cid, split)
            public.append(pub)
            truth[split][cid] = tru
    return public, truth


# ---------------------------------------------------------------------------
# empirical categorical contexts: real human response distributions
# ---------------------------------------------------------------------------


def empirical_questions():
    """Read GlobalOpinionQA without exposing question/country lookup keys.

    Keep full survey question groups together. Each context uses a real country
    distribution conditioned on a subset of positive-response options; no
    parametric prior generates its probability vector. Reference probabilities
    are independent of the survey answers.
    """
    if hashlib.sha256(SURVEY_PATH.read_bytes()).hexdigest() != SURVEY_SHA256:
        raise ValueError("GlobalOpinionQA checksum mismatch; run scripts/download.sh")
    questions = {}
    with SURVEY_PATH.open() as f:
        for row in csv.DictReader(f):
            key = (row["source"], row["question"].strip())
            s = row["selections"]
            # The upstream field is defaultdict(<class 'list'>, {...}), not JSON.
            selections = ast.literal_eval(s[s.index("{"):s.rindex("}") + 1])
            for country, p in selections.items():
                if not isinstance(p, list) or any(not math.isfinite(x) or x < 0 for x in p):
                    continue
                active = [i for i, x in enumerate(p) if x >= 0.002]
                if len(active) >= 2:
                    questions.setdefault(key, []).append((country, p, active))
    return questions


def gen_empirical_categorical(rng):
    questions = empirical_questions()
    # Disjoint original questions across experiments AND splits, including
    # duplicate source rows and countries. No train question supplies test truth.
    keys = sorted(questions)
    rng.shuffle(keys)
    four_keys = [key for key in keys if any(len(a) >= 4 for _, _, a in questions[key])]
    if len(four_keys) < 1200:
        raise ValueError("not enough distinct four-option survey questions")
    four_keys = four_keys[:1200]
    used = set(four_keys)
    two_keys = [key for key in keys if key not in used][:600]
    if len(two_keys) < 600:
        raise ValueError("not enough disjoint two-option survey questions")
    provenance = {}

    def build(selected, K, prefix):
        public, truth = [], {"train": {}, "test": {}}
        half = len(selected) // 2
        for idx, key in enumerate(selected):
            split = "train" if idx < half else "test"
            cid = f"{prefix}_{split}_{idx % half:03d}"
            rows = [r for r in questions[key] if len(r[2]) >= K]
            country, full_p, active = rng.choice(rows)
            options = rng.sample(active, K)
            p = [full_p[i] for i in options]
            sp = sum(p)
            p = [x / sp for x in p]
            # Independent, mildly non-uniform pre-alignment policy. In
            # particular, it is NOT a noisy square root of the target vector.
            ref = normalize6([math.exp(rng.gauss(0.0, 0.45)) for _ in range(K)])
            n = rng.randint(32, 128) if K == 2 else rng.randint(40, 120)
            counts = {}
            for _ in range(n):
                i = random_choice(rng, ref)
                j = i
                while j == i:
                    j = random_choice(rng, ref)
                w, l = (i, j) if rng.random() < p[i] / (p[i] + p[j]) else (j, i)
                counts[(w, l)] = counts.get((w, l), 0) + 1
            comps = [[w, l, c] for (w, l), c in sorted(counts.items())]
            public.append({"id": cid, "split": split, "num_actions": K,
                           "reference_probability": ref, "comparisons": comps})
            t = {"reference_probability": ref}
            if K == 2:
                t["population_preference"] = p
            else:
                t["correct_answer_distribution"] = p
                net = [0] * K
                for w, l, c in comps:
                    net[w] += c
                    net[l] -= c
                t["comparison_vote_winners"] = [k for k in range(K) if net[k] == max(net)]
            truth[split][cid] = t
            provenance[cid] = {"source": key[0], "question": key[1],
                               "country": country, "option_indices": options}
        return public, truth

    two_pub, two_truth = build(two_keys, 2, "two_outcome")
    # The paper's exact analytic example remains a visible prompt with extra unscored diagnostics.
    two_pub.insert(0, {"id": "two_outcome_paper_80_20", "split": "train", "num_actions": 2,
                       "reference_probability": [0.5, 0.5], "comparisons": [[0, 1, 80], [1, 0, 20]]})
    two_truth["train"]["two_outcome_paper_80_20"] = {
        "population_preference": [0.8, 0.2], "reference_probability": [0.5, 0.5]}
    cal_pub, cal_truth = build(four_keys, 4, "calibration")
    return two_pub, two_truth, cal_pub, cal_truth, provenance


def random_choice(rng, probs):
    u = rng.random()
    c = 0.0
    for k, x in enumerate(probs):
        c += x
        if u <= c:
            return k
    return len(probs) - 1


# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    root = Path(args.root)
    rng = random.Random(args.seed)
    legacy_two_outcome(rng, 300, 300)
    seq_pub, seq_truth = gen_sequence(rng, 32, 32)
    two_pub, two_truth, cal_pub, cal_truth, provenance = gen_empirical_categorical(random.Random(args.seed + 1))
    provenance_out = root / "eval" / "test" / "survey_provenance.json"
    provenance_out.parent.mkdir(parents=True, exist_ok=True)
    provenance_out.write_text(json.dumps(provenance, separators=(",", ":")) + "\n")

    data = root / "data"
    data.mkdir(parents=True, exist_ok=True)
    (data / "two_outcome_contexts.json").write_text(
        json.dumps({"name": "two_outcome_contexts", "contexts": two_pub}, separators=(",", ":")) + "\n"
    )
    (data / "sequence_contexts.json").write_text(
        json.dumps(
            {
                "name": "sequence_contexts",
                "vocab_size": SEQ_CONTENT + SEQ_GARBAGE,
                "length": SEQ_LEN,
                "token_alphabet": TOKEN_CHARS[: SEQ_CONTENT + SEQ_GARBAGE],
                "contexts": seq_pub,
            },
            separators=(",", ":"),
        )
        + "\n"
    )
    (data / "calibration_contexts.json").write_text(
        json.dumps({"name": "calibration_contexts", "contexts": cal_pub}, separators=(",", ":")) + "\n"
    )
    for split, out in (("train", root / "eval" / "train" / "truth.json"), ("test", root / "eval" / "test" / "truth.json")):
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "split": split,
            "two_outcome_mode_collapse": two_truth[split],
            "sequence_diversity_best_of_n": {
                "vocab_size": SEQ_CONTENT + SEQ_GARBAGE,
                "length": SEQ_LEN,
                "contexts": seq_truth[split],
            },
            "calibration_multiple_choice": cal_truth[split],
        }
        out.write_text(json.dumps(payload, separators=(",", ":")) + "\n")
    print("wrote benchmark inputs and truth files")


if __name__ == "__main__":
    main()
