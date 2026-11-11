# Hardening report (iteration 11)

**Verdict.** Better results now score higher and worse results lower, by margins well outside the
paired test noise. Every major shortcut in `REDTEAM.md` now either fails the run (worst value in
every metric) or ties the unchanged reference. The reference method (SPL, alpha = 1, beta = 0.1)
still beats DPO and DPO + token temperature on every experiment.

One "write down the generator" shortcut still pays. It is listed under "Shortcuts left" with its
score.

Evidence:
- Audit job `fe21a55a-560` ([log](jobs/fe21a55a-560/stdout.log)).
- Cohort ranking: [summary.md](redteam/hardening/summary.md) and
  [summary.json](redteam/hardening/summary.json).
- The same cohort under the old coefficients:
  [summary_old_coefficients.md](redteam/hardening/summary_old_coefficients.md).
- Case builders: [build_cases.py](redteam/hardening/build_cases.py) and
  [run_hardening.sh](redteam/hardening/run_hardening.sh).

The work came in two passes within this iteration:
1. **Earlier session (commit `30417a3`).**
   - Rebuilt the benchmark: hidden truth in `eval/test/`, a sequence-level best-of-N experiment, and
     noisy finite comparisons.
   - Rebuilt the evaluator: it reads only probability tables, scores every method directory found on
     disk, enforces task gates, and gives failed runs the worst values.
   - Ran the first audit job.
2. **This pass.** Fixed what that audit still showed (calibration scoring), added shortcut cases,
   and verified everything in one job.

## How gains are computed

`gain = Σ_exp weight_exp · Σ_metric |coef| · dir · (value − value_ref) / σ_metric`

- σ is the population standard deviation of the metric over the 13 non-failed honest runs
  (`honest_cohort` in `summary.json`). Failed runs are excluded from σ, because their worst values
  would inflate it.
- Gains computed over all non-failed runs, shortcuts included, are in the same file (`all σ`).
- The 95% intervals come from a paired bootstrap over test prompts: the method and the reference
  are resampled on the same prompts, with σ held fixed.
- Training and evaluation are deterministic. The job retrained the reference and both baselines,
  and the policy tables matched the packaged ones exactly (`[run] … identical … True`). Seed noise
  is therefore zero; the only noise is which test prompts were drawn, and the paired intervals
  measure that.

## Do better results score higher? (variants, before → after)

**Before:** the original gym, as reported in `REDTEAM.md`.

| Variant | Before (REDTEAM) | After: test gain [95% paired CI] | After: train gain |
|---|---:|---:|---:|
| Genuine improvement (REDTEAM: reduced reference bias) | +0.0008, while uniform output got **+0.429** | — | — |
| `variant_improved` (label smoothing + shared token bias) | — | **+0.160** [+0.138, +0.183] | +0.147 |
| `variant_truncate_rare_tokens` (min-p-style filter on the reference run) | — | **+0.155** [+0.149, +0.162] | +0.150 |
| `variant_alpha_1.2` (SPL, global temperature 12) | — | **+0.045** [+0.021, +0.070] | +0.039 |
| `variant_beta_0.01` | tie (REDTEAM: train-selected settings) | +0.003 [−0.012, +0.019] (tie) | −0.027 |
| `variant_alpha_1.5` | — | −0.051 [−0.109, −0.001] | −0.064 |
| `variant_undertrained` (300 / 150 steps) | −0.090 (10 steps) | **−0.107** [−0.174, −0.048] | −0.057 |
| `variant_alpha_0.5` | — | −0.890 | −0.848 |
| `variant_global_temperature_1.2` (post-hoc on the reference) | — | −0.224 | −0.214 |
| DPO + min-p 0.1 + token temperature 3 (paper baseline) | — | −2.418 | −1.989 |
| DPO + token temperature 1.5 / 2.0 | — | −3.254 / −3.239 | −2.961 / −2.914 |
| DPO (alpha = beta = 0.1) | −4.774 | **−4.534** | −4.356 |

The calibration fix also matters for the variants. Before it (the earlier session's audit), the
undertrained variant scored −0.010 with an interval spanning zero. And `variant_improved` scored
+0.163 on test but only +0.013 on train. Both artefacts came from accuracy noise (next section).
After the fix, train and test gains agree for every honest variant.

## Major shortcuts: fix and score before → after

| Shortcut (REDTEAM, or new in this pass) | Before | Fix | After |
|---|---:|---|---:|
| Uniform policy | **+0.429** | Task gates: majority on top in ≥ 75% of two-outcome prompts; sequence reward ≥ reference policy's; multiple-choice accuracy ≥ vote floor. A gate failure fails the whole run. | failed run, −26.0 |
| Brute force (cover every action) | **+0.429** | Sequence experiment over 21^7 ≈ 1.8e9 outputs, scored on hidden verifier sets | coherent-uniform enumeration: failed (gate); 30% mix: −0.123 |
| Copy public answers | +0.014 | Answers and generator live only in `eval/test/`; `data/` holds noisy comparisons | memorise visible train truth: test **0.000**, train +0.340 |
| Forge labels / truth / reference / success flags / samples / metric fields | +0.636 / +0.256 / +0.00001 / 0 (paying tie) | The evaluator reads only `initial` / `transitions` tables. Truth, reference policies, verifiers and thresholds come from `eval/`. | 0.000 (equals the unchanged reference artifacts it carries) |
| Single context; omit experiments from a manifest; extra inferior method | +0.153; 0 (paying tie) | Every test prompt is required. Methods are discovered from `checkpoints/` with no manifest. A missing experiment fails the run. | failed run, −26.0 |
| NaN / null / negative / unnormalised / empty | no scores written (rank skipped them) | Recorded as failed runs with the worst value of every metric in every experiment | −26.0 each |
| Score injection | already failed | Unchanged: the file is overwritten on every run | overwritten (`score_injection_check.txt`) |
| Near-constant scored metric (REDTEAM: reference cross-entropy, σ ≈ 1e-16) | ±0.05 to ±0.9 from round-off | Removed earlier. In this pass, calibration `accuracy` (σ ≈ 0.0012 across honest runs, so near-tie argmax flips dominated) now has coefficient 0, is still reported, and is enforced by a stricter evaluator-owned gate: accuracy ≥ comparison-vote accuracy − 0.01 (floor 0.4717; reference 0.4797). | Laplace-count estimator (a simple baseline): **+0.400 → +0.008** [−0.019, +0.034] |
| **New:** ECE gaming with constant confidence (reference argmax, one confidence everywhere) | **+0.045** under the old coefficients (ECE 1.0, Brier 0.75, accuracy 0.5) | Brier (a proper scoring rule) made primary at coefficient 1.0; ECE 0.5 | **−0.028** |
| **New:** invert the generator's reference (population ≈ reference², rare tokens −8) | — | None needed: per-row normalisation of the reference hides the potentials | −1.157 |

Gates and thresholds are constants in `eval/` or fields of `eval/*/truth.json`. The accuracy floor
comes from per-prompt vote winners. The generator now writes these into the truth files without
consuming randomness, so `data/` is byte-identical: the job regenerated it and confirmed this.

## Shortcuts left, and why

- **Writing down the generator for the categorical experiments (`shortcut_bayes_exact_prior`):
  +0.099 [+0.078, +0.117].** This is the posterior mean under the exact prior and likelihood copied
  from the hidden generator. The sequence part is the unchanged reference.
  - On two-outcome it gains only +0.018, the same as plain Laplace counting (KL 0.0177 vs 0.0180).
  - On calibration it gains +0.081: Brier 0.6296 vs 0.6448 (oracle 0.6107), ECE 0.009. It exploits
    how the synthetic reference is built (roughly the square root of the answer distribution, times
    noise).
  - It ranks below the best honest variants (+0.160, +0.155).
  - It requires either the hidden generator code or the visible train truth, and
    `scoring/CONSTRAINTS.md` forbids both as training signals, so the post-hoc grader can flag it.
  - The real fix is to draw calibration targets from real survey or answer distributions. I did not
    do that this iteration: it means new data, retraining, and a new audit.
- **Partial constant-confidence shrinkage (`shortcut_constant_confidence_half`): +0.014
  [−0.000, +0.030].** It leaves Brier unchanged (0.6450 vs 0.6448) and improves ECE. It is a crude
  but real recalibration of an over-confident policy, and is marginal at this test size.
- **Coherence can be pushed above the population's own level.** Truncating rare tokens reaches
  0.9999; the hidden population policy itself is at 0.9961. Truncation also lowers population KL
  and raises best-of-128, so I treat it as an honest min-p-style improvement (+0.155), not a
  loophole. Agents will likely find it.
- **The train evaluator checks only train prompts.** A NaN in a test-prompt table passes train
  scoring. The hidden test fails it.

## Reference score, before and after

The rank gain is 0 by definition. The raw reference numbers changed with the gym.

**Before (original gym, REDTEAM):**
- Tabular KL 0.000216, ECE 0.0087.
- Analytic best-of-128 0.99999, empirical best-of-128 1.
- Accuracy 1. Several metrics were at their bounds.

**After (`scoring/scores_test.json`, hidden test, regenerated from the reference runs only):**

| Experiment | spl_alpha_1.0_beta_0.1 | DPO | DPO + token temperature 1.5 |
|---|---|---|---|
| two_outcome | KL 0.0378, minority log-ratio error 0.513 | 2.002 / 11.45 | 1.186 / 7.362 |
| sequence | best-of-128 (hard) 0.386, population KL 0.249, coherence 0.970 | 0.087 / 7.712 / 0.995 | 0.172 / 3.674 / 0.927 |
| calibration | Brier 0.645, ECE 0.068, accuracy 0.4797 | 0.943 / 0.449 / 0.4821 | 0.898 / 0.414 / 0.4821 |

The paper's analytic 80/20 example is the visible train prompt `two_outcome_paper_80_20`:

| | Gym | Paper |
|---|---|---|
| DPO minority probability | 9.5e-7 | 1e-6 |
| DPO majority overweight | 1.25 | 1.25 |
| SPL minority probability | 0.200 | 0.2 |

## Headroom check

**No scored reference metric is at its bound.**

| Metric (test) | Reference | Bound / oracle | Best run found |
|---|---:|---:|---:|
| preference_kl | 0.0378 | 0 | 0.0177 |
| minority_log_ratio_error | 0.513 | 0 | 0.266 |
| best_of_128_success_hard | 0.386 | 1 (population policy: 0.417) | 0.394 |
| population_kl | 0.249 | 0 | 0.216 |
| coherence | 0.970 | 1 (population policy: 0.996) | 0.9999 |
| brier_score | 0.645 | 0.611 (true answer distribution) | 0.630 |
| expected_calibration_error | 0.068 | 0 | 0.009 |

- **Variants above the reference:** three honest variants score above it, by more than the noise.
  No single simple variant reaches a bound.
- **Brute force:** uniform enumeration of coherent sequences fails the reward gate, and mixing 30%
  of it in loses. Brute force does not outrank the reference.
- **Raising temperature doesn't pay:** higher sequence-level temperature on the reference policy
  loses best-of-128, because hard problems need coherent, population-plausible sequences.
- **Simple baselines are below the reference:** DPO, token temperature, DPO + min-p, majority vote,
  and the untrained reference policy (which fails the gate).
- **No new experiment was added in this pass.** The sequence best-of-N experiment, added earlier in
  this iteration, is the "where the paper's method falls short" experiment. Its hard problems sit
  away from the population mode, and the reference keeps the pre-alignment probability on
  incoherent tokens that are rarely seen in the comparisons.

## Weight changes

- **Experiment weights unchanged:** 0.25 two-outcome, 0.5 sequence, 0.25 calibration. The sequence
  experiment carries the paper's main empirical claims (diversity vs quality, hard best-of-N).
- **Calibration coefficients changed:**

| Metric | Before | After |
|---|---:|---:|
| Brier | 0.75 | 1.0 (primary) |
| ECE | 1.0 (primary) | 0.5 |
| accuracy | 0.5 | 0 (gate-enforced constraint) |

  - The paper reports both Brier and ECE.
  - ECE alone is minimised by uninformative constant confidence (the shortcut above); Brier is not.
  - Accuracy is the paper's "without sacrificing accuracy" constraint. It is now a hard gate at
    DPO-level accuracy (the vote floor is 0.4817; DPO scores 0.4821), rather than a ranked metric
    that only measured near-tie flips.

## What I could not fix

- **Synthetic data.** The gym is a synthetic, scaled-down substitute: tabular chain policies
  trained on generated comparisons. It has no Mistral-7B, HH-RLHF, UltraFeedback, GSM8K/MATH or
  MMLU/TruthfulQA. The paper's LLM numbers are kept unchanged under `_paper_reported` in
  `scoring/reference.json`.
- **Generator knowledge still pays on calibration** (+0.099, above).
- **Methods see the comparisons for test prompts**, because each prompt has its own policy table.
  The test measures estimation of hidden population or answer distributions from noisy data, not
  generalisation to unseen prompts.
- **The reference's settings differ from the paper's best-of-N runs.** The reference uses alpha = 1,
  beta = 0.1 (global temperature 10). That setting is inside the paper's sweep (1 to 11) and gives
  the proportional representation of Proposition 3.2. The paper's best-of-N math runs used global
  temperatures of 1.1 to 1.2.
