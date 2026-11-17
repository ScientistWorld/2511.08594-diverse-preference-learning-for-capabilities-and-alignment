# Overview

- **Paper ID:** 2511.08594
- **Title:** Diverse Preference Learning for Capabilities and Alignment
- **Domain:** Language model alignment / preference learning
- **TL;DR:** Soft Preference Learning decouples entropy from reference-policy regularisation so
  aligned models keep useful output diversity.

## Short Summary

KL-regularised preference learning (RLHF, DPO) both anchors the policy to a reference model and
sharpens it, so a split population becomes a near-deterministic majority answer. SPL separates the
entropy and cross-entropy terms, giving a sequence-level temperature that restores proportional
representation, keeps generations coherent better than token-level temperature, improves hard
best-of-N problem solving and improves calibration.

## This Gym

A scaled finite reproduction with hidden ground truth, empirical survey categorical targets and synthetic sequences: two-outcome social-choice prompts,
7-token sequence generation over a 21-token vocabulary with hidden verifier sets and incoherent
tokens, and four-option multiple-choice calibration. All training is pure-Python, CPU-only, and
deterministic. See `scoring/EXPERIMENTS.md` and `briefing/evaluation.md`.

---

# Problem Definition

## Research Question

Preference-tuned language models tend to collapse onto a few high-probability responses: a
population that is split 80/20 between two answers ends up with a model that says the majority
answer ~100% of the time, generations lose variety, repeated sampling stops finding minority
solution strategies, and answer probabilities become overconfident. The problem is to **learn a
policy from pairwise human preferences that represents the diversity of the population's
preferences while keeping outputs coherent and correct.**

## Setting

Every prompt comes with (a) a fixed *reference policy* (the pre-alignment model) and (b) a finite
set of *pairwise comparisons* "y preferred to y'" collected from a heterogeneous population of
raters. Nothing else about the population is observed. Three families of prompts are provided
(all in `data/`, each split into a visible `train` part and a held-out `test` part whose answers
are hidden):

1. **Two-outcome prompts** (601 prompts). Two possible completions; the population prefers one
   of them according to an empirical human response distribution; 32-128 sampled comparisons per prompt. The visible 80/20 example has 100 exact comparisons.
2. **Sequence prompts** (64 prompts). Completions are 7-token sequences over a 21-token
   vocabulary (about 1.8 billion possible completions, far too many to enumerate). Most tokens at
   each position are rare "incoherent" tokens (the analogue of non-words). Each prompt has 2000
   comparisons between pairs of reference-policy samples, judged by raters with
   sequence-level preferences.
3. **Multiple-choice prompts** (1200 prompts). Four survey response options with a hidden human response distribution; 40-120 sampled comparisons per prompt. Calibration measures probabilities of a randomly drawn human response, not factual correctness.

Categorical targets are empirical response frequencies from GlobalOpinionQA (Pew Global Attitudes
and World Values Survey), conditioned on a subset of two or four positive-response options.
Original questions are disjoint across both experiments and both splits. Question text, country,
source row and option mapping are withheld; only the preference observations are inputs.
Reference probabilities are independent of the empirical targets. The sequence task remains
synthetic. This is a finite preference-learning benchmark, not a reproduction of full LLM tasks.

A solution outputs, for every prompt (train and test), a full probability distribution over its
completions (a first-order chain over tokens for the sequence prompts).

## Why It Matters

Language models are now information sources for many people. If alignment converts a split
population into a near-deterministic answer, minority viewpoints disappear, repeated-sampling
problem solving wastes its budget on duplicates, and confidence stops reflecting uncertainty.

## Success Criteria

A better solution, on held-out prompts:

- represents the population's split instead of amplifying the majority (two-outcome and
  sequence-level distribution fit),
- solves more hard problems when 128 samples are drawn (it keeps probability on less popular but
  valid solutions),
- produces calibrated answer probabilities,

**without** giving up what the alignment step is for: the population's preferred completion must
stay on top, generations must stay coherent and at least as preferred by the population as the
reference policy's, and multiple-choice accuracy must not drop below what simply counting which
option wins the most comparisons achieves (with evaluator-defined slack).

---

# Paper's Method

## Key Contribution

Soft Preference Learning (SPL) decouples the two roles of the KL regulariser in RLHF/DPO: the
cross-entropy to the reference policy (prior strength, coefficient `beta`) and the entropy of the
learned policy (diversity, coefficient `alpha`).

## Approach

DPO-style SPL objective (paper Eq. 5), maximised over the policy:

`E_{(y_w > y_l)} [ log sigmoid( alpha * log pi(y_w)/pi(y_l) - beta * log pi_ref(y_w)/pi_ref(y_l) ) ]`

Its optimum is `pi(y) ∝ pi_ref(y)^(beta/alpha) * exp(r(y)/alpha)` (Prop. A.2); standard DPO is
`alpha = beta`, whose optimum `pi_ref * p^(1/beta)` raises population preferences to the 10th power
at `beta = 0.1` (Prop. 3.1). `alpha = 1` gives proportional representation (Cor. 3.1);
`alpha/beta` acts as a sequence-level (global) temperature on top of the DPO policy, unlike
token-level temperature, which rescales every next-token distribution.

## This gym's implementation (`method/spl.py`)

- Policy per prompt: tabular first-order chain (softmax logits per row), initialised at the
  reference policy; the length-1 case for two-outcome and multiple-choice prompts.
- Full-batch gradient descent on the mean SPL loss over the comparisons: 3000 steps at lr 1.0
  (categorical prompts), 1500 steps at lr 5.0 (sequence prompts). Deterministic.
- Reference run: `alpha = 1.0`, `beta = 0.1` (global temperature 10). DPO baseline:
  `alpha = beta = 0.1` (`baseline/run_dpo.py`); token-temperature baseline:
  `baseline/token_temperature.py`.
- Optional variant knobs (off in the reference): `--label-smoothing`,
  `--label-smoothing-sequence`, `--shared-token-bias`.

## Main Claims

SPL avoids majority mode collapse, keeps generation diversity with less quality loss than
token-level temperature, improves best-of-N accuracy on hard problems, and improves calibration
relative to DPO.

---

# Evaluation

## Protocol

A method writes one directory per run under `checkpoints/<run_name>/` containing three files:

| file | content |
|---|---|
| `two_outcome.json` | `{"contexts": {<id>: {"initial": [p0, p1]}}}` |
| `calibration.json` | `{"contexts": {<id>: {"initial": [p0, p1, p2, p3]}}}` |
| `sequence.json` | `{"contexts": {<id>: {"initial": [21 probs], "transitions": [6 x 21 x 21 row-stochastic]}}}` |

`initial` is the distribution of the first token, `transitions[i][a]` the distribution of token
`i+1` given token `i` = `a`. Every prompt id in `data/*_contexts.json` (train and test split) must be
present. Only these probability tables are read; extra fields inside a context are rejected. Top-level descriptive metadata and unrelated files never enter a score. Ground truth
(population preferences, rewards, verifier sets, human response distributions), the reference
policies used for gates, and all thresholds are owned by the evaluator.

- `bash scripts/evaluate_train.sh` scores every `checkpoints/*` directory on the visible train
  prompts and writes `scoring/scores_train.json` (same code path as `scripts/evaluate.sh` ->
  `scoring/scores.json`).
- `bash scripts/evaluate_test.sh` (no arguments) scores every `checkpoints/*` directory on the
  hidden test prompts and writes `scoring/scores_test.json`. The hidden test only reads the saved
  artifacts; it never trains.

All numbers are computed exactly from the probability tables (dynamic programming for
sequences); there is no sampling noise and the evaluation is deterministic.

## Metrics (scored)

**two_outcome_mode_collapse**
- `preference_kl` (lower): mean KL(population split || policy), capped at 20 per prompt.
- `minority_log_ratio_error` (lower): mean |log(policy minority mass / population minority
  share)|, capped at 30 per prompt.

**sequence_diversity_best_of_n**
- `best_of_128_success_hard` (higher, primary): probability that 128 samples contain a solution,
  averaged over 6 hidden hard problems per prompt (coherent sequences matching a hidden pattern,
  away from the most popular completion).
- `population_kl` (lower): sequence-level KL(population distribution || policy), cap 60.
- `coherence` (higher): probability that a sampled sequence contains no incoherent token, capped per prompt at the hidden population policy's own coherence. Increasing coherence beyond that level receives no extra credit. Raw and population coherence are diagnostic only.

**calibration_multiple_choice** (expectations over a hidden randomly drawn human response)
- `brier_score` (lower, primary): multi-class Brier score.
- `expected_calibration_error` (lower): 10-bin ECE of top-answer confidence (weighted below Brier:
  a constant, uninformative confidence can make ECE small but not Brier).
- `accuracy` (higher): accuracy of the most probable option (ties -> lowest index). Reported and
  enforced by the gate below; it carries no rank weight (coefficient 0), because methods that pass
  the gate differ in it only by a handful of near-tie decisions.

## Task gates and failures

A run **fails** if, in any experiment, an artifact is missing, malformed, not a probability
distribution (non-finite, negative, > 1, rows not summing to 1 within 1e-4), misses a prompt, or
violates a task gate:

- two-outcome: the population-majority completion is the most probable one on < 75% of prompts;
- sequence: expected population reward of the policy below that of the reference policy;
- multiple choice: expected agreement with a human response below that of simply picking the option with the best
  (wins - losses) count in the prompt's comparisons, minus 0.01. Floors are recomputed from evaluator-owned truth for each split and are reported in diagnostics.

Hidden-named method directories are also scored. An empty checkpoint tree produces a failed `missing_submission` record.

A failed run receives the **worst value of every metric in every experiment** (categorical KL 20,
minority log-ratio error 30, sequence KL 60,
success 0, coherence 0, ECE 1, Brier 2, accuracy 0); the reason is recorded under
`failed_methods`. Gate quantities are reported under `diagnostics`.

## Baselines

- `dpo_alpha_equals_beta`: standard KL-regularised preference learning (DPO, beta = 0.1).
- `dpo_token_temperature_1.5`: the same policy with token-level temperature 1.5 applied to every
  conditional distribution.

Both stay coherent or nearly so but collapse onto the majority, so they lose on distribution fit,
hard best-of-128 and calibration. Higher token temperatures recover some diversity but quickly
produce incoherent tokens. A uniform policy fails the gates.

Rank contributions use experiment weight times the magnitude of each coefficient times the
direction-correct improvement over the reference, divided by the metric's standard deviation
across the attempts being ranked. Coefficient signs and measurement units do not change priority.
Every failed attempt has every metric, so no task failure can buy a higher rank through another
metric. Metrics, operating points and test sizes are fixed by the evaluator.

---

# Reproduction Log

### Iteration 1: gpt-5.5
- **Milestone**: `none` | **Status**: error
- **Working time**: 15m | **GPU**: 0.0h

<details>
<summary>Progress Log</summary>

<!-- Write your progress here -->

</details>

### Iteration 1: gpt-5.5
- **Milestone**: `none` | **Status**: error
- **Working time**: 30m | **GPU**: 0.0h

### Iteration 1: gpt-5.5
- **Milestone**: `none` | **Status**: error
- **Working time**: 30m | **GPU**: 0.0h

### Iteration 1: gpt-5.5
- **Milestone**: `core_claim` | **Status**: done
- **Working time**: 12m | **GPU**: 0.0h
- **Jobs**: 1 total (1 completed, 0 failed)

<details>
<summary>Progress Log</summary>

### [2026-06-01 08:32] - none
- Read the paper markdown and identified Soft Preference Learning as a DPO-style objective with separate `alpha` and `beta` terms.
- Identified the core analytic claim: standard KL-regularized preference learning overweights majority preferences, while SPL can recover proportional representation.

### [2026-06-01 08:32] - method_runs
- Implemented tabular SPL training using the paper's pairwise objective.
- Implemented DPO as the `alpha == beta` special case and wrote reusable policy artifacts.

### [2026-06-01 08:32] - core_claim
- Ran the paper's 80/20 two-outcome example locally.
- DPO assigned the minority option approximately `9.54e-7` probability, matching the paper's mode-collapse prediction.
- SPL with `alpha=1.0` and `beta=0.1` assigned the minority option `0.20` probability while preserving the majority option as top choice.

### [2026-06-01 08:37] - core_claim
- Continuation audit verified that the implementation uses the paper's SPL pairwise objective, with DPO represented as the `alpha == beta` special case.
- Re-ran `scripts/reproduce.sh` and `python validate.py --compare`; regenerated scores match the two-outcome reference and pass separation checks.

### [2026-06-01 08:39] - core_claim
- CPU smoke job `e57fca5a-e7d` completed successfully in the packaged container.
- The job regenerated DPO and SPL outputs, wrote `scoring/scores.json`, and passed `python validate.py --compare`.

</details>

### Iteration 2: gpt-5.5
- **Milestone**: `core_claim_plus` | **Status**: done
- **Working time**: 8m | **GPU**: 0.0h
- **Jobs**: 1 total (1 completed, 0 failed)

<details>
<summary>Progress Log</summary>

### [2026-06-01 08:39] - method_runs
- Packaged the paper's DPO-style Soft Preference Learning objective in a finite tabular setting.
- Verified that the baseline and proposed method execute end-to-end and write evaluable artifacts.

### [2026-06-01 08:43] - core_claim
- Reproduced the paper's central two-outcome social-choice example: DPO with `beta=0.1` collapses minority support near zero, while SPL with `alpha=1.0, beta=0.1` restores the 0.2 minority preference probability.
- Scored both benefit and constraint metrics: minority probability, majority overweight, and top-choice preservation.
- Confirmed `python validate.py --compare` passes with import separation and evaluation reading saved artifacts.

### [2026-06-01 08:55] - core_claim_plus
- Added a separate multi-outcome finite Plackett-Luce benchmark following the paper's generalized Appendix A.1 setting.
- Reproduced the same SPL-vs-DPO pattern beyond the two-outcome example: SPL sharply reduces preference KL and improves best-of-128 success while preserving the population mode.
- Updated `scripts/reproduce.sh` and `scoring/scores.json` so both experiments are produced in one end-to-end run.

</details>

### Iteration 3: gpt-5.5
- **Milestone**: `core_claim_plus` | **Status**: done
- **Working time**: 4m | **GPU**: 0.0h
- **Jobs**: 2 total (2 completed, 0 failed)

<details>
<summary>Progress Log</summary>

### [2026-06-01 08:52] - core_claim_plus
- Managed CPU verification job `9b0d9378-ef7` completed with exit code 0.
- The job reran `scripts/reproduce.sh`, regenerated DPO and SPL artifacts for both experiments, and ran `python validate.py --compare`.
- Container validation passed import separation, artifact-based evaluation, and reproduced-score comparison checks.

### [2026-06-01 08:51] - core_claim_plus
- Continuation audit confirmed `method/train_tabular_spl.py` directly optimizes the paper's DPO-style SPL objective rather than a surrogate.
- Confirmed `eval/evaluate_tabular.py` reads saved `policy.json` and `samples.json` artifacts only and does not import from `method/`.
- Local `python validate.py --compare` passes for both the two-outcome core claim and the multi-outcome tabular extension.
- Submitting a fresh CPU container verification because the only managed job log on disk predates the current two-experiment package.

### [2026-06-01 08:49] - core_claim_plus
- Implemented and packaged the paper's actual DPO-style Soft Preference Learning objective for finite policies.
- Reproduced the paper's two-outcome 80/20 mode-collapse example with DPO as the `alpha == beta` baseline and SPL with separate `alpha` and `beta`.
- Added and scored a finite multi-outcome Plackett-Luce extension that tests distributional representation, repeated-sampling coverage, and preference-order preservation.
- Verified `eval/` reads saved artifacts and does not import `method/`.

</details>

### Iteration 4: gpt-5.5 (designer_stage_1)
- **Milestone**: `core_claim_plus` | **Status**: done
- **Working time**: 7m | **GPU**: 0.0h
- **Jobs**: 2 total (2 completed, 0 failed)

<details>
<summary>Progress Log</summary>

### [2026-06-01 08:52] - core_claim_plus
- Managed CPU verification job `9b0d9378-ef7` completed with exit code 0.
- The job reran `scripts/reproduce.sh`, regenerated DPO and SPL artifacts for both experiments, and ran `python validate.py --compare`.
- Container validation passed import separation, artifact-based evaluation, and reproduced-score comparison checks.

### [2026-06-01 08:51] - core_claim_plus
- Continuation audit confirmed `method/train_tabular_spl.py` directly optimizes the paper's DPO-style SPL objective rather than a surrogate.
- Confirmed `eval/evaluate_tabular.py` reads saved `policy.json` and `samples.json` artifacts only and does not import from `method/`.
- Local `python validate.py --compare` passes for both the two-outcome core claim and the multi-outcome tabular extension.
- Submitting a fresh CPU container verification because the only managed job log on disk predates the current two-experiment package.

### [2026-06-01 08:49] - core_claim_plus
- Implemented and packaged the paper's actual DPO-style Soft Preference Learning objective for finite policies.
- Reproduced the paper's two-outcome 80/20 mode-collapse example with DPO as the `alpha == beta` baseline and SPL with separate `alpha` and `beta`.
- Added and scored a finite multi-outcome Plackett-Luce extension that tests distributional representation, repeated-sampling coverage, and preference-order preservation.
- Verified `eval/` reads saved artifacts and does not import `method/`.

</details>

### Iteration 5: gpt-5.5 (designer_stage_2)
- **Milestone**: `core_claim_plus` | **Status**: done
- **Working time**: 3m | **GPU**: 0.0h
- **Jobs**: 2 total (2 completed, 0 failed)

<details>
<summary>Progress Log</summary>

### [2026-06-01 08:52] - core_claim_plus
- Managed CPU verification job `9b0d9378-ef7` completed with exit code 0.
- The job reran `scripts/reproduce.sh`, regenerated DPO and SPL artifacts for both experiments, and ran `python validate.py --compare`.
- Container validation passed import separation, artifact-based evaluation, and reproduced-score comparison checks.

### [2026-06-01 08:51] - core_claim_plus
- Continuation audit confirmed `method/train_tabular_spl.py` directly optimizes the paper's DPO-style SPL objective rather than a surrogate.
- Confirmed `eval/evaluate_tabular.py` reads saved `policy.json` and `samples.json` artifacts only and does not import from `method/`.
- Local `python validate.py --compare` passes for both the two-outcome core claim and the multi-outcome tabular extension.
- Submitting a fresh CPU container verification because the only managed job log on disk predates the current two-experiment package.

### [2026-06-01 08:49] - core_claim_plus
- Implemented and packaged the paper's actual DPO-style Soft Preference Learning objective for finite policies.
- Reproduced the paper's two-outcome 80/20 mode-collapse example with DPO as the `alpha == beta` baseline and SPL with separate `alpha` and `beta`.
- Added and scored a finite multi-outcome Plackett-Luce extension that tests distributional representation, repeated-sampling coverage, and preference-order preservation.
- Verified `eval/` reads saved artifacts and does not import `method/`.

</details>

### Iteration 6: gpt-5.5 (designer_stage_3)
- **Milestone**: `core_claim_plus` | **Status**: done
- **Working time**: 4m | **GPU**: 0.0h
- **Jobs**: 4 total (4 completed, 0 failed)

<details>
<summary>Progress Log</summary>

### [2026-06-01 09:10] - core_claim_plus
- Added a deterministic multi-context two-outcome split benchmark that keeps the paper's original 80/20 example as the first reported context while adding real held-out two-action contexts for anti-shortcut train/test scoring.
- Updated the standard evaluator to report the same two-outcome distribution-fit constraints used by the split evaluators.
- Submitting a full CPU reproduction run that regenerates DPO/SPL artifacts, refreshes `scores.json`, refreshes train/test scores, and runs validation.

### [2026-06-01 08:52] - core_claim_plus
- Managed CPU verification job `9b0d9378-ef7` completed with exit code 0.
- The job reran `scripts/reproduce.sh`, regenerated DPO and SPL artifacts for both experiments, and ran `python validate.py --compare`.
- Container validation passed import separation, artifact-based evaluation, and reproduced-score comparison checks.

### [2026-06-01 08:51] - core_claim_plus
- Continuation audit confirmed `method/train_tabular_spl.py` directly optimizes the paper's DPO-style SPL objective rather than a surrogate.
- Confirmed `eval/evaluate_tabular.py` reads saved `policy.json` and `samples.json` artifacts only and does not import from `method/`.
- Local `python validate.py --compare` passes for both the two-outcome core claim and the multi-outcome tabular extension.
- Submitting a fresh CPU container verification because the only managed job log on disk predates the current two-experiment package.

### [2026-06-01 08:49] - core_claim_plus
- Implemented and packaged the paper's actual DPO-style Soft Preference Learning objective for finite policies.
- Reproduced the paper's two-outcome 80/20 mode-collapse example with DPO as the `alpha == beta` baseline and SPL with separate `alpha` and `beta`.
- Added and scored a finite multi-outcome Plackett-Luce extension that tests distributional representation, repeated-sampling coverage, and preference-order preservation.
- Verified `eval/` reads saved artifacts and does not import `method/`.

</details>

### Iteration 7: gpt-5.5 (packager)
- **Milestone**: `core_claim_plus` | **Status**: done
- **Working time**: 2m | **GPU**: 0.0h
- **Jobs**: 4 total (4 completed, 0 failed)

<details>
<summary>Progress Log</summary>

### [2026-06-01 09:10] - core_claim_plus
- Added a deterministic multi-context two-outcome split benchmark that keeps the paper's original 80/20 example as the first reported context while adding real held-out two-action contexts for anti-shortcut train/test scoring.
- Updated the standard evaluator to report the same two-outcome distribution-fit constraints used by the split evaluators.
- Submitting a full CPU reproduction run that regenerates DPO/SPL artifacts, refreshes `scores.json`, refreshes train/test scores, and runs validation.

### [2026-06-01 08:52] - core_claim_plus
- Managed CPU verification job `9b0d9378-ef7` completed with exit code 0.
- The job reran `scripts/reproduce.sh`, regenerated DPO and SPL artifacts for both experiments, and ran `python validate.py --compare`.
- Container validation passed import separation, artifact-based evaluation, and reproduced-score comparison checks.

### [2026-06-01 08:51] - core_claim_plus
- Continuation audit confirmed `method/train_tabular_spl.py` directly optimizes the paper's DPO-style SPL objective rather than a surrogate.
- Confirmed `eval/evaluate_tabular.py` reads saved `policy.json` and `samples.json` artifacts only and does not import from `method/`.
- Local `python validate.py --compare` passes for both the two-outcome core claim and the multi-outcome tabular extension.
- Submitting a fresh CPU container verification because the only managed job log on disk predates the current two-experiment package.

### [2026-06-01 08:49] - core_claim_plus
- Implemented and packaged the paper's actual DPO-style Soft Preference Learning objective for finite policies.
- Reproduced the paper's two-outcome 80/20 mode-collapse example with DPO as the `alpha == beta` baseline and SPL with separate `alpha` and `beta`.
- Added and scored a finite multi-outcome Plackett-Luce extension that tests distributional representation, repeated-sampling coverage, and preference-order preservation.
- Verified `eval/` reads saved artifacts and does not import `method/`.

</details>

### Iteration 8: gpt-5.5 (verifier)
- **Milestone**: `core_claim_plus` | **Status**: done
- **Working time**: 7m | **GPU**: 0.0h
- **Jobs**: 4 total (4 completed, 0 failed)

<details>
<summary>Progress Log</summary>

### [2026-06-01 09:25] - core_claim_plus
- Final verification passed `python validate.py --compare`.
- Cleaned method-agnostic documentation and added notes explaining which reference targets are generated finite-benchmark targets rather than published LLM table values.
- Kept milestone at `core_claim_plus` because the verified package is a finite-action reproduction rather than the paper's full LLM experiment suite.

### [2026-06-01 09:23] - core_claim_plus
- Added a finite multiple-choice calibration benchmark that uses the same pairwise SPL/DPO objective and artifact format.
- Local `scripts/reproduce.sh`, split evaluators, and `python validate.py --compare` pass for the expanded three-experiment package.
- Submitting a managed CPU verification job before claiming `secondary_claims`.

### [2026-06-01 09:10] - core_claim_plus
- Added a deterministic multi-context two-outcome split benchmark that keeps the paper's original 80/20 example as the first reported context while adding real held-out two-action contexts for anti-shortcut train/test scoring.
- Updated the standard evaluator to report the same two-outcome distribution-fit constraints used by the split evaluators.
- Submitting a full CPU reproduction run that regenerates DPO/SPL artifacts, refreshes `scores.json`, refreshes train/test scores, and runs validation.

### [2026-06-01 08:52] - core_claim_plus
- Managed CPU verification job `9b0d9378-ef7` completed with exit code 0.
- The job reran `scripts/reproduce.sh`, regenerated DPO and SPL artifacts for both experiments, and ran `python validate.py --compare`.
- Container validation passed import separation, artifact-based evaluation, and reproduced-score comparison checks.

### [2026-06-01 08:51] - core_claim_plus
- Continuation audit confirmed `method/train_tabular_spl.py` directly optimizes the paper's DPO-style SPL objective rather than a surrogate.
- Confirmed `eval/evaluate_tabular.py` reads saved `policy.json` and `samples.json` artifacts only and does not import from `method/`.
- Local `python validate.py --compare` passes for both the two-outcome core claim and the multi-outcome tabular extension.
- Submitting a fresh CPU container verification because the only managed job log on disk predates the current two-experiment package.

### [2026-06-01 08:49] - core_claim_plus
- Implemented and packaged the paper's actual DPO-style Soft Preference Learning objective for finite policies.
- Reproduced the paper's two-outcome 80/20 mode-collapse example with DPO as the `alpha == beta` baseline and SPL with separate `alpha` and `beta`.
- Added and scored a finite multi-outcome Plackett-Luce extension that tests distributional representation, repeated-sampling coverage, and preference-order preservation.
- Verified `eval/` reads saved artifacts and does not import `method/`.

</details>

### Iteration 9: gpt-5.5 (verifier)
- **Milestone**: `core_claim_plus` | **Status**: done
- **Working time**: 23m | **GPU**: 0.0h
- **Jobs**: 4 total (4 completed, 0 failed)

<details>
<summary>Progress Log</summary>

### [2026-06-01 09:35] - core_claim_plus
- Final verifier reran `scripts/reproduce.sh`, both split evaluators, and `python validate.py --compare`; all checks passed.
- Renamed the problem-facing multi-outcome benchmark files to method-agnostic names and removed stale paper-method wording from `data/`, `eval/`, and standard scripts.
- Confirmed no `shared/` references, no `eval/` or `data/` imports from `method/`, and no external download URLs requiring verification.

### [2026-06-01 09:25] - core_claim_plus
- Final verification passed `python validate.py --compare`.
- Cleaned method-agnostic documentation and added notes explaining which reference targets are generated finite-benchmark targets rather than published LLM table values.
- Kept milestone at `core_claim_plus` because the verified package is a finite-action reproduction rather than the paper's full LLM experiment suite.

### [2026-06-01 09:23] - core_claim_plus
- Added a finite multiple-choice calibration benchmark that uses the same pairwise SPL/DPO objective and artifact format.
- Local `scripts/reproduce.sh`, split evaluators, and `python validate.py --compare` pass for the expanded three-experiment package.
- Submitting a managed CPU verification job before claiming `secondary_claims`.

### [2026-06-01 09:10] - core_claim_plus
- Added a deterministic multi-context two-outcome split benchmark that keeps the paper's original 80/20 example as the first reported context while adding real held-out two-action contexts for anti-shortcut train/test scoring.
- Updated the standard evaluator to report the same two-outcome distribution-fit constraints used by the split evaluators.
- Submitting a full CPU reproduction run that regenerates DPO/SPL artifacts, refreshes `scores.json`, refreshes train/test scores, and runs validation.

### [2026-06-01 08:52] - core_claim_plus
- Managed CPU verification job `9b0d9378-ef7` completed with exit code 0.
- The job reran `scripts/reproduce.sh`, regenerated DPO and SPL artifacts for both experiments, and ran `python validate.py --compare`.
- Container validation passed import separation, artifact-based evaluation, and reproduced-score comparison checks.

### [2026-06-01 08:51] - core_claim_plus
- Continuation audit confirmed `method/train_tabular_spl.py` directly optimizes the paper's DPO-style SPL objective rather than a surrogate.
- Confirmed `eval/evaluate_tabular.py` reads saved `policy.json` and `samples.json` artifacts only and does not import from `method/`.
- Local `python validate.py --compare` passes for both the two-outcome core claim and the multi-outcome tabular extension.
- Submitting a fresh CPU container verification because the only managed job log on disk predates the current two-experiment package.

### [2026-06-01 08:49] - core_claim_plus
- Implemented and packaged the paper's actual DPO-style Soft Preference Learning objective for finite policies.
- Reproduced the paper's two-outcome 80/20 mode-collapse example with DPO as the `alpha == beta` baseline and SPL with separate `alpha` and `beta`.
- Added and scored a finite multi-outcome Plackett-Luce extension that tests distributional representation, repeated-sampling coverage, and preference-order preservation.
- Verified `eval/` reads saved artifacts and does not import `method/`.

</details>

### Iteration 10: gpt-6.1-sol (redteam)
- **Milestone**: `core_claim_plus` | **Status**: done
- **Working time**: 8m | **GPU**: 0.0h
- **Jobs**: 4 total (4 completed, 0 failed)

<details>
<summary>Progress Log</summary>

### [2026-06-01 09:35] - core_claim_plus
- Final verifier reran `scripts/reproduce.sh`, both split evaluators, and `python validate.py --compare`; all checks passed.
- Renamed the problem-facing multi-outcome benchmark files to method-agnostic names and removed stale paper-method wording from `data/`, `eval/`, and standard scripts.
- Confirmed no `shared/` references, no `eval/` or `data/` imports from `method/`, and no external download URLs requiring verification.

### [2026-06-01 09:25] - core_claim_plus
- Final verification passed `python validate.py --compare`.
- Cleaned method-agnostic documentation and added notes explaining which reference targets are generated finite-benchmark targets rather than published LLM table values.
- Kept milestone at `core_claim_plus` because the verified package is a finite-action reproduction rather than the paper's full LLM experiment suite.

### [2026-06-01 09:23] - core_claim_plus
- Added a finite multiple-choice calibration benchmark that uses the same pairwise SPL/DPO objective and artifact format.
- Local `scripts/reproduce.sh`, split evaluators, and `python validate.py --compare` pass for the expanded three-experiment package.
- Submitting a managed CPU verification job before claiming `secondary_claims`.

### [2026-06-01 09:10] - core_claim_plus
- Added a deterministic multi-context two-outcome split benchmark that keeps the paper's original 80/20 example as the first reported context while adding real held-out two-action contexts for anti-shortcut train/test scoring.
- Updated the standard evaluator to report the same two-outcome distribution-fit constraints used by the split evaluators.
- Submitting a full CPU reproduction run that regenerates DPO/SPL artifacts, refreshes `scores.json`, refreshes train/test scores, and runs validation.

### [2026-06-01 08:52] - core_claim_plus
- Managed CPU verification job `9b0d9378-ef7` completed with exit code 0.
- The job reran `scripts/reproduce.sh`, regenerated DPO and SPL artifacts for both experiments, and ran `python validate.py --compare`.
- Container validation passed import separation, artifact-based evaluation, and reproduced-score comparison checks.

### [2026-06-01 08:51] - core_claim_plus
- Continuation audit confirmed `method/train_tabular_spl.py` directly optimizes the paper's DPO-style SPL objective rather than a surrogate.
- Confirmed `eval/evaluate_tabular.py` reads saved `policy.json` and `samples.json` artifacts only and does not import from `method/`.
- Local `python validate.py --compare` passes for both the two-outcome core claim and the multi-outcome tabular extension.
- Submitting a fresh CPU container verification because the only managed job log on disk predates the current two-experiment package.

### [2026-06-01 08:49] - core_claim_plus
- Implemented and packaged the paper's actual DPO-style Soft Preference Learning objective for finite policies.
- Reproduced the paper's two-outcome 80/20 mode-collapse example with DPO as the `alpha == beta` baseline and SPL with separate `alpha` and `beta`.
- Added and scored a finite multi-outcome Plackett-Luce extension that tests distributional representation, repeated-sampling coverage, and preference-order preservation.
- Verified `eval/` reads saved artifacts and does not import `method/`.

</details>

### Iteration 11: claude-opus-5-5 (hardener)
- **Milestone**: `core_claim_plus` | **Status**: error
- **Working time**: 38m | **GPU**: 0.0h
- **Jobs**: 5 total (5 completed, 0 failed)

<details>
<summary>Progress Log</summary>

### [2026-06-01 09:35] - core_claim_plus
- Final verifier reran `scripts/reproduce.sh`, both split evaluators, and `python validate.py --compare`; all checks passed.
- Renamed the problem-facing multi-outcome benchmark files to method-agnostic names and removed stale paper-method wording from `data/`, `eval/`, and standard scripts.
- Confirmed no `shared/` references, no `eval/` or `data/` imports from `method/`, and no external download URLs requiring verification.

### [2026-06-01 09:25] - core_claim_plus
- Final verification passed `python validate.py --compare`.
- Cleaned method-agnostic documentation and added notes explaining which reference targets are generated finite-benchmark targets rather than published LLM table values.
- Kept milestone at `core_claim_plus` because the verified package is a finite-action reproduction rather than the paper's full LLM experiment suite.

### [2026-06-01 09:23] - core_claim_plus
- Added a finite multiple-choice calibration benchmark that uses the same pairwise SPL/DPO objective and artifact format.
- Local `scripts/reproduce.sh`, split evaluators, and `python validate.py --compare` pass for the expanded three-experiment package.
- Submitting a managed CPU verification job before claiming `secondary_claims`.

### [2026-06-01 09:10] - core_claim_plus
- Added a deterministic multi-context two-outcome split benchmark that keeps the paper's original 80/20 example as the first reported context while adding real held-out two-action contexts for anti-shortcut train/test scoring.
- Updated the standard evaluator to report the same two-outcome distribution-fit constraints used by the split evaluators.
- Submitting a full CPU reproduction run that regenerates DPO/SPL artifacts, refreshes `scores.json`, refreshes train/test scores, and runs validation.

### [2026-06-01 08:52] - core_claim_plus
- Managed CPU verification job `9b0d9378-ef7` completed with exit code 0.
- The job reran `scripts/reproduce.sh`, regenerated DPO and SPL artifacts for both experiments, and ran `python validate.py --compare`.
- Container validation passed import separation, artifact-based evaluation, and reproduced-score comparison checks.

### [2026-06-01 08:51] - core_claim_plus
- Continuation audit confirmed `method/train_tabular_spl.py` directly optimizes the paper's DPO-style SPL objective rather than a surrogate.
- Confirmed `eval/evaluate_tabular.py` reads saved `policy.json` and `samples.json` artifacts only and does not import from `method/`.
- Local `python validate.py --compare` passes for both the two-outcome core claim and the multi-outcome tabular extension.
- Submitting a fresh CPU container verification because the only managed job log on disk predates the current two-experiment package.

### [2026-06-01 08:49] - core_claim_plus
- Implemented and packaged the paper's actual DPO-style Soft Preference Learning objective for finite policies.
- Reproduced the paper's two-outcome 80/20 mode-collapse example with DPO as the `alpha == beta` baseline and SPL with separate `alpha` and `beta`.
- Added and scored a finite multi-outcome Plackett-Luce extension that tests distributional representation, repeated-sampling coverage, and preference-order preservation.
- Verified `eval/` reads saved artifacts and does not import `method/`.

</details>

### Iteration 11: claude-opus-5-5 (hardener)
- **Milestone**: `core_claim_plus` | **Status**: done
- **Working time**: 14m | **GPU**: 0.0h
- **Jobs**: 6 total (6 completed, 0 failed)

<details>
<summary>Progress Log</summary>

### [2026-11-11 08:55] - core_claim_plus (hardening, final)
- Hidden-test gym hardened (see HARDENING.md): evaluator-owned truth/reference/verifiers, sequence-level best-of-N experiment, task gates with worst-value failures, disk-discovered methods.
- Calibration: accuracy is a gate-enforced constraint (comparison-vote floor - 0.01), Brier primary (1.0), ECE 0.5; closes accuracy-noise amplification and constant-confidence ECE gaming.
- Job fe21a55a-560: retrained reference/baselines bit-identical; better variants +0.160/+0.155/+0.045 over the reference, weaker variants and baselines below; validate.py passes.

### [2026-06-01 09:35] - core_claim_plus
- Final verifier reran `scripts/reproduce.sh`, both split evaluators, and `python validate.py --compare`; all checks passed.
- Renamed the problem-facing multi-outcome benchmark files to method-agnostic names and removed stale paper-method wording from `data/`, `eval/`, and standard scripts.
- Confirmed no `shared/` references, no `eval/` or `data/` imports from `method/`, and no external download URLs requiring verification.

### [2026-06-01 09:25] - core_claim_plus
- Final verification passed `python validate.py --compare`.
- Cleaned method-agnostic documentation and added notes explaining which reference targets are generated finite-benchmark targets rather than published LLM table values.
- Kept milestone at `core_claim_plus` because the verified package is a finite-action reproduction rather than the paper's full LLM experiment suite.

### [2026-06-01 09:23] - core_claim_plus
- Added a finite multiple-choice calibration benchmark that uses the same pairwise SPL/DPO objective and artifact format.
- Local `scripts/reproduce.sh`, split evaluators, and `python validate.py --compare` pass for the expanded three-experiment package.
- Submitting a managed CPU verification job before claiming `secondary_claims`.

### [2026-06-01 09:10] - core_claim_plus
- Added a deterministic multi-context two-outcome split benchmark that keeps the paper's original 80/20 example as the first reported context while adding real held-out two-action contexts for anti-shortcut train/test scoring.
- Updated the standard evaluator to report the same two-outcome distribution-fit constraints used by the split evaluators.
- Submitting a full CPU reproduction run that regenerates DPO/SPL artifacts, refreshes `scores.json`, refreshes train/test scores, and runs validation.

### [2026-06-01 08:52] - core_claim_plus
- Managed CPU verification job `9b0d9378-ef7` completed with exit code 0.
- The job reran `scripts/reproduce.sh`, regenerated DPO and SPL artifacts for both experiments, and ran `python validate.py --compare`.
- Container validation passed import separation, artifact-based evaluation, and reproduced-score comparison checks.

### [2026-06-01 08:51] - core_claim_plus
- Continuation audit confirmed `method/train_tabular_spl.py` directly optimizes the paper's DPO-style SPL objective rather than a surrogate.
- Confirmed `eval/evaluate_tabular.py` reads saved `policy.json` and `samples.json` artifacts only and does not import from `method/`.
- Local `python validate.py --compare` passes for both the two-outcome core claim and the multi-outcome tabular extension.
- Submitting a fresh CPU container verification because the only managed job log on disk predates the current two-experiment package.

### [2026-06-01 08:49] - core_claim_plus
- Implemented and packaged the paper's actual DPO-style Soft Preference Learning objective for finite policies.
- Reproduced the paper's two-outcome 80/20 mode-collapse example with DPO as the `alpha == beta` baseline and SPL with separate `alpha` and `beta`.
- Added and scored a finite multi-outcome Plackett-Luce extension that tests distributional representation, repeated-sampling coverage, and preference-order preservation.
- Verified `eval/` reads saved artifacts and does not import `method/`.

</details>

### Iteration 11: gpt-6.1-sol (hardener)
- **Milestone**: `core_claim_plus` | **Status**: done
- **Working time**: 14m | **GPU**: 0.0h
- **Jobs**: 7 total (7 completed, 0 failed)

<details>
<summary>Progress Log</summary>

### [2026-11-11 08:55] - core_claim_plus (hardening, final)
- Hidden-test gym hardened (see HARDENING.md): evaluator-owned truth/reference/verifiers, sequence-level best-of-N experiment, task gates with worst-value failures, disk-discovered methods.
- Calibration: accuracy is a gate-enforced constraint (comparison-vote floor - 0.01), Brier primary (1.0), ECE 0.5; closes accuracy-noise amplification and constant-confidence ECE gaming.
- Job fe21a55a-560: retrained reference/baselines bit-identical; better variants +0.160/+0.155/+0.045 over the reference, weaker variants and baselines below; validate.py passes.

### [2026-06-01 09:35] - core_claim_plus
- Final verifier reran `scripts/reproduce.sh`, both split evaluators, and `python validate.py --compare`; all checks passed.
- Renamed the problem-facing multi-outcome benchmark files to method-agnostic names and removed stale paper-method wording from `data/`, `eval/`, and standard scripts.
- Confirmed no `shared/` references, no `eval/` or `data/` imports from `method/`, and no external download URLs requiring verification.

### [2026-06-01 09:25] - core_claim_plus
- Final verification passed `python validate.py --compare`.
- Cleaned method-agnostic documentation and added notes explaining which reference targets are generated finite-benchmark targets rather than published LLM table values.
- Kept milestone at `core_claim_plus` because the verified package is a finite-action reproduction rather than the paper's full LLM experiment suite.

### [2026-06-01 09:23] - core_claim_plus
- Added a finite multiple-choice calibration benchmark that uses the same pairwise SPL/DPO objective and artifact format.
- Local `scripts/reproduce.sh`, split evaluators, and `python validate.py --compare` pass for the expanded three-experiment package.
- Submitting a managed CPU verification job before claiming `secondary_claims`.

### [2026-06-01 09:10] - core_claim_plus
- Added a deterministic multi-context two-outcome split benchmark that keeps the paper's original 80/20 example as the first reported context while adding real held-out two-action contexts for anti-shortcut train/test scoring.
- Updated the standard evaluator to report the same two-outcome distribution-fit constraints used by the split evaluators.
- Submitting a full CPU reproduction run that regenerates DPO/SPL artifacts, refreshes `scores.json`, refreshes train/test scores, and runs validation.

### [2026-06-01 08:52] - core_claim_plus
- Managed CPU verification job `9b0d9378-ef7` completed with exit code 0.
- The job reran `scripts/reproduce.sh`, regenerated DPO and SPL artifacts for both experiments, and ran `python validate.py --compare`.
- Container validation passed import separation, artifact-based evaluation, and reproduced-score comparison checks.

### [2026-06-01 08:51] - core_claim_plus
- Continuation audit confirmed `method/train_tabular_spl.py` directly optimizes the paper's DPO-style SPL objective rather than a surrogate.
- Confirmed `eval/evaluate_tabular.py` reads saved `policy.json` and `samples.json` artifacts only and does not import from `method/`.
- Local `python validate.py --compare` passes for both the two-outcome core claim and the multi-outcome tabular extension.
- Submitting a fresh CPU container verification because the only managed job log on disk predates the current two-experiment package.

### [2026-06-01 08:49] - core_claim_plus
- Implemented and packaged the paper's actual DPO-style Soft Preference Learning objective for finite policies.
- Reproduced the paper's two-outcome 80/20 mode-collapse example with DPO as the `alpha == beta` baseline and SPL with separate `alpha` and `beta`.
- Added and scored a finite multi-outcome Plackett-Luce extension that tests distributional representation, repeated-sampling coverage, and preference-order preservation.
- Verified `eval/` reads saved artifacts and does not import `method/`.

</details>


---

# Reproduction Milestones

**Current: core_claim_plus**

## Stop Justification
- Completed at milestone `core_claim_plus`.
