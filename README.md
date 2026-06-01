# Overview

- **Paper ID:** 2511.08594
- **Title:** Diverse Preference Learning for Capabilities and Alignment
- **Domain:** Language model alignment / preference learning
- **TL;DR:** Soft Preference Learning decouples entropy from reference-policy regularization so aligned LLMs can retain useful output diversity.

## Short Summary

The paper studies why preference-tuned language models often lose diversity and become overconfident. It argues that the KL regularizer in RLHF and DPO both anchors the policy to a reference model and sharpens the policy distribution, which can amplify majority preferences and collapse alternatives. Soft Preference Learning separates these effects with independent entropy and cross-entropy coefficients, making sequence-level diversity controllable during preference learning. Empirically, the paper reports better diversity-quality tradeoffs, stronger hard best-of-N math performance, and improved calibration.

## Key Results

- In the paper's 80/20 two-outcome example with `beta=0.1`, standard preference learning assigns about 99.9999% probability to the majority option and about 0.0001% to the minority option.
- With the calibrated entropy setting analyzed by the paper, the target distribution recovers proportional representation of the 80/20 population preference.
- SPL is presented as a global, sequence-level alternative to token temperature scaling, intended to preserve coherence while increasing diversity.

---

# Problem Definition

## Research Question

Preference-tuned language models often become too concentrated on a small set of high-probability responses. The benchmark problem is to learn from heterogeneous pairwise preferences while preserving a calibrated distribution over valid alternatives instead of collapsing all probability mass onto the plurality choice.

For the minimal core-claim reproduction, the task is a finite two-outcome social-choice setting: 80% of a population prefers one completion and 20% prefers another. A successful learner should keep the majority-preferred completion as the most likely output while still assigning probability to the minority-preferred completion in proportion to its support.

## Why It Matters

Many downstream uses of aligned models require representation of minority viewpoints or alternative solution paths. If alignment turns an 80/20 population preference into a nearly deterministic 100/0 model distribution, the model no longer represents the population it was trained to serve.

## Success Criteria

A successful solution should reduce majority over-weighting, increase the probability assigned to the minority-supported option, and preserve the majority option as the top-ranked output. This captures the benefit side of diversity and the constraint that the learned distribution should not invert the population's preference ordering.

---

# Paper's Method

## Key Contribution

The paper introduces Soft Preference Learning (SPL), a preference-learning objective designed to reduce diversity loss caused by standard KL-regularized alignment methods such as RLHF and DPO.

## Approach

Standard RLHF and DPO couple two roles inside the KL penalty: keeping the trained policy close to a reference model and controlling the entropy of the trained policy. The paper argues that this coupling causes majority-preferred outputs to be overweighted, producing mode collapse and poor representation of minority preferences or alternative reasoning paths.

SPL separates these terms. Its RL-style objective optimizes expected reward plus an entropy bonus with coefficient `alpha`, while separately applying a cross-entropy penalty toward the reference policy with coefficient `beta`. The paper also gives a DPO-style pairwise objective:

`log sigmoid(alpha * log(pi(y|x) / pi(y'|x)) - beta * log(pi_ref(y|x) / pi_ref(y'|x)))`

where `y` is preferred to `y'`. Standard DPO is recovered when `alpha = beta`; increasing `alpha / beta` acts like a sequence-level, or global, temperature that flattens probabilities over complete responses rather than token-by-token.

## Main Claims

The primary claim is that SPL improves the diversity-quality tradeoff of aligned language models: it can produce more varied outputs and improve hard best-of-N problem solving without the rapid coherence degradation seen with token-level temperature scaling.

For the core mathematical reasoning result, SPL improves hard-split best-of-N accuracy on GSM8K and MATH when many samples are allowed. In Table 1, the strongest SPL setting outperforms DPO and temperature-scaled DPO at best-of-128 on both hard splits.

Secondary claims are that SPL improves semantic and lexical diversity in chat generation, better represents diverse societal viewpoints, and improves multiple-choice logit calibration while maintaining similar accuracy.

---

# Evaluation

## Metrics

- `minority_probability`: learned probability assigned to the 20% minority-supported option. Higher is better up to the population target; near-zero values indicate mode collapse.
- `majority_overweight`: learned majority-option probability divided by its 80% population support. Lower is better, with 1.0 indicating proportional representation and values above 1 indicating over-concentration.
- `top_choice_preserved`: whether the majority-supported option remains the highest-probability output. Higher is better; this is a quality/order constraint.
- `expected_calibration_error`: finite-action calibration gap between learned probability mass and population preference mass. Lower is better.
- `preference_kl` and `brier_score`: distributional fit metrics between the learned policy and population preferences. Lower is better.

## Baselines and Targets

The included baseline artifact for the 80/20 task assigns nearly all probability to the majority option and almost none to the minority option. The target behavior for a calibrated learner assigns 80% to the majority option and 20% to the minority option while preserving the majority as the top choice.

## Evaluation Protocol

Evaluate saved policy artifacts for the fixed two-outcome preference task in `data/two_outcome_benchmark.json`, the generated multi-outcome task in `data/spl_tabular_benchmark.json`, and the generated multiple-choice calibration task in `data/calibration_benchmark.json`. All methods receive the same pairwise preference counts, reference distributions, and population targets. Evaluation code reads only method output artifacts and must not import training code.

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


---

# Reproduction Milestones

**Current: core_claim_plus**

## Stop Justification
- Completed at milestone `core_claim_plus`.
