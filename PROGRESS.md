# Progress

## What Works

- `scripts/reproduce.sh` runs a complete tabular reproduction of the paper's core SPL/DPO objective.
- `baseline/run_dpo.py` runs standard DPO as the `alpha == beta` special case.
- `method/train_tabular_spl.py` runs the paper's SPL DPO-style objective with separate `alpha` and `beta`.
- `eval/evaluate_tabular.py` scores saved artifacts only and does not import from `method/`.
- Continuation audit on 2026-06-01 confirmed the implementation is the paper's Eq. 5 finite-policy objective, not a surrogate model.
- Continuation audit on 2026-06-01 re-ran `scripts/reproduce.sh` and `python validate.py --compare`; both passed locally.
- CPU smoke job `e57fca5a-e7d` passed in the packaged container with exit code 0.
- Managed CPU verification job `9b0d9378-ef7` passed in the packaged container with exit code 0 and reran the full two-experiment reproduction.
- Current milestone is `core_claim_plus`; the reproduction now covers the central two-outcome claim and a multi-outcome finite Plackett-Luce extension from the paper's Appendix A.1 setting.
- Added a finite multiple-choice calibration benchmark for the paper's calibration claim. Local artifact-based reproduction runs successfully; a managed CPU verification job is being submitted for the expanded three-experiment package.

## Results

- `two_outcome_mode_collapse / dpo_alpha_equals_beta`: minority probability `9.54e-7`, majority overweight `1.25`, top choice preserved `1.0`.
- `two_outcome_mode_collapse / spl_alpha_1.0_beta_0.1`: minority probability `0.20`, majority overweight `1.0`, top choice preserved `1.0`.
- These match the paper's analytic prediction for the 80/20 example: DPO collapses minority support to near zero at `beta=0.1`, while SPL with `alpha=1` recovers proportional representation.
- Managed job validation re-created these numbers through `scripts/run.sh`, `scripts/reproduce.sh`, and `scripts/evaluate.sh`.
- `tabular_preference_learning / dpo_alpha_equals_beta`: preference KL `3.896`, normalized entropy `0.024`, best-of-128 success probability `0.151`, population-mode accuracy `1.0`.
- `tabular_preference_learning / spl_alpha_1.0_beta_0.1`: preference KL `0.000138`, normalized entropy `0.858`, best-of-128 success probability `0.999994`, population-mode accuracy `1.0`.
- The multi-outcome result shows the same mechanism beyond the paper's minimum two-outcome example: SPL restores distributional representation and repeated-sampling coverage while preserving the top-ranked output.
- `calibration_multiple_choice / dpo_alpha_equals_beta`: ECE `0.467`, preference KL `3.869`, Brier `0.307`, top-answer accuracy `1.0`.
- `calibration_multiple_choice / spl_alpha_1.0_beta_0.1`: ECE `0.00411`, preference KL `0.000069`, Brier `0.000036`, top-answer accuracy `1.0`.
- The calibration result is a finite multiple-choice version of the paper's calibration claim: SPL greatly reduces overconfidence while preserving the most supported answer.
- `python validate.py --compare` passes after the continuation changes.

## Remaining

- Higher milestones would require a scaled LLM LoRA reproduction on HH-RLHF or Ultrafeedback plus GSM8K/MATH, chat diversity, or full MCQ calibration.
- The finite-action benchmarks are sufficient for `core_claim_plus`, and likely `secondary_claims` after managed verification, because they implement the paper's SPL computation directly and test two-outcome representation, multi-outcome repeated-sampling diversity, and calibration.

## Issues

- The original 7B experiments require large model downloads and GPU training. This environment starts with the analytic core claim to provide a fast, reusable benchmark.
- The current two-experiment package has now been verified by managed CPU job `9b0d9378-ef7`.
- The calibration extension has only been verified locally so far; the next managed job should rerun `scripts/run.sh` and validate all three experiments in the packaged container.
- `README.md` is a pre-existing character-device entry in the repository and was left untouched during the audit.

## Deviations from Paper

- Paper experiments use Mistral-7B LoRA finetuning on HH-RLHF/Ultrafeedback and evaluate chat diversity, best-of-N math, and MCQ calibration. This reproduction uses the paper's finite two-outcome social-choice example instead, because it is the central theorem/example and can be run without heavyweight LLM training.
- The continuation adds a generated finite multi-outcome Plackett-Luce benchmark derived from Appendix A.1. It is still a finite-action reproduction, not a substitute for the paper's full LLM empirical suite.
- The multi-outcome reference values are closed-form/theory-derived targets for this generated finite benchmark, not a reported LLM table from the paper.
- The calibration benchmark is a finite synthetic substitute for the paper's full multiple-choice logit calibration experiments. It preserves the SPL/DPO objective and distributional calibration phenomenon but does not use an LLM or MMLU-style data.
- The algorithm is not substituted: the implemented objective is the paper's SPL pairwise objective, with DPO recovered by `alpha == beta`.
