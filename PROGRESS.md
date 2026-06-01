# Progress

## What Works

- `scripts/reproduce.sh` runs a complete tabular reproduction of the paper's core SPL/DPO objective.
- `baseline/run_dpo.py` runs standard DPO as the `alpha == beta` special case.
- `method/train_tabular_spl.py` runs the paper's SPL DPO-style objective with separate `alpha` and `beta`.
- `eval/evaluate_tabular.py` scores saved artifacts only and does not import from `method/`.
- Continuation audit on 2026-06-01 re-ran `scripts/reproduce.sh` and `python validate.py --compare`; both passed locally.
- CPU smoke job `e57fca5a-e7d` passed in the packaged container with exit code 0.
- Current milestone is `core_claim_plus`; the reproduction now covers the central two-outcome claim and a multi-outcome finite Plackett-Luce extension from the paper's Appendix A.1 setting.

## Results

- `two_outcome_mode_collapse / dpo_alpha_equals_beta`: minority probability `9.54e-7`, majority overweight `1.25`, top choice preserved `1.0`.
- `two_outcome_mode_collapse / spl_alpha_1.0_beta_0.1`: minority probability `0.20`, majority overweight `1.0`, top choice preserved `1.0`.
- These match the paper's analytic prediction for the 80/20 example: DPO collapses minority support to near zero at `beta=0.1`, while SPL with `alpha=1` recovers proportional representation.
- Managed job validation re-created these numbers through `scripts/run.sh`, `scripts/reproduce.sh`, and `scripts/evaluate.sh`.
- `tabular_preference_learning / dpo_alpha_equals_beta`: preference KL `3.896`, normalized entropy `0.024`, best-of-128 success probability `0.151`, population-mode accuracy `1.0`.
- `tabular_preference_learning / spl_alpha_1.0_beta_0.1`: preference KL `0.000138`, normalized entropy `0.858`, best-of-128 success probability `0.999994`, population-mode accuracy `1.0`.
- The multi-outcome result shows the same mechanism beyond the paper's minimum two-outcome example: SPL restores distributional representation and repeated-sampling coverage while preserving the top-ranked output.
- `python validate.py --compare` passes after the continuation changes.

## Remaining

- Higher milestones would require a scaled LLM LoRA reproduction on HH-RLHF or Ultrafeedback plus GSM8K/MATH, chat diversity, or MCQ calibration.
- The finite-action benchmarks are sufficient for `core_claim_plus` because they implement the paper's SPL computation directly and test it on both two-outcome and multi-outcome preference settings.

## Issues

- The original 7B experiments require large model downloads and GPU training. This environment starts with the analytic core claim to provide a fast, reusable benchmark.
- `README.md` is a pre-existing character-device entry in the repository and was left untouched during the audit.

## Deviations from Paper

- Paper experiments use Mistral-7B LoRA finetuning on HH-RLHF/Ultrafeedback and evaluate chat diversity, best-of-N math, and MCQ calibration. This reproduction uses the paper's finite two-outcome social-choice example instead, because it is the central theorem/example and can be run without heavyweight LLM training.
- The continuation adds a generated finite multi-outcome Plackett-Luce benchmark derived from Appendix A.1. It is still a finite-action reproduction, not a substitute for the paper's full LLM empirical suite.
- The multi-outcome reference values are closed-form/theory-derived targets for this generated finite benchmark, not a reported LLM table from the paper.
- The algorithm is not substituted: the implemented objective is the paper's SPL pairwise objective, with DPO recovered by `alpha == beta`.
