# Progress

## What Works

- `scripts/reproduce.sh` runs a complete tabular reproduction of the paper's core SPL/DPO objective.
- `baseline/run_dpo.py` runs standard DPO as the `alpha == beta` special case.
- `method/train_tabular_spl.py` runs the paper's SPL DPO-style objective with separate `alpha` and `beta`.
- `eval/evaluate_tabular.py` scores saved artifacts only and does not import from `method/`.
- `python validate.py --compare` passes locally.

## Results

- `two_outcome_mode_collapse / dpo_alpha_equals_beta`: minority probability `9.54e-7`, majority overweight `1.25`, top choice preserved `1.0`.
- `two_outcome_mode_collapse / spl_alpha_1.0_beta_0.1`: minority probability `0.20`, majority overweight `1.0`, top choice preserved `1.0`.
- These match the paper's analytic prediction for the 80/20 example: DPO collapses minority support to near zero at `beta=0.1`, while SPL with `alpha=1` recovers proportional representation.

## Remaining

- Higher milestones would require a scaled LLM LoRA reproduction on HH-RLHF or Ultrafeedback plus GSM8K/MATH, which is beyond this first smoke job.
- The existing tabular benchmark is sufficient for `core_claim` because it implements and verifies the paper's central computational claim directly.

## Issues

- The original 7B experiments require large model downloads and GPU training. This environment starts with the analytic core claim to provide a fast, reusable benchmark.

## Deviations from Paper

- Paper experiments use Mistral-7B LoRA finetuning on HH-RLHF/Ultrafeedback and evaluate chat diversity, best-of-N math, and MCQ calibration. This reproduction uses the paper's finite two-outcome social-choice example instead, because it is the central theorem/example and can be run without heavyweight LLM training.
- The algorithm is not substituted: the implemented objective is the paper's SPL pairwise objective, with DPO recovered by `alpha == beta`.
