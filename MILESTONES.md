# Reproduction Milestones

**Current: core_claim**

## Progress Log

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

## Stop Justification
