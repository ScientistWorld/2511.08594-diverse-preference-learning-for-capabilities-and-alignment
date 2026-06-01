# Reproduction Milestones

**Current: core_claim_plus**

## Progress Log

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

## Stop Justification
- Completed at milestone `core_claim_plus`.
