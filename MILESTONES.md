# Reproduction Milestones

**Current: core_claim_plus**

## Progress Log

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

## Stop Justification
- Completed at milestone `core_claim_plus`.
