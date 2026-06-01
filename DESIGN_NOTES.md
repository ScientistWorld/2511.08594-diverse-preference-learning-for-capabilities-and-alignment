# Train/Test Evaluation Design Notes

## Checks Performed

- Read `scripts/evaluate_train.sh` and `scripts/evaluate_test.sh` side by side. Train invokes `eval/train/evaluate.py` and writes `scoring/scores_train.json`; test invokes `eval/test/evaluate.py` and writes `scoring/scores_test.json`.
- Checked the split evaluators for method detachment. Both slices read saved `policy.json` artifacts, plus optional `samples.json`, and do not import or rerun `method/`.
- Checked for hardcoded method defaults. Both scripts enumerate checkpoint directories when no explicit run directory is supplied; test can also resolve the same method names from `scoring/scores_train.json`.
- Checked for train/test leakage. No Python cross-imports between `eval/train/` and `eval/test/` were found.
- Checked for forbidden paper-agreement metric names in the edited reference schema. None were found.
- Confirmed no `scoring/scores_train.json` or `scoring/scores_test.json` stubs are present.
- Ran `bash -n` on both train/test shell entry points.

## Changes Made

- Added distribution-validity checks to both train and test evaluators. Invalid, negative, non-finite, or non-normalized probability artifacts now fail instead of being scoreable shortcuts.
- Strengthened `two_outcome_mode_collapse` against over-flattening shortcuts. The first pass rewarded minority mass, majority overweight, and top-choice preservation, but a method could still improve the visible score by pushing minority mass far above the population target while keeping the majority barely on top. Both slices now also emit `preference_kl`, `brier_score`, `expected_calibration_error`, and `population_l1_error` for the two-outcome artifact.
- Added required metric descriptions throughout `scoring/reference.json`.
- Reweighted the two experiments from 0.65/0.35 to 0.55/0.45. The two-outcome case remains primary, but the complementary tabular context split now has more influence because it provides the real held-out data variation.

## Runtime Concerns For Next Iteration

- The two-outcome experiment still has only one artifact context, so train/test separation there is metric triangulation rather than a true data split. The new distribution-fit metrics reduce shortcut pressure, but the tabular experiment is the stronger held-out signal.
- `empirical_best_of_128_success` depends on optional `samples.json`. The analytic `best_of_128_success_probability` is the more robust signal; if future artifacts omit samples, this empirical metric may become a format-compliance penalty rather than a pure scientific metric.
- I did not run the Python evaluators or regenerate scores, per this iteration's no-experiment constraint.
