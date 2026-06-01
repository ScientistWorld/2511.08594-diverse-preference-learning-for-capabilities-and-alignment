# Evaluation Targets

## Primary: two_outcome_mode_collapse

Tests whether a preference learner avoids collapsing a minority-supported option to near-zero probability in the fixed 80/20 two-outcome setting.

- **Primary metric**: `minority_probability` (higher is better, up to the 0.2 population target)
- **Constraint metrics**: `majority_overweight` (lower is better) and `top_choice_preserved` (higher is better)
- **How to run**: `bash scripts/reproduce.sh`, or run `bash scripts/evaluate.sh <artifact-dir> [...]` after producing policy artifacts.

## Extension: tabular_preference_learning

Tests generalized multi-outcome preference representation and its implication for repeated sampling.

- **Primary metric**: `preference_kl` from population preferences to learned policy (lower is better)
- **Benefit metrics**: `best_of_128_success_probability`, `empirical_best_of_128_success`, `normalized_entropy`, and `effective_actions` (higher is better)
- **Constraint metrics**: `argmax_population_mode_accuracy` and `expected_population_preference` (higher is better), plus `reference_cross_entropy`, `brier_score`, and `expected_calibration_error` (lower is better)
- **How to run**: `bash scripts/reproduce.sh`, or generate `data/multi_outcome_benchmark.json` and run `EXPERIMENT=tabular_preference_learning bash scripts/evaluate.sh <artifact-dir> [...]`.

## Secondary: calibration_multiple_choice

Tests whether a preference learner produces calibrated multiple-choice probabilities rather than overconfident majority choices.

- **Primary metric**: `expected_calibration_error` (lower is better)
- **Benefit metrics**: `preference_kl`, `brier_score`, `normalized_entropy`, and `effective_actions`
- **Constraint metric**: `argmax_population_mode_accuracy` must remain high so calibration cannot be improved by inverting answer preferences.
- **How to run**: `bash scripts/reproduce.sh`, or generate `data/calibration_benchmark.json` and run `EXPERIMENT=calibration_multiple_choice bash scripts/evaluate.sh <artifact-dir> [...]`.
