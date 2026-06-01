# Evaluation

## Metrics

- `minority_probability`: learned probability assigned to the 20% minority-supported option. Higher is better up to the population target; near-zero values indicate mode collapse.
- `majority_overweight`: learned majority-option probability divided by its 80% population support. Lower is better, with 1.0 indicating proportional representation and values above 1 indicating over-concentration.
- `top_choice_preserved`: whether the majority-supported option remains the highest-probability output. Higher is better; this is a quality/order constraint.
- `expected_calibration_error`: finite-action calibration gap between learned probability mass and population preference mass. Lower is better.
- `preference_kl` and `brier_score`: distributional fit metrics between the learned policy and population preferences. Lower is better.

## Baselines and Targets

The paper's analytic 80/20 example reports that standard preference learning with `beta=0.1` assigns approximately 99.9999% probability to the majority option and 0.0001% to the minority option. The target distribution for the calibrated setting assigns 80% to the majority option and 20% to the minority option while preserving the majority as the top choice.

## Evaluation Protocol

Evaluate saved policy artifacts for the fixed two-outcome preference task in `data/two_outcome_benchmark.json`, the generated multi-outcome task in `data/spl_tabular_benchmark.json`, and the generated multiple-choice calibration task in `data/calibration_benchmark.json`. All methods receive the same pairwise preference counts, reference distributions, and population targets. Evaluation code reads only method output artifacts and must not import training code.
