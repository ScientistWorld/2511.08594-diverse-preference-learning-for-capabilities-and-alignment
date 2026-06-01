# Scientist Constraints

## Data

Use the fixed benchmark in `data/two_outcome_benchmark.json` for the reported `two_outcome_mode_collapse` score. Do not change the 80/20 population preference, the 50/50 reference policy, or the pairwise comparison counts when reporting against this experiment.

Use the generated benchmark from `data/spl_benchmark.py` for the reported `tabular_preference_learning` score. Keep its seeded prompt set, four-outcome action spaces, pairwise comparison counts, and marked best-of-N solution actions fixed when comparing against the reported results.

Use the generated benchmark from `data/calibration_benchmark.py` for the reported `calibration_multiple_choice` score. Keep its prompt set, four-answer action spaces, pairwise comparison counts, reference policy, and answer preference distributions fixed when comparing against the reported results.

## Evaluation

Scores must be produced from saved policy artifacts with the same output format consumed by `eval/evaluate_tabular.py`. Do not edit the evaluator to call training code or regenerate method behavior internally.

## Compute and Scale

This benchmark is intentionally finite and tabular. Do not improve the score by adding more outcomes, extra prompts, external preference data, larger language models, or additional labels for this experiment.

## Compliance Checklist

- [ ] `data/two_outcome_benchmark.json` is unchanged for the reported score
- [ ] `data/spl_tabular_benchmark.json` is generated from `data/spl_benchmark.py` with the default seed for the multi-outcome score
- [ ] `data/calibration_benchmark.json` is generated from `data/calibration_benchmark.py` with default settings for the calibration score
- [ ] Evaluation reads saved artifacts rather than importing training code
- [ ] No external data, extra prompts, or larger model resources are used for this experiment
- [ ] Reported methods preserve the same problem formulation for each experiment
