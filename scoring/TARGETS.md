# Evaluation Targets

Every target below is an entry of `scoring/scores_test.json` (hidden test prompts) and
`scoring/scores_train.json` (visible train prompts). Run `bash scripts/reproduce.sh` for the
reference runs, then `bash scripts/evaluate_train.sh` (and, for the hidden slice,
`bash scripts/evaluate_test.sh`).

## Primary: sequence_diversity_best_of_n (weight 0.5)

- **Primary metric**: `best_of_128_success_hard` (higher is better).
- **Benefit / fit metric**: `population_kl` (lower is better).
- **Constraint metric (scored)**: `coherence` (higher is better).
- **Constraint gate**: expected population reward >= reference policy's (else the run fails).

## Core claim: two_outcome_mode_collapse (weight 0.25)

- **Primary metric**: `preference_kl` (lower is better).
- **Benefit metric**: `minority_log_ratio_error` (lower is better).
- **Constraint gate**: majority completion stays most probable on >= 75% of prompts.
- The visible train prompt `two_outcome_paper_80_20` is the paper's analytic 80/20 example; its
  minority probability and majority overweight are reported under train `diagnostics`.

## Secondary: calibration_multiple_choice (weight 0.25)

- **Primary metric**: `expected_calibration_error` (lower is better).
- **Benefit metric**: `brier_score` (lower is better).
- **Constraint metric (scored)**: `accuracy` (higher is better).
- **Constraint gate**: accuracy >= reference-policy accuracy - 0.02.

## Ablation / baseline targets

`dpo_alpha_equals_beta` (standard preference learning) and `dpo_token_temperature_1.5` are scored
by the same scripts and must stay clearly below the reference method.
