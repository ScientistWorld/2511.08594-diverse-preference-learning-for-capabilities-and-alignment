# Evaluation Targets

## Primary: two_outcome_mode_collapse

Tests whether a preference learner avoids collapsing a minority-supported option to near-zero probability in the paper's fixed 80/20 two-outcome setting.

- **Primary metric**: `minority_probability` (higher is better, up to the 0.2 population target)
- **Constraint metrics**: `majority_overweight` (lower is better) and `top_choice_preserved` (higher is better)
- **How to run**: `bash scripts/reproduce.sh`, or run `bash scripts/evaluate.sh checkpoints/dpo_two_outcome checkpoints/spl_two_outcome` after producing artifacts.
