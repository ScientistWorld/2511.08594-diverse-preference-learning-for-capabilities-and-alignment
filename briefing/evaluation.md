# Evaluation

## Protocol

A method writes one directory per run under `checkpoints/<run_name>/` containing three files:

| file | content |
|---|---|
| `two_outcome.json` | `{"contexts": {<id>: {"initial": [p0, p1]}}}` |
| `calibration.json` | `{"contexts": {<id>: {"initial": [p0, p1, p2, p3]}}}` |
| `sequence.json` | `{"contexts": {<id>: {"initial": [21 probs], "transitions": [6 x 21 x 21 row-stochastic]}}}` |

`initial` is the distribution of the first token, `transitions[i][a]` the distribution of token
`i+1` given token `i` = `a`. Every prompt id in `data/*_contexts.json` (train and test split) must be
present. Only these probability tables are read; extra fields inside a context are rejected. Top-level descriptive metadata and unrelated files never enter a score. Ground truth
(population preferences, rewards, verifier sets, human response distributions), the reference
policies used for gates, and all thresholds are owned by the evaluator.

- `bash scripts/evaluate_train.sh` scores every `checkpoints/*` directory on the visible train
  prompts and writes `scoring/scores_train.json` (same code path as `scripts/evaluate.sh` ->
  `scoring/scores.json`).
- `bash scripts/evaluate_test.sh` (no arguments) scores every `checkpoints/*` directory on the
  hidden test prompts and writes `scoring/scores_test.json`. The hidden test only reads the saved
  artifacts; it never trains.

All numbers are computed exactly from the probability tables (dynamic programming for
sequences); there is no sampling noise and the evaluation is deterministic.

## Metrics (scored)

**two_outcome_mode_collapse**
- `preference_kl` (lower): mean KL(population split || policy), capped at 20 per prompt.
- `minority_log_ratio_error` (lower): mean |log(policy minority mass / population minority
  share)|, capped at 30 per prompt.

**sequence_diversity_best_of_n**
- `best_of_128_success_hard` (higher, primary): probability that 128 samples contain a solution,
  averaged over 6 hidden hard problems per prompt (coherent sequences matching a hidden pattern,
  away from the most popular completion).
- `population_kl` (lower): sequence-level KL(population distribution || policy), cap 60.
- `coherence` (higher): probability that a sampled sequence contains no incoherent token, capped per prompt at the hidden population policy's own coherence. Increasing coherence beyond that level receives no extra credit. Raw and population coherence are diagnostic only.

**calibration_multiple_choice** (expectations over a hidden randomly drawn human response)
- `brier_score` (lower, primary): multi-class Brier score.
- `expected_calibration_error` (lower): 10-bin ECE of top-answer confidence (weighted below Brier:
  a constant, uninformative confidence can make ECE small but not Brier).
- `accuracy` (higher): accuracy of the most probable option (ties -> lowest index). Reported and
  enforced by the gate below; it carries no rank weight (coefficient 0), because methods that pass
  the gate differ in it only by a handful of near-tie decisions.

## Task gates and failures

A run **fails** if, in any experiment, an artifact is missing, malformed, not a probability
distribution (non-finite, negative, > 1, rows not summing to 1 within 1e-4), misses a prompt, or
violates a task gate:

- two-outcome: the population-majority completion is the most probable one on < 75% of prompts;
- sequence: expected population reward of the policy below that of the reference policy;
- multiple choice: expected agreement with a human response below that of simply picking the option with the best
  (wins - losses) count in the prompt's comparisons, minus 0.01. Floors are recomputed from evaluator-owned truth for each split and are reported in diagnostics.

Hidden-named method directories are also scored. An empty checkpoint tree produces a failed `missing_submission` record.

A failed run receives the **worst value of every metric in every experiment** (categorical KL 20,
minority log-ratio error 30, sequence KL 60,
success 0, coherence 0, ECE 1, Brier 2, accuracy 0); the reason is recorded under
`failed_methods`. Gate quantities are reported under `diagnostics`.

## Baselines

- `dpo_alpha_equals_beta`: standard KL-regularised preference learning (DPO, beta = 0.1).
- `dpo_token_temperature_1.5`: the same policy with token-level temperature 1.5 applied to every
  conditional distribution.

Both stay coherent or nearly so but collapse onto the majority, so they lose on distribution fit,
hard best-of-128 and calibration. Higher token temperatures recover some diversity but quickly
produce incoherent tokens. A uniform policy fails the gates.

Rank contributions use experiment weight times the magnitude of each coefficient times the
direction-correct improvement over the reference, divided by the metric's standard deviation
across the attempts being ranked. Coefficient signs and measurement units do not change priority.
Every failed attempt has every metric, so no task failure can buy a higher rank through another
metric. Metrics, operating points and test sizes are fixed by the evaluator.
