# Scientist Constraints

## Data

- Learn only from the public inputs in `data/`: the reference policies and the pairwise
  comparisons of `data/two_outcome_contexts.json`, `data/sequence_contexts.json` and
  `data/calibration_contexts.json`. Do not modify these files.
- Do not use evaluator ground truth as a training signal: `eval/train/truth.json` exists so the
  train slice can be scored, and must not be read by a method (copying it improves train scores
  only and is treated as a shortcut). Do not try to recover hidden test answers.
- No external preference data, labels, or pretrained models.

## Output

- One directory per run under `checkpoints/`, with `two_outcome.json`, `sequence.json` and
  `calibration.json` covering every prompt id (train and test) in the format described in
  `briefing/evaluation.md`. Policies must be genuine probability distributions; the reference
  policies, population answers and verifier sets are fixed by the evaluator and cannot be supplied
  by a method. Extra fields inside a context are rejected; supply probability tables only.
- The sequence policy format is a first-order chain over tokens (initial distribution plus
  position-dependent transition matrices).

## Evaluation

- Do not edit `eval/`, the gates, caps, or metric definitions. `scripts/evaluate_test.sh` scores
  every directory under `checkpoints/`; a missing or invalid experiment fails the whole run.

## Compute

- The reference runs are pure Python on one CPU (~20 minutes per method). There is no limit on a
  method's training compute, but the hidden test only scores saved artifacts.

## Compliance Checklist

- [ ] Only `data/` inputs used for training; no reads of `eval/*/truth.json` from method code
- [ ] All three artifact files present for every run, covering all prompt ids
- [ ] Probabilities are valid distributions; no evaluator-owned fields supplied or relied on
- [ ] Evaluator code and thresholds unchanged
