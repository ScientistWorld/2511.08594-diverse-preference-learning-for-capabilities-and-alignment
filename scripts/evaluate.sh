#!/bin/bash
# Evaluation script — the standard way to evaluate work in this environment.
#
# This script must be SELF-CONTAINED using only code in eval/.
# Do NOT import from method/ — evaluate.sh must work even when the
# paper's method code is hidden from a scientist agent.
#
# Design it so that anyone following the same output format can run
# this script to evaluate their results.
#
# OUTPUT CONTRACT: This script MUST write scoring/scores.json containing
# a JSON dict nested by experiment, e.g.:
#   {"imagenet_classification": {"top1_accuracy": 82.1, "top5_accuracy": 94.8}}
# Experiment and metric names must match the keys in scoring/reference.json.
# This file is read by the system for automated comparison.
#
# Example usage:
#   bash scripts/evaluate.sh checkpoints/spl_tabular
#   bash scripts/evaluate.sh checkpoints/dpo_tabular checkpoints/spl_tabular
#
# Called from run.sh after training, or independently.

set -e

cd /home/user

mkdir -p scoring

EXPERIMENT="${EXPERIMENT:-two_outcome_mode_collapse}"

if [ "$#" -lt 1 ]; then
  set -- checkpoints/dpo_two_outcome checkpoints/spl_two_outcome
fi

MERGE_ARG=""
if [ "${MERGE_SCORES:-0}" = "1" ]; then
  MERGE_ARG="--merge"
fi

python eval/evaluate_tabular.py "$@" --experiment "$EXPERIMENT" --scores-out scoring/scores.json $MERGE_ARG
