#!/bin/bash
# Visible train-slice evaluator.
#
# Mirrors scripts/evaluate.sh's artifact contract: pass one or more checkpoint
# directories, or pass none to enumerate every checkpoints/*/policy.json.
# Writes scoring/scores_train.json; real score files are produced only when this
# script is run in a later evaluation iteration.

set -e

cd /home/user

mkdir -p scoring

EXPERIMENT_ARG=()
if [ -n "${EXPERIMENT:-}" ]; then
  EXPERIMENT_ARG=(--experiment "$EXPERIMENT")
fi

MERGE_ARG=()
if [ "${MERGE_SCORES:-0}" = "1" ]; then
  MERGE_ARG=(--merge)
fi

python eval/train/evaluate.py "$@" --scores-out scoring/scores_train.json "${EXPERIMENT_ARG[@]}" "${MERGE_ARG[@]}"
