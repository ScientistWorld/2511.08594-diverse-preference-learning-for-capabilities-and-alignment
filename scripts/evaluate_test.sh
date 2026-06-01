#!/bin/bash
# Held-out test-slice evaluator.
#
# Mirrors scripts/evaluate.sh's artifact contract. With explicit checkpoint
# directories, it scores those artifacts. With no arguments, it scores the same
# experiment/method entries already present in scoring/scores_train.json; if
# that file is absent, it enumerates every checkpoints/*/policy.json.
# Writes scoring/scores_test.json.

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

METHODS_FROM_ARG=()
if [ "$#" -eq 0 ] && [ -f scoring/scores_train.json ]; then
  METHODS_FROM_ARG=(--methods-from-scores scoring/scores_train.json)
fi

python eval/test/evaluate.py "$@" --scores-out scoring/scores_test.json "${EXPERIMENT_ARG[@]}" "${METHODS_FROM_ARG[@]}" "${MERGE_ARG[@]}"
