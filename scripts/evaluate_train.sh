#!/bin/bash
# Visible train-slice evaluation.
#
# Runs with no arguments: scores every method directory under checkpoints/ on
# the visible train contexts (same metrics, gates and failure rules as the
# hidden test slice) and writes scoring/scores_train.json.

set -euo pipefail

cd /home/user

mkdir -p scoring
python3 eval/train/evaluate.py "$@" --scores-out scoring/scores_train.json
