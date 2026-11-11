#!/bin/bash
# Standard evaluation entry point.
#
# Scores method directories (default: every checkpoints/* directory) on the
# visible train contexts with eval/train/evaluate.py and writes
# scoring/scores.json.  Uses only code in eval/ (never method/).
#
# Artifact contract (per method directory):
#   two_outcome.json   {"contexts": {<id>: {"initial": [p0, p1]}}}
#   calibration.json   {"contexts": {<id>: {"initial": [p0, p1, p2, p3]}}}
#   sequence.json      {"contexts": {<id>: {"initial": [V probs],
#                                           "transitions": [L-1 x V x V row-stochastic]}}}
# Every context of data/*_contexts.json (train AND test split) must be present.
#
# Usage:
#   bash scripts/evaluate.sh                      # all checkpoints/*
#   bash scripts/evaluate.sh checkpoints/my_run   # explicit directories

set -euo pipefail

cd /home/user

mkdir -p scoring
python3 eval/train/evaluate.py "$@" --scores-out scoring/scores.json
