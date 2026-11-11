#!/bin/bash
# Hidden test-slice evaluation.
#
# Runs with no arguments: scores EVERY method directory under checkpoints/
# (one directory = one method, named by the directory) on the held-out test
# contexts and writes scoring/scores_test.json.  Methods are discovered from
# disk; no manifest is consulted.  A method with a missing, malformed or
# invalid artifact, or one that fails a task gate, is recorded as a failed run
# with the worst value of every metric in every experiment.

set -euo pipefail

cd /home/user

mkdir -p scoring
python3 eval/test/evaluate.py "$@" --scores-out scoring/scores_test.json
