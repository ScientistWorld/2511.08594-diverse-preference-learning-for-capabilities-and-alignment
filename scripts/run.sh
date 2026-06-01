#!/bin/bash
# Score the visible train slice and held-out test slice against every saved
# checkpoint artifact. The evaluators discover compatible methods under
# checkpoints/ and write scoring/scores_{train,test}.json.

set -euo pipefail

cd /home/user

bash scripts/evaluate_train.sh
bash scripts/evaluate_test.sh
