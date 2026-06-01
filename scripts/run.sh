#!/bin/bash
# Run the actual tabular SPL reproduction, validate the core scoring file, then
# refresh the visible train and held-out test split scores from the new artifacts.

set -euo pipefail

cd /home/user

bash scripts/reproduce.sh
bash scripts/evaluate_train.sh
bash scripts/evaluate_test.sh
python validate.py --compare
