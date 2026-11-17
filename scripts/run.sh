#!/bin/bash
# Reproduce the reference and both baselines end to end, then score both slices.
# Designer-only combined audit: scripts/verify_hardening.sh.
set -euo pipefail
cd /home/user
bash scripts/reproduce.sh
bash scripts/evaluate_train.sh
bash scripts/evaluate_test.sh
bash scripts/evaluate.sh
python3 validate.py --compare
