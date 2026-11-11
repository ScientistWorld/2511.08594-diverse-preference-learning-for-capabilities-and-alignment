#!/bin/bash
# Run the paper's method: Soft Preference Learning (alpha=1, beta=0.1) on all
# three experiments.  Writes checkpoints/spl_alpha_1.0_beta_0.1/.
#
# Usage: bash scripts/method.sh   (env: ALPHA, BETA, OUTPUT_DIR, RUN_EVAL)

set -euo pipefail

cd /home/user

ALPHA="${ALPHA:-1.0}"
BETA="${BETA:-0.1}"
OUTPUT_DIR="${OUTPUT_DIR:-checkpoints/spl_alpha_${ALPHA}_beta_${BETA}}"

python3 method/spl.py --data-dir data --output-dir "$OUTPUT_DIR" --alpha "$ALPHA" --beta "$BETA"

if [ "${RUN_EVAL:-1}" = "1" ]; then
  bash scripts/evaluate.sh "$OUTPUT_DIR"
fi
