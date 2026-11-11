#!/bin/bash
# Run a baseline from the paper.
#
#   bash scripts/baseline.sh dpo                 # DPO (alpha == beta = 0.1)
#   bash scripts/baseline.sh token_temperature   # DPO + token-level temperature 1.5 (needs dpo first)
#   bash scripts/baseline.sh reference_policy    # the untrained reference policy

set -euo pipefail

cd /home/user

NAME="${1:-dpo}"
BETA="${BETA:-0.1}"
TEMPERATURE="${TEMPERATURE:-1.5}"

case "$NAME" in
  dpo)
    OUT="${OUTPUT_DIR:-checkpoints/dpo_alpha_equals_beta}"
    python3 baseline/run_dpo.py --data-dir data --output-dir "$OUT" --beta "$BETA"
    ;;
  token_temperature)
    OUT="${OUTPUT_DIR:-checkpoints/dpo_token_temperature_${TEMPERATURE}}"
    python3 baseline/token_temperature.py --source-dir "${SOURCE_DIR:-checkpoints/dpo_alpha_equals_beta}" \
      --output-dir "$OUT" --temperature "$TEMPERATURE"
    ;;
  reference_policy)
    OUT="${OUTPUT_DIR:-checkpoints/reference_policy}"
    python3 baseline/reference_policy.py --data-dir data --output-dir "$OUT"
    ;;
  *)
    echo "Unknown baseline '$NAME'. Available: dpo, token_temperature, reference_policy" >&2
    exit 2
    ;;
esac

if [ "${RUN_EVAL:-1}" = "1" ]; then
  bash scripts/evaluate.sh "$OUT"
fi
