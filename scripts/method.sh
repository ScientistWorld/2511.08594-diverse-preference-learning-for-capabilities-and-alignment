#!/bin/bash
# Run the paper's proposed method.
#
# This script trains/runs the paper's method using code in method/.
#
# Usage:
#   bash scripts/method.sh

set -e

cd /home/user

BENCHMARK="${BENCHMARK:-data/two_outcome_benchmark.json}"
OUTPUT_DIR="${OUTPUT_DIR:-checkpoints/spl_two_outcome}"
ALPHA="${ALPHA:-1.0}"
BETA="${BETA:-0.1}"
STEPS="${STEPS:-3000}"
LR="${LR:-0.8}"
SAMPLES_PER_CONTEXT="${SAMPLES_PER_CONTEXT:-128}"
SEED="${SEED:-11}"

if [ ! -f "$BENCHMARK" ]; then
  python data/spl_benchmark.py --output "$BENCHMARK"
fi
python method/train_tabular_spl.py \
  --benchmark "$BENCHMARK" \
  --output-dir "$OUTPUT_DIR" \
  --method-name "spl_alpha_${ALPHA}_beta_${BETA}" \
  --alpha "$ALPHA" \
  --beta "$BETA" \
  --steps "$STEPS" \
  --lr "$LR" \
  --samples-per-context "$SAMPLES_PER_CONTEXT" \
  --seed "$SEED"

if [ "${RUN_EVAL:-1}" = "1" ]; then
  bash scripts/evaluate.sh "$OUTPUT_DIR"
fi
