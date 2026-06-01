#!/bin/bash
# Run a baseline method from the paper.
#
# Baselines are simple reference methods the paper compares against. 
# Implement this only if applicable, i.e. the paper has baselines.
# You do not need to implement all baselines to begin — 
# just enough to validate that the evaluation pipeline works 
# and provide a reference point. You can decide to expand more baselines later.
# Each baseline must be actually run and scored through evaluate.sh
# so the numbers are verified, not self-reported.
# You can decide what this baseline script should do yourself.
#
# Usage:
#   bash scripts/baseline.sh <name>

set -e

cd /home/user

NAME="${1:-dpo}"
BENCHMARK="${BENCHMARK:-data/two_outcome_benchmark.json}"
OUTPUT_DIR="${OUTPUT_DIR:-checkpoints/dpo_two_outcome}"
BETA="${BETA:-0.1}"
STEPS="${STEPS:-3000}"
LR="${LR:-0.8}"
SAMPLES_PER_CONTEXT="${SAMPLES_PER_CONTEXT:-128}"
SEED="${SEED:-11}"

if [ "$NAME" != "dpo" ]; then
  echo "Unknown baseline '$NAME'. Available: dpo" >&2
  exit 2
fi

if [ ! -f "$BENCHMARK" ]; then
  python data/spl_benchmark.py --output "$BENCHMARK"
fi
python baseline/run_dpo.py \
  --benchmark "$BENCHMARK" \
  --output-dir "$OUTPUT_DIR" \
  --beta "$BETA" \
  --steps "$STEPS" \
  --lr "$LR" \
  --samples-per-context "$SAMPLES_PER_CONTEXT" \
  --seed "$SEED"

if [ "${RUN_EVAL:-1}" = "1" ]; then
  bash scripts/evaluate.sh "$OUTPUT_DIR"
fi
