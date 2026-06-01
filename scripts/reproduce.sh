#!/bin/bash
# Reproduce the paper's results end-to-end.
#
# This script runs inside the compute container and should call
# scripts/baseline.sh and scripts/method.sh to run baselines and
# the paper's method, each of which calls scripts/evaluate.sh.
#
# Assumes:
#   - Container is built (environment/container.sif)
#   - Data and models are downloaded (via scripts/download.sh)
#   - Working directory is /home/user
#
# This is the "gold standard" reproduction — 
# You do not need to run this, but keep this in good shape
# so other agents can easily run a full loop of reproduction and get the paper's numbers.

set -e

cd /home/user

BENCHMARK="${BENCHMARK:-data/two_outcome_benchmark.json}"
STEPS="${STEPS:-6000}"
LR="${LR:-0.8}"
SAMPLES_PER_CONTEXT="${SAMPLES_PER_CONTEXT:-128}"
SEED="${SEED:-11}"

BENCHMARK="$BENCHMARK" \
OUTPUT_DIR="checkpoints/dpo_two_outcome" \
BETA="${BETA:-0.1}" \
STEPS="$STEPS" \
LR="$LR" \
SAMPLES_PER_CONTEXT="$SAMPLES_PER_CONTEXT" \
SEED="$SEED" \
RUN_EVAL=0 \
bash scripts/baseline.sh dpo

BENCHMARK="$BENCHMARK" \
OUTPUT_DIR="checkpoints/spl_two_outcome" \
ALPHA="${ALPHA:-1.0}" \
BETA="${BETA:-0.1}" \
STEPS="$STEPS" \
LR="$LR" \
SAMPLES_PER_CONTEXT="$SAMPLES_PER_CONTEXT" \
SEED="$SEED" \
RUN_EVAL=0 \
bash scripts/method.sh

python data/spl_benchmark.py --output data/spl_tabular_benchmark.json

BENCHMARK="data/spl_tabular_benchmark.json" \
OUTPUT_DIR="checkpoints/dpo_tabular" \
BETA="${BETA:-0.1}" \
STEPS="$STEPS" \
LR="$LR" \
SAMPLES_PER_CONTEXT="$SAMPLES_PER_CONTEXT" \
SEED="$SEED" \
RUN_EVAL=0 \
bash scripts/baseline.sh dpo

BENCHMARK="data/spl_tabular_benchmark.json" \
OUTPUT_DIR="checkpoints/spl_tabular" \
ALPHA="${ALPHA:-1.0}" \
BETA="${BETA:-0.1}" \
STEPS="$STEPS" \
LR="$LR" \
SAMPLES_PER_CONTEXT="$SAMPLES_PER_CONTEXT" \
SEED="$SEED" \
RUN_EVAL=0 \
bash scripts/method.sh

bash scripts/evaluate.sh checkpoints/dpo_two_outcome checkpoints/spl_two_outcome
EXPERIMENT="tabular_preference_learning" MERGE_SCORES=1 \
  bash scripts/evaluate.sh checkpoints/dpo_tabular checkpoints/spl_tabular
