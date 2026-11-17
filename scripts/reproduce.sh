#!/bin/bash
# Reproduce the reference runs end to end (CPU only, pure Python, ~25 min).
#
#   checkpoints/spl_alpha_1.0_beta_0.1      the paper's method (SPL, alpha=1, beta=0.1)
#   checkpoints/dpo_alpha_equals_beta       DPO baseline (alpha = beta = 0.1)
#   checkpoints/dpo_token_temperature_1.5   DPO + token-level temperature 1.5
#
# Training is deterministic (full-batch gradient descent from the reference
# policy on fixed public comparisons), so reruns are bit-identical.
# Reference and baseline trainings run alone, sequentially. No wall-clock
# budget or throughput measurement participates in their training or scores.

set -euo pipefail

cd /home/user

mkdir -p checkpoints logs
RUN_EVAL=0 bash scripts/method.sh > logs/train_spl.log 2>&1
RUN_EVAL=0 bash scripts/baseline.sh dpo > logs/train_dpo.log 2>&1
RUN_EVAL=0 bash scripts/baseline.sh token_temperature

bash scripts/evaluate.sh checkpoints/spl_alpha_1.0_beta_0.1 checkpoints/dpo_alpha_equals_beta checkpoints/dpo_token_temperature_1.5
