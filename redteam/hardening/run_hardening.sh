#!/bin/bash
# Hardening audit: honest variants, baselines and shortcuts, all scored by the
# unmodified hidden-test evaluator (eval/test/evaluate.py) in one cohort.
# Expects the reference runs in checkpoints/ (scripts/reproduce.sh).
# Resumable: a trained variant is skipped when its config.json exists.

set -uo pipefail

cd /home/user

H=redteam/hardening
RUNS=$H/runs/checkpoints
mkdir -p "$RUNS" "$H/logs"
MAXJ="${MAXJ:-$(nproc)}"
echo "[hardening] parallel slots: $MAXJ"

run_bg() {
  while [ "$(jobs -rp | wc -l)" -ge "$MAXJ" ]; do wait -n; done
  "$@" &
}

train_variant() {
  local name="$1"; shift
  if [ -f "$RUNS/$name/config.json" ]; then
    echo "[hardening] $name already trained"
    return 0
  fi
  echo "[hardening] training $name: $*"
  python3 method/spl.py --data-dir data --output-dir "$RUNS/$name" "$@" > "$H/logs/$name.log" 2>&1 \
    && echo "[hardening] $name done" || echo "[hardening] $name FAILED (see $H/logs/$name.log)"
}

# --- honest variants (better / weaker) -------------------------------------
run_bg train_variant variant_improved --label-smoothing 0.05 --label-smoothing-sequence 0 --shared-token-bias
run_bg train_variant variant_alpha_1.2 --alpha 1.2
run_bg train_variant variant_beta_0.01 --beta 0.01
run_bg train_variant variant_alpha_0.5 --alpha 0.5
run_bg train_variant variant_undertrained --steps-categorical 300 --steps-sequence 150
wait

# --- reference runs and post-hoc baselines into the same cohort ------------
for m in spl_alpha_1.0_beta_0.1 dpo_alpha_equals_beta dpo_token_temperature_1.5; do
  rm -rf "$RUNS/$m"
  cp -r "checkpoints/$m" "$RUNS/$m"
done
python3 baseline/token_temperature.py --source-dir checkpoints/dpo_alpha_equals_beta \
  --output-dir "$RUNS/dpo_token_temperature_2.0" --temperature 2.0
python3 baseline/token_temperature.py --source-dir checkpoints/dpo_alpha_equals_beta \
  --output-dir "$RUNS/dpo_token_temperature_3.0" --temperature 3.0

# --- shortcut / control cases ----------------------------------------------
python3 $H/build_cases.py --ref-dir checkpoints/spl_alpha_1.0_beta_0.1 --out "$RUNS"

# --- score everything with the hidden-test and train evaluators -------------
python3 eval/test/evaluate.py --checkpoints "$RUNS" --scores-out "$H/scores_test_all.json" \
  --per-context-out "$H/runs/per_context_test.json" > "$H/logs/eval_test.log" 2>&1
python3 eval/train/evaluate.py --checkpoints "$RUNS" --scores-out "$H/scores_train_all.json" \
  --per-context-out "$H/runs/per_context_train.json" > "$H/logs/eval_train.log" 2>&1
grep "^\[test\]" "$H/logs/eval_test.log"

python3 $H/analyze.py --scores "$H/scores_test_all.json" --per-context "$H/runs/per_context_test.json" \
  --train-scores "$H/scores_train_all.json" --bootstrap 400 --out "$H/summary.json"

# --- score injection: a pre-written scores_test.json must be overwritten ----
cat > scoring/scores_test.json <<'JSON'
{"experiments": {"two_outcome_mode_collapse": {"results": {"spl_alpha_1.0_beta_0.1": {"preference_kl": -1e9}}}}}
JSON
bash scripts/evaluate_test.sh > "$H/logs/score_injection.log" 2>&1
python3 - <<'PY' | tee "$H/score_injection_check.txt"
import json
s = json.load(open("scoring/scores_test.json"))
v = s["experiments"]["two_outcome_mode_collapse"]["results"]["spl_alpha_1.0_beta_0.1"]["preference_kl"]
print("injected value overwritten:", v != -1e9, "value now", v)
PY

# --- determinism: retrain the categorical experiments and compare bytes ------
python3 method/spl.py --data-dir data --output-dir "$H/runs/determinism_check" \
  --experiments two_outcome_mode_collapse,calibration_multiple_choice > "$H/logs/determinism.log" 2>&1
{
  python3 -c "import sys; sys.exit(open(sys.argv[1],'rb').read() != open(sys.argv[2],'rb').read())" "$H/runs/determinism_check/two_outcome.json" checkpoints/spl_alpha_1.0_beta_0.1/two_outcome.json \
    && echo "two_outcome.json bit-identical on retrain" || echo "two_outcome.json DIFFERS on retrain"
  python3 -c "import sys; sys.exit(open(sys.argv[1],'rb').read() != open(sys.argv[2],'rb').read())" "$H/runs/determinism_check/calibration.json" checkpoints/spl_alpha_1.0_beta_0.1/calibration.json \
    && echo "calibration.json bit-identical on retrain" || echo "calibration.json DIFFERS on retrain"
} | tee "$H/determinism_check.txt"
echo "[hardening] done"
