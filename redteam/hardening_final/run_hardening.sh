#!/bin/bash
# One combined audit, after isolated reference/baseline reproduction.
# Never runs from evaluate_test.sh: the hidden test only scores raw artifacts.
set -euo pipefail
cd /home/user
H=redteam/hardening_final
RUNS=$H/runs/checkpoints
mkdir -p "$RUNS" "$H/logs" eval/test/runtime/final_audit

# All variants use the final public inputs. No stale config-based skip.
# Reference and baselines have already finished before this starts.
python3 method/spl.py --data-dir data --output-dir "$RUNS/variant_improved" \
  --label-smoothing 0.05 --label-smoothing-sequence 0 --shared-token-bias \
  > "$H/logs/variant_improved.log" 2>&1 &
pid_improved=$!
python3 method/spl.py --data-dir data --output-dir "$RUNS/variant_alpha_1.2" --alpha 1.2 \
  > "$H/logs/variant_alpha_1.2.log" 2>&1 &
pid_alpha=$!
python3 method/spl.py --data-dir data --output-dir "$RUNS/variant_undertrained" \
  --steps-categorical 300 --steps-sequence 150 > "$H/logs/variant_undertrained.log" 2>&1 &
pid_weak=$!
wait "$pid_improved"
wait "$pid_alpha"
wait "$pid_weak"
for m in spl_alpha_1.0_beta_0.1 dpo_alpha_equals_beta dpo_token_temperature_1.5; do
  rm -rf "$RUNS/$m"
  cp -r "checkpoints/$m" "$RUNS/$m"
done
python3 "$H/build_cases.py" --out "$RUNS"

# Exercise no-argument entry points with the whole cohort on disk.
# Restore the reference-only checkpoint tree even on an error.
mv checkpoints "$H/runs/reference_checkpoints"
ln -s "$PWD/$RUNS" checkpoints
restore_reference() {
  rm -f checkpoints
  mv "$H/runs/reference_checkpoints" checkpoints
}
trap restore_reference EXIT
bash scripts/evaluate_test.sh > "$H/logs/eval_test.log" 2>&1
cp scoring/scores_test.json "$H/scores_test_all.json"
bash scripts/evaluate_train.sh > "$H/logs/eval_train.log" 2>&1
cp scoring/scores_train.json "$H/scores_train_all.json"
# Raw paired statistics, kept exclusively in the hidden-test directory.
python3 eval/test/evaluate.py --per-context-out eval/test/runtime/final_audit/per_context_test.json \
  --scores-out "$H/scores_test_all_repeat.json" > "$H/logs/paired_test.log" 2>&1
cmp "$H/scores_test_all.json" "$H/scores_test_all_repeat.json"
rm "$H/scores_test_all_repeat.json"
python3 eval/train/evaluate.py --per-context-out "$H/runs/per_context_train.json" \
  --scores-out "$H/runs/scores_train_repeat.json" > "$H/logs/paired_train.log" 2>&1
cmp "$H/scores_train_all.json" "$H/runs/scores_train_repeat.json"
python3 "$H/analyze.py" --scores "$H/scores_test_all.json" \
  --per-context eval/test/runtime/final_audit/per_context_test.json \
  --train-scores "$H/scores_train_all.json" --bootstrap 1000 --out "$H/summary.json"
restore_reference
trap - EXIT

# Forged pre-written scores must be overwritten.
cp scoring/scores_test.json "$H/runs/scores_before_injection.json"
echo '{"experiments":{"two_outcome_mode_collapse":{"results":{"forged":{"preference_kl":-1e9}}}}}' > scoring/scores_test.json
bash scripts/evaluate_test.sh > "$H/logs/injection.log" 2>&1
python3 - <<'PY'
import json
s=json.load(open('scoring/scores_test.json'))
assert 'forged' not in s['experiments']['two_outcome_mode_collapse']['results']
print('[audit] injected scores overwritten')
PY

# Repeat deterministic categorical training with a differently seeded process.
PYTHONHASHSEED=71 python3 method/spl.py --data-dir data --output-dir "$H/runs/determinism_check" \
  --experiments two_outcome_mode_collapse,calibration_multiple_choice > "$H/logs/determinism.log" 2>&1
python3 - <<'PY' | tee redteam/hardening_final/determinism_check.txt
import json
for f in ('two_outcome.json', 'calibration.json'):
 a=json.load(open('redteam/hardening_final/runs/determinism_check/'+f))['contexts']
 b=json.load(open('checkpoints/spl_alpha_1.0_beta_0.1/'+f))['contexts']
 assert a==b, f
 print(f, 'policy tables identical under a changed process seed:',a==b)
PY
