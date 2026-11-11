#!/bin/bash
# Job entry point: reproduce the reference runs end to end, refresh the train /
# test / standard score files, and run the hardening audit (redteam/hardening/).

set -euo pipefail

cd /home/user

echo "[run] nproc=$(nproc)"
# the committed inputs and truth files are exactly what the hidden generator produces
python3 eval/test/generate_benchmark.py --root tmp/regen_check
for f in data/two_outcome_contexts.json data/sequence_contexts.json data/calibration_contexts.json eval/train/truth.json eval/test/truth.json; do
  python3 -c "import sys; sys.exit(open(sys.argv[1],'rb').read() != open(sys.argv[2],'rb').read())" "$f" "tmp/regen_check/$f" \
    && echo "[run] $f regenerates bit-identically" || echo "[run] WARNING: $f differs from regenerated copy"
done

# snapshot the previously packaged reference artifacts (if any) to check that retraining reproduces them
rm -rf tmp/prev_checkpoints && mkdir -p tmp/prev_checkpoints
for m in spl_alpha_1.0_beta_0.1 dpo_alpha_equals_beta dpo_token_temperature_1.5; do
  [ -d "checkpoints/$m" ] && cp -r "checkpoints/$m" "tmp/prev_checkpoints/$m"
done

bash scripts/reproduce.sh

python3 - <<'PY'
import json, os
for m in ("spl_alpha_1.0_beta_0.1", "dpo_alpha_equals_beta", "dpo_token_temperature_1.5"):
    for f in ("two_outcome.json", "sequence.json", "calibration.json"):
        old = f"tmp/prev_checkpoints/{m}/{f}"
        if not os.path.exists(old):
            print(f"[run] {m}/{f}: no previous artifact to compare")
            continue
        same = json.load(open(old))["contexts"] == json.load(open(f"checkpoints/{m}/{f}"))["contexts"]
        print(f"[run] {m}/{f}: retrained policy tables identical to packaged ones: {same}")
PY

bash redteam/hardening/run_hardening.sh
bash scripts/evaluate_train.sh
bash scripts/evaluate_test.sh
bash scripts/evaluate.sh
python3 validate.py --compare || echo "[run] validate.py reported issues"
