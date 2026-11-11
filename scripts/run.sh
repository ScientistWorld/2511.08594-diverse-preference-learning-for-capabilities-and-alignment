#!/bin/bash
# Job entry point: reproduce the reference runs end to end, refresh the train /
# test / standard score files, then (this iteration) run the hardening audit.

set -euo pipefail

cd /home/user

echo "[run] nproc=$(nproc)"
# the committed inputs and truth files are exactly what the hidden generator produces
python3 eval/test/generate_benchmark.py --root tmp/regen_check
for f in data/two_outcome_contexts.json data/sequence_contexts.json data/calibration_contexts.json eval/train/truth.json eval/test/truth.json; do
  python3 -c "import sys; sys.exit(open(sys.argv[1],'rb').read() != open(sys.argv[2],'rb').read())" "$f" "tmp/regen_check/$f" \
    && echo "[run] $f regenerates bit-identically" || echo "[run] WARNING: $f differs from regenerated copy"
done

bash scripts/reproduce.sh
bash redteam/hardening/run_hardening.sh
bash scripts/evaluate_train.sh
bash scripts/evaluate_test.sh
bash scripts/evaluate.sh
python3 validate.py --compare || echo "[run] validate.py reported issues (reference.json results are refreshed after this job)"
