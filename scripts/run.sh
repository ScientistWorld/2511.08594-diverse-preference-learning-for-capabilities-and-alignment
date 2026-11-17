#!/bin/bash
# Final hardening verification: all fixes checked together in one CPU job.
# Reference/baseline training is isolated and has no wall-clock stopping rule.
set -euo pipefail
cd /home/user
mkdir -p logs eval/test/runtime/final_regen
python3 - <<'PY'
import hashlib,json
from pathlib import Path
p=Path('eval/test/runtime/sequence_before_sha256.json')
p.write_text(json.dumps({str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in
 [Path('data/sequence_contexts.json')]}))
PY
python3 eval/test/generate_benchmark.py
python3 - <<'PY'
import hashlib,json
from pathlib import Path
old=json.loads(Path('eval/test/runtime/sequence_before_sha256.json').read_text())
assert all(hashlib.sha256(Path(f).read_bytes()).hexdigest()==h for f,h in old.items())
print('[run] sequence inputs unchanged; only categorical inputs replaced')
PY
# All generated copies, including answers and lookup maps, stay under eval/test/.
python3 eval/test/generate_benchmark.py --root eval/test/runtime/final_regen
for f in data/two_outcome_contexts.json data/sequence_contexts.json data/calibration_contexts.json eval/train/truth.json eval/test/truth.json eval/test/survey_provenance.json; do
  cmp "$f" "eval/test/runtime/final_regen/$f"
done
bash scripts/reproduce.sh
bash redteam/hardening_final/run_hardening.sh
python3 eval/test/verify_integrity.py
# Restore public baseline files from the reference/baseline runs alone.
bash scripts/evaluate_train.sh
bash scripts/evaluate_test.sh
bash scripts/evaluate.sh
python3 scripts/package_reference.py
python3 validate.py --compare
