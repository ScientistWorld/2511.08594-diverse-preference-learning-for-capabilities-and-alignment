#!/bin/bash
# Public model inputs are tracked under data/. The upstream survey includes
# test answers, so its only local copy is in eval/test/assets/ (never data/).
# No packages or computation are needed here. Run before submitting a job.
set -euo pipefail
cd /home/user
mkdir -p eval/test/assets data/downloads
asset=eval/test/assets/global_opinions.csv
if [ ! -s "$asset" ]; then
  curl --fail --location --retry 3 \
    https://huggingface.co/datasets/Anthropic/llm_global_opinions/resolve/cb2880488749218abb81802a94c2c62ebfde2f35/data/global_opinions.csv \
    --output "$asset.part"
  mv "$asset.part" "$asset"
fi
echo '789e99ebb0eb1d0768de565ec80fab7a25bfef5baa1cccebdda75777eea86051  eval/test/assets/global_opinions.csv' | sha256sum --check
for f in data/two_outcome_contexts.json data/sequence_contexts.json data/calibration_contexts.json; do
  python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); print(sys.argv[1],len(d["contexts"]),"contexts")' "$f"
done
