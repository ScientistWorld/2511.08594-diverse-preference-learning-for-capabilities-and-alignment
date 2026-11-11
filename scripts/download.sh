#!/bin/bash
# Download all data and models needed for this environment.
#
# Estimated total download size: 0 GB
# Estimated disk usage after extraction: 0 GB
#
# This environment has no external datasets or model weights.  Every input a
# method needs (reference policies and pairwise comparisons for all three
# experiments) is agent-authored, small, and tracked in git under data/:
#   data/two_outcome_contexts.json
#   data/sequence_contexts.json
#   data/calibration_contexts.json
# Nothing is fetched into data/downloads/ or shared/.  The script only checks
# that the tracked inputs are present and parse as JSON.

set -euo pipefail

cd /home/user

mkdir -p data/downloads
for f in data/two_outcome_contexts.json data/sequence_contexts.json data/calibration_contexts.json; do
  python3 -c "import json,sys; d=json.load(open(sys.argv[1])); print(sys.argv[1], len(d['contexts']), 'contexts')" "$f"
done
