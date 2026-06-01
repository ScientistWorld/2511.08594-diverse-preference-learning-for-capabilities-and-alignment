#!/bin/bash
# Download all data and models needed for this environment.
#
# Run on a node with internet access BEFORE building the container
# or submitting compute jobs.
#
# This script should be idempotent — safe to run multiple times.
# It should skip files that already exist.
#
# Update the storage estimate below as you add downloads.
#
# Estimated total download size: 0 GB
# Estimated disk usage after extraction: 0 GB
#
# No external data or models are needed. The two-outcome benchmark is tracked
# under data/, and the optional multi-outcome/calibration benchmarks are
# generated locally.

set -e

cd /home/user

echo "No external downloads are required for the finite preference-learning reproduction."
test -f data/two_outcome_benchmark.json
python data/multi_outcome_benchmark.py --output data/multi_outcome_benchmark.json
python data/calibration_benchmark.py --output data/calibration_benchmark.json
