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
# No external data or models are needed. The benchmark data are small,
# agent-authored JSON files tracked under data/.

set -e

cd /home/user

echo "No external downloads are required for the tabular SPL reproduction."
python data/spl_benchmark.py --output data/spl_tabular_benchmark.json
