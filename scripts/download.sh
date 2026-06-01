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
# During reproduction, prefer downloading to shared directories
# (shared/datasets/, shared/models/) so other papers can reuse them.
# You do not need to run this script during reproduction — it exists
# for future portability. Write it to download to workspace-local
# paths (data/, checkpoints/) instead of the shared directories
# so a future user with just the workspace can run this script and get everything they need.

set -e

cd /home/user

echo "No external downloads are required for the tabular SPL reproduction."
python data/spl_benchmark.py --output data/spl_tabular_benchmark.json
