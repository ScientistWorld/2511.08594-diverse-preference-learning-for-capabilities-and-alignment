#!/bin/bash
set -euo pipefail
cd /home/user
bash redteam/honest_prior_fix/run.sh
bash redteam/honest_train_selected/run.sh
bash redteam/shortcut_uniform/run.sh
bash redteam/shortcut_brute_force/run.sh
python redteam/analyze.py
