# Scoring audit runs

`bash redteam/<case>/run.sh` installs that case's actual policy artifacts,
runs the visible evaluator, and invokes **`bash scripts/evaluate_test.sh`
with no arguments** in a separate process. It saves both scores and exit codes
inside the case folder, then restores the checkpoint inventory and scoring
files. `method.py`, `config.json`, and `common.py` implement each candidate;
`runner.py` provides only isolation and evaluator invocation.

Run all cases on a scheduled CPU job with `bash redteam/run_all.sh`, then
`python redteam/analyze.py`. No package installation, GPU, network, evaluator
modification, Python-environment modification, or persistent process is used.
The original trained artifacts are cached under the ignored `.cache/` folder
for the annotation-only shortcuts. On a fresh clone, first produce reference
artifacts with the gym's normal reproduction script.

Candidate implementations access only the shipped data, method (improve-mode
cases), and visible evaluation module. They do not consult paper text, hidden
test source, hidden context lists, initial test scores, or rank coefficients.
The audit analyzer reads those scores and coefficients only after evaluation.
All contexts in the shipped data are treated identically by each candidate,
except `shortcut_single_context`, which retains a **visible training context**.
The shortcut names describe scoring deficiencies even where direct attempts
are rejected. A zero-gain successful shortcut is explicitly a paying tie.

`summary.json` and each `rank.json` report raw metrics and the weighted,
direction-correct, absolute-coefficient normalized improvements. Since the
variance of future attempts is not available, they declare the audit cohort,
also use the honest-check cohort, and give two-attempt comparisons. Zero is
the initial hidden-test reference. Invalid runs have no claimed numeric rank.
