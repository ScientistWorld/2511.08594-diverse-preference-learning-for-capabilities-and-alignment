> **Superseded.** This file describes the gym before the iteration-11 hardening (old tabular benchmark, `method/train_tabular_spl.py`, `eval/evaluate_tabular.py`). The current gym is described in `briefing/`, `scoring/` and `HARDENING.md`.

# Verification Report

## Initial Assessment

The workspace claimed `core_claim_plus`. The implementation is a finite-action reproduction of the paper's preference-learning objective, with a baseline special case, artifact-based evaluation, train/test split evaluators, and three scored experiments. The claimed milestone is justified for the scaled finite setting.

## Issues Found

- Minor: problem-facing data and evaluator descriptions still contained paper-method terms and an old method-specific benchmark filename.
- Cosmetic: root contains pre-existing system/device entries (`README.md`, `manifest.json`) and ignored runtime artifacts; these were not introduced by this verification and were left untouched.

## Fixes Applied

- Renamed the problem-facing finite multi-outcome benchmark from method-specific filenames to `data/multi_outcome_benchmark.py` and `data/multi_outcome_benchmark.json`.
- Reworded data, evaluation, and script descriptions so `briefing/problem.md`, `briefing/evaluation.md`, `scoring/EXPERIMENTS.md`, `data/`, and `eval/` remain method-agnostic.
- Re-ran `bash scripts/reproduce.sh`, `bash scripts/evaluate_train.sh`, `bash scripts/evaluate_test.sh`, and `python validate.py --compare`; all checks passed.
- Confirmed there are no `shared/` references in committed code, `eval/` and `data/` do not import from `method/`, and `scripts/download.sh` has no external URLs to verify because all benchmark data is generated locally.

## Final Judgment

The workspace is ready to serve as a finite-action research gym. Improve-mode agents can compare against the packaged method and baseline, and from-scratch agents can use the method-agnostic problem, data, and evaluators without importing from `method/`. The main limitation is scale: this gym reproduces the paper's computational mechanism on finite synthetic benchmarks, not the full LLM LoRA experiments.

- Milestone: core_claim_plus
- Ready for gym use: yes
- Confidence: medium
- Key limitation: finite synthetic data only
