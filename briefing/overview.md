# Overview

- **Paper ID:** 2511.08594
- **Title:** Diverse Preference Learning for Capabilities and Alignment
- **Domain:** Language model alignment / preference learning
- **TL;DR:** Soft Preference Learning decouples entropy from reference-policy regularisation so
  aligned models keep useful output diversity.

## Short Summary

KL-regularised preference learning (RLHF, DPO) both anchors the policy to a reference model and
sharpens it, so a split population becomes a near-deterministic majority answer. SPL separates the
entropy and cross-entropy terms, giving a sequence-level temperature that restores proportional
representation, keeps generations coherent better than token-level temperature, improves hard
best-of-N problem solving and improves calibration.

## This Gym

A scaled synthetic reproduction with hidden ground truth: two-outcome social-choice prompts,
7-token sequence generation over a 21-token vocabulary with hidden verifier sets and incoherent
tokens, and four-option multiple-choice calibration. All training is pure-Python, CPU-only, and
deterministic. See `scoring/EXPERIMENTS.md` and `briefing/evaluation.md`.
