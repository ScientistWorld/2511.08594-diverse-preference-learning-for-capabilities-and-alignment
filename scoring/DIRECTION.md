# Research Direction

## Research Problem

This benchmark studies preference learning under heterogeneous population preferences. The core question is whether alignment from pairwise preferences can avoid majority mode collapse while preserving the correct preference ordering.

## Core Contribution Scope

Scientists should work on preference-learning objectives or policy-optimization procedures that control the learned distribution over outputs. Improvements should address how the learner balances reward/preference fit, entropy, and reference-policy anchoring.

## Approach Scope

In-bounds approaches may change the preference objective, regularization coefficients, optimization procedure, or policy parameterization as long as they still learn from the provided pairwise preference data and produce a probability distribution over the same two outputs.

## Out of Bounds

- Switching to a different task such as answer classification, supervised label prediction, or generative language modeling for the reported score.
- Using external data or additional preference labels to infer the target distribution.
- Hardcoding the evaluator's target probabilities into output artifacts instead of learning from the provided comparisons.
- Changing the benchmark distribution or metric definitions for the reported experiment.
