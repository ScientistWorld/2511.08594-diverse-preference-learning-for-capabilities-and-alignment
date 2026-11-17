# Research Direction

## Research Problem

Learning policies from pairwise preferences of a heterogeneous population so that the policy keeps
the population's diversity (proportional representation, coverage of less popular but valid
solutions, calibrated uncertainty) while staying coherent and at least as preferred as the
pre-alignment model.

## Core Contribution Scope

Preference-learning objectives, regularisation toward the reference policy, controls on the
entropy / temperature of the learned distribution, estimation under few and noisy comparisons,
parameter sharing across prompts, positions or tokens, and inference-time transformations of the
learned distribution are all in scope, as long as the result is a policy table per prompt learned
from the provided comparisons and reference policies.

## Where the reference method leaves room

- Finite comparisons per prompt with real survey response targets (two-outcome, multiple choice): the reference over-fits noisy or
  unanimous comparisons and becomes overconfident.
- Sequence coherence is credited only up to the population's own value, so further suppression alone cannot raise that metric.
- Sequence prompts: incoherent tokens are rarely seen in the comparisons, so they keep their
  pre-alignment probability; hard problems reward keeping mass on valid minority solutions; the
  population distribution itself is far from fully recovered.

## Out of Bounds

- Using evaluator ground truth (`eval/*/truth.json`) or external labels.
- Editing the evaluator, data files, gates or metric definitions.
- Producing artifacts that are not valid probability distributions or that omit prompts or
  experiments.
