# Experiment Overview

## two_outcome_mode_collapse

Tests whether a learned preference policy represents a heterogeneous population rather than collapsing to the majority-supported option. The experiment measures both the probability retained for a minority-supported output and whether the model still preserves the population's top-ranked output, so improvements cannot come from simply flattening or inverting the distribution.

## tabular_preference_learning

Tests whether the same preference-learning setup extends beyond two choices to several prompts with four possible outputs each. The experiment measures distributional fit, repeated-sampling success on minority-supported useful outputs, and whether the population's top option remains top-ranked.
