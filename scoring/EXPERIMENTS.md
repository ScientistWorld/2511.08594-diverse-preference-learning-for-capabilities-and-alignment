# Experiment Overview

All three experiments are scored on held-out test prompts whose answers are visible only to the
hidden evaluator. A run that fails any experiment (invalid output or a failed task gate) gets the
worst value in every experiment.

## two_outcome_mode_collapse (weight 0.25)

Establishes whether preference learning *represents* a split population or *amplifies* its
majority. Each prompt has two completions and a population that prefers one of them 55-95% of
the time; only a few dozen noisy comparisons are observed. The measurement is how far the learned
split is from the population split (KL and the log-factor by which the minority is
under- or over-represented), with the constraint that the majority completion stays on top.

## sequence_diversity_best_of_n (weight 0.5)

Establishes whether diversity can be kept at the level of whole generations without losing
coherence, and whether that diversity pays off in repeated-sampling problem solving. Completions
are token sequences from a space far too large to enumerate, most tokens are incoherent, and
preferences are expressed over whole sequences. The measurement is (i) the chance that 128 samples
contain a verified solution for hard problems whose solutions are valid but not the most popular
completions, (ii) sequence-level distribution fit to the population, and (iii) coherence, with the
constraint that generations are at least as preferred by the population as the pre-alignment
model's.

## calibration_multiple_choice (weight 0.25)

Establishes whether learned answer probabilities are calibrated with respect to which answer is
actually correct, rather than overconfident. The measurement is calibration error and Brier score
against the hidden correct answer, plus accuracy, with the constraint that accuracy does not drop
below the pre-alignment model's.
