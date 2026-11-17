# Problem Definition

## Research Question

Preference-tuned language models tend to collapse onto a few high-probability responses: a
population that is split 80/20 between two answers ends up with a model that says the majority
answer ~100% of the time, generations lose variety, repeated sampling stops finding minority
solution strategies, and answer probabilities become overconfident. The problem is to **learn a
policy from pairwise human preferences that represents the diversity of the population's
preferences while keeping outputs coherent and correct.**

## Setting

Every prompt comes with (a) a fixed *reference policy* (the pre-alignment model) and (b) a finite
set of *pairwise comparisons* "y preferred to y'" collected from a heterogeneous population of
raters. Nothing else about the population is observed. Three families of prompts are provided
(all in `data/`, each split into a visible `train` part and a held-out `test` part whose answers
are hidden):

1. **Two-outcome prompts** (601 prompts). Two possible completions; the population prefers one
   of them according to an empirical human response distribution; 32-128 sampled comparisons per prompt. The visible 80/20 example has 100 exact comparisons.
2. **Sequence prompts** (64 prompts). Completions are 7-token sequences over a 21-token
   vocabulary (about 1.8 billion possible completions, far too many to enumerate). Most tokens at
   each position are rare "incoherent" tokens (the analogue of non-words). Each prompt has 2000
   comparisons between pairs of reference-policy samples, judged by raters with
   sequence-level preferences.
3. **Multiple-choice prompts** (1200 prompts). Four survey response options with a hidden human response distribution; 40-120 sampled comparisons per prompt. Calibration measures probabilities of a randomly drawn human response, not factual correctness.

Categorical targets are empirical response frequencies from GlobalOpinionQA (Pew Global Attitudes
and World Values Survey), conditioned on a subset of two or four positive-response options.
Original questions are disjoint across both experiments and both splits. Question text, country,
source row and option mapping are withheld; only the preference observations are inputs.
Reference probabilities are independent of the empirical targets. The sequence task remains
synthetic. This is a finite preference-learning benchmark, not a reproduction of full LLM tasks.

A solution outputs, for every prompt (train and test), a full probability distribution over its
completions (a first-order chain over tokens for the sequence prompts).

## Why It Matters

Language models are now information sources for many people. If alignment converts a split
population into a near-deterministic answer, minority viewpoints disappear, repeated-sampling
problem solving wastes its budget on duplicates, and confidence stops reflecting uncertainty.

## Success Criteria

A better solution, on held-out prompts:

- represents the population's split instead of amplifying the majority (two-outcome and
  sequence-level distribution fit),
- solves more hard problems when 128 samples are drawn (it keeps probability on less popular but
  valid solutions),
- produces calibrated answer probabilities,

**without** giving up what the alignment step is for: the population's preferred completion must
stay on top, generations must stay coherent and at least as preferred by the population as the
reference policy's, and multiple-choice accuracy must not drop below what simply counting which
option wins the most comparisons achieves (with evaluator-defined slack).
