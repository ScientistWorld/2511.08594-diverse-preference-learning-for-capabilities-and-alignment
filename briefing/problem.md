# Problem Definition

## Research Question

Preference-tuned language models often become too concentrated on a small set of high-probability responses. The benchmark problem is to learn from heterogeneous pairwise preferences while preserving a calibrated distribution over valid alternatives instead of collapsing all probability mass onto the plurality choice.

For the minimal core-claim reproduction, the task is a finite two-outcome social-choice setting: 80% of a population prefers one completion and 20% prefers another. A successful learner should keep the majority-preferred completion as the most likely output while still assigning probability to the minority-preferred completion in proportion to its support.

## Why It Matters

Many downstream uses of aligned models require representation of minority viewpoints or alternative solution paths. If alignment turns an 80/20 population preference into a nearly deterministic 100/0 model distribution, the model no longer represents the population it was trained to serve.

## Success Criteria

A successful solution should reduce majority over-weighting, increase the probability assigned to the minority-supported option, and preserve the majority option as the top-ranked output. This captures the benefit side of diversity and the constraint that the learned distribution should not invert the population's preference ordering.
