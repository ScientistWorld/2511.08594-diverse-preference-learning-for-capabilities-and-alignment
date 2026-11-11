# Data

Public inputs for all three experiments (agent-authored, deterministic, tracked in git). Each file
holds a list of `contexts`, each with an `id` and a `split` (`train` = visible scoring slice,
`test` = hidden scoring slice). A method must output a policy for every context of both splits.

- `two_outcome_contexts.json`: `reference_probability` (2 numbers) and `comparisons` as
  `[winner, loser, count]` over option indices 0/1. The first train context,
  `two_outcome_paper_80_20`, is the paper's 80/20 example (uniform reference, 100 comparisons).
- `calibration_contexts.json`: `reference_probability` (4 numbers) and `comparisons`
  `[winner, loser, count]` over options 0-3.
- `sequence_contexts.json`: `vocab_size` 21, `length` 7, `token_alphabet` (token i is character i);
  per context `reference_initial` (21), `reference_transitions` (6 x 21 x 21) and 2000 `pairs`
  `[winner_sequence, loser_sequence]` encoded as 7-character strings.

Ground truth is not stored here.
