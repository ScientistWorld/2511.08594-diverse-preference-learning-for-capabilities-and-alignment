# Data

Public inputs for all three experiments (deterministic, tracked in git). Categorical comparisons
use real human response distributions from GlobalOpinionQA; sequence comparisons are synthetic. Each file
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

The raw upstream survey includes test answers and is fetched only into `eval/test/assets/` by
`scripts/download.sh` (pinned revision and SHA-256). Neither the upstream responses nor their
source-row/option mappings are inputs to a method. Reference distributions for categorical
contexts are independent of the targets. Original questions are disjoint across all slices and
both categorical experiments. The hidden generator and survey provenance live only in `eval/test/`.

Source: [GlobalOpinionQA](https://huggingface.co/datasets/Anthropic/llm_global_opinions),
[dataset paper](https://arxiv.org/abs/2306.16388); CC BY-NC-SA 4.0. Survey-derived targets measure
human response representation/calibration, not correctness on MMLU or TruthfulQA.
