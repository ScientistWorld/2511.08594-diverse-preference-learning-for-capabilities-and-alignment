# Final hardening

Verification is pending in one combined CPU job. The previous measured audit is archived in
[notes/HARDENING_previous.md](notes/HARDENING_previous.md); its links are relative to the repository root.

The changes address both findings in REVIEW.md:

- Categorical targets now come from real GlobalOpinionQA human response frequencies, conditioned
  on two or four positive-response options. Original survey questions are disjoint across both
  experiments and splits. Reference probabilities are independent of these targets. The old
  truncated-uniform/Dirichlet prior and square-root reference relation no longer generate answers.
- Scored coherence is min(policy coherence, population coherence), separately for each prompt.
  Actual and population coherence remain diagnostics. Suppressing population variation past that
  level cannot increase the scored metric.

Other small fixes: context records accept only the documented policy fields; hidden-named method
folders are scored; an empty submission gets complete worst-value records. Old scratch copies of
truth and test per-context results were moved under eval/test/runtime/. Reference and baseline
trainings now run sequentially, without a wall-clock stopping condition. No training method changed.

The job regenerates inputs twice, confirms unchanged sequence inputs, retrains SPL and both
baselines end to end, runs improved/alpha-1.2/undertrained variants, replays the affected shortcuts,
and checks paired ranking intervals. It uses the no-argument evaluation entry points on the full
cohort, then regenerates the public score files from the reference runs alone. Test paired values
are kept only under eval/test/. The cohort analysis includes normalization over all attempts,
including complete failed records, as well as honest-only and successful-only sensitivity analyses.

Before the fixes (previous audit): improved variant +0.160, truncation +0.155, alpha 1.2 +0.045,
undertrained -0.107, DPO -4.534, exact-prior shortcut +0.099. Reference gain is zero by definition;
its raw hidden-test scores are preserved in redteam/hardening_final/before_scores_test.json.
These gains used that audit's declared honest-run normalization and are not directly comparable
with a changed cohort's numerical gain scale.

Experiment weights and coefficients are unchanged. The paper's reported results are unchanged
under _paper_reported in scoring/reference.json. This remains a finite reproduction of the actual
SPL pairwise objective, not the paper's Mistral-7B LoRA empirical suite. Human-response calibration
is not factual MMLU/TruthfulQA accuracy; the sequence task remains synthetic. Methods observe noisy
comparisons for test contexts and estimate hidden distributions, rather than generalizing to
unseen prompts. Final scores, shortcut outcomes and headroom conclusions will be recorded after
the job returns.
