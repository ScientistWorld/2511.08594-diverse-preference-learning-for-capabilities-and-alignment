# Final hardening

The improved variants beat the reference on paired hidden-test comparisons. DPO and reduced
training score lower. Forged fields, omitted experiments, impossible probabilities and empty
submissions now receive complete worst-value records. Coherence above the population's own
level earns no extra credit.

**One requested result remains unmet:** the old categorical-prior estimator still beats SPL,
although its gain is smaller after replacing synthetic targets with real survey responses.
The strict requirement that this reviewed shortcut must not pay has not been achieved. A
smaller positive gain is not counted as a closed shortcut.

Evidence: combined verification job [`2e18944d-0e8`](jobs/2e18944d-0e8/stdout.log), exit 0;
[aggregate results and paired intervals](redteam/hardening_final/summary.json);
[case builder](redteam/hardening_final/build_cases.py). The scheduler preempted the first
attempt; the successful attempt restarted the full pipeline. Reference and baseline training
completed sequentially before the variant audit. `python validate.py --compare` passed.

## Changes after REVIEW.md

- **Replaced the known categorical generating prior.** Targets now use empirical human response
  frequencies from [GlobalOpinionQA](https://huggingface.co/datasets/Anthropic/llm_global_opinions),
  conditioned on two or four positive-response options. There are 1,800 distinct original survey
  questions, disjoint across both experiments and both splits. Country, question and option
  mapping remain hidden. Reference probabilities are independent of the targets. The previous
  truncated-uniform/Dirichlet targets and noisy square-root relationship no longer generate
  answers. The pinned source CSV, generator and provenance live only under `eval/test/`.
- **Capped coherence credit.** Each prompt scores `min(policy coherence, population coherence)`.
  Both probabilities are computed by dynamic programming from evaluator-owned labels/potentials
  and the submitted policy. Uncapped values remain diagnostics. A regression case with perfect
  token suppression confirms that its credit equals the population value exactly.
- **Closed output and inventory gaps.** Extra context fields are rejected. Hidden-named method
  directories are included. An empty checkpoint tree emits a failed `missing_submission` record.
  An error or task-gate failure anywhere assigns every experiment's worst values to that method.
- **Removed accidental answer copies.** Scratch truth, generator simulations and per-context test
  results outside `eval/test/` were removed or moved there. Only aggregate audit results remain
  in `redteam/`.
- **Reproduced final baselines in isolation.** SPL and both DPO baselines were rerun end to end on
  final data, sequentially. No method path stops on a wall-clock budget. Input comparisons and
  iteration counts are fixed. The SPL objective and optimisation procedure did not change.

All fixes preceded the single combined job. It regenerated inputs/truth/provenance twice and
compared them byte for byte; sequence inputs stayed unchanged from the preceding gym. It trained
the reference/baselines and three variants, replayed affected shortcuts together, checked complete
failures, exercised both no-argument evaluation entry points, checked score-file injection, and
repeated categorical reference training under a changed process seed. Policy tables were identical;
see [determinism_check.txt](redteam/hardening_final/determinism_check.txt).

## Ranking and scientific ordering

`gain = sum(experiment_weight * abs(coefficient) * direction * (value - reference) / sigma)`

Direction is +1 for `higher_is_better` and -1 otherwise. Sigma is population standard deviation
across the attempts being ranked; zero-variance metrics contribute zero. The primary final
column uses **all 33 audited attempts, including 14 complete failed records**. Honest-only and
successful-only normalisations are sensitivity checks, not additional scored metrics. Only
`results` in freshly evaluated score files supply ranked numbers.

Intervals use 1,000 paired bootstrap resamples of prompts. Method and reference receive the same
indices; sigma is fixed. Sequence metrics are exact expectations, with no sampling noise.
Training is deterministic. These intervals measure finite-test uncertainty, not uncertainty over
all possible tasks or ranking cohorts. Before gains used the previous audit's different cohort,
excluding failures, so gain magnitudes across columns are not directly comparable.

| Variant | Before: previous honest sigma | After: honest sigma | After: all-attempt sigma [95% paired CI] | Train: all-attempt sigma |
|---|---:|---:|---:|---:|
| Reference SPL | 0 | 0 | 0 | 0 |
| Improved: shared token bias and categorical label smoothing | +0.160 | +0.264 | **+0.02518 [+0.01314, +0.03576]** | +0.01598 |
| Min-p-style truncation of reference | +0.155 | +0.266 | **+0.03605 [+0.03426, +0.03784]** | +0.03499 |
| Alpha 1.2 | +0.045 | -0.012 | +0.02093 [+0.00654, +0.03277] | +0.01752 |
| Reduced training: 300 categorical / 150 sequence steps | -0.107 | -0.230 | -0.02697 [-0.05948, +0.00504] | -0.00318 |
| DPO | -4.534 | -3.977 | **-1.24768 [-1.33561, -1.16533]** | -1.13994 |
| DPO + token temperature 1.5 | -3.254 | -3.171 | **-0.91773 [-0.99950, -0.84324]** | -0.75848 |

The improved and truncation variants establish headroom beyond paired uncertainty. DPO is a
clearly weaker scientific control. Reduced training has a negative measured gain, but its
all-attempt interval crosses zero; this test distinguishes it less firmly than DPO. Alpha 1.2
trades coherence for distribution fit and calibration, so its ordering changes with cohort metric
variances. No universal ordering of such tradeoffs is claimed.

## Major shortcuts: before, fix and after

Every final case passed through the same no-argument hidden-test entry point as ordinary methods.
Final gains below use all-attempt sigma. A failed run has gain **-3.89874**, from explicit worst
values in every metric, rather than skipped rank entries. Historical REDTEAM and previous-audit
figures are distinguished because the task and normalisation changed.

| Case | Before | Defence or change | Final outcome |
|---|---:|---|---:|
| Uniform output | REDTEAM +0.42947 | Majority, population-reward and response-accuracy gates; failure propagates to every experiment | Failed, -3.89874 |
| Enumerate every output / coherent-uniform policy | REDTEAM +0.42947 | 21^7 sequence space; hidden verifier patterns and population reward unavailable to private search | Coherent uniform fails; 30% mixture -0.05719 |
| Agent-written targets, references, success flags, samples and metrics | REDTEAM up to +0.63647; previous audit paying tie 0 | Evaluator-owned truth and reference policies; reject undocumented context fields | Failed, -3.89874 |
| Single context or omitted experiment/manifest | REDTEAM +0.15251 / paying tie 0 | All required prompts/experiments checked; disk inventory independent of score files/manifests | Failed, -3.89874 |
| Hidden-named incomplete directory | Not previously tested | Included in inventory | Failed, -3.89874 |
| NaN, null, negative, unnormalised, empty output | REDTEAM wrote no fresh results | Strict probabilities and complete worst-value records | Failed, -3.89874 each |
| Pre-written score injection | REDTEAM did not improve real output | Overwrite scores from raw policies | Forged entry removed; genuine SPL 0 |
| Copy visible answers | REDTEAM +0.01358 | Test truth and anything regenerating it only under `eval/test/` | Train memorisation gives no test improvement: 0 |
| Fixed confidence on SPL argmax | Previous -0.02754; half mixture +0.01429 | Brier primary, ECE secondary; empirical response targets | -0.00873; half mixture -0.00091 |
| Laplace win counting | Previous +0.00814 | Richer empirical targets, same evaluator-owned gates | -0.02018 |
| Invert sequence reference without comparisons | Previous -1.15706 | Hidden reward potentials not recovered by rowwise squaring | -0.52030 |
| Write down old categorical generator as prior | Previous **+0.09885** | Replace target family and remove target/reference relationship | **Still +0.00854: not closed** |
| Excess coherence from truncation | Previous raw 0.9999; population 0.9961 | Per-prompt population cap | Raw 0.9999; scored 0.996083; no excess credit |

The train-memorisation control deliberately retains genuine SPL outputs on test prompts; these
explain its test tie. Its train gain is +0.06381. It does not show that memorising train answers
solves hidden contexts. Truncation's remaining gain is an ordinary decoding improvement: it also
improves best-of-128 and distribution fit. Further suppression cannot earn extra coherence credit.

## Unresolved shortcuts and limitations

**The reviewed old-prior case still pays.** Its honest-normalised gain fell from
**+0.09885 [0.07778, 0.11746]** to **+0.02466 [0.00719, 0.04203]**. Under final all-attempt sigma it
is **+0.00854 [0.00296, 0.01329]**. The calibration-only component is +0.00839; binary-only is
+0.00015 with an interval spanning zero. Calibration Brier is 0.62229 versus SPL's 0.62309,
and ECE is 0.02401 versus 0.05318. It retains SPL's sequence output.

The old prior is no longer the generating process and its square-root reference likelihood is
now wrong. Nevertheless its shrinkage still yields a measured improvement on empirical responses.
That improvement was not removed by special-casing the estimator, rejecting valid policies, or
making scores disagree with Brier/ECE. The requested non-paying outcome is unachieved; it is not
delegated to a later shortcut grader as though fixed. A harder empirical estimation or
language-model task remains needed to satisfy that requirement fully.

Other limits left in place:

- Finite categorical tasks admit short statistical estimators. Real targets do not make these
  tasks equivalent to LLM training. The sequence task remains synthetic and token rarity is easy
  to identify from the public reference.
- Min-p-style truncation remains a simple successful alternative. It reaches the coherence cap,
  but not bounds on distribution fit or best-of-128. Its gain slightly exceeds the trained
  improvement in this cohort; the richer training variant does not dominate decoding changes.
- Methods observe noisy comparisons for test contexts and estimate hidden populations. This
  tests estimation, not generalisation to unseen prompts. Train-only memorisation does not
  transfer to hidden contexts.
- Train checks only train prompts. An invalid test-only table is caught by test evaluation.
- The audit establishes outcomes for the replayed private-search cases, not every possible
  search. Hidden reward/verifier answers have no public objective oracle. Full LLM diversity,
  natural-language quality and mathematical reasoning are not established here.

## Reference before and after

Reference gain is zero by definition. Previous raw scores are preserved in
[before_scores_test.json](redteam/hardening_final/before_scores_test.json). Final
[`scores_test.json`](scoring/scores_test.json), [`scores_train.json`](scoring/scores_train.json)
and [`scores.json`](scoring/scores.json) contain only the isolated reference and two baseline
runs, not shortcut or variant entries.

| Hidden-test reference quantity | Before | After |
|---|---:|---:|
| Two-outcome KL | 0.0377616 | 0.00729553 |
| Minority log-ratio error | 0.512554 | 0.225800 |
| Hard best-of-128 | 0.386275 | 0.386275 |
| Sequence population KL | 0.249488 | 0.249488 |
| Coherence | 0.969756 | 0.969756 |
| Four-option Brier | 0.644800 | 0.623087 |
| Four-option ECE | 0.0678132 | 0.0531802 |
| Response accuracy (gate, coefficient 0) | 0.479749 | 0.493438 |

DPO's final test binary KL is 1.63950, hard best-of-128 0.0874553, sequence KL 7.71187, Brier
0.910139 and ECE 0.432080. SPL improves population representation, repeated-sampling success and
calibration in the paper's direction. DPO has higher coherence, so that quality tradeoff remains
measured. All three reproduced methods pass every gate. The exact visible 80/20 example still
gives SPL minority probability 0.200000 and DPO 9.5430e-7, matching the analytical result.

## Headroom and paper coverage

**No scored reference metric is at its bound.**

| Scored metric | Reference | Bound / population operating point |
|---|---:|---:|
| Two-outcome KL | 0.00729553 | 0 |
| Minority log-ratio error | 0.225800 | 0 |
| Hard best-of-128 | 0.386275 | At most 1 |
| Sequence population KL | 0.249488 | 0 |
| Coherence | 0.969756 | Mean per-prompt cap 0.996083 |
| Expected Brier | 0.623087 | Population optimum 0.595320 |
| ECE | 0.0531802 | 0 |

Brier's optimum is the fixed test mean of `1 - sum(population_probability**2)`, not zero:
human responses have irreducible variation. No tested variant saturates all scored metrics.
The trained improvement attains Brier 0.617529 and ECE 0.0117293; truncation raises hard
best-of-128 to 0.394388. Both beat reference rank beyond paired uncertainty. DPO baselines are
clearly below SPL, but the positive old-prior result qualifies the shortcut requirement.

No experiment was added in this review pass. The preceding hardening's sequence experiment
remains the scored headroom task: 7-token chains, 21 tokens, 2,000 comparisons per prompt,
32 hidden prompts and six hard verifier patterns per prompt away from the population mode.
Uniform enumeration fails the reward gate; its tested partial mixture loses. This is a finite
version of useful diversity versus coherence and hard repeated sampling, not GSM8K/MATH.

The code implements the paper's actual Eq. 5 SPL pairwise objective, with DPO recovered by
alpha = beta. It does not reproduce Mistral-7B LoRA training on HH-RLHF/UltraFeedback, factual
MMLU/TruthfulQA calibration or natural-language generation. Calibration now measures real human
response agreement, as documented in the briefings. Milestone remains `core_claim_plus`;
no full empirical-paper claim is added.

## Packaging and weights

Weights remain 0.25 / 0.5 / 0.25. Coefficients are unchanged from the previous hardening:
distribution fit, useful sequence diversity and population-capped coherence are scored; Brier
is primary, ECE secondary, accuracy is gate-enforced with coefficient zero. Coefficients were
not tuned to reverse an observed attack's rank in this pass.

Paper-reported numbers remain unchanged under `_paper_reported` in
[`reference.json`](scoring/reference.json). Its scored `results` now contain final reproduced
gym numbers at full precision rather than incomparable paper-scale targets.

[`scripts/run.sh`](scripts/run.sh) again reproduces and evaluates the three reference methods
end to end. [`scripts/verify_hardening.sh`](scripts/verify_hardening.sh) preserves the combined
audit. Hidden test reads raw policy artifacts and never trains. Public fetching is pinned and
checksum-verified in `scripts/download.sh`; no committed code references `shared/` or builds
process/environment defences. Test per-context values remain only under `eval/test/runtime/`;
aggregate shortcut evidence is in [the audit table](redteam/hardening_final/summary.md).
