# Paper's Method

## Key Contribution

Soft Preference Learning (SPL) decouples the two roles of the KL regulariser in RLHF/DPO: the
cross-entropy to the reference policy (prior strength, coefficient `beta`) and the entropy of the
learned policy (diversity, coefficient `alpha`).

## Approach

DPO-style SPL objective (paper Eq. 5), maximised over the policy:

`E_{(y_w > y_l)} [ log sigmoid( alpha * log pi(y_w)/pi(y_l) - beta * log pi_ref(y_w)/pi_ref(y_l) ) ]`

Its optimum is `pi(y) ∝ pi_ref(y)^(beta/alpha) * exp(r(y)/alpha)` (Prop. A.2); standard DPO is
`alpha = beta`, whose optimum `pi_ref * p^(1/beta)` raises population preferences to the 10th power
at `beta = 0.1` (Prop. 3.1). `alpha = 1` gives proportional representation (Cor. 3.1);
`alpha/beta` acts as a sequence-level (global) temperature on top of the DPO policy, unlike
token-level temperature, which rescales every next-token distribution.

## This gym's implementation (`method/spl.py`)

- Policy per prompt: tabular first-order chain (softmax logits per row), initialised at the
  reference policy; the length-1 case for two-outcome and multiple-choice prompts.
- Full-batch gradient descent on the mean SPL loss over the comparisons: 3000 steps at lr 1.0
  (categorical prompts), 1500 steps at lr 5.0 (sequence prompts). Deterministic.
- Reference run: `alpha = 1.0`, `beta = 0.1` (global temperature 10). DPO baseline:
  `alpha = beta = 0.1` (`baseline/run_dpo.py`); token-temperature baseline:
  `baseline/token_temperature.py`.
- Optional variant knobs (off in the reference): `--label-smoothing`,
  `--label-smoothing-sequence`, `--shared-token-bias`.

## Main Claims

SPL avoids majority mode collapse, keeps generation diversity with less quality loss than
token-level temperature, improves best-of-N accuracy on hard problems, and improves calibration
relative to DPO.
