# Paper's Method

## Key Contribution

The paper introduces Soft Preference Learning (SPL), a preference-learning objective designed to reduce diversity loss caused by standard KL-regularized alignment methods such as RLHF and DPO.

## Approach

Standard RLHF and DPO couple two roles inside the KL penalty: keeping the trained policy close to a reference model and controlling the entropy of the trained policy. The paper argues that this coupling causes majority-preferred outputs to be overweighted, producing mode collapse and poor representation of minority preferences or alternative reasoning paths.

SPL separates these terms. Its RL-style objective optimizes expected reward plus an entropy bonus with coefficient `alpha`, while separately applying a cross-entropy penalty toward the reference policy with coefficient `beta`. The paper also gives a DPO-style pairwise objective:

`log sigmoid(alpha * log(pi(y|x) / pi(y'|x)) - beta * log(pi_ref(y|x) / pi_ref(y'|x)))`

where `y` is preferred to `y'`. Standard DPO is recovered when `alpha = beta`; increasing `alpha / beta` acts like a sequence-level, or global, temperature that flattens probabilities over complete responses rather than token-by-token.

## Main Claims

The primary claim is that SPL improves the diversity-quality tradeoff of aligned language models: it can produce more varied outputs and improve hard best-of-N problem solving without the rapid coherence degradation seen with token-level temperature scaling.

For the core mathematical reasoning result, SPL improves hard-split best-of-N accuracy on GSM8K and MATH when many samples are allowed. In Table 1, the strongest SPL setting outperforms DPO and temperature-scaled DPO at best-of-128 on both hard splits.

Secondary claims are that SPL improves semantic and lexical diversity in chat generation, better represents diverse societal viewpoints, and improves multiple-choice logit calibration while maintaining similar accuracy.
