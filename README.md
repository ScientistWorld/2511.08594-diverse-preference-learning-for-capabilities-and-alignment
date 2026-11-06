# Diverse Preference Learning for Capabilities and Alignment

| This paper | Scientist World, the framework this environment belongs to |
|:---:|:---:|
| <a href="https://arxiv.org/abs/2511.08594"><img alt="This paper on arXiv" src="https://img.shields.io/badge/arXiv-2511.08594-b31b1b?logo=arxiv&logoColor=white"></a> | <a href="https://arxiv.org/abs/2611.08478"><img alt="Scientist World on arXiv" src="https://img.shields.io/badge/arXiv-2611.08478-b31b1b?logo=arxiv&logoColor=white"></a> <a href="https://github.com/zlab-princeton/scientist_world"><img alt="Scientist World code" src="https://img.shields.io/badge/Code-Scientist_World-181717?logo=github&logoColor=white"></a> <a href="https://github.com/zlab-princeton/scientist_world_cli"><img alt="Command line code" src="https://img.shields.io/badge/Code-Command_line-181717?logo=github&logoColor=white"></a> |

| | |
|---|---|
| **Paper ID** | [2511.08594](https://arxiv.org/abs/2511.08594) |
| **Domain** | Language model alignment / preference learning |
| **TL;DR** | Soft Preference Learning decouples entropy from reference-policy regularization so aligned LLMs can retain useful output diversity. |

## Summary

The paper studies why preference-tuned language models often lose diversity and become overconfident. It argues that the KL regularizer in RLHF and DPO both anchors the policy to a reference model and sharpens the policy distribution, which can amplify majority preferences and collapse alternatives. Soft Preference Learning separates these effects with independent entropy and cross-entropy coefficients, making sequence-level diversity controllable during preference learning. Empirically, the paper reports better diversity-quality tradeoffs, stronger hard best-of-N math performance, and improved calibration.

## Problem

Preference-tuned language models often become too concentrated on a small set of high-probability responses. The benchmark problem is to learn from heterogeneous pairwise preferences while preserving a calibrated distribution over valid alternatives instead of collapsing all probability mass onto the plurality choice.

For the minimal core-claim reproduction, the task is a finite two-outcome social-choice setting: 80% of a population prefers one completion and 20% prefers another. A successful learner should keep the majority-preferred completion as the most likely output while still assigning probability to the minority-preferred completion in proportion to its support.

## The paper's method

The paper introduces Soft Preference Learning (SPL), a preference-learning objective designed to reduce diversity loss caused by standard KL-regularized alignment methods such as RLHF and DPO.

## Key results of the paper

- In the paper's 80/20 two-outcome example with `beta=0.1`, standard preference learning assigns about 99.9999% probability to the majority option and about 0.0001% to the minority option.
- With the calibrated entropy setting analyzed by the paper, the target distribution recovers proportional representation of the 80/20 population preference.
- SPL is presented as a global, sequence-level alternative to token temperature scaling, intended to preserve coherence while increasing diversity.

## Run it

```bash
pip install sciworld
sw research 2511.08594 --mode innovate --agent claude-code
sw evaluate 2511.08594
```

## More

The full problem statement, method notes, evaluation protocol and reproduction log are in [DETAILS.md](DETAILS.md).

