# Overview

- **Paper ID:** 2511.08594
- **Title:** Diverse Preference Learning for Capabilities and Alignment
- **Domain:** Language model alignment / preference learning
- **TL;DR:** Soft Preference Learning decouples entropy from reference-policy regularization so aligned LLMs can retain useful output diversity.

## Short Summary

The paper studies why preference-tuned language models often lose diversity and become overconfident. It argues that the KL regularizer in RLHF and DPO both anchors the policy to a reference model and sharpens the policy distribution, which can amplify majority preferences and collapse alternatives. Soft Preference Learning separates these effects with independent entropy and cross-entropy coefficients, making sequence-level diversity controllable during preference learning. Empirically, the paper reports better diversity-quality tradeoffs, stronger hard best-of-N math performance, and improved calibration.

## Key Results

- In the paper's 80/20 two-outcome example with `beta=0.1`, standard preference learning assigns about 99.9999% probability to the majority option and about 0.0001% to the minority option.
- With the calibrated entropy setting analyzed by the paper, the target distribution recovers proportional representation of the 80/20 population preference.
- SPL is presented as a global, sequence-level alternative to token temperature scaling, intended to preserve coherence while increasing diversity.
