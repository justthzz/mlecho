# Research framing

The research questions, gaps, hypotheses and objectives from the project proposal, and where the repo answers each one so far. The plan itself is the proposal; how the build differs from it, and why, is in [ROADMAP → Changes from the proposal](ROADMAP.md#changes-from-the-proposal). Papers: [REFERENCES.md](REFERENCES.md). Numbers: [RESULTS.md](RESULTS.md).

## Problem

Current MLOps pipelines lack a mechanism to detect training data poisoning attacks in real time during active training and to respond to them automatically, leaving accumulative and adaptive poisoning strategies undetected until a model is already compromised.

## Gaps (hedged, as in the proposal)

| Gap | Statement | Closest prior work |
|---|---|---|
| G1 Timing | Limited evidence was found of systems that detect poisoning in the live training stream **and** respond to it automatically inside an MLOps pipeline. | Zhu et al. (2023), in-training discovery of accumulative poison via model dynamics; Lakshmikantha and Giri (2026), streaming detect-and-respond on event data; Li et al. (2021), training-time backdoor isolation |
| G2 Method | Limited evidence was found of CNN-VAE reconstruction error used as an in-training, batch-level poisoning signal, or of its features used to classify the attack type. | Archa and Kartheeban (2024, 2026), offline autoencoder/GAN poisoning detection |
| G3 Response | RL security response has been studied mainly in network settings and mostly in simulation; limited evidence was found of it inside ML training pipelines, compared with a deterministic baseline. | Hammar (2026), CSLE |

## Research questions, hypotheses and evidence so far

| RQ | Hypothesis | Evidence in the repo | Status |
|---|---|---|---|
| **RQ1** To what extent can a CNN-VAE batch-level detector identify training data poisoning in real time during training, compared with existing defences? | **H1** Higher recall than a statistical drift baseline at a comparable false-positive rate | Exp 07, 08, 13 (picture check: backdoor AUC 0.994; 88.9% of poisoned images caught at 0.91% false alarms); exp 09 (label check); exp 12 (PSI/KS drift checks caught **none** of the attacks); exp 11 (false alarms under harmless change); exp 10 (overhead) | Answered for the backdoor, slow-drip and distinct-class label-flips; **not** for gradient matching (never detected) or cat → dog at low rates. Comparison with post-training defences not run |
| **RQ2** How accurately can a classifier distinguish the attack types using detector-derived features? | **H2** Accuracy varies by attack type; clean-label and accumulative attacks are harder than the backdoor | Exp 14: clean 97%, backdoor 86.5%, label-flip 98%, slow-drip 76.5–99% (1–5 per batch); gradient matching can't be classified because it is never detected | Answered; H2 supported. The built classifier's types differ from the proposal's (see ROADMAP) |
| **RQ3** Does a deterministic rule-based response restore training integrity after a detected poisoning event? | **H3** Faster recovery than no intervention | Exp 15 (final seeds 1–3): backdoor ASR 99.8% → 1.4%, slow-drip 97–100% → ~1%, recovered within 20–100 batches; label-flip damage halved; 0 false actions on clean data; harmless change escalated, no data deleted | Answered; H3 supported for trigger attacks, partly for label-flip |
| **RQ4** (enhancement) Does a PPO-trained response agent improve recovery relative to the rule-based baseline? | **H4** PPO matches or exceeds the rules | Exp 18–19 (practice world): copy-then-PPO −3.88, PPO from scratch −4.22, rules −4.69. Exp 15 on unseen seeds 4–6: copy-then-PPO matches the rules (backdoor 1.5% vs 1.7%, the rest identical); PPO from scratch fails slow-drip (97.3%) | Answered: H4 supported in the "matches" sense with a rules warm start; PPO alone does not transfer from practice to the real attacks. One agent seed each |

## Objectives

| Objective | Where |
|---|---|
| RO1 Critically analyse existing defences to establish the real-time gap | The proposal's literature review; `REFERENCES.md` |
| RO2 Define requirements (FR-M1–M4, FR-S1, NFR-P1, NFR-R1, NFR-A1) | The proposal §3.5.3; status in the ROADMAP |
| RO3 Design the detection, classification and rule-based response architecture | ROADMAP steps 3–5; `ARCHITECTURE.md` |
| RO4 Implement the detector, classifier, response and MLOps integration | `src/mlecho/` (steps 3–6) |
| RO5 Evaluate detection, response, recovery and overhead | RESULTS exp 07–16 |
| RO6 (enhancement) Design, implement and evaluate a PPO response agent | Step 7, exp 18–19 |
| RO7 Analyse the legal, ethical, social and professional implications | The proposal, chapter 4 |
