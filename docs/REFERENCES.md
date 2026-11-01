# References

The papers and tools the implementation is based on, what each was used for, and where. Before citing in a report, check the details (authors, venue, year) against the original paper and your proposal's reference list.

## Attacks (step 2)

| Reference | Used for | Where |
|---|---|---|
| Gu, T., Dolan-Gavitt, B. and Garg, S. (2017). *BadNets: Identifying Vulnerabilities in the Machine Learning Model Supply Chain.* arXiv:1708.06733. | the backdoor attack: a small patch trigger plus a changed label | `attacks/backdoor.py`, RESULTS exp 02 |
| Pang, T., Yang, X., Dong, Y., Su, H. and Zhu, J. (2021). *Accumulative Poisoning Attacks on Real-time Data.* NeurIPS 2021. | the accumulative attack in online learning (partially reproduced at our scale), and the online-learning setting of the response episodes | `attacks/accumulative.py`, `bench/online.py`, RESULTS exp 04 |
| Geiping, J., Fowl, L., Huang, W. R., Czaja, W., Taylor, G., Moeller, M. and Goldstein, T. (2021). *Witches' Brew: Industrial Scale Data Poisoning via Gradient Matching.* ICLR 2021. | the gradient-based, clean-label targeted attack | `attacks/gradient_matching.py`, RESULTS exp 06 |
| Cinà, A. E. et al. (2023). *Wild Patterns Reloaded: A Survey of Machine Learning Security against Training Data Poisoning.* ACM Computing Surveys. | the attack taxonomy; label-flip poisoning | `attacks/labelflip.py`, RESULTS exp 03 |
| Wu, B. et al. (2022). *BackdoorBench: A Comprehensive Benchmark of Backdoor Learning.* NeurIPS 2022, Datasets and Benchmarks. | reference settings and protocol for backdoor experiments (poison rate, trigger, ASR) | RESULTS exp 02 |

## Detection and classification (steps 3–4)

| Reference | Used for | Where |
|---|---|---|
| Kingma, D. P. and Welling, M. (2014). *Auto-Encoding Variational Bayes.* ICLR 2014. | the VAE behind the picture check (reconstruction error as the anomaly score) | `detection/vae.py`, RESULTS exp 07–08 |
| Chernozhukov, V. et al. (2018). *Double/debiased machine learning for treatment and structural parameters.* The Econometrics Journal. | the cross-fitting idea: score each trusted image with a model that never saw it | `detection/folds.py`, RESULTS exp 09, 13 |
| Population Stability Index and the Kolmogorov–Smirnov test (standard statistical drift checks) | the drift baselines MLEcho is compared with, and the label-mix feature | `bench/baselines/drift.py`, `detection/label_check.py`, RESULTS exp 12 |

## Response and PPO (steps 5, 7)

| Reference | Used for | Where |
|---|---|---|
| Schulman, J., Wolski, F., Dhariwal, P., Radford, A. and Klimov, O. (2017). *Proximal Policy Optimization Algorithms.* arXiv:1707.06347. | the learned response policy | `response/ppo.py`, `experiments/19_train_ppo.py` |
| Raffin, A. et al. (2021). *Stable-Baselines3: Reliable Reinforcement Learning Implementations.* Journal of Machine Learning Research 22(268). | the PPO implementation | `experiments/19_train_ppo.py` |
| Towers, M. et al. (2024). *Gymnasium: A Standard Interface for Reinforcement Learning Environments.* arXiv:2407.17032. | the environment interface of the practice world | `bench/env/env.py` |

## Data, models and tools

| Reference | Used for | Where |
|---|---|---|
| Krizhevsky, A. (2009). *Learning Multiple Layers of Features from Tiny Images.* Technical report, University of Toronto. | CIFAR-10, the test bench dataset | `bench/datasets.py` |
| He, K., Zhang, X., Ren, S. and Sun, J. (2016). *Deep Residual Learning for Image Recognition.* CVPR 2016. | ResNet-18, the second victim model (overhead, final evaluation) | `bench/models.py`, RESULTS exp 10 |
| Zaharia, M. et al. (2018). *Accelerating the Machine Learning Lifecycle with MLflow.* IEEE Data Engineering Bulletin. | experiment tracking and the guard's event log | `tracking/mlflow_logger.py`, all experiments |

## Considered, not used

| Reference | Why not |
|---|---|
| Memorization Discrepancy (Zhu et al., 2023), a defence against accumulative poisoning | planned as a baseline (ROADMAP step 3); not reproduced, since the accumulative attack itself was only partly reproduced at our scale |
