# Alignment of Autonomous Cyber Defense Agents via Adaptive Policy Shaping

Code for our paper **Alignment of Autonomous Cyber Defense Agents via Adaptive Policy Shaping**. This repository extends CybORG CAGE Challenge 2 to study how autonomous defenders can be steered towards operational priorities while retaining their ability to defend the network.

## Differences from the original CybORG CAGE Challenge 2

The original CAGE 2 challenge evaluates blue defenders against built-in red agents using the CAGE reward. This modified version adds operational objectives, modified attacker policies, and alignment methods for blue defenders. The simulator and experiment scripts remain in the `CybORG/` layout. The [original CAGE 2 guide](docs/CAGE2.md) retains the original challenge description.

### Modified scenario and resilience metrics

The modified network includes authentication (`Auth`), database (`Database`), and front-end (`Front`) servers in addition to the user, enterprise, operational, and defender hosts. These servers represent operational priorities expressed through confidentiality (C), integrity (I), and availability (A):

| Server type | Operational priorities |
| --- | --- |
| Authentication (`Auth`) | Confidentiality, integrity, availability |
| Database (`Database`) | Confidentiality, availability |
| Front-end (`Front`) | Integrity, availability |

Resilience scoring is adapted from Cadet et al. **[Quantitative Resilience Modeling for Autonomous Cyber Defense](https://rlj.cs.umass.edu/2025/papers/Paper99.html)**. The implementation builds on the associated [CyberResilience code](https://github.com/xcadet/CyberResilience).


Our [resilience metric](CybORG/CybORG/AlignmentMetric/resilience_measure.py) records CIA drops at each step and combines them into a normalized resilience drop, with default weights of 0.6 for confidentiality, 0.2 for integrity, and 0.2 for availability. It also scores candidate blue actions for the attribute-based policy and alignment analysis. The generated scenarios assign CIA-critical servers zero ordinary CAGE host reward values, allowing operational alignment and general network defense performance to be evaluated separately.

Additional internal changes made to the CAGE 2 codebase are detailed in [experiment README](CybORG/experiment_configs/README.md#differences-from-the-original-cage-2).

### CAGE 2 red agent modifications

The original Meander and B-line attackers are extended to target the added operational servers and support host names across the generated network sizes. Evaluation uses five adversaries:

| Paper adversary | Python class | Impact targets |
| --- | --- | --- |
| M-CIA | `RedMeanderAgent_Combination_CIA` | CIA-critical `Auth`, `Database`, and `Front` servers |
| M-CAGE | `RedMeanderAgent_Combination_CAGE` | Other eligible compromised hosts contributing to CAGE performance |
| Bline-C | `B_lineAgent_Resilience_C` | `Auth` or `Database` servers |
| Bline-I | `B_lineAgent_Resilience_I` | `Auth` or `Front` servers |
| Bline-A | `B_lineAgent_Resilience_A` | `Auth`, `Database`, or `Front` servers |

Training uses `RedMeanderAgent_Combination`, which selects between CIA and CAGE impact targets with equal probability at each impact-selection decision. The B-line variants select a target server type once per episode and resolve matching hosts from observations. See [Meander_Resilience.py](CybORG/CybORG/Agents/SimpleAgents/Meander_Resilience.py) and [B_line_resilience.py](CybORG/CybORG/Agents/SimpleAgents/B_line_resilience.py).


## Installation

Use Python 3.10 or newer and [install uv](https://docs.astral.sh/uv/getting-started/installation/). From the repository root:

```bash
uv sync
cd CybORG
```

`uv sync` creates the project environment and installs CybORG and its dependencies. Run scripts with `uv run ...` and shell launchers with `uv run bash ...`; manual environment activation is not required.


## Repository experiment scripts

| Location | Contents |
| --- | --- |
| `CybORG/CybORG/` | Simulator, agents, rewards, metrics, scenarios, and upstream tests |
| `CybORG/experiment_configs/` | Test-time, training-time, adaptive, scaling, and red-agent evaluation configurations |
| `CybORG/train_agents.py` | Train baseline or reward-shaped blue agents |
| `CybORG/test_experiment.py` | Evaluate baseline, fixed, and adaptive policies |
| `CybORG/test_experiment_training_time.py` | Evaluate agents aligned during training |
| `CybORG/alignment_analysis.py` | Compute alignment scores and analyze reward trade-offs |
| `CybORG/CybORG/Shared/Scenarios/scenario_generation.py` | Generate scenarios with configurable host populations and reward settings |

## Training

Use [train_agents.py](CybORG/train_agents.py) to train a PPO defender against the mixed meandering attacker. `PPOAgent` trains the baseline; `AlignedPPOAgent` adds a reward term for the selected CIA attribute. The training-time bonus is called lambda in the paper and `alpha` in the CLI only; it is separate from the test-time interpolation weight.

To train one baseline checkpoint for `base`:

```bash
uv run python train_agents.py \
  --scenario scalability_experiments/base \
  --agent_type PPOAgent \
  --red_agent RedMeanderAgent_Combination \
  --alpha 0 --seed 0 --game_steps 200 --training_iter 500000 \
  --agent_export checkpoints/PPO_base_0_seed0
```

To train a defender aligned to confidentiality with a bonus of 1.5:

```bash
uv run python train_agents.py \
  --scenario scalability_experiments/base \
  --agent_type AlignedPPOAgent \
  --red_agent RedMeanderAgent_Combination \
  --alignment_attribute C --alpha 1.5 \
  --seed 0 --game_steps 200 --training_iter 500000 \
  --agent_export checkpoints/PPO_base_1.5_C_seed0
```

For the complete training sweeps:

```bash
uv run bash experiment_configs/train_agents/train_base.sh
uv run bash experiment_configs/train_agents/train_trainingtime_alignment.sh
```

These launchers cover `base`, `hosts_2`, `hosts_3`, and `hosts_4` with training seeds `0 1 2 3 4`. Checkpoints are saved under `CybORG/checkpoints/` as `PPO_{scenario}_0_seed{seed}.zip` for baselines and `PPO_{scenario}_{bonus}_{attribute}_seed{seed}.zip` for reward-shaped agents. After saving a checkpoint, `train_agents.py` also runs a short evaluation. Logging is disabled by default; add `--wandb` to a Python training command to enable.

## Evaluation and analysis

Evaluation scripts load trained checkpoints and run the agents against the configured red adversary. The supplied configurations use 100 episodes, an evaluation seed starting at 42, and five training seeds. Episode horizons are 200, 300, 400, and 500 steps for `base`, `hosts_2`, `hosts_3`, and `hosts_4`, respectively.

[test_experiment.py](CybORG/test_experiment.py) evaluates Blue Monitor and PPO baselines alongside fixed or adaptive test-time alignment:

```bash
# Baselines and fixed test-time interpolation
uv run python test_experiment.py --config experiment_configs/base.yaml
uv run python alignment_analysis.py --config experiment_configs/base.yaml

# Adaptive test-time policy selection
uv run python test_experiment.py --config experiment_configs/adaptive_alpha_config/base/cia.yaml
uv run python alignment_analysis.py --config experiment_configs/adaptive_alpha_config/base/cia.yaml
```

[test_experiment_training_time.py](CybORG/test_experiment_training_time.py) evaluates the reward-shaped checkpoints:

```bash
uv run python test_experiment_training_time.py --config experiment_configs/training_time_config/base.yaml
uv run python alignment_analysis.py --config experiment_configs/training_time_config/base.yaml
```

Train all checkpoints requested by a configuration before running it. To run the full evaluation and analysis sweeps:

```bash
uv run bash experiment_configs/run_all.sh
uv run bash experiment_configs/run_training_time_experiments.sh
uv run bash experiment_configs/export_alignment_analysis.sh
```

See the [configuration guide](CybORG/experiment_configs/README.md) for the method and adversary configuration.


## Scenario generation

Generated scenarios are already included, so generation is not required to run the supplied experiments:

- [Training scenarios](CybORG/CybORG/Shared/Scenarios/scalability_experiments/) are in `CybORG/CybORG/Shared/Scenarios/scalability_experiments/` and use `AlignedReward` for blue agents.
- [Evaluation scenarios](CybORG/CybORG/Shared/Scenarios/scalability_eval_scenarios/) are in `CybORG/CybORG/Shared/Scenarios/scalability_eval_scenarios/` and use `HybridAvailabilityConfidentiality` to measure ordinary CAGE performance.

Both folders contain the following paper scenarios. Each has three subnets (User, Enterprise, and Operational); totals include one defender host. Operational hosts below count `Op_Server` and `Op_Host` together, separately from the CIA-critical servers.

| Scenario file | User hosts | Enterprise hosts | Operational hosts | Auth | Database | Front | Defender | Total hosts |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `base.yaml` | 5 | 3 | 4 | 1 | 1 | 1 | 1 | 16 |
| `hosts_2.yaml` | 10 | 6 | 8 | 2 | 2 | 2 | 1 | 31 |
| `hosts_3.yaml` | 15 | 9 | 12 | 3 | 3 | 3 | 1 | 46 |
| `hosts_4.yaml` | 20 | 12 | 16 | 4 | 4 | 4 | 1 | 61 |


Use [scenario_generation.py](CybORG/CybORG/Shared/Scenarios/scenario_generation.py) to create a custom scenario. For example, this creates a new 31-host training scenario:

```bash
uv run python CybORG/Shared/Scenarios/scenario_generation.py \
  --export_path CybORG/Shared/Scenarios/scalability_experiments/custom \
  --operational_n 8 --enterprise_n 6 --user_n 10 --resilience_n 2 \
  --blue_reward AlignedReward
```

`--resilience_n` is the number of **each** CIA-critical server type, so `2` creates two Auth, two Database, and two Front servers. The generator also accepts reward levels (`--operational_w`, `--enterprise_w`, `--user_w`, `--resilience_w`) and subnet placement options (`--operational_s`, `--enterprise_s`, `--user_s`, `--resilience_s`, `--defender_s`). Subnet indices are Enterprise `0`, Operational `1`, and User `2`. The current generator keeps these three subnets; `--subnets` does not yet change their number.

The existing training-scenario launcher can regenerate `base` through `hosts_5`, overwriting those training YAML files:

```bash
uv run bash CybORG/Shared/Scenarios/generate_resilience_scenario.sh
```

Action-space mappings for the four paper evaluation scenarios are already included in `CybORG/CybORG/Shared/Scenarios/scalability_eval_scenarios/action_spaces/`. Regenerate them without a trained checkpoint using:

```bash
uv run python export_action_space_mapping.py
```

## Citation

If you find this work useful, please consider citing our paper:

```bibtex
@inproceedings{mujtaba2026autonomous,
  title={Alignment of Autonomous Cyber Defense Agents via Adaptive Policy Shaping},
  author={Dena Mujtaba and Bharadwaj Ravichandran and Brian Hu and Arslan Basharat},
  booktitle={Proceedings of the 2026 Conference on Applied Machine Learning for Information Security},
  year={2026}
}
```

The code is licensed under the Apache License, Version 2.0 (see [LICENSE](LICENSE)). The codebase builds on [CAGE Challenge 2](https://github.com/cage-challenge/cage-challenge-2) and CybORG. Upstream simulator components retain their original copyright and license terms under the MIT License and CC0 Public Domain Dedication (see [NOTICE](NOTICE) and [CybORG/LICENSE.txt](CybORG/LICENSE.txt)). 

## Disclaimer
We emphasize that our work should be considered academic research, as we cannot fully guarantee model outputs are free of inaccuracies or biases that may pose risks if relied upon for decision-making.

This material is based upon work supported by the Defense Advanced Research Projects Agency and the Air Force Research Laboratory, under contract number FA8650-23-C-7316. Any opinions, findings and conclusions or recommendations expressed in this material are those of the author(s) and do not necessarily reflect the views of AFRL or DARPA.
