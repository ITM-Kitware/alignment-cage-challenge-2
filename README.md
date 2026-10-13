# Alignment of Autonomous Cyber Defense Agents via Adaptive Policy Shaping

Research code accompanying **Alignment of Autonomous Cyber Defense Agents via Adaptive Policy Shaping**. This repository extends CybORG CAGE Challenge 2 to study how autonomous defenders can follow operational priorities while retaining their ability to defend a network.

The paper shapes a pretrained reinforcement learning policy at test time using action scores for confidentiality, integrity, and availability (CIA). A fixed interpolation weight controls the trade-off between the PPO policy and an attribute-based policy. An adaptive variant switches between them as the observed network state changes. The experiments compare these approaches with training-time reward shaping across network sizes and attacker strategies.

## How this differs from CAGE Challenge 2

The original CAGE 2 task evaluates blue defenders against built-in red agents using the CAGE reward. This fork adds:

- **Operational objectives:** authentication, database, and front-end servers with CIA importance weights, attribute predictors, and resilience metrics. CIA-critical assets are excluded from the ordinary blue reward so that operational alignment and CAGE performance can be studied separately.
- **Alignment methods:** PPO blue agents, training-time alignment rewards, fixed test-time policy interpolation, and adaptive policy shaping.
- **Configurable scale:** scenario generation and experiments with 16, 31, 46, and 61 hosts; extended host addressing and action spaces support larger topologies.
- **Paper adversaries:** meandering attackers targeting CIA assets or CAGE assets, and B-line attackers targeting individual CIA priorities. Training uses a mixed meandering attacker.
- **Experiment tooling:** five-seed evaluation, trajectory and action-score exports, alignment analysis, trade-off plots, and optional Weights & Biases logging. The RL interfaces use Gymnasium.

The simulator and experiment scripts remain in their existing `CybORG/` layout. The [historical CAGE 2 guide](docs/CAGE2.md) retains the original challenge description and upstream references; its competition instructions are historical.

## Repository layout

| Location | Contents |
| --- | --- |
| `CybORG/CybORG/` | Simulator, agents, rewards, metrics, scenarios, and upstream tests |
| `CybORG/experiment_configs/` | Test-time, training-time, adaptive, scaling, and red-agent evaluation configurations |
| `CybORG/train_agents.py` | Train baseline or reward-shaped blue agents |
| `CybORG/test_experiment.py` | Evaluate baseline, fixed, and adaptive policies |
| `CybORG/test_experiment_training_time.py` | Evaluate agents aligned during training |
| `CybORG/alignment_analysis.py` | Compute alignment scores and analyze reward trade-offs |
| `CybORG/plot_*.py` | Plot gameplay, training curves, and method comparisons |
| `docs/RELEASE_REVIEW.md` | Artifact availability and remaining reproduction checks |

## Installation

Use Python 3.10 or newer. From this repository's root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
cd CybORG
```

Run the Python experiment commands below from `CybORG/`. Configuration paths are relative to that directory. Shell launchers locate it from their own location and use the active Python environment. No particular checkout location, shared drive, or user account is required.

## Training and evaluation

**Pretrained checkpoints and paper result data are not bundled.** Train checkpoints first, or supply a separately reviewed checkpoint release. Evaluating an untrained model is not a reproduction of the paper. See the [release review](docs/RELEASE_REVIEW.md) for the missing artifacts.

The complete baseline and training-time sweeps are:

```bash
bash experiment_configs/train_agents/train_base.sh
bash experiment_configs/train_agents/train_trainingtime_alignment.sh
```

These scripts cover four network sizes and training seeds `0 1 2 3 4`. Training budgets are 500,000, 1,000,000, 1,500,000, and 2,000,000 steps, respectively. Reward-shaped agents additionally sweep C, I, A and bonuses 0.5 through 3.0. These are substantial training jobs. To select a smaller run, set `CAGE_SCENARIOS`, `CAGE_TRAIN_SEEDS`, `CAGE_ATTRIBUTES`, or `CAGE_ALPHAS` as space-separated values before invoking a launcher.

After training, run an individual experiment and its analysis:

```bash
python test_experiment.py --config experiment_configs/base.yaml
python alignment_analysis.py --config experiment_configs/base.yaml
python test_experiment.py --config experiment_configs/adaptive_alpha_config/base/cia.yaml
python test_experiment_training_time.py --config experiment_configs/training_time_config/base.yaml
```

Evaluations use held-out seed 42 and run for 100 episodes. The four scenario horizons are 200, 300, 400, and 500 steps. `bash experiment_configs/run_all.sh` runs the fixed and adaptive matrix; `bash experiment_configs/run_training_time_experiments.sh` runs training-time evaluation and analysis. See the [configuration guide](CybORG/experiment_configs/README.md) for method and adversary mappings, and the [release review](docs/RELEASE_REVIEW.md) for settings to reconcile before exact numerical reproduction.

Checkpoints are written under `CybORG/checkpoints/`; experiment results under `CybORG/results/experiments/`. These generated artifacts are ignored by Git. Action-space mappings already reside beside the evaluation scenarios and can be regenerated with `python export_action_space_mapping.py` without a trained checkpoint.

Training runs locally without a hosted logging account by default. To enable Weights & Biases, add `--wandb` to `train_agents.py` (or set `CAGE_WANDB=1` for a training launcher), and configure your own `WANDB_PROJECT` and `WANDB_ENTITY`. Exported training histories for plotting belong in `CybORG/results/wandb_training/`.

## Paper scope and attribution

This release covers the submitted paper's original CIA/resilience metrics and its heuristic red-agent experiments. Later technique-based CIA labeling, evolutionary-agent experiments, and additional learned red agents are excluded.

The code is licensed under the Apache License, Version 2.0 (see [LICENSE](LICENSE)).

The codebase builds on [CAGE Challenge 2](https://github.com/cage-challenge/cage-challenge-2) and CybORG. Upstream simulator components retain their original copyright and license terms under the MIT License and CC0 Public Domain Dedication (see [NOTICE](NOTICE) and [CybORG/LICENSE.txt](CybORG/LICENSE.txt)). Citation information for the underlying environment is in the [CybORG guide](CybORG/README.md#citing-this-project). The supplied manuscript is anonymized, so author names and final publication metadata are not inferred here.

## Disclaimer
We emphasize that our work should be considered academic research, as we cannot fully guarantee model outputs are free of inaccuracies or biases that may pose risks if relied upon for decision-making.

This material is based upon work supported by the Defense Advanced Research Projects Agency and the Air Force Research Laboratory, under contract number FA8650-23-C-7316. Any opinions, findings and conclusions or recommendations expressed in this material are those of the author(s) and do not necessarily reflect the views of AFRL or DARPA.