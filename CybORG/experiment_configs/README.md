# Experiment Configuration Guide

This guide explains how to choose and customize the experiment YAML files. See the main README for [installation](../../README.md#installation), [training](../../README.md#training), [evaluation commands](../../README.md#evaluation-and-analysis), and [scenario generation](../../README.md#scenario-generation).

Run commands from the outer `CybORG/` directory. Configuration paths are relative to that directory unless noted below.

## Choose a Configuration

Replace `{scenario}` with `base`, `hosts_2`, `hosts_3`, or `hosts_4`, and choose one value from each brace-delimited list. Paths below are relative to this `experiment_configs/` folder. The [red-agent descriptions](../../README.md#cage-2-red-agent-modifications) explain each adversary.

| Method / opponent | Configuration |
| --- | --- |
| Fixed test-time / M-CIA | `{scenario}.yaml` |
| Fixed test-time / M-CAGE | `cage_agent_config/{scenario}.yaml` |
| Fixed test-time / B-line | `bline_agent_config/{C,I,A}/{scenario}.yaml` |
| Adaptive / all opponents | `adaptive_alpha_config/{scenario}/{cia,cage,bline_C,bline_I,bline_A}.yaml` |
| Training-time / M-CIA | `training_time_config/{scenario}.yaml` |
| Training-time / M-CAGE | `training_time_config/{scenario}_cage.yaml` |
| Training-time / B-line | `training_time_config/bline_agent_config/{scenario}_{C,I,A}.yaml` |

Use `test_experiment.py` for fixed and adaptive policies, and `test_experiment_training_time.py` for policies aligned during training. Both include Blue Monitor and PPO baselines.

## Example Configuration and Parameters


The example below evaluates fixed test-time policy shaping on `base` against M-CIA. It follows [base.yaml](base.yaml), makes optional settings explicit, and uses a separate output directory. It requires the five baseline checkpoints `checkpoints/PPO_base_0_seed0.zip` through `checkpoints/PPO_base_0_seed4.zip`.

Save this YAML as `experiment_configs/example.yaml`:

```yaml
seed: 42  # Starting evaluation seed; episode e uses seed + e.

paths:
  experiment_dir: results/experiments/RedMeanderAgent_Combination_CIA/example
  export_path: test_results.csv  # Aggregate results filename inside experiment_dir.
  scenario: scalability_eval_scenarios/base  # Scenario path without .yaml.
  blue_agent_base_path: checkpoints/PPO_base_0.zip  # Template for seeded checkpoints.

env:
  episodes: 100  # Episodes evaluated for each policy and training checkpoint.
  steps: 200  # Maximum steps per episode.
  verbose: false  # Enable detailed trajectory messages and printed result summaries.
  average_metric: false  # Use sums for the metric's running CIA score accessor.
  save_trajectories: false  # Export episode trajectories as JSON.
  auto_plot_trajectories: false  # Plot saved trajectories; requires save_trajectories.

agents:
  red_agent: RedMeanderAgent_Combination_CIA  # Attacker class used by the simulator.
  blue_agent: PPOAgent  # Class used to load the baseline PPO checkpoint.
  train_seeds: [0, 1, 2, 3, 4]  # Checkpoint training seeds to evaluate and aggregate.

alignment:
  adaptive: false  # Use a fixed interpolation weight instead of adaptive selection.
  alpha_range:
    start: 0.25  # First interpolation weight, inclusive.
    stop: 1.0  # Last interpolation weight, inclusive.
    num: 4  # Number of evenly spaced weights: 0.25, 0.5, 0.75, 1.0.
  attributes_to_run: [C, I, A]  # Evaluate a separate aligned policy for each attribute.
  use_kneepoint: false  # Replace the sweep with per-attribute weights from knee_point.json.
  resilience_weights:
    C: 0.6  # Confidentiality's weight in the combined resilience-drop metric.
    I: 0.2  # Integrity's weight in the combined resilience-drop metric.
    A: 0.2  # Availability's weight in the combined resilience-drop metric.

analysis:
  metric: Reward  # Compare alignment with Reward or Resilience Drop.
  training_time: false  # Use test-time policy names when locating and labeling results.
  recalculate: true  # Recompute alignment scores from .npz exports.
  alpha_sampling_range: 4  # Analyze four evenly selected entries from alpha_range.
  calculate_kneepoint: true  # Calculate knee points and write knee_point.json.
  action_space_file: CybORG/Shared/Scenarios/scalability_eval_scenarios/action_spaces/base.json
```

Run evaluation first, then analysis:

```bash
uv run python test_experiment.py --config experiment_configs/example.yaml
uv run python alignment_analysis.py --config experiment_configs/example.yaml
```

### Parameter Details

- **Paths:** `paths.scenario` is relative to `CybORG/Shared/Scenarios/` and omits `.yaml`. Checkpoint and action-map paths are relative to the outer `CybORG/` directory. Match the scenario, checkpoint network size, and `analysis.action_space_file`; the JSON mapping lets analysis identify and filter `Remove` actions.
- **Outputs:** use a distinct `paths.experiment_dir` for each experiment. `paths.export_path` is relative to it unless absolute; keep the filename `test_results.csv` because the analysis script reads that name directly.
- **Checkpoints:** with multiple `agents.train_seeds`, the evaluator adds `_seed{seed}` before `.zip`. With one seed, supply the actual suffixed filename. This selects existing checkpoints; it does not train agents. Keep `agents.blue_agent: PPOAgent` for the supplied test-time experiments.
- **Metric averaging:** `env.average_metric: true` averages the metric's running CIA score accessor. It does not change final CSV aggregation: the runner sums C, I, A, and reward over steps and averages resilience drop.
- **Policy weights:** alpha `0` uses PPO and alpha `1` uses the attribute-based policy. `alignment.resilience_weights` only changes the combined resilience-drop metric; use nonnegative weights with at least one positive value.
- **Knee points:** `alignment.use_kneepoint: true` requires a prior `knee_point.json` in the experiment directory. Generate it by evaluating an alpha sweep and running analysis with `analysis.calculate_kneepoint: true`.
- **Analysis sampling:** `analysis.alpha_sampling_range` counts evenly selected entries from the alpha sweep. Keep it between `1` and `alignment.alpha_range.num` to avoid duplicate entries. `analysis.training_time` selects the agent-name and export conventions for analysis; it does not launch training.

For **adaptive evaluation**, use the supplied adaptive configuration: `alignment.adaptive: true` enables selection based on observed CIA-host conditions. These configurations omit `alpha_range`, analyze one policy per attribute, and disable knee-point calculation.

For **training-time evaluation**, use a supplied training-time configuration. It combines `paths.scenario_root` with `scenario_setting`, stores the baseline path under `agents.blue_agent_base_path`, and locates aligned models using `paths.checkpoint_dir` and `paths.checkpoint_prefix`. Here, `alignment.alpha_range` selects saved training reward bonuses rather than interpolation weights, and `analysis.training_time` is `true`.

## Outputs and Cached Analysis

All outputs below are inside `paths.experiment_dir`:

| Output | Contents |
| --- | --- |
| `test_results.csv` | Aggregate CIA, resilience-drop, and CAGE reward statistics. |
| `scenarios/*.npz` | Per-action CIA scores, selected action indices, episode indices, and step indices for PPO and aligned policies; Blue Monitor does not export these arrays. |
| `plots/` | Per-step means and standard errors as CSVs; PNG figures for Blue Monitor and the first training seed of each policy. |
| `trajectories/` | Episode JSON files when `env.save_trajectories` is enabled; `env.auto_plot_trajectories` also creates figures in `trajectory_plots/`. |
| `all_alignment_scores.csv` | Computed alignment scores. |
| `alignment_barplot_with_reward_over.pdf`, `pareto_front.png` | Alignment and trade-off plots. |
| `knee_point.json` | Selected knee points when `analysis.calculate_kneepoint` is enabled. |

Set `analysis.recalculate: false`, or pass `--no-recalculate` to `alignment_analysis.py`, to redraw analysis plots using the saved alignment scores. Trade-off plots are still generated when knee-point calculation is disabled.

The training-time evaluator also reuses existing per-seed plot CSVs by default. Its top-level `regenerate: true` setting reruns PPO and aligned evaluations; it is separate from `analysis.recalculate`.

## Differences from the Original CAGE 2

This repository includes the following internal changes to the original CAGE 2 environment:

- Bug fixes identified by the [CybORG++ library](https://github.com/alan-turing-institute/CybORG_plus_plus).
- Scenario updates adding authentication (`Auth`), database (`Database`), and front-end web (`Front`) servers, plus OT services on Enterprise and Operational hosts to enable red agent `Impact` actions.
- Additional host images in `CybORG/Shared/Scenarios/images/` to support OT services.
- Migration from `gym` to `gymnasium`.
- Updates to `AgentInterface` and `CybORG` to support custom reward functions with input parameters, used for alpha-based reward shaping.
- Tracking of impacted hostnames in `ExploitRemoteService` for CIA calculations.
- Integration of `wandb` and tracking of additional parameters, including alignment, in `EnvironmentController`.
- Updated IP generation in `CybORG/Simulator/State.py` to support more hosts per subnet.
