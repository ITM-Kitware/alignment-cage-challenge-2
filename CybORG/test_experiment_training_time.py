from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
from CybORG.Agents.SimpleAgents.PPOAgent import PPOAgent, PPOAgentReactDecoys, AlignedPPOAgent
from CybORG.Agents.SimpleAgents.RandomAgent import RandomAgent
from CybORG.Agents.SimpleAgents.BlueMonitorAgent import BlueMonitorAgent
from CybORG.Agents.SimpleAgents.InterpolatedAgent import InterpolatedAgent

from cage_experiment import CAGEExperiment
from CybORG.AlignmentMetric.resilience_measure import ResilienceMetric
from CybORG.Agents.SimpleAgents.B_line_resilience import (
    B_lineAgent_Resilience_A,
    B_lineAgent_Resilience_C,
    B_lineAgent_Resilience_I,
    B_lineAgent_Resilience,
)
from CybORG.Agents.SimpleAgents.Meander_Resilience import (
    RedMeanderAgent_Resilience,
    RedMeanderAgent_RandomImpact,
    RedMeanderAgent_Combination_CIA,
    RedMeanderAgent_Combination_CAGE
)
from CybORG.Agents.SimpleAgents.CIAAttributeAgent import CIAAttributeAgent


def load_yaml(path: str):
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"Config file did not load as a dict: {path}")
    return cfg


def resolve_alpha_schedule(cfg: dict):
    sched = cfg["alignment"]["alpha_range"]
    return np.linspace(
        float(sched["start"]),
        float(sched["stop"]),
        int(sched["num"]),
    ).tolist()


def checkpoint_path(cfg: dict, checkpoint_id: str):
    ckpt_dir = cfg["paths"]["checkpoint_dir"]
    prefix = cfg["paths"]["checkpoint_prefix"]
    return f"{ckpt_dir}/{prefix}_{checkpoint_id}"


def aggregate_seeds(seed_results):
    arr = np.array(seed_results)  # (n_seeds, 10): cols 0-4 = means, cols 5-9 = within-episode stds
    return arr[:, :5].mean(axis=0).tolist() + arr[:, 5:].mean(axis=0).tolist()


def _load_from_plot_csv(path: Path):

    if not path.exists():
        return None
    df = pd.read_csv(path)
    return [
        df["confidentiality_mean"].sum(),
        df["integrity_mean"].sum(),
        df["availability_mean"].sum(),
        df["resilience_mean"].mean(),
        df["reward_mean"].sum(),
        0.0, 0.0, 0.0, 0.0, 0.0,
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        help="Path to yaml config file",
    )
    args = parser.parse_args()

    cfg = load_yaml(args.config)

    seed = int(cfg["seed"])
    scenario_setting = cfg["scenario_setting"]

    experiment_dir = Path(cfg["paths"]["experiment_dir"])
    experiment_dir.mkdir(parents=True, exist_ok=True)

    export_path = Path(cfg["paths"]["export_path"])
    if not export_path.is_absolute():
        export_path = experiment_dir / export_path

    scenario = f"{cfg['paths']['scenario_root']}/{scenario_setting}"
    episodes = int(cfg["env"]["episodes"])
    steps = int(cfg["env"]["steps"])
    verbose = bool(cfg["env"].get("verbose", False))
    average_metric = bool(cfg["env"].get("average_metric", False))
    red_agent = globals()[cfg["agents"]["red_agent"]]
    blue_agent_base_path = cfg["agents"]["blue_agent_base_path"]
    attributes_to_run = cfg["alignment"]["attributes_to_run"]
    save_trajectories = bool(cfg["env"].get("save_trajectories", False))
    auto_plot_trajectories = bool(cfg["env"].get("auto_plot_trajectories", False))

    alphas = resolve_alpha_schedule(cfg)
    train_seeds = cfg["agents"].get("train_seeds", [0])
    base_path_stem = blue_agent_base_path.replace('.zip', '')
    regenerate = bool(cfg.get("regenerate", False))
    plots_dir = experiment_dir / "plots"

    all_results = [[
        "Agent", "Steps", "Episodes",
        "C Mean", "I Mean", "A Mean",
        "Resilience Drop", "Reward",
        "C Std.", "I Std.", "A Std.",
        "Resilience Std.", "Reward Std."
    ]]

    # Blue Monitor: deterministic agent, no seed sweep, no scenario export
    metric = ResilienceMetric()
    cage = CAGEExperiment(
        BlueMonitorAgent(),
        red_agent=red_agent,
        scenario=scenario,
        seed=seed,
        metric=metric,
        experiment_export_dir=str(experiment_dir),
        average_metric=average_metric,
        alpha=0.0,
        use_wrapper=False,
    )

    all_results.append(
        ["Blue Monitor", steps, episodes]
        + cage.run_experiment(
            episodes=episodes,
            steps=steps,
            export_scenario_choices=False,
            plot_export_path=f"{steps}_blue_monitor.png",
            verbose=verbose,
            save_trajectories=save_trajectories,
            auto_plot_trajectories=auto_plot_trajectories
        )
    )

    # RL baseline and aligned agents: sweep over training seeds
    rl_seed_results = []
    aligned_seed_results = {}  # (cia, alpha_r) -> list of results

    for train_seed in train_seeds:
        seed_model_file = (
            f"{base_path_stem}_seed{train_seed}.zip"
            if len(train_seeds) > 1
            else blue_agent_base_path
        )

        rl_plot_csv = plots_dir / f"{steps}_ppo_agent_seed{train_seed}.csv"
        existing_rl = None if regenerate else _load_from_plot_csv(rl_plot_csv)
        if existing_rl is not None:
            print(f"[skip] RL baseline seed {train_seed} (already done)")
            rl_seed_results.append(existing_rl)
        else:
            agent1 = PPOAgent(model_file=seed_model_file)
            metric = ResilienceMetric()
            cage = CAGEExperiment(
                agent1,
                red_agent=red_agent,
                scenario=scenario,
                seed=seed,
                metric=metric,
                experiment_export_dir=str(experiment_dir),
                average_metric=average_metric,
                use_wrapper=True,
            )
            agent1.set_env(cage.env)
            rl_seed_results.append(cage.run_experiment(
                episodes=episodes,
                steps=steps,
                export_scenario_choices=True,
                plot_export_path=f"{steps}_ppo_agent_seed{train_seed}.png",
                verbose=verbose,
                save_trajectories=save_trajectories,
                auto_plot_trajectories=auto_plot_trajectories,
                scenario_seed=train_seed,
                save_plots=(train_seed == train_seeds[0]),
            ))

        for cia in attributes_to_run:
            for alpha in alphas:
                alpha_r = round(float(alpha), 3)
                key = (cia, alpha_r)

                aligned_plot_csv = plots_dir / f"{steps}_{cia}_a{alpha_r}_seed{train_seed}.csv"
                existing_aligned = None if regenerate else _load_from_plot_csv(aligned_plot_csv)
                if existing_aligned is not None:
                    print(f"[skip] {cia} alpha={alpha_r} seed {train_seed} (already done)")
                    aligned_seed_results.setdefault(key, []).append(existing_aligned)
                else:
                    ckpt_id = f"{scenario_setting}_{alpha_r}_{cia}_seed{train_seed}"
                    model_file = checkpoint_path(cfg, ckpt_id)

                    agent = AlignedPPOAgent(
                        model_file=model_file,
                        alpha=alpha_r,
                        align_attribute=cia,
                    )

                    metric = ResilienceMetric()
                    cage = CAGEExperiment(
                        agent,
                        red_agent=red_agent,
                        scenario=scenario,
                        seed=seed,
                        metric=metric,
                        experiment_export_dir=str(experiment_dir),
                        average_metric=average_metric,
                        use_wrapper=True,
                        alpha=0.0,
                        align_attribute=cia,
                    )
                    agent.set_env(cage.env)

                    result = cage.run_experiment(
                        episodes=episodes,
                        steps=steps,
                        export_scenario_choices=True,
                        plot_export_path=f"{steps}_{cia}_a{alpha_r}_seed{train_seed}.png",
                        verbose=verbose,
                        save_trajectories=save_trajectories,
                        auto_plot_trajectories=auto_plot_trajectories,
                        scenario_seed=train_seed,
                        save_plots=(train_seed == train_seeds[0]),
                    )
                    aligned_seed_results.setdefault(key, []).append(result)

    all_results.append(["RL", steps, episodes] + aggregate_seeds(rl_seed_results))

    for cia in attributes_to_run:
        for alpha in alphas:
            alpha_r = round(float(alpha), 3)
            key = (cia, alpha_r)
            all_results.append(
                [f"RL-a{alpha_r}-{cia}", steps, episodes] + aggregate_seeds(aligned_seed_results[key])
            )

    pd.DataFrame(all_results).to_csv(export_path, index=False, header=False)
    print(f"Saved results to: {export_path}")


if __name__ == "__main__":
    main()