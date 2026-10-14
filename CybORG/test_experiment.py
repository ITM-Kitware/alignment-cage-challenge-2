####
# Runs several experiments with various agents
####

from __future__ import annotations

import argparse
from pathlib import Path
import json
import pandas as pd
import yaml
import numpy as np
from CybORG.Agents.SimpleAgents.PPOAgent import PPOAgent, AlignedPPOAgent
from CybORG.Agents.SimpleAgents.RandomAgent import RandomAgent
from CybORG.Agents.SimpleAgents.BlueMonitorAgent import BlueMonitorAgent
from CybORG.Agents.SimpleAgents.InterpolatedAgent import InterpolatedAgent
from CybORG.Agents.SimpleAgents.B_line import B_lineAgent
from CybORG.Agents.SimpleAgents.B_line_resilience import (
    B_lineAgent_Resilience_A,
    B_lineAgent_Resilience_C,
    B_lineAgent_Resilience_I,
    B_lineAgent_Resilience
)
from CybORG.Agents.SimpleAgents.Meander_Resilience import (
    RedMeanderAgent_Resilience,
    RedMeanderAgent_RandomImpact,
    RedMeanderAgent_Combination_CIA,
    RedMeanderAgent_Combination_CAGE
)
from CybORG.Agents.SimpleAgents.CIAAttributeAgent import CIAAttributeAgent
from CybORG.AlignmentMetric.resilience_measure import ResilienceMetric
from cage_experiment import CAGEExperiment
from CybORG.Agents.Castle.scorer import AVAIL_DROP_KEY, CONF_DROP_KEY, INTEGRITY_DROP_KEY


def load_yaml(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def aggregate_seeds(seed_results):
    arr = np.array(seed_results)  # (n_seeds, 10): cols 0-4 = means, cols 5-9 = within-episode stds
    return arr[:, :5].mean(axis=0).tolist() + arr[:, 5:].mean(axis=0).tolist()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config file",
    )
    args = parser.parse_args()

    cfg = load_yaml(args.config)

    seed = int(cfg["seed"])
    experiment_dir = Path(cfg["paths"]["experiment_dir"])
    experiment_dir.mkdir(parents=True, exist_ok=True)

    export_path = Path(cfg["paths"]["export_path"])
    if not export_path.is_absolute():
        export_path = experiment_dir / export_path

    scenario = cfg["paths"]["scenario"]
    blue_agent_base_path = cfg["paths"]["blue_agent_base_path"]
    blue_agent_name = cfg["agents"]["blue_agent"]
    blue_agent_cls = globals()[blue_agent_name]
    episodes = int(cfg["env"]["episodes"])
    steps = int(cfg["env"]["steps"])
    verbose = bool(cfg["env"].get("verbose", False))
    save_trajectories = bool(cfg["env"].get("save_trajectories", False))
    auto_plot_trajectories = bool(cfg["env"].get("auto_plot_trajectories", False))
    average_metric = bool(cfg["env"].get("average_metric", False))
    red_agent_name = cfg["agents"]["red_agent"]
    red_agent_cls = globals()[red_agent_name]
    adaptive_alpha = cfg["alignment"].get("adaptive", False)
    alphas = cfg["alignment"].get("alpha_range", {"start": 0, "stop": 0, "num": 0})
    alphas = np.linspace(alphas["start"], 
                        alphas["stop"], 
                        alphas["num"])
    attributes_to_run = cfg["alignment"]["attributes_to_run"]
    use_kneepoint = cfg["alignment"].get("use_kneepoint", False)
    reslience_weights = cfg["alignment"].get("resilience_weights", {"C": 0.6, "I": 0.2, "A": 0.2})
    confidentiality_w = float(reslience_weights["C"])
    integrity_w = float(reslience_weights["I"])
    availability_w = float(reslience_weights["A"])

    if use_kneepoint:
        #load json
        with open(experiment_dir/ 'knee_point.json') as f:
            knee_points = json.load(f)
            print("Using knee-point alpha=", knee_points)
            
    if use_kneepoint or adaptive_alpha:
        alphas = [0] #will be overwritten below
            
    all_results = [[
        "Agent", "Steps", "Episodes",
        "C Mean", "I Mean", "A Mean",
        "Resilience Drop", "Reward",
        "C Std.", "I Std.", "A Std.",
        "Resilience Std.", "Reward Std."
    ]]

    metric_weights = {
        CONF_DROP_KEY: confidentiality_w,
        INTEGRITY_DROP_KEY: integrity_w,
        AVAIL_DROP_KEY: availability_w
    }

    def make_metric():
        return ResilienceMetric(weighter=metric_weights)

    train_seeds = cfg["agents"].get("train_seeds", [0])
    base_path_stem = blue_agent_base_path.replace('.zip', '')

    # Blue Monitor (no seed variation — deterministic agent)
    metric = make_metric()
    cage = CAGEExperiment(
        BlueMonitorAgent(),
        red_agent=red_agent_cls,
        scenario=scenario,
        seed=seed,
        metric=metric,
        experiment_export_dir=str(experiment_dir),
        average_metric=average_metric,
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

    # RL baseline and test-time aligned agents: sweep over training seeds
    rl_seed_results = []
    aligned_seed_results = {}  # (cia, alpha_r) -> list of results

    for train_seed in train_seeds:
        seed_model_file = (
            f"{base_path_stem}_seed{train_seed}.zip"
            if len(train_seeds) > 1
            else blue_agent_base_path
        )

        metric = make_metric()
        agent1 = blue_agent_cls(model_file=seed_model_file)
        cage = CAGEExperiment(
            agent1,
            red_agent=red_agent_cls,
            scenario=scenario,
            seed=seed,
            metric=metric,
            experiment_export_dir=str(experiment_dir),
            average_metric=average_metric,
            use_wrapper=True,
        )
        agent1.set_env(cage.env)
        rl_seed_results.append(
            cage.run_experiment(
                episodes=episodes,
                steps=steps,
                export_scenario_choices=True,
                plot_export_path=f"{steps}_ppo_agent_seed{train_seed}.png",
                verbose=verbose,
                save_trajectories=save_trajectories,
                auto_plot_trajectories=auto_plot_trajectories,
                scenario_seed=train_seed,
                save_plots=(train_seed == train_seeds[0]),
            )
        )

        # Test-time policy shaping for this seed
        for cia in attributes_to_run:
            for alpha in alphas:
                alpha_r = round(float(alpha), 3)
                if use_kneepoint:
                    alpha_r = float(knee_points[cia]['alpha'])

                key = (cia, alpha_r)
                agent2 = CIAAttributeAgent({cia: 1})
                agent = InterpolatedAgent(agent1, agent2, alpha=alpha_r, adaptive=adaptive_alpha)
                metric = make_metric()
                cage = CAGEExperiment(
                    agent,
                    red_agent=red_agent_cls,
                    scenario=scenario,
                    seed=seed,
                    metric=metric,
                    experiment_export_dir=str(experiment_dir),
                    average_metric=average_metric,
                    use_wrapper=True,
                )
                agent2.set_env(cage)
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

    # Aggregate across seeds and record results
    all_results.append(["RL", steps, episodes] + aggregate_seeds(rl_seed_results))

    for cia in attributes_to_run:
        for alpha in alphas:
            alpha_r = round(float(alpha), 3)
            if use_kneepoint:
                alpha_r = float(knee_points[cia]['alpha'])
            key = (cia, alpha_r)
            all_results.append(
                [f"RL-a{alpha_r}-{cia}", steps, episodes] + aggregate_seeds(aligned_seed_results[key])
            )

    pd.DataFrame(all_results).to_csv(export_path, index=False, header=False)
    print(f"Saved results to: {export_path}")


if __name__ == "__main__":
    main()
