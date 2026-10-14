from CybORG.Agents import B_lineAgent, RedMeanderAgent, GreenAgent, BlueMonitorAgent
from CybORG import CybORG
from CybORG.Agents.Wrappers.ChallengeWrapper import ChallengeWrapper
import inspect
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, MaxNLocator, MultipleLocator
import seaborn as sns
import statistics as stats
import numpy as np
import torch as th
import pandas as pd
from CybORG.AlignmentMetric.cia_triad_measure import CIATriadMetric
from agent_trajectory import AgentTrajectory
from datetime import datetime
import os

# Shared plot styling (matches plot_drops_from_csv.py)
SPINE_LW = 2.0
SPINE_COLOR = "#2E2E2E"

# (color,     title,                  invert, no_bounds, ylim)
_CIA_PLOT_STYLE = [
    ("#1f6f9e", "Confidentiality Drop", True,  False, (-0.25, 10.25)),
    ("#2a9d8f", "Integrity Drop",       True,  False, (-0.25, 10.25)),
    ("#52a675", "Availability Drop",    True,  False, (-0.25, 10.25)),
    ("#6a4c93", "Resilience Drop",      True,  False, (-0.02,  1.1)),
    ("#b5384d", "CAGE Reward",          False, True,  None),
]

# Class runs an instance of CAGE
# exports trajectories and results
class CAGEExperiment:
    def __init__(self, 
                blue_agent,
                red_agent = B_lineAgent,
                seed: int = 1,
                
                average_metric=False, # whether to calculate CIA as average or sum
                
                metric = CIATriadMetric(),
                scenario: str = "Scenario2",
                use_wrapper = False, #set true when using RL agent, false for text-based
                experiment_export_dir: str = "results",
                            
                alpha: int = 0.0,
                align_attribute: str = "C",
                alignment_metric: str = "ResilienceMetric",
                

                ):
        path = str(inspect.getfile(CybORG))
        path = path[:-10] + f'/Shared/Scenarios/{scenario}.yaml'
    
        self.cyborg = CybORG(path, 'sim', agents={
             'Red': red_agent, 'Green': GreenAgent},
                             agent_rewards={
                                 "Red":{},
                                 "Blue": {
                                        "alpha": alpha,
                                        "align_attribute": align_attribute,
                                        "alignment_metric": alignment_metric  
                                        },
                                 "Green": {}
                             })
        self.blue_agent = blue_agent
        self.red_agent = red_agent

        self.seed = seed
        self.cyborg.set_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        th.manual_seed(seed)
        
        self.use_wrapper = use_wrapper
        
        self.env = self.wrap(self.cyborg)
        
        self.average_metric = average_metric
        self.current_step = 0
        self.done = False
        self.current_obs = None
        self.rewards = []
        self.action_space = None
        self.metric = metric
        
        self.experiment_export_dir = experiment_export_dir
        
    def wrap(self, env):
        return ChallengeWrapper(env=env, agent_name='Blue')

    def update_metric(self):
        self.metric.calculate_scores(self.cyborg)
        return self.metric.scores(use_avg=self.average_metric)
    
    # Update observation, action, and status
    def set_state(self, obs, reward, done, info):
        self.current_obs = obs
        self.done = done
        self.rewards.append(reward)
        self.action_space = info['action_space']
        
        return self.current_obs, self.action_space, self.rewards[-1], self.done
    
    # reset environment    
    def reset(self):
        self.current_step = 0
        self.rewards = []
        self.done = False
        self.current_obs = None
        if self.use_wrapper:
            obs, _ = self.env.reset()
            self.action_space = self.env.get_action_space("Blue")
        else:
            results = self.cyborg.reset(agent="Blue")
            self.action_space = results.action_space
            obs = results.observation
            
        self.metric.reset(self.cyborg.get_agent_state("Blue"))
        
        return obs, None, False, self.action_space
        
    # step in the environment
    def step(self, action):
        self.current_step += 1
        if self.use_wrapper:
            obs, reward, done, truncated, info = self.env.step(action=action)
        else:
            results = self.cyborg.step(agent="Blue", action=action)
            obs = results.observation
            reward = results.reward
            done = results.done
            info = {"action_space": results.action_space}
        self.update_metric()
        return self.set_state(obs, reward, done, info)

    def train(self, agent_export_path, training_iter=1e6, wandb=True):
        self.blue_agent.set_env(self.env)
        self.blue_agent.train(agent_export_path, 
                              training_iter=training_iter,
                              use_wandb=wandb)

    def plot_mean_with_smoothing(self, ax, runs, color, title, window=5,
                                 no_bounds=False, invert=True, ylim=None):
        runs = np.array(runs)
        mean = runs.mean(axis=0)
        # Compute SEM from raw runs
        std = runs.std(axis=0, ddof=1)  # sample std
        std = std / np.sqrt(runs.shape[0])

        x = np.arange(len(mean))

        mean_smooth = pd.Series(mean).rolling(window, min_periods=1).mean().values

        upper = mean_smooth + std
        lower = mean_smooth - std if no_bounds else np.maximum(mean_smooth - std, 0)

        ax.plot(x, mean_smooth, color=color, linewidth=2.25)
        ax.fill_between(x, lower, upper, color=color, alpha=0.3)

        ax.set_title(title, fontsize=14)
        ax.tick_params(which="major", labelsize=11.5, pad=2)
        ax.tick_params(which="minor", length=3, width=0.8)
        ax.set_xlim(left=0, right=len(mean))
        x_interval = 100 if len(mean) >= 400 else 50
        ax.xaxis.set_major_locator(MultipleLocator(x_interval))
        ax.xaxis.set_minor_locator(AutoMinorLocator(5))
        ax.yaxis.set_minor_locator(AutoMinorLocator(4))
        if ylim is not None:
            ax.set_ylim(ylim)
        if invert:
            ax.invert_yaxis()
        for spine in ax.spines.values():
            spine.set_linewidth(SPINE_LW)
            spine.set_edgecolor(SPINE_COLOR)

    def run_experiment(self,
                       episodes: int = 10,
                       steps: int = 30,
                       plot_export_path = "experiment_results.png",
                       verbose: float= True,
                       export_scenario_choices = False,
                       save_trajectories = False,
                       auto_plot_trajectories = False,
                       scenario_seed = None,
                       save_plots = True):
                
        os.makedirs(f"{self.experiment_export_dir}/plots", exist_ok=True)
        os.makedirs(f"{self.experiment_export_dir}/scenarios", exist_ok=True)

        obs, action, reward, done = self.reset()
        
        rightnow = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        
        #gets possible actions spelled-out
        action_map = self.action_space if not self.use_wrapper else self.env.get_attr("possible_actions")
        
        sns.set_context("paper")
        sns.set_style("whitegrid")
        fig, axs = plt.subplots(1, 5, figsize=(13, 3.5), constrained_layout=True)

        rewards, confidentiality, integrity, availability, resiliences = [], [], [], [], []
        
        if export_scenario_choices:
            scenario_selected = []
            scenario_choices_arr = []
            scenario_episodes = []
            scenario_steps = []

        for e in tqdm(range(episodes), desc="Episodes"):
            
            self.cyborg.set_seed(self.seed + e)
            random.seed(self.seed + e)
            np.random.seed(self.seed + e)
            th.manual_seed(self.seed + e)
            
            ip_map = self.cyborg.get_ip_map()
            ip_to_hostname = {ip: hostname for hostname, ip in ip_map.items()}

            subnet_cidr_map = self.cyborg.environment_controller.subnet_cidr_map
            cidr_to_subnet_name = {cidr: name for name, cidr in subnet_cidr_map.items()}

            agent_trajectory = AgentTrajectory(str(self.blue_agent),
            self.red_agent.__name__,
            e,
            rightnow,
            self.cyborg.get_agent_state("Blue"),
            export_path=f"{self.experiment_export_dir}/trajectories",
            verbose=verbose)

            for s in tqdm(range(steps), leave=False):
                if export_scenario_choices:
                    last_scenario_action_cia = self.metric.calculate_per_action_scores(self.cyborg if not self.use_wrapper else self.env, action_map)
                
                action = self.blue_agent.get_action(
                    obs,
                    self.action_space
                )
                obs, _, reward, done = self.step(action)
                
                agent_trajectory.add_blue_action(self.env.get_last_action("Blue"), self.env.get_observation("Blue"))
                agent_trajectory.add_red_action(self.env.get_last_action("Red"), self.env.get_observation("Red"), ip_to_hostname, cidr_to_subnet_name)
                agent_trajectory.add_metric_scores(self.metric.step_scores())

                if done:
                    break
            
                if export_scenario_choices:
                    scenario_selected.append(int(action))
                    scenario_choices_arr.append(last_scenario_action_cia)
                    scenario_episodes.append(e)
                    scenario_steps.append(s)
                        
            rewards.append(self.rewards)
            confidentiality.append(self.metric.confidentialities)
            integrity.append(self.metric.integrities)
            availability.append(self.metric.availabilities)
            resiliences.append(self.metric.resiliences)
                
            if save_trajectories:
                agent_trajectory.export()

                if auto_plot_trajectories:
                    trajectory_plot_dir = f"{self.experiment_export_dir}/trajectory_plots"
                    agent_trajectory.view(trajectory_plot_dir, agent_type="red")
                    agent_trajectory.view(trajectory_plot_dir, agent_type="blue")

            obs, action, reward, done = self.reset()
            self.blue_agent.end_episode()
            
        #TODO: Better organize plotting/metric tracking
        confidentiality = np.array(confidentiality)
        integrity = np.array(integrity)
        availability = np.array(availability)
        resiliences = np.array(resiliences)
        rewards = np.array(rewards)
        
        _series_mean_sem = lambda arr: (arr.mean(axis=0), arr.std(axis=0, ddof=1) / np.sqrt(arr.shape[0]))

        c_mean, c_sem = _series_mean_sem(confidentiality)
        i_mean, i_sem = _series_mean_sem(integrity)
        a_mean, a_sem = _series_mean_sem(availability)
        r_mean, r_sem = _series_mean_sem(resiliences)
        rw_mean, rw_sem = _series_mean_sem(rewards)

        plot_csv_path = (
            f"{self.experiment_export_dir}/plots/{plot_export_path}".replace(".png", ".csv")
        )
        pd.DataFrame({
            "step": np.arange(len(c_mean)),
            "confidentiality_mean": c_mean, "confidentiality_sem": c_sem,
            "integrity_mean": i_mean, "integrity_sem": i_sem,
            "availability_mean": a_mean, "availability_sem": a_sem,
            "resilience_mean": r_mean, "resilience_sem": r_sem,
            "reward_mean": rw_mean, "reward_sem": rw_sem,
        }).to_csv(plot_csv_path, index=False)

        if save_plots:
            series = [confidentiality, integrity, availability, resiliences, rewards]
            for i, (ax, runs, (color, title, invert, no_bounds, ylim)) in enumerate(
                zip(axs, series, _CIA_PLOT_STYLE)
            ):
                self.plot_mean_with_smoothing(
                    ax, runs, color, title,
                    invert=invert, no_bounds=no_bounds, ylim=ylim,
                )
                # Hide y tick labels on Integrity and Availability; keep ticks visible
                if i in (1, 2):
                    ax.tick_params(labelleft=False)
                # CAGE Reward: increase tick density (auto-range often yields only 2 ticks)
                if i == 4:
                    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))

            fig.supxlabel("Game Step", fontsize=14)
            fig.supylabel("Score", fontsize=14)
            plt.savefig(f"{self.experiment_export_dir}/plots/{plot_export_path}", dpi=300, bbox_inches="tight")

        plt.close(fig)
        

        confidentiality = confidentiality.sum(axis=1)
        integrity = integrity.sum(axis=1)
        availability = availability.sum(axis=1) 
        #compute avg. resilience based on paper sec. 6.2:
        #resiliences = (resiliences - resiliences.mean(axis=0)).mean(axis=1)
        # above uses centered matrix which removes the need for a mean, essentially
        # results in a 0.0 mean since differences are removed
        resiliences = resiliences.mean(axis=0) #across timesteps
        rewards = rewards.sum(axis=1) 
        
        if export_scenario_choices:
            seed_suffix = f"_seed{scenario_seed}" if scenario_seed is not None else ""
            np.savez_compressed(
                f"{self.experiment_export_dir}/scenarios/scenario_choices_{str(self.blue_agent)}{seed_suffix}.npz",
                choices=np.array(scenario_choices_arr, dtype=np.float32),
                selected=np.array(scenario_selected, dtype=np.int32),
                episode=np.array(scenario_episodes, dtype=np.int32),
                step=np.array(scenario_steps, dtype=np.int32),
            )
            
        if verbose:
            print("C, I, A, Reward")
            print(f"{confidentiality.mean():.2f}", end=",")
            print(f"{integrity.mean():.2f}", end=",")
            print(f"{availability.mean():.2f}", end=",")
            print(f"{resiliences.mean():.2f}", end=",")
            print(f"{rewards.mean():.2f}")
            
        return [
            confidentiality.mean(),
            integrity.mean(), 
            availability.mean(),
            resiliences.mean(),
            rewards.mean(),
            confidentiality.std(),
            integrity.std(),
            availability.std(),
            resiliences.std(),
            rewards.std(),
        ]
        
