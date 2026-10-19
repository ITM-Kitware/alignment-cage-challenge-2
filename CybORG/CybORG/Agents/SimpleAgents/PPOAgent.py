import inspect
import os

from stable_baselines3 import PPO
from CybORG import CybORG
from CybORG.Agents.SimpleAgents.BaseAgent import BaseAgent
from CybORG.Agents.Wrappers.ChallengeWrapper import ChallengeWrapper
from CybORG.Agents import B_lineAgent, GreenAgent
import torch
from torch import nn
from CybORG.Shared.Actions import *
from CybORG.Agents.Cadet.agent_utils import *
import copy
from sb3_contrib import MaskablePPO
from gymnasium import Space, spaces
import wandb
from wandb.integration.sb3 import WandbCallback
from CybORG.Shared.WandbIntegration import CAGEWandbCallback
from loguru import logger
    
USE_ALL_DECOYS = True
ASSETS_ONLY = False


class PPOAgent(BaseAgent):
    # agent that loads a StableBaselines3 PPO model file
    def train(self, agent_export_name, training_iter=10e6, use_wandb=True):
        callback = []
        
        if use_wandb:
            run = wandb.init(
                    project=os.environ.get("WANDB_PROJECT", "cyborg-alignment"),
                    name=f'{str(self)}-{training_iter}-{self.scenario}',
                    config={
                            "policy_type": "MlpPolicy",
                            "total_timesteps": training_iter
                            },
                    sync_tensorboard=False,
                    monitor_gym=False, #works if env has .render()
                    save_code=False
                )

            callback = [WandbCallback(
                log="rollout",
                model_save_freq=training_iter/5,
                model_save_path=f"{agent_export_name}-saves/{run.id}",
                verbose=2
            ), CAGEWandbCallback()]

        self.model.learn(total_timesteps=training_iter,
                            progress_bar=True,
                            callback=callback)
        self.model.save(agent_export_name)
        if use_wandb:
            run.finish()

    def end_episode(self):
        pass

    def set_initial_values(self, action_space, observation):
        pass

    def __init__(self, 
                 model_file: str = None,
                 scenario_config: str = "Scenario2_resilience",
                 red_agent = B_lineAgent,
                 seed: int = 1,
                 game_steps: int = 100,
                 learning_rate:float=0.0003,
                 reward_params:dict = {}):
        self.scenario = scenario_config
        self.seed = seed
        if model_file is not None:
            self.model = PPO.load(model_file)
            self.cyborg_env = None
        else:
            path = str(inspect.getfile(CybORG))
            path = path[:-10] + f'/Shared/Scenarios/{scenario_config}.yaml'
            cyborg = ChallengeWrapper('Blue', 
                                    CybORG(path, 'sim', 
                                    agents={'Red': red_agent,
                                    'Green': GreenAgent},
                                    agent_rewards=reward_params),
                                    max_steps=game_steps)
            self.cyborg_env = cyborg
            self.model = PPO('MlpPolicy', cyborg,
                stats_window_size=100,
                n_steps=2048,
                batch_size=1024,#32768, # Match sgd_minibatch_size of Dartmouth - doesn't work well with our env
                n_epochs=10,
                policy_kwargs=dict(net_arch=[256, 256], activation_fn=nn.Tanh),
                seed=seed,
                learning_rate=learning_rate,
                device="cpu"
            )

    def get_action(self, observation, action_space):
        """gets an action from the agent that should be performed based on the agent's internal state and provided observation and action space"""
        action, _states = self.model.predict(observation)
        return action
    
    def set_env(self, env):
        self.cyborg_env = env
        
    def get_action_probabilities(self, observation, action_space=None):
        obs = torch.tensor(observation, device = self.model.device).unsqueeze(0)
        dis = self.model.policy.get_distribution(obs)
        probs = dis.distribution.probs
        probs_np = probs.squeeze(0).detach().cpu().numpy()
        return probs_np
    
    def __str__(self):
        return "PPO agent"
    
class AlignedPPOAgent(PPOAgent):
    def __init__(self, 
                base_agent: str = None, #path to orig RL agent checkpoint for tuning
                alpha: float = 0.5,
                align_attribute:str = "C", #required
                game_steps: int = 100,
                model_file: str = None,
                scenario_config: str = "Scenario2_resilience",
                red_agent = B_lineAgent,
                seed: int = 1,
                learning_rate:float=0.0003,
                alignment_metric: str = "ResilienceMetric"):
        self.alpha = alpha
        self.align_attribute = align_attribute
        super().__init__(model_file,
                         scenario_config,
                         red_agent,
                         seed,             
                         game_steps=game_steps,
                         learning_rate=learning_rate,
                         reward_params={
                             "Blue": {
                                 "alpha": alpha,
                                 "align_attribute": align_attribute,
                                 "alignment_metric": alignment_metric
                             }
                         })
        if base_agent is not None:
            print(f"LOADING BASE AGENT {base_agent}")
            self.model = PPO.load(base_agent)
            self.model.set_env(self.cyborg_env)
        self.align_attribute = align_attribute
        
    def __str__(self):
        return f"Aligned RL {self.alpha} {self.align_attribute} agent"
    
class ModifiedObservationWrapper(gym.Wrapper):
    #overwrite cage env for Cadet action masking
    def __init__(self, env):
        super().__init__(env)
        self.decoy_history_init = None
        self.decoy_history = None
        self.observation_space = None
        self.action_mask = None
        self.extend_obs_space()
        
    def extend_obs_space(self):
        #print(self.action_space, self.action_space.n)
        self.observation_space = self.env.observation_space
        self.action_mask = spaces.MultiDiscrete([2] * self.action_space.n)


    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)

        self.set_initial_values(self.env.get_attr('possible_actions'), 
        observation=self.env.get_observation("Blue"))
        
        modified_obs = self.get_modified_observation(obs,
                        self.env.get_attr('possible_actions'))

        return np.array(modified_obs, dtype=np.float32), {}

    def set_initial_values(self, action_space=None, observation=None):
        """ 
        action_space: self.env.get_attr('possible_actions')
        observation: self.env.get_agent_state("Blue")
        """
        decoy_history = reset_decoy_history(action_space)
        self.decoy_history_init = prune_decoys_on_used_ports(
            self.env.cyborg_env, action_space, observation, decoy_history
        )
        self.decoy_history = copy.deepcopy(self.decoy_history_init)

        for (
            hostname,
            _,
        ) in self.env.cyborg_env.environment_controller.state.hosts.items():
            print_valid_decoys(action_space, self.decoy_history, hostname)
            


    def get_modified_observation(self, observation, action_space):
        if isinstance(observation, tuple):
            observation = observation[0]
            
        user_hosts, escalated_hosts = get_compromised_hosts(
            self.env.cyborg_env, observation
        )

        if len(user_hosts) != 0 or len(escalated_hosts) != 0:  # recover
            possible_reactions = get_possible_reactions(action_space, [Restore])
            filtered_reactions = filter_reactions(
                user_hosts, escalated_hosts, possible_reactions, action_space
            )
            action_mask = filter_action_mask(action_space, filtered_reactions)
        else:
            indexes_to_deploy = []
            deployed, available = deployed_decoys(
                action_space,
                self.decoy_history_init,
                self.decoy_history,
                assets_only=ASSETS_ONLY
            )
            for host in available:
                if USE_ALL_DECOYS:  # deploy all available decoys
                    indexes_to_deploy += available[host]
                elif len(deployed[host]) == 0:  # ensure at least one decoy per host
                    indexes_to_deploy += available[host]

            if len(indexes_to_deploy) > 0:  # harden
                action_mask = filter_action_mask(action_space, indexes_to_deploy)
            else:  # try any valid action
                action_mask = valid_action_mask(action_space, self.decoy_history)

        self.observation_space = observation
        self.action_mask = action_mask
        return self.observation_space
    
    
    def action_masks(self):
        return self.action_mask
    
    def step(self, action):
        obs, reward, done, _, info = self.env.step(action)

        # apply modified observation logic w/ action mask
        modified_obs = self.get_modified_observation(obs, self.env.get_attr('possible_actions'))
        numeric_obs = np.array(modified_obs, dtype=np.float32)

        valid_actions = np.where(self.action_mask)[0]
        if action not in valid_actions:
            action = np.random.choice(valid_actions)

        self.decoy_history = update_decoy_history(
            self.env.get_attr('possible_actions'), action, self.decoy_history_init, self.decoy_history
        )
        
        return numeric_obs, reward, done, False, info
    
    
class PPOAgentReactDecoys(PPOAgent):
    

    def set_initial_values(self, action_space=None, observation=None):
        self.cyborg_env.set_initial_values(
            self.cyborg_env.env.get_attr('possible_actions'), 
            self.cyborg_env.env.get_observation("Blue"))
            
            
    def __init__(self, model_file=None, scenario_config="Scenario2_resilience", red_agent=B_lineAgent, seed=1, learning_rate=0.0003):
        print("Loading Blue-RD PPO Agent")
        if model_file is not None:
            self.model = MaskablePPO.load(model_file)
        else:
            path = str(inspect.getfile(CybORG))
            path = path[:-10] + f'/Shared/Scenarios/{scenario_config}.yaml'
            cyborg = ChallengeWrapper('Blue', 
                                    CybORG(path, 'sim', 
                                    agents={'Red': red_agent,
                                    'Green': GreenAgent}),
                                    max_steps=500)
            
            env = ModifiedObservationWrapper(cyborg)
            self.model = MaskablePPO('MlpPolicy', env,
                stats_window_size=500,
                batch_size=32768,      # Match sgd_minibatch_size
                policy_kwargs=dict(net_arch=[256, 256], activation_fn=nn.Tanh),
                seed=seed,
                learning_rate=learning_rate,
            )
        self.cyborg_env = None
        
    def set_env(self, env):
        self.cyborg_env = ModifiedObservationWrapper(env)
        self.set_initial_values()
        
    def get_action(self, observation, action_space):
        obs = self.cyborg_env.get_modified_observation(
             observation, 
             self.cyborg_env.env.get_attr('possible_actions'))
        action, _ = self.model.predict(obs, action_masks=self.cyborg_env.action_mask)
        valid_actions = np.where(self.cyborg_env.action_mask)[0]
        if action not in valid_actions:
            action = np.random.choice(valid_actions)

        self.cyborg_env.decoy_history = update_decoy_history(
            self.cyborg_env.env.get_attr('possible_actions'), action, self.cyborg_env.decoy_history_init, self.cyborg_env.decoy_history
        )
        return action
       
    def update_state_by_action(self, action):
        #update the env for maintained decoys in internal env state
        self.cyborg_env.decoy_history = update_decoy_history(
            self.cyborg_env.env.get_attr('possible_actions'), action, self.cyborg_env.decoy_history_init, self.cyborg_env.decoy_history
        )
        
    def get_action_probabilities(self, observation, action_space=None):
        obs = self.cyborg_env.get_modified_observation(
             observation, 
             self.cyborg_env.env.get_attr('possible_actions'))
        obs = torch.tensor(obs, device = self.model.device).unsqueeze(0)
        dis = self.model.policy.get_distribution(obs, action_masks=self.cyborg_env.action_mask)
        probs = dis.distribution.probs
        probs_np = probs.squeeze(0).detach().cpu().numpy()
        return probs_np
    
    