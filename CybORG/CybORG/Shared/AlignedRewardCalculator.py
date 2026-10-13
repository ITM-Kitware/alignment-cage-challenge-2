from CybORG.Shared.BlueRewardCalculator import HybridAvailabilityConfidentialityRewardCalculator
from CybORG.Shared import Scenario
from loguru import logger
from CybORG.AlignmentMetric.resilience_measure import ResilienceMetric 
from CybORG.AlignmentMetric.cia_triad_measure import CIATriadMetric
import numpy as np

WARMUP_EPISODES = 10 #for normalization

class AlignedRewardCalculator(HybridAvailabilityConfidentialityRewardCalculator):
    
    def __init__(self, agent_name: str, scenario: Scenario, agent_reward_params={}):
        super().__init__(agent_name, scenario)
        self.target_attribute = agent_reward_params.get('align_attribute', 'C')
        self.metric = self.__set_metric(agent_reward_params.get('alignment_metric', None))
        self.gamma = agent_reward_params.get("alpha", 0.0)
        self.alpha = self.gamma
        self.count = 0
        self.mean = 0.0
        self.M2 = 0.0
        self.eps = 1e-8
        self.clip_range = 5.0  # PPO-friendly
        self.warm_up_episodes = 0
        
    def reset(self):
        super().reset()
        self.warm_up_episodes +=1

    def __set_metric(self, metric):
        #Factory method to initialize metric for reward calculation

        if metric == "ResilienceMetric":
            return ResilienceMetric()
        elif metric == "CIATriad":
            return CIATriadMetric() #TODO: setup older CIA triad metric for reward
        else:
            return None 
       
    def _update_running_stats(self, x: float):
        self.count += 1
        delta = x - self.mean
        self.mean += delta / self.count
        delta2 = x - self.mean
        self.M2 += delta * delta2
        
    def _get_std(self):
        if self.count < 2:
            return 1.0
        return np.sqrt(self.M2 / (self.count - 1))
    
    def reward_penalty(self, cage_reward, metric_score):
        #apply a standard penalty
        #assumes metric_score is scaled same way as cage_reward
        #cage_norm = cage_reward  / self.cage_rms.rms
        reward = cage_reward + (self.gamma * metric_score)

        return reward

    
    def calculate_reward(self, current_state: dict, action: dict, agent_observations: dict, done: bool) -> float:
        cage_reward = super().calculate_reward(current_state, action, agent_observations, done)
        # normalize only after warm-up episodes
        if self.warm_up_episodes >= WARMUP_EPISODES:
            std = self._get_std()
            cage_reward_norm = (cage_reward - self.mean) / (std + self.eps)
            cage_reward_norm = np.clip(
                cage_reward_norm, -self.clip_range, self.clip_range
            )
        else:
            cage_reward_norm = cage_reward  # raw during warm-up
            self._update_running_stats(cage_reward)
        
        align_score = 0
        if self.metric is not None:
            metric_scores = self.metric.calculate_score_from_actions(
                red_action=action['Red'],
                red_obs=agent_observations['Red'].data,
                blue_action=action['Blue'], 
                blue_obs=agent_observations['Blue'].data)
            align_score = metric_scores[self.target_attribute]*-1  / self.metric.get_max_scores()[self.metric.index_for_target(self.target_attribute)]
        


        reward = self.reward_penalty(cage_reward_norm, align_score)
        # if self.warm_up_episodes >= WARMUP_EPISODES:
        #     print(self.warm_up_episodes, "\n")
        #     print(action['Red'], action['Blue'])
        #     print(align_score, metric_scores, cage_reward, cage_reward_norm, reward)

        return reward, cage_reward, align_score