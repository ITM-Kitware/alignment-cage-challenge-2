from CybORG.Agents.SimpleAgents.BaseAgent import BaseAgent
import numpy as np

class CIAAttributeAgent(BaseAgent):
    def __init__(self, attributes=None):
        # attributes is a dict with {attribute: val}
        # where val = 1 for high and 0 for low
        self.attributes = attributes

    def set_env(self, cage):
        self.env = cage.env
        self.set_metric(cage.metric)
        
    def set_metric(self, metric):
        self.metric = metric
        
    def get_action(self, observation, action_space=None):
        probs = self.get_action_probabilities()
        action = np.random.choice(np.flatnonzero(probs == probs.max()))
        return action

    def get_action_probabilities(self, observation=None, action_space=None):
        scores = self.metric.calculate_per_action_scores(self.env, self.env.get_attr('possible_actions'))
        
        #define some target attribute score arr        
        #assume attr array is always in shape [c,i,a]
        target, target_attr_idx = [], []
        for a, v in self.attributes.items():
            if v >= 0.5: #max
                target.append(self.metric.get_max_scores())
            else: #min
                target.append(self.metric.get_min_scores())
            target_attr_idx.append(self.metric.index_for_target(a))
        #only pick by attrs of focus for alignment
        scores = scores[:, target_attr_idx]
        distances = np.linalg.norm(scores - np.array(target), axis=1)
        #softmax
        probs = np.exp(-distances) / np.sum(np.exp(-distances))

        return probs
    
    
    def train(self):
        pass

    def end_episode(self):
        pass

    def set_initial_values(self, action_space, observation):
        pass
    
    def __str__(self):
        attrstr = "_".join(f"{k.upper()}{v}" for k, v in sorted(self.attributes.items()))
        return f"Attribute {attrstr} agent"