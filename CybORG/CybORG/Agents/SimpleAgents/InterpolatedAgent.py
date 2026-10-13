import inspect

from CybORG.Agents.SimpleAgents.BaseAgent import BaseAgent
from CybORG.Agents.Wrappers import BlueTableWrapper

import numpy as np

class InterpolatedAgent(BaseAgent):
    def train(self):
        pass

    def end_episode(self):
        pass

    def set_initial_values(self, action_space, observation):
        pass

    def __init__(self, agent1, agent2, alpha=0.5, 
                 target_kl=0.25,
                 lr=0.01, 
                 beta=0.9, 
                 adaptive=False):
        self.agent1 = agent1
        self.agent2 = agent2
        self.alpha_base = alpha
        self.alpha = alpha
        self.adaptive = adaptive
        
        # parameters to stay within original policy (of reward focus), e.g., 0.1 =10% of original
        self.target_kl = target_kl
        self.lr = lr
        self.beta = beta
        self.ema_kl = 0.0  # initialize running average
        
        self.attribute_to_host_mapping = {
            "Auth": set(("A", "I", "C")),
            "Database": set(("A", "C")),
            "Front": set(("A", "I"))
        }
    
    def kl_divergence(self, p, q):
        #compute divergence between old/new policy
        p = np.clip(p, 1e-12, 1.0)
        q = np.clip(q, 1e-12, 1.0)
        return np.sum(p * np.log(p / q))
    
    def update_alpha(self, probs_new, orig_prob ):
        kl = self.kl_divergence(orig_prob, probs_new)
        # ema = expected moving average
        #self.ema_kl = self.beta * self.ema_kl + (1 - self.beta) * kl
        # smooth update of alpha based on kl divergence
        self.alpha -= self.lr * (kl - self.target_kl)
        self.alpha = np.clip(self.alpha, 0.0, 1.0)
        
    def convex_combination(self, p1, p2):
        probs = (1.0 - self.alpha)*p1 + self.alpha*p2
        probs /= probs.sum() #normalize to fix fp error with numpy
        return probs
    
    def product_of_experts(self, p1, p2):
        probs = (np.clip(p1, 1e-12, None)**(1-self.alpha)) * (np.clip(p2, 1e-12, None)**self.alpha) 
        probs /= probs.sum()
        return probs
    
    def top_k_filter(self, p1, p2, k=10): #preserve top k actions by p1
        topk_idx = np.argsort(p1)[-k:] #identify top
        probs = np.zeros_like(p1) #default 0
        
        probs[topk_idx] = (1 - self.alpha)*p1[topk_idx] + self.alpha*p2[topk_idx] #only shape top k indexes, else 0
        
        probs /= probs.sum()
        return probs
    
    def set_env(self, env):
        self.env = env
        

    def adaptive_selection(self, probs1, probs2, alignment_attributes):
        #
        # "Adaptive" policy selection, where based on the 
        # state of the environment, one of the policies is picked
        #
        # in cyber, we want to maintaina a lower bound of safety, so if the obs reveals an unsafe scenario, it 
        # prioritizes alignment-based selection versus reward-based
        tbl = self.env.get_observation_table()

        alignment_targets = set(alignment_attributes.keys())

        if hasattr(tbl, "_rows"):
            rows = tbl._rows
        else:
            rows = []
            for line in str(tbl).splitlines():
                line = line.strip()
                if not line.startswith("|"):
                    continue
                if "Hostname" in line and "Activity" in line and "Compromised" in line:
                    continue
                parts = [p.strip() for p in line.split("|") if p.strip()]
                if len(parts) == 5:
                    rows.append(parts)

        impacted_now = False

        # Define what "unsafe" means from Blue's partial belief
        unsafe_compromised = {"User", "Privileged", "Unknown"}

        for row in rows:
            # row format: [Subnet, IP Address, Hostname, Activity, Compromised]
            hostname = row[2]
            activity = row[3]
            compromised = row[4]

            if hostname.startswith("Auth") and (self.attribute_to_host_mapping["Auth"] & alignment_targets):
                if (compromised in unsafe_compromised) or (activity == "Exploit"):
                    impacted_now = True
                    break
            if hostname.startswith("Front") and (self.attribute_to_host_mapping["Front"] & alignment_targets):
                if (compromised in unsafe_compromised) or (activity == "Exploit"):
                    impacted_now = True
                    break
            if hostname.startswith("Database") and (self.attribute_to_host_mapping["Database"] & alignment_targets):
                if (compromised in unsafe_compromised) or (activity == "Exploit"):
                    impacted_now = True
                    break     
            
        # Switch policy to safety one
        self.alpha = 1.0 if impacted_now else 0.0

        return self.convex_combination(probs1, probs2)

    def get_action(self, observation, action_space):
        
        probs1 = self.agent1.get_action_probabilities(observation, action_space)
        probs2 = self.agent2.get_action_probabilities(observation, action_space)
        
        if not self.adaptive:
            probs = self.convex_combination(probs1, probs2)
        else:
            #NOTE: Assumes one of the agents has the attributes for alignment
            alignment_attributes = self.agent2.attributes if hasattr(self.agent2, "attributes") else self.agent1.attributes
            probs = self.adaptive_selection(probs1, 
                                            probs2, 
                                            alignment_attributes)
            
        #self.update_alpha(probs, probs1) #update alpha using kl divergence
        #print max index of probs
        action = np.random.choice([i for i in range(len(probs))], p=probs)

        if hasattr(self.agent1, 'update_state_by_action'):
            self.agent1.update_state_by_action(action)
        if hasattr(self.agent2, 'update_state_by_action'):
            self.agent2.update_state_by_action(action)
            
        return action

    def __str__(self):
        attrstr = "_".join(f"{k.upper()}{v}" for k, v in sorted(self.agent2.attributes.items()))
        return f"RL-a{self.alpha_base}-{attrstr} agent"