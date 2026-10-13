# Adapted from Quantitative Resilience work
# by Dartmouth team:
# https://github.com/xcadet/CyberResilience/tree/main

from CybORG.Agents.Castle.weighter import WEIGHTER_V2
from CybORG.Agents.Castle.impact_reporters import ImpactReporter, IMPACT_AUTH, IMPACT_DATABASE, IMPACT_FRONT
from CybORG.Agents.Castle.scorer_mappings import CONF_MAP_ORIGINAL, INTEG_MAP_ORIGINAL, AVAIL_MAP_ORIGINAL
from CybORG.Agents.Castle.scorer import AVAIL_DROP_KEY, CONF_DROP_KEY, INTEGRITY_DROP_KEY
import statistics as stats
import numpy as np
from CybORG.Shared.Enums import TrinaryEnum
from CybORG.Shared.Actions import Impact, PrivilegeEscalate, ExploitRemoteService, Restore, Remove

# Update for use in our CAGE experiments
# as per-time-step calculation
# with default CAGE 2
class ResilienceMetric:
    
    def __init__(self, weighter = {
        CONF_DROP_KEY: 0.6,
        INTEGRITY_DROP_KEY: 0.2,
        AVAIL_DROP_KEY: 0.2,
        }, cia_scoring = {
        CONF_DROP_KEY:  CONF_MAP_ORIGINAL, 
        INTEGRITY_DROP_KEY:  INTEG_MAP_ORIGINAL, 
        AVAIL_DROP_KEY:  AVAIL_MAP_ORIGINAL, 
    }):
        self.weighter = weighter
        self.scorer = cia_scoring
        self.impact_reporter = ImpactReporter()
        #Impacts
        self.confidentialities = []
        self.integrities = []
        self.availabilities = []
        self.resiliences = []
    
    
    def calculate_scores(self, env, blue_action=None, blue_obs=None):
        # Assumes a network topology of three operational servers, 
        # called "Database" "Front" and "Auth"
        red_agent_env_action = env.get_last_action("Red")
        red_agent_obs = env.get_observation("Red")

        scores =  self.calculate_score_from_actions(
            red_agent_env_action,
            red_agent_obs,
            blue_action,
            blue_obs
        )
        return scores["C"], scores["I"], scores["A"]
    
    def calculate_score_from_actions(self, red_action, red_obs, blue_action=None, blue_obs=None):
        c, i, a, = 0,0,0
        impact_type, _ = self.impact_reporter(red_action,
                                         red_obs["success"])
        
        # custom map to c, i, and a based on paper
        # Auth: C, I, A
        # Database: C, A
        # Front: I, A
        # Using Costs-2 from paper, where Auth server has 2x impact
        if impact_type == IMPACT_AUTH:
            c += self.scorer[CONF_DROP_KEY][IMPACT_AUTH]
            a += self.scorer[AVAIL_DROP_KEY][IMPACT_AUTH]
            i += self.scorer[INTEGRITY_DROP_KEY][IMPACT_AUTH]
        elif impact_type == IMPACT_DATABASE:
            c += self.scorer[CONF_DROP_KEY][IMPACT_DATABASE]
            a += self.scorer[AVAIL_DROP_KEY][IMPACT_DATABASE]
        elif impact_type == IMPACT_FRONT:
            a += self.scorer[AVAIL_DROP_KEY][IMPACT_FRONT]
            i += self.scorer[INTEGRITY_DROP_KEY][IMPACT_FRONT]
            
        resilience = c*self.weighter[CONF_DROP_KEY] + i*self.weighter[INTEGRITY_DROP_KEY] + a*self.weighter[AVAIL_DROP_KEY]
        # normalization of resilience
        r_max = sum([max(self.scorer[t].values())*self.weighter[t] for t in [CONF_DROP_KEY, INTEGRITY_DROP_KEY, AVAIL_DROP_KEY]])
        resilience /= r_max
        
        if blue_action is None:
            self.confidentialities.append(c)
            self.integrities.append(i)
            self.availabilities.append(a)
            self.resiliences.append(resilience)
        else:
            #calculate c, i, a if blue took action x
            if type(red_action) is ExploitRemoteService and red_obs["success"] == TrinaryEnum.TRUE:
                #blue should Remove. If blue takes a Restore, will also remove but causes down time so is not the prefered action
                exploited_host = red_action.impacted_hostname
                
                if type(blue_action) is Remove and exploited_host == blue_action.hostname:
                    #NOTE: Will attempt the action, and fail unless the exploited_host allows the "Remove" action. See: https://github.com/alan-turing-institute/CybORG_plus_plus?tab=readme-ov-file#action-space
                    if "Auth" in exploited_host:
                        c -= self.scorer[CONF_DROP_KEY][IMPACT_AUTH]
                        a -= self.scorer[AVAIL_DROP_KEY][IMPACT_AUTH]
                        i -= self.scorer[INTEGRITY_DROP_KEY][IMPACT_AUTH]
                    elif "Database" in exploited_host:
                        c -= self.scorer[CONF_DROP_KEY][IMPACT_DATABASE]
                        a -= self.scorer[AVAIL_DROP_KEY][IMPACT_DATABASE]
                    elif "Front" in exploited_host:
                        a -= self.scorer[AVAIL_DROP_KEY][IMPACT_FRONT]
                        i -= self.scorer[INTEGRITY_DROP_KEY][IMPACT_FRONT]
                        
            elif (type(red_action) is Impact or type(red_action) is PrivilegeEscalate) and red_obs["success"] == TrinaryEnum.TRUE:
                #blue should Restore
                exploited_host = red_action.hostname                    
                if type(blue_action) is Restore and exploited_host == blue_action.hostname:
                    if "Auth" in exploited_host:
                        c -= self.scorer[CONF_DROP_KEY][IMPACT_AUTH]
                        a -= self.scorer[AVAIL_DROP_KEY][IMPACT_AUTH]
                        i -= self.scorer[INTEGRITY_DROP_KEY][IMPACT_AUTH]
                    elif "Database" in exploited_host:
                        c -= self.scorer[CONF_DROP_KEY][IMPACT_DATABASE]
                        a -= self.scorer[AVAIL_DROP_KEY][IMPACT_DATABASE]
                    elif "Front" in exploited_host:
                        a -= self.scorer[AVAIL_DROP_KEY][IMPACT_FRONT]
                        i -= self.scorer[INTEGRITY_DROP_KEY][IMPACT_FRONT]

        return {"C":c, "I": i, "A": a}
    
    def reset(self, obs=None):
        self.confidentialities = []
        self.integrities = []
        self.availabilities = []
        self.resiliences = []
        return
    
    def step_scores(self):
        #Get last used scores for the step
        return {
                "C": self.confidentialities[-1],
                "I": self.integrities[-1],
                "A": self.availabilities[-1],
                "Resilience": self.resiliences[-1]
        }
        
    def scores(self, use_avg=False):
        if use_avg:
            results =  {
                "C": stats.mean(self.confidentialities),
                "I": stats.mean(self.integrities),
                "A": stats.mean(self.availabilities)
            }
        else:
            results =  {
                "C": sum(self.confidentialities),
                "I": sum(self.integrities),
                "A": sum(self.availabilities)
            }
        return results
    
    def get_max_scores(self):
        return max(self.scorer[CONF_DROP_KEY].values()), max(self.scorer[INTEGRITY_DROP_KEY].values()), max(self.scorer[AVAIL_DROP_KEY].values())
    
    def get_min_scores(self):
        return 0, 0, 0
    
    def index_for_target(self, attr):
        #returns index of c, i, a for distance fn
        mapping = {"c": 0, "i": 1, "a": 2}
        return mapping.get(attr.lower(), -1)  # returns -1 if not found
    
    def calculate_per_action_scores(self, env, action_possibilities):
        cias = []
        for idx, action in enumerate(action_possibilities):
            if hasattr(action, "hostname") and action.hostname == "Defender":
                c = i = a = 100
            else:
                c, i, a = self.calculate_scores(env, action) #no blue obs b/c outcome unknown
            cias.append([c,i,a])
        return np.array(cias)*-1 #inverse b/c measuring drop