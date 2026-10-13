# Test file to run CybORG challenge 1


from CybORG.Shared.Enums import TrinaryEnum
from pprint import pprint
import statistics as stats
import pandas as pd
import copy
import numpy as np


        
# Calculates CIA attributes given a CAGE simulation environment step
# NOTE: OLD CIA METRIC FOR CAGE 1
class CIATriadMetric:
    
    def __init__(self,
            user_host_w:float=0.2,
            enterprise_host_w:float=0.3,
            operational_host_w:float=0.5,
            availability_multiplier:float=1.0,
            restoration_weight:float=0.8
            ):
        
        #weights assigned to different host types
        self.user_host_w = user_host_w
        self.enterprise_host_w = enterprise_host_w
        self.operational_host_w = operational_host_w
        
        #multiplier for availability. Each step a system is down, availability
        # is impacted x multipler
        self.availability_multiplier = availability_multiplier
        self.restoration_weight = restoration_weight
        
        #holds CIA over each step of the game
        self.confidentialities = []
        self.integrities = []
        self.availabilities = []
        
        self.network_topology = {} #holds information about each host on the network over time
        
        #path to random agent files for normalizing scores
        self.path_to_random_agent_runs = "./results/random_baselines"
        self.normalization_df = None
        
    def calculate_scores(self, env, blue_action=None, blue_obs=None):
        """Calculates and returns c, i, and a scores for the current step

        Args:
            env (CybORG): Cyborg environment class
            blue_action (Action): Blue agent action to overwrite one in the env
            blue_obs (Observation): Blue agent obs to overwrite one in the env

        """

        c, i, a = self._calculate_cia_host_and_action(env, blue_action, blue_obs)
            
        if blue_action is None:
            #only add to metric if not testing
            self.confidentialities.append(c)
            self.integrities.append(i)
            self.availabilities.append(a)
        return c, i, a

    def _calculate_cia_host_and_action(self, env, blue_action=None, blue_obs=None):
        """Calculate CIA attributes from cyborg observation table. 

        Args:
            env (CybORG): CybORG class environment with dictionary-like observations
        """

        confidentiality, integrity, availability = 0,0,0
        
        # Blue agent action posibilities: ['Sleep', 'Monitor', 'Analyse', 'Remove', 'Restore']
        blue_agent_env_action = env.get_last_action("Blue") if blue_action is None else blue_action
        blue_agent_action = blue_agent_env_action.__class__.__name__
        blue_agent_obs = env.get_observation("Blue") if blue_obs is None else blue_obs

        #temporary to replace later if not in 'test' mode
        network = copy.deepcopy(self.network_topology)
        
        red_agent_env_action = env.get_last_action("Red")
        red_agent_action = red_agent_env_action.__class__.__name__
        red_agent_obs = env.get_observation("Red")

        
        # Confidentiality: 
        # -1.0 x host type if red agent successfully exploits a host
        # 1.0 x host type if blue agent successfully removes red agent access to a host
        if red_agent_action == "ExploitRemoteService" and red_agent_obs["success"] == TrinaryEnum.TRUE:
            exploited_host = red_agent_obs[str(red_agent_env_action.ip_address)]["System info"]["Hostname"]
            if exploited_host != "Defender": #TODO: MeanderAgent atacks defender? 
                network[exploited_host]["Confidentiality"] = -1.0
                
        if blue_agent_action == "Remove" and blue_agent_obs["success"] == TrinaryEnum.TRUE:
            # only count as an improvement in confidentiality if red agent had a shell open in the host
            if blue_agent_env_action.hostname in network and network[blue_agent_env_action.hostname]["Confidentiality"] < 0:
                network[blue_agent_env_action.hostname]["Confidentiality"] = 1.0

        # calculate confidentiality
        for sys, details in network.items():
            #weigh system based on host type
            confidentiality += network[sys]["Confidentiality"]*self.get_sys_weight(sys)
            
            
        # Integrity: 
        # -1.0 x host type if red agent successfully gets root access on a host
        # 1.0 x host type if blue agent successfully removes root access of a red agent
        #       or if red agent's root access fails (was removed in a prior step)
        if red_agent_action == "PrivilegeEscalate":
            if red_agent_obs["success"] == TrinaryEnum.TRUE: 
                network[red_agent_env_action.hostname]["Integrity"] = -1.0
                
        # calculate integrity
        for sys, details in network.items():
            #weigh system based on host type
            integrity += network[sys]["Integrity"]*self.get_sys_weight(sys)
            
            
        # Availability: 
        # 1.0 x host type if service is available
        # -1.0 x host type x availability_multiplier if service is not available
        #       and will be set to 1.0 if blue agent performs a "Restore"
        
        # first check any hosts that were restored prior
        for host in network.keys():
            if network[host]["Restoring"] == 1.0:
                #next step restore
                network[host]["Restoring"] = 0.0
                network[host]["Available"] = 1.0
                
        if red_agent_action == "Impact" and red_agent_obs["success"] == TrinaryEnum.TRUE:
            #if this was already impacted, apply multiplier
            if network[red_agent_env_action.hostname]["Restoring"] != 1.0:
                if network[red_agent_env_action.hostname]["Available"] < 0:
                    network[red_agent_env_action.hostname]["Available"] *= self.availability_multiplier
                else:
                    network[red_agent_env_action.hostname]["Available"] = -1.0
        if blue_agent_action == "Restore" and blue_agent_obs["success"] == TrinaryEnum.TRUE and \
            blue_agent_env_action.hostname in network:
            # doing restore resets the host to its initial state, so everything is restored
            # causes disruption for next step though - reward reflects
            network[blue_agent_env_action.hostname]["Available"] = 0 #next step will be ready
            network[blue_agent_env_action.hostname]["Confidentiality"] = 1.0
            network[blue_agent_env_action.hostname]["Integrity"] = 1.0*self.restoration_weight
            network[blue_agent_env_action.hostname]["Restoring"] = 1.0

        # calculate availability
        for sys, details in network.items():
            #weigh system based on host type
            availability += network[sys]["Available"]*self.get_sys_weight(sys)
        
        if blue_obs is None:
            self.network_topology = network
            
        return confidentiality, integrity, availability
    
    def get_sys_weight(self, system_name):
        # Calculate weight based on system_name string
        sys_w = self.user_host_w
        if "Enterprise" in system_name:
            sys_w = self.enterprise_host_w
        elif "Op" in system_name:
            sys_w = self.operational_host_w
        return sys_w
    
    def reset(self, network_observation):
        # Reset network topology for referencing later
        # and CIA scores
        self.initialize(network_observation)
        self.confidentialities = []
        self.integrities = []
        self.availabilities = []
        
    def initialize(self, network_observation):
        # Set all the hosts and subnets based on intiial network observation contianing all
        # CIA computation is based on this established network topology
        self.network_topology = {}
        for system, details in network_observation.items():
            if system in ["Defender", "success"]: 
                continue
            
            self.network_topology[system] = {
                "Confidentiality": 1,
                "Available": 1,
                "Integrity": 1,
                "Restoring": 0 #whether restoration is in progress, delay availability by 1 step
            }

    def get_max_scores(self):
        c = i = a = len(self.network_topology.keys())
        return c, i, a
    
    def get_min_scores(self):
        #TODO: Actual calculation based on steps and multipliers
        c = i = a = -1e3
        return c, i, a
    
    def index_for_target(self, attr):
        #returns index of c, i, a for distance fn
        mapping = {"c": 0, "i": 1, "a": 2}
        return mapping.get(attr.lower(), -1)  # returns -1 if not found
            
    def scores(self, use_avg=False, 
                normalize=False,
                norm_step_size = 30, 
                red_agent_name = "Bline"):
        """Function to output final CIA scores for an entire episode.

        Args:
            use_avg (bool): _description_. Switch between using sum of CIA over steps (False), or the mean (True)
            normalize (bool): _description_. Whether values should be normalized. Requires csv file for norm.
            norm_step_size (int): _description_. Game steps to norm against
            red_agent_name (str): _description_. Red agent results to norm against.
        """
        #TODO: Does average make more sense than the sum here?
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
        if normalize:
            if self.normalization_df is None:
                pathtodf = self.path_to_random_agent_runs
                if use_avg:
                    pathtodf + "_averaged"
                self.normalization_df = pd.read_csv(pathtodf + ".csv")
                self.normalization_df = (
                    self.normalization_df.set_index(["Agent", "Steps"])
                    .apply(lambda row: {
                        "C": {"mean": row["C Mean"], "std": row["C Std."]},
                        "I": {"mean": row["I Mean"], "std": row["I Std."]},
                        "A": {"mean": row["A Mean"], "std": row["A Std."]}
                    }, axis=1)
                    .groupby(level=0).apply(lambda s: s.droplevel(0).to_dict())
                    .to_dict())
            else:
                #need csv file for normalization values
                raise ValueError("Missing Normalization File With Mean and Std. Dev.")
            
            for a, v in results.items():
                mean = self.normalization_df[(red_agent_name, norm_step_size)][a]["mean"]
                std = self.normalization_df[(red_agent_name, norm_step_size)][a]["std"]
                results[a] = (v - mean) / std
            
        return results
    
    def calculate_per_action_scores(self, env, action_possibilities):
        """Returns C, I, and A per blue agent action

        Args:
            env (_type_): _description_
        """
        
        #determine c, i, and a of each by overwritting blue obs and action in above cia calculate fns
        
        cias = []
        blue_obs_filler = {"success": TrinaryEnum.TRUE}
        idx = 0 
        for action in action_possibilities:
            if hasattr(action, "hostname") and action.hostname == "Defender":
                c = i = a = -1
            else:
                c, i, a = self.calculate_scores(env, action, blue_obs_filler)
            cias.append([c, i, a])
            idx += 1
        return np.array(cias)
    
    def __str__(self):
        return str(self.scores())
    
class CIATriadBlueOnlyMetric(CIATriadMetric):
    
    def __init__(self):
        pass
    
    def calculate_scores(self, env, blue_action=None, blue_obs=None):
        c, i, a = self._calculate_cia_blue_only(env, blue_action, blue_obs)
        if blue_action is None:
            #only add to metric if not testing
            self.confidentialities.append(c)
            self.integrities.append(i)
            self.availabilities.append(a)
        return c, i, a
    
    def _calculate_cia_blue_only(self, env, blue_action=None, blue_obs=None):
        """An alternative scoring approach that only relies on blue agent actions,
        regardless of system status or other factors E.g., availability over time

        Args:
            env (CybORG): CybORG class environment with dictionary-like observations
        """
        confidentiality, integrity, availability = 0,0,0
        blue_agent_env_action = env.get_last_action("Blue") if blue_action is None else blue_action
        blue_agent_action = blue_agent_env_action.__class__.__name__
        blue_agent_obs = env.get_observation("Blue") if blue_obs is None else blue_obs
        
        if blue_agent_obs["success"] == TrinaryEnum.TRUE:
            if blue_agent_action == "Remove":
                integrity += 1.0
            elif blue_agent_action == "Restore":
                availability += 1.0
            elif blue_agent_action == "Misinform":
                confidentiality += 1.0

        return confidentiality, integrity, availability
        
        

#
# Example:
# metric = CIATriadMetric()
# 
# Initialize network topology: 
# metric.initialize(env.get_action_state("Blue"))
#
# Calculate CIA for one step:
# c, i, a = metric.calculate_scores(env)
#
# After finishing run to get CIA scores across an entire episode:
# results = metric.scores()
#
# or to calculate CIA scores as the average across an entire episode:
# results = metric.scores(use_avg=True)
#
#
# To output normalized CIA scores, use the random agent baselines 
#   csv files in the 'results' folder. norm_step_size is the number of
#   steps in one episode, and red_agent_name is the red agent, either 'Bline' or 'Meander'
# results = metric.scores(normalize=True, norm_step_size=30, red_agent_name='Bline')
# 
