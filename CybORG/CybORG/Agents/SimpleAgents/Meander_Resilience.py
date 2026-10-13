import random

from CybORG.Agents.SimpleAgents.BaseAgent import BaseAgent
from CybORG.Shared import Results
from CybORG.Shared.Actions import PrivilegeEscalate, ExploitRemoteService, DiscoverRemoteSystems, Impact, \
    DiscoverNetworkServices

import numpy as np

class RedMeanderAgent_Resilience(BaseAgent):
    # a red agent that meanders through scenario 1b
    def __init__(self):
        self.scanned_subnets = []
        self.scanned_ips = []
        self.exploited_ips = []
        self.escalated_hosts = []
        self.host_ip_map = {}
        self.last_host = None
        self.last_ip = None
        self.focused_impact = True
        
    def train(self, results: Results):
        """allows an agent to learn a policy"""
        pass

    def resilience_impact(self):
        for target in ["Auth", "Front", "Database"]:
            matches = [h for h in self.escalated_hosts if h.startswith(target)]
            if matches:
                chosen = matches[0]
                self.last_host = chosen
                return chosen
            
    def non_resilience_impact(self):
        #randomly decide host to impact
        allowed_escalated_hosts = [e for e in self.escalated_hosts if "User" not in e and "Auth" not in e and "Database" not in e and "Front" not in e]
        if len(allowed_escalated_hosts) > 0:
            #impact randomly
            host_to_impact = random.choice(allowed_escalated_hosts)
            self.last_host = host_to_impact
            return host_to_impact
            
    def run_impact(self):
        if self.focused_impact:
            return self.resilience_impact()
        else:
            if random.choice([0,1]):
                return self.non_resilience_impact()
            
    def get_action(self, observation, action_space):
        """gets an action from the agent that should be performed based on the agent's internal state and provided observation and action space"""
        self._process_success(observation)

        session = list(action_space['session'].keys())[0]
        
        impacted_host = self.run_impact()
        if impacted_host is not None:
            return Impact(agent='Red', hostname=impacted_host, session=session)
            
        # start by scanning
        for subnet in action_space["subnet"]:
            if not action_space["subnet"][subnet] or subnet in self.scanned_subnets:
                continue
            self.scanned_subnets.append(subnet)
            return DiscoverRemoteSystems(subnet=subnet, agent='Red', session=session)
        # discover network services
        # # act on ip addresses discovered in first subnet
        addresses = [i for i in action_space["ip_address"]]
        random.shuffle(addresses)
        for address in addresses:
            if not action_space["ip_address"][address] or address in self.scanned_ips:
                continue
            self.scanned_ips.append(address)

            return DiscoverNetworkServices(ip_address=address, agent='Red', session=session)
        # priv esc on owned hosts
        hostnames = [x for x in action_space['hostname'].keys()]
        random.shuffle(hostnames)
        for hostname in hostnames:
            # test if host is not known
            if not action_space["hostname"][hostname]:
                continue
            # test if host is already priv esc
            if hostname in self.escalated_hosts:
                continue
            # test if host is exploited
            if hostname in self.host_ip_map and self.host_ip_map[hostname] not in self.exploited_ips:
                continue
            
            self.escalated_hosts.append(hostname)
            self.last_host = hostname
            return PrivilegeEscalate(hostname=hostname, agent='Red', session=session)

        # access unexploited hosts
        for address in addresses:
            # test if output of observation matches expected output
            if not action_space["ip_address"][address] or address in self.exploited_ips:
                continue
            self.exploited_ips.append(address)
            self.last_ip = address
            return ExploitRemoteService(ip_address=address, agent='Red', session=session)

        raise NotImplementedError('Red Meander has run out of options!')



    def _process_success(self, observation):
        if self.last_ip is not None:
            if observation['success'] == True:
                self.host_ip_map[[value['System info']['Hostname'] for key, value in observation.items()
                                  if key != 'success' and 'System info' in value
                                  and 'Hostname' in value['System info']][0]] = self.last_ip
            else:
                self._process_failed_ip()
            self.last_ip = None
        if self.last_host is not None:
            if observation['success'] == False:
                if self.last_host in self.escalated_hosts:
                    self.escalated_hosts.remove(self.last_host)
                if self.last_host in self.host_ip_map and self.host_ip_map[self.last_host] in self.exploited_ips:
                    self.exploited_ips.remove(self.host_ip_map[self.last_host])
            self.last_host = None

    def _process_failed_ip(self):
        self.exploited_ips.remove(self.last_ip)

        for prefix in ["Auth", "Database", "Front", "Ent"]:
            hosts = [h for h in self.escalated_hosts if h.startswith(prefix)]
            for host in hosts:
                self.escalated_hosts.remove(host)
                ip = self.host_ip_map[host]
                self.exploited_ips.remove(ip)

    def end_episode(self):
        self.scanned_subnets = []
        self.scanned_ips = []
        self.exploited_ips = []
        self.escalated_hosts = []
        self.host_ip_map = {}
        self.last_host = None
        self.last_ip = None

    def set_initial_values(self, action_space, observation):
        pass

class RedMeanderAgent_Combination(RedMeanderAgent_Resilience):
    def __init__(self, p_cia_impact=0.5):
        super().__init__()
        self.focused_impact = False
        self.p_cia_impact = p_cia_impact #50% of the time, impact a resilience-tied host
        
    #switches between alignment targeting and the other hosts
    def run_impact(self):
        if np.random.choice([0, 1], p=[1.0 - self.p_cia_impact, self.p_cia_impact]):
            return self.resilience_impact()
        else:
            return self.non_resilience_impact()
            
class RedMeanderAgent_Combination_CIA(RedMeanderAgent_Combination):
    def __init__(self):
        super().__init__()
        self.focused_impact = False
        self.p_cia_impact = 1.0 #100% of the time, impact a resilience-tied host
   
class RedMeanderAgent_Combination_CAGE(RedMeanderAgent_Combination):
    def __init__(self):
        super().__init__()
        self.focused_impact = False
        self.p_cia_impact = 0.0 #0% of the time, impact a resilience-tied host
             
class RedMeanderAgent_RandomImpact(RedMeanderAgent_Resilience):
    def __init__(self):
        super().__init__()
        self.focused_impact = False #randomly impact or explore
