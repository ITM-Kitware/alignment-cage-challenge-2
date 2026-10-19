import gymnasium as gym
import numpy as np
import sys
import warnings
from collections import defaultdict
from loguru import logger

from CybORG.Shared.Actions import *
from CybORG.Shared.Enums import OperatingSystemType, TrinaryEnum

sys.modules["gym"] = gym
warnings.filterwarnings("ignore")

decoy_map = {
    DecoyApache: {
        "port": 80,
    },
    DecoyFemitter: {"port": 21, "OSType": OperatingSystemType.WINDOWS},
    DecoyHarakaSMPT: {"port": 25, "OSType": OperatingSystemType.LINUX},
    DecoySmss: {"port": 139, "OSType": OperatingSystemType.WINDOWS},
    DecoySSHD: {
        "port": 22,
    },
    DecoySvchost: {"port": 3389, "OSType": OperatingSystemType.WINDOWS},
    DecoyTomcat: {
        "port": 443,
    },
    DecoyVsftpd: {"port": 21, "OSType": OperatingSystemType.LINUX},
}

# importance-based cost for confidentiality
confidentiality_mapping = {
    "None": 0.0,
    "Low": 0.1,
    "Medium": 5.0,
    "High": 10.0,
}

COST_INTEGRITY = 10.0  # per altered record
COST_AVAILABILITY = 10.0  # per unavailable service

REWARD_MAX_DECIMAL_PLACES = 1


def valid_action_mask(action_space, decoy_history):

    amsize = len(action_space)
    new_mask = [True] * amsize

    for i in range(amsize):
        # if type(action_space[i]) in [Sleep, Remove, Block]:
        if type(action_space[i]) in [Sleep, Remove]:
            new_mask[i] = False
        elif decoy_history[i] == True:
            new_mask[i] = False

    action_mask = np.array(new_mask, dtype=np.int64)
    return action_mask


def valid_action_mask_v2(action_space, filter_out_actions):

    amsize = len(action_space)
    new_mask = [True] * amsize

    for i in range(amsize):
        for act_str in filter_out_actions:
            if act_str in str(action_space[i]):
                #print(str(action_space[i]))
                new_mask[i] = False

    action_mask = np.array(new_mask, dtype=np.int64)
    return action_mask


# react_list could be [Remove, Restore]
# or just [Restore]
def get_possible_reactions(action_space, react_list):
    possible_reactions = []
    for act in action_space:
        if type(act) in react_list:
            possible_reactions.append(action_space.index(act))
    return possible_reactions


def get_compromised_hosts(cyborg_env, obs_vector):
    user_hosts = []
    escalated_hosts = []

    hosts = list(cyborg_env.environment_controller.state.hosts.keys())

    i = 1
    j = 0
    while i < len(obs_vector):
        if obs_vector[i] != -1:
            compromised_status = [obs_vector[i + 2], obs_vector[i + 3]]
            if compromised_status == [0, 1]:  # user
                user_hosts.append(hosts[j])
            elif compromised_status == [1, 1]:  # escalated
                escalated_hosts.append(hosts[j])
            j = j + 1
        i = i + 5
    return user_hosts, escalated_hosts


def filter_reactions_v2(user_hosts, escalated_hosts, possible_reactions, action_space):
    filtered_reactions = []
    for act_index in possible_reactions:
        act = action_space[act_index]
        if act.hostname in escalated_hosts and type(act) == Restore:
            filtered_reactions.append(act_index)  # just restores on escalated hosts
        elif act.hostname in user_hosts:
            filtered_reactions.append(
                act_index
            )  # either removes or restores on user hosts

    return filtered_reactions


def filter_reactions(user_hosts, escalated_hosts, possible_reactions, action_space):

    filtered_reactions_root = []
    filtered_reactions_user = []

    for act_index in possible_reactions:
        act = action_space[act_index]

        if act.hostname in escalated_hosts:
            filtered_reactions_root.append(act_index)

        elif act.hostname in user_hosts:
            filtered_reactions_user.append(act_index)

    if len(filtered_reactions_root) > 0:
        return filtered_reactions_root

    return filtered_reactions_user


def filter_action_mask(action_space, keep_indexes):

    amsize = len(action_space)
    new_mask = [False] * amsize
    for i in keep_indexes:
        new_mask[i] = True

    return np.array(new_mask, dtype=bool)


def reset_decoy_history(action_space):
    action_space_size = len(action_space)
    decoy_history = [False] * action_space_size
    return decoy_history


def print_valid_decoys(action_space, decoy_history, host):
    action_space_size = len(action_space)
    for i in range(action_space_size):
        if decoy_history[i] == False:
            act = action_space[i]
            if "Decoy" not in str(act):
                continue
            if host not in act.hostname:
                continue
            #print(type(act), act.hostname, i)


def get_host_os(observation):
    host_os = {}
    for h, v in observation.items():
        if h == "success":
            continue
        # print(h, v['System info']['OSType'])
        host_os[h] = v["System info"]["OSType"]
    return host_os


# compute scanning freq to host based on observation
def host_scanning(cyborg_env, obs_dict, host_dict, host):
    host_ip_map = cyborg_env.environment_controller.hostname_ip_map

    # print("db_ip", db_ip)
    if host not in obs_dict:
        return host_dict
    if "Interface" not in obs_dict[host]:
        return host_dict

    host_ip = obs_dict[host]["Interface"][0]["IP Address"]
    procs_dict = obs_dict[host]["Processes"][0]
    for conn, val in procs_dict.items():
        if conn != "Connections":
            continue
        conn_info = val[0]
        remote_addr = conn_info["remote_address"]
        local_addr = conn_info["local_address"]
        if local_addr != host_ip:
            continue  # should not happen!?
        remote_host = list(host_ip_map.keys())[
            list(host_ip_map.values()).index(remote_addr)
        ]
        host_dict[remote_host] += 1


def prune_decoys_on_used_ports(cyborg_env, action_space, observation, decoy_history):
    processes_per_host = {}
    for hostname, host in cyborg_env.environment_controller.state.hosts.items():
        processes_per_host[hostname] = set()
        for proc in host.processes:
            if proc.open_ports == None:
                continue
            for p in proc.open_ports:
                processes_per_host[hostname].add(p["local_port"])

    # print(processes_per_host)

    host_os = get_host_os(observation)

    action_space_size = len(action_space)
    for i in range(action_space_size):
        if "Decoy" not in str(action_space[i]):
            continue
        decoy_action = type(action_space[i])

        decoy_os = None
        if "OSType" in decoy_map[decoy_action]:
            decoy_os = decoy_map[decoy_action]["OSType"]

        hostname = action_space[i].hostname
        if decoy_os and decoy_os != host_os[hostname]:
            decoy_history[i] = True

        # if a port is already in use by other processes, we can not deploy decoys on that port
        # will just mask out that action
        decoy_port = decoy_map[decoy_action]["port"]
        if decoy_port in processes_per_host[hostname]:
            decoy_history[i] = True

    return decoy_history


def update_decoy_history(action_space, action, decoy_history_init, decoy_history):

    # once a decoy on a host+port was deployed, trying to re-deploy it will fail and waste time steps
    # will mask out corresponding decoy action as invalid so it is not called again during the same episode
    if "Decoy" in str(action_space[action]):
        decoy_history[action] = True

    # after restore, decoys can be deployed again if available
    elif type(action_space[action]) == Restore:
        hostname = action_space[action].hostname
        for i in range(len(action_space)):
            if "Decoy" not in str(action_space[i]):
                continue
            if action_space[i].hostname == hostname:
                decoy_history[i] = decoy_history_init[i]

    return decoy_history


def deployed_decoys(action_space, decoy_history_init, decoy_history, assets_only=True):
    deployed = defaultdict(lambda: [])
    available = defaultdict(lambda: [])
    for i in range(len(action_space)):
        act = action_space[i]
        if "Decoy" not in str(act):
            continue
        if (
            assets_only
            and "Auth" not in act.hostname
            and "Database" not in act.hostname
            and "Front" not in act.hostname
        ):
            continue
        if decoy_history_init[i] == False and decoy_history[i] == True:
            deployed[act.hostname].append(i)
        elif decoy_history_init[i] == False and decoy_history[i] == False:
            available[act.hostname].append(i)

    return deployed, available


def calculate_reward_conf(scenario, red_action, red_success):
    disrupted = []
    if type(red_action) is Impact and red_success == TrinaryEnum.TRUE:
        # just these two services for confidentiality
        if "Auth" in red_action.hostname or "Database" in red_action.hostname:
            disrupted.append(red_action.hostname)

    scores = [
        confidentiality_mapping[scenario.get_host(i).get("AvailabilityValue", "Medium")]
        for i in disrupted
    ]
    impacted_hosts = dict(zip(disrupted, scores))
    reward = sum(scores)

    return round(reward, REWARD_MAX_DECIMAL_PLACES), impacted_hosts


def calculate_reward_integrity(scenario, red_action, red_success):
    disrupted = []
    if type(red_action) is Impact and red_success == TrinaryEnum.TRUE:
        # just these two services for integrity
        if "Auth" in red_action.hostname or "Front" in red_action.hostname:
            disrupted.append(red_action.hostname)

    scores = [COST_INTEGRITY for i in disrupted]
    impacted_hosts = dict(zip(disrupted, scores))
    reward = sum(scores)

    return round(reward, REWARD_MAX_DECIMAL_PLACES), impacted_hosts
