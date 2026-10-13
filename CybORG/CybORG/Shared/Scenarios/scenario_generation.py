###
# File generates scenario.yaml files according to configs
###

import yaml
from pprint import pprint
import argparse
import math
from collections import defaultdict
import random

REWARD_MAP = {
    0: "None",
    1: "Low",
    2: "Medium",
    3: "High"
}

def random_user_image():
    return random.choice(["Internal", "Gateway"])   # Windows / Linux mix

def parse_subnet_indices(value):
    # Returns subnet idxs (accepts int, list/tuple, or "0,1" string)
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [int(v) for v in value]
    if isinstance(value, str):
        parts = [p.strip() for p in value.split(',') if p.strip() != '']
        return [int(p) for p in parts]
    return [int(value)]

def get_allowed_subnets(subnets):
    """
    Return AllowedSubnets in the example order:
    prefer ['User','Enterprise','Operational'] but only include ones actually present.
    """
    preferred_order = ['User', 'Enterprise', 'Operational']
    return [name for name in preferred_order if name in subnets]

######
# Make agents
######

def make_red_agent(hosts, subnets):
    user_hosts = [h for h in hosts if h.startswith("User")]
    red_host = random.choice(user_hosts) if user_hosts else None

    red_actions = [
        "Sleep",
        "DiscoverRemoteSystems",
        "DiscoverNetworkServices",
        "ExploitRemoteService",
        "BlueKeep",
        "EternalBlue",
        "FTPDirectoryTraversal",
        "HarakaRCE",
        "HTTPRFI",
        "HTTPSRFI",
        "SQLInjection",
        "PrivilegeEscalate",
        "Impact",
        "SSHBruteForce"
    ]

    red_sessions = []
    if red_host:
        red_sessions.append({
            "username": "SYSTEM",
            "hostname": red_host,
            "type": "RedAbstractSession",
            "name": "RedPhish",
        })

    red_int = {"Hosts": {}}
    if red_host:
        red_int["Hosts"][red_host] = {"Interfaces": "All", "System info": "All"}

    return {
        "agent_type": "SleepAgent",
        "wrappers": [],
        "actions": red_actions,
        "starting_sessions": red_sessions,
        "reward_calculator_type": "HybridImpactPwn",
        "INT": red_int,
        "AllowedSubnets": get_allowed_subnets(subnets),
    }

def make_blue_agent(hosts, subnets, reward_type):
    blue_actions = [
        "Sleep",
        "Monitor",
        "Analyse",
        "Remove",
        "DecoyApache",
        "DecoyFemitter",
        "DecoyHarakaSMPT",
        "DecoySmss",
        "DecoySSHD",
        "DecoySvchost",
        "DecoyTomcat",
        "DecoyVsftpd",
        "Restore"
    ]

    blue_host_dict = {
        host: {"Interfaces": "All", "System info": "All", "User info": "All"}
        for host in hosts
    }
    blue_int = {"Hosts": blue_host_dict}

    blue_sessions = []
    # Server session: Defender
    blue_sessions.append({
        "hostname": "Defender",
        "name": "VeloServer",
        "type": "VelociraptorServer",
        "username": "ubuntu",
        "num_children_sessions": 2,
        "artifacts": ["NetworkConnections", "ProcessCreation"]
    })

    # Client on every host
    for host in hosts:
        blue_sessions.append({
            "hostname": host,
            "name": f"Velo{host}",
            "type": "VelociraptorClient",
            "username": "ubuntu",
            "parent": "VeloServer"
        })

    return {
        "agent_type": "SleepAgent",
        "wrappers": [],
        "actions": blue_actions,
        "starting_sessions": blue_sessions,
        "reward_calculator_type": reward_type,
        "INT": blue_int,
        "AllowedSubnets": get_allowed_subnets(subnets),
        "adversary": "Red"
    }

def make_green_agent(hosts, subnets):
    
    visible_hosts = [h for h in hosts if h != "Defender"]

    green_actions = ["Sleep", "GreenPingSweep", "GreenPortScan", "GreenConnection"]

    green_sessions = [{
        "username": "GreenAgent",
        "hostname": host,
        "type": "green_session",
        "name": "GreenSession",
    } for host in visible_hosts]

    green_host_dict = {
        host: {"Interfaces": "All", "System info": "All", "User info": "All"}
        for host in visible_hosts
    }
    green_int = {"Hosts": green_host_dict}

    return {
        "agent_type": "SleepAgent",
        "wrappers": [],
        "actions": green_actions,
        "starting_sessions": green_sessions,
        "reward_calculator_type": "None",
        "INT": green_int,
        "AllowedSubnets": get_allowed_subnets(subnets),
    }

def host_meta(name, idx, subnet_name, host_type, image_map, reward_weight):
    # create host entry
    return {
        'image': (
            random_user_image() if host_type == "User" else image_map.get(host_type, "InternalOT")
            ),
        'AWS_Info': [],
        'ConfidentialityValue': REWARD_MAP.get(reward_weight, "None"),
        'AvailabilityValue': REWARD_MAP.get(reward_weight, "None"),
        'subnet': subnet_name,
        'Type': host_type
    }

def allocate_to_subnets(subnet_names, hosts, subnets, target_indices, count, hostname_prefix, host_type, image_map, reward_weight):
    """
    Allocate hosts across one or more subnets round-robin.
    """
    if count <= 0 or not target_indices:
        return
    for i in range(count):
        subnet_idx = target_indices[i % len(target_indices)]
        subnet_name = subnet_names[subnet_idx]
        idx_within_type = sum(1 for h in hosts if h.startswith(hostname_prefix))
        hostname = f"{hostname_prefix}{idx_within_type}"
        hosts[hostname] = host_meta(hostname, idx_within_type, subnet_name, host_type, image_map, reward_weight)
        subnets[subnet_name]['Hosts'].append(hostname)
        subnets[subnet_name]['Size'] = len(subnets[subnet_name]['Hosts'])

def build_host_connections(hosts):
    user_hosts = []
    enterprise_hosts = []
    op_hosts = []
    res_hosts = []
    defender_hosts = []

    for h, meta in hosts.items():
        t = meta["Type"]
        if t == "User":
            user_hosts.append(h)
        elif t == "Enterprise":
            enterprise_hosts.append(h)
        elif t in ("Op_Server", "Op_Host"):
            op_hosts.append(h)
        elif t in ("Front", "Auth", "Database"):
            res_hosts.append(h)
        elif t == "Defender":
            defender_hosts.append(h)

    # assign connections
    for hostname, meta in hosts.items():
        meta["info"] = {}
        meta["info"][hostname] = {"Interfaces": "All"}

        # defender only gets self
        if meta["Type"] == "Defender":
            continue

        if meta["Type"] == "User":
            # pick 2–4 enterprise connections
            num_conn = min(4, max(2, len(enterprise_hosts)))
            conns = random.sample(enterprise_hosts, num_conn)

            for target in conns:
                meta["info"][target] = {"Interfaces": "IP Address"}

            continue

        if meta["Type"] == "Enterprise":
            for u in user_hosts:
                meta["info"][u] = {"Interfaces": "IP Address"}

            for e in enterprise_hosts:
                if e != hostname:
                    meta["info"][e] = {"Interfaces": "IP Address"}

            # connect to operational + resilience
            for op in op_hosts:
                meta["info"][op] = {"Interfaces": "IP Address"}
            for r in res_hosts:
                meta["info"][r] = {"Interfaces": "IP Address"}

            meta["info"][hostname]["Services"] = ["OTService"]
            continue

        if meta["Type"].startswith("Op_"):
            # connect to other operational hosts
            for op in op_hosts:
                if op != hostname:
                    meta["info"][op] = {"Interfaces": "IP Address"}

            for r in res_hosts:
                meta["info"][r] = {"Interfaces": "IP Address"}

            meta["info"][hostname]["Services"] = ["OTService"]
            continue

        # RESILIENCE HOSTS
        if meta["Type"] in ("Front", "Auth", "Database"):
            # connect to all resilience hosts
            for r in res_hosts:
                if r != hostname:
                    meta["info"][r] = {"Interfaces": "IP Address"}

            # connect to operational hosts
            for op in op_hosts:
                meta["info"][op] = {"Interfaces": "IP Address"}

            meta["info"][hostname]["Services"] = ["OTService"]
            continue

    return hosts

def create_cage_scenario(args):
    # Semantic subnet names in the same order as your example
    subnet_names = ['Enterprise', 'Operational', 'User']
    image_map = {
        'User': 'windows_user_host1',
        'Enterprise': 'InternalOT',
        'Op_Server': 'OP_Server',
        'Op_Host': 'GatewayOT',
        'Front': 'OP_Server',
        'Auth': 'OP_Server',
        'Database': 'OP_Server',
        'Defender': 'Velociraptor_Server',
    }

    subnets = {name: {'Hosts': [], 'Size': 0, 'NACLs': {'all': {'in': 'all', 'out': 'all'}}}
               for name in subnet_names}
    hosts = {}

    user_indices = parse_subnet_indices(getattr(args, 'user_s', 2))
    enterprise_indices = parse_subnet_indices(getattr(args, 'enterprise_s', 0))
    operational_indices = parse_subnet_indices(getattr(args, 'operational_s', 1))
    resilience_indices = parse_subnet_indices(getattr(args, 'resilience_s', 1))
    defender_indices = parse_subnet_indices(getattr(args, 'defender_s', 0))

    # Allocate users: placed in subnet index(s) specified by user_indices
    allocate_to_subnets(
        subnet_names, hosts, subnets, user_indices,
        getattr(args, 'user_n', 5), 'User', 'User', image_map, getattr(args, 'user_w', 1)
    )

    # Enterprise
    allocate_to_subnets(
        subnet_names, hosts, subnets, enterprise_indices,
        getattr(args, 'enterprise_n', 3), 'Enterprise', 'Enterprise', image_map, getattr(args, 'enterprise_w', 2)
    )

    # Operational: 1 server + remaining hosts
    op_count = getattr(args, 'operational_n', 4)  
    if op_count > 0:
        server_count = 1
        host_count = max(0, op_count - server_count)
        allocate_to_subnets(
            subnet_names, hosts, subnets, operational_indices,
            server_count, 'Op_Server', 'Op_Server', image_map, getattr(args, 'operational_w', 3)
        )
        allocate_to_subnets(
            subnet_names, hosts, subnets, operational_indices,
            host_count, 'Op_Host', 'Op_Host', image_map, getattr(args, 'operational_w', 3)
        )

    # Resilience types: create if resilience_n > 0 (each type gets resilience_n hosts)
    resilience_types = ['Front', 'Auth', 'Database']
    resilience_count = getattr(args, 'resilience_n', 0)  
    for r_type in resilience_types:
        allocate_to_subnets(
            subnet_names, hosts, subnets, resilience_indices,
            resilience_count, r_type, r_type, image_map,
            getattr(args, 'resilience_w', getattr(args, 'operational_w', 3))
        )

    # Defender host: create a single host named exactly "Defender"
    if defender_indices:
        idx = defender_indices[0]
        subnet_name = subnet_names[idx]
        hostname = "Defender"
        hosts[hostname] = host_meta(hostname, 0, subnet_name, 'Defender', image_map, 0)
        # Not appended to subnets[subnet_name]['Hosts']
        # Not counted in Size
        subnets[subnet_name]['Hosts'].append(hostname)

    # After host allocation is done:
    hosts = build_host_connections(hosts)

    # Build agents using current hosts/subnets
    agents = {
        "Red": make_red_agent(list(hosts.keys()), subnets),
        "Blue": make_blue_agent(list(hosts.keys()), subnets, args.blue_reward),
        "Green": make_green_agent(list(hosts.keys()), subnets),
    }

    scenario = {
        'Agents': agents,
        'Hosts': hosts,
        'Subnets': subnets
    }

    with open(f"{args.export_path}.yaml", "w") as f:
        yaml.dump(scenario, f, sort_keys=False)
    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--export_path", type=str, default="scenario_example")

    #TODO: Set up subnet generation to increase number of subnets
    parser.add_argument("--subnets", type=int, default=3)
    parser.add_argument("--operational_n", type=int, default=4)   # 1 server + 3 hosts -> Op_Host0..2
    parser.add_argument("--resilience_n", type=int, default=0)    # match example (none created)
    parser.add_argument("--enterprise_n", type=int, default=3)   # Enterprise0..2
    parser.add_argument("--user_n", type=int, default=5)         # User0..4

    # reward levels
    parser.add_argument("--resilience_w", type=int, default=0)
    parser.add_argument("--operational_w", type=int, default=2)
    parser.add_argument("--enterprise_w", type=int, default=3)
    parser.add_argument("--user_w", type=int, default=2)

    # important: set these so indices reference the semantic subnet_names:
    # subnet_names = ['Enterprise'(0), 'Operational'(1), 'User'(2)]
    parser.add_argument("--operational_s", type=int, default=1)
    parser.add_argument("--enterprise_s", type=int, default=0)
    parser.add_argument("--user_s", type=int, default=2)
    parser.add_argument("--resilience_s", type=int, default=1)
    parser.add_argument("--defender_s", type=int, default=0)

    parser.add_argument("--blue_reward", type=str, default="HybridAvailabilityConfidentiality")
    
    
    
    args = parser.parse_args()
    create_cage_scenario(args)