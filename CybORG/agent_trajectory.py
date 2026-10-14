
from loguru import logger
import os
from datetime import datetime
import ipaddress
import networkx as nx
import matplotlib.pyplot as plt
import json
import sys
import enum
from dataclasses import asdict, is_dataclass
from collections import defaultdict
import numpy as np


def to_json_safe(obj):
    """Recursively convert any Python object to something JSON-serializable."""
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj

    if isinstance(obj, enum.Enum):
        return obj.name  # or obj.value

    if isinstance(obj, (set, tuple)):
        return [to_json_safe(x) for x in obj]

    if is_dataclass(obj):
        return {k: to_json_safe(v) for k, v in asdict(obj).items()}

    if isinstance(obj, dict):
        return {k: to_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_json_safe(x) for x in obj]

    if hasattr(obj, "__dict__"):
        return {k: to_json_safe(v) for k, v in obj.__dict__.items()}
    return str(obj)

# Maintains trajectory of agent throughout the network for debugging
class AgentTrajectory:
    def __init__(self, blue_agent, 
                 red_agent, 
                 episode_num=0, 
                 experiment_time=datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
                 network={}, 
                 export_path="trajectories/", verbose=False):
        self.blue_agent_name = blue_agent
        self.red_agent_name = red_agent
        self.episode = episode_num
        
        self.export_path = export_path
        self.experiment_time = experiment_time

        if not verbose:
            logger.remove()
            

        logger.info(f"[AGENT BLUE] {self.blue_agent_name} [EPISODE] {self.episode}")
        
        self.network_state = {} #has state of network
        self.blue_actions = [] #has action, hostname, and state
        self.red_actions = [] #has action, hostname, and state
        self.metric_scores = [] #has scores per step of metric
        
        self.set_network_state(network)
        
    def set_network_state(self, initial_obs):
        self.network_topology = {}
        for system, details in initial_obs.items():
            if system == 'success': 
                continue

            self.network_topology[system] = to_json_safe(details)
            
    def add_metric_scores(self, metric):
        self.metric_scores.append(metric)
        
    def add_blue_action(self, cage_action, action_status):
        self.blue_actions.append(
            {"Action": cage_action.__class__.__name__,
             "Status": str(action_status["success"]),
             "Host": cage_action.__class__.__name__ if not hasattr(cage_action, "hostname") else cage_action.hostname}
        )
        logger.opt(colors=True).info(f'<blue>[BLUE AGENT ACTION] {self.blue_actions[-1]["Action"]} {self.blue_actions[-1]["Host"]} {action_status["success"]}</blue>')

    def add_red_action(self, cage_action, action_status, ip_to_hostname, cidr_to_subnet_name):
        if hasattr(cage_action, "hostname"):
            target = {"Host": cage_action.hostname}
        elif hasattr(cage_action, "impacted_hostname"):
            target = {"Host": str(cage_action.impacted_hostname)}
        elif hasattr(cage_action, "ip_address"):
            target = {"Host": ip_to_hostname.get(cage_action.ip_address, str(cage_action.ip_address))}
        elif hasattr(cage_action, "subnet"):
            target = {"Subnet": cidr_to_subnet_name.get(cage_action.subnet, str(cage_action.subnet))}
        else:
            target = {}

        action_data = {
            "Action": cage_action.__class__.__name__,
            "Status": str(action_status["success"]),
            **target
        }

        self.red_actions.append(action_data)
        logger.opt(colors=True).info(f'<red>[RED AGENT ACTION] {action_data["Action"]} {list(target.values())[0] if target else ""} {action_status["success"]}</red>')

    def get_experimentname(self):
        return f"{self.blue_agent_name}-{self.red_agent_name}"
    
    def export(self):
        save_dir = os.path.join(self.export_path, self.get_experimentname())
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f'E{self.episode}.json')

        data = {
            "blue_agent_name": self.blue_agent_name,
            "red_agent_name": self.red_agent_name,
            "episode": self.episode,
            "experiment_time": self.experiment_time,
            "network_topology": self.network_topology,
            "blue_actions": self.blue_actions,
            "red_actions": self.red_actions,
            "metric_scores": self.metric_scores,
        }

        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)

        logger.info(f"[SAVED AGENT JSON] {save_path}")

    @staticmethod
    def load(path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Create the object with whatever minimal info needed for init
        obj = AgentTrajectory(
            blue_agent=data.get("blue_agent_name", "unknown"),
            red_agent=data.get("red_agent_name", "unknown"),
            episode_num=data.get("episode", 0),
            experiment_time=data.get("experiment_time", None),
            network=data.get("network_topology", {}),
        )

        # Restore everything else
        obj.network_state = data.get("network_state", {})
        obj.network_topology = data.get("network_topology", {})
        obj.blue_actions = data.get("blue_actions", [])
        obj.red_actions = data.get("red_actions", [])
        obj.metric_scores = data.get("metric_scores", [])

        logger.info(f"[LOADED AGENT JSON FULL] {path}")
        return obj
    
    
    def view(self, export_path, agent_type="", show_actions=True, limit_steps_to_range=[None, None]):
        #
        # Main function to export trajectory plot
        #
        
        save_dir = os.path.join(export_path, self.get_experimentname())
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f'E{self.episode} {agent_type}.png')
        
        plt.figure(figsize=(12, 7))

        # Create graph
        G = nx.Graph()
        #Add network topology nodes
        for host, data in self.network_topology.items():
            G.add_node(host)

        subnet_groups = defaultdict(list)
        for host, data in self.network_topology.items():
            for iface in data.get("Interface", []):
                if iface["Interface Name"] == "eth0": 
                    # TODO: Only outer IPs for now, need to look into this in CAGE sim
                    # if this changes red agent 
                    # TODO: Also doesn't look like there is a visible connection
                    # between the subnets being drawn
                    
                    ip = iface["Subnet"]["network_address"]
                    prefix = ".".join(ip.split(".")[:3])
                    subnet_groups[prefix].append(host)

        # Connect all hosts within the same subnet
        for subnet, hosts in subnet_groups.items():
            for i in range(len(hosts)):
                for j in range(i + 1, len(hosts)):
                    G.add_edge(hosts[i], hosts[j], subnet=subnet)

        #cluster the subnets
        pos = {}
        cluster_spacing = 0  # distance between clusters
        node_spacing = 0.95   # spacing inside cluster

        for i, (subnet, hosts) in enumerate(subnet_groups.items()):
            cluster_x = (i % 3) * cluster_spacing
            cluster_y = (i // 3) * cluster_spacing

            # local spring layout for nodes in this cluster
            subG = G.subgraph(hosts)
            sub_pos = nx.spring_layout(subG, seed=42, k=node_spacing)
            for node, p in sub_pos.items():
                pos[node] = p + (cluster_x, cluster_y)
                
        extract_path = lambda actions, G: [
            a["Host"]
            for a in actions
            if a.get("Status") == "TRUE" and a.get("Host") and a["Host"] in G.nodes
        ] #ignores non-host-tied actions like scans

        pos = nx.spring_layout(G, seed=42, k=1.25)

        # base network topology
        nx.draw_networkx_nodes(G, pos, node_color="lightgray", node_size=500, edgecolors="black")
        nx.draw_networkx_labels(G, pos, font_size=9, font_weight="semibold")
        nx.draw_networkx_edges(G, pos, width=1, edge_color="gray", alpha=0.4)
        
        # draw agent trajectory, either blue or red
        # gets messy if includes both
        if agent_type.lower() in ["red", "blue"]:
            if agent_type.lower() == "red":
                agent_actions = self.red_actions
                agent_color = "darkred"
                node_color = "#ff7f7f"
            elif agent_type.lower() == "blue":
                agent_actions = self.blue_actions
                agent_color = "navy"
                node_color = "#87cefa"

            if limit_steps_to_range[0] is not None:
                #limit start actions of trajectory
                agent_actions = agent_actions[limit_steps_to_range[0] - 1:]

            if limit_steps_to_range[1] is not None: 
                #limit end actions of trajectory
                start = limit_steps_to_range[0] or 0
                agent_actions = agent_actions[:limit_steps_to_range[1] - 1 - start]
                
            agent_path = extract_path(agent_actions, G)
            agent_edges = list(zip(agent_path, agent_path[1:]))


            nx.draw_networkx_nodes(G, pos, nodelist=agent_path, node_color=node_color, node_size=500, edgecolors="black")

            nx.draw_networkx_edges(
                G,
                pos,
                edgelist=agent_edges,
                width=1,
                edge_color=agent_color,
                style="dashed",
                arrows=True,
                arrowstyle='-|>',
                arrowsize=12
            )
            
            # Print the action names on the lines and on the right
            if show_actions:
                node_label_count = defaultdict(int)
                node_actions = defaultdict(list)
                
                
                for i, (src, dst) in enumerate(agent_edges):
                    host = agent_actions[i].get("Host")
                    if host and host in pos:
                        #only draw actions with hosts (ignore scans)
                        count = node_label_count[(src, dst)]
                        offset = 0.05 * count
        
                        if src == dst:
                            x, y = pos[src]
                            loop_radius = 0.1 # how far from the node the label sits
                            angle = np.pi / 3 + count * 0.2 
                            x_label = x + loop_radius * np.cos(angle)
                            y_label = y + loop_radius * np.sin(angle)
                        else:
                            # Normal edge midpoint
                            x_label = (pos[src][0] + pos[dst][0]) / 2
                            y_label = (pos[src][1] + pos[dst][1]) / 2  + offset

                        # only print the action number on the edge
                        plt.text(
                            x_label,
                            y_label,
                            f"{i+1}",
                            fontsize=10,
                            ha="center",
                            va="center",
                            color=agent_color,
                            bbox=dict(facecolor='white', edgecolor='none', alpha=0.7, pad=0.5)
                        )

                        node_label_count[(src, dst)] += 1

                        node_actions[host].append(f"{i+1}:{agent_actions[i]['Action']}")

                # Make a stacked legend/list on the right side of the plot
                # TODO: Cleaner view for this or animation of traversal in a video
                grouped_actions = []
                start = 0

                for i in range(1, len(agent_actions) + 1):
                    if i == len(agent_actions) or agent_actions[i]["Action"] != agent_actions[i - 1]["Action"]:
                        end = i
                        action_name = agent_actions[i - 1]["Action"]
                        host_name = agent_actions[i - 1].get("Host") or agent_actions[i - 1].get("Subnet", "")
                        if host_name == action_name:
                            host_name = ""
                        if end - start > 1:
                            grouped_actions.append(f"{start+1} - {end}: {action_name} {host_name}")
                        else:
                            grouped_actions.append(f"{start+1}: {action_name}  {host_name}")
                        start = i

                actions_text = grouped_actions

                x_max = max(x for x, y in pos.values())
                y_max = max(y for x, y in pos.values())
                y_min = min(y for x, y in pos.values())
                y_range = y_max - y_min

                # spacing for stacked text
                spacing = y_range / max(len(actions_text), 1)

                for j, text in enumerate(actions_text):
                    plt.text(
                        x_max + 0.2,  # right of plot
                        y_max - j * spacing,  # stack vertically
                        text,
                        fontsize=6,
                        ha="left",
                        va="top",
                        color=agent_color,
                        bbox=dict(facecolor='white', edgecolor='none', alpha=0.7, pad=0.5)
                    )
                        
        plt.title(f"{self.blue_agent_name} vs. {self.red_agent_name} - {agent_type.capitalize()}")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(save_path, format="png", dpi=300)
        plt.close()
