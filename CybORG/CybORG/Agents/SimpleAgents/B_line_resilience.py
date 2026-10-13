import random

from CybORG.Agents import BaseAgent
from CybORG.Shared import Results
from CybORG.Shared.Actions import (
    PrivilegeEscalate,
    ExploitRemoteService,
    DiscoverRemoteSystems,
    Impact,
    DiscoverNetworkServices,
    Sleep,
)
from CybORG.Shared.Enums import TrinaryEnum


class B_lineAgent_Resilience(BaseAgent):
    def __init__(self):
        self.action = 0
        self.target_ip_address = None
        self.last_subnet = None
        self.last_ip_address = None

        self.action_history = {}

        self.jumps = [0, 1, 2, 2, 2, 2, 5, 5, 5, 5, 9, 9, 9, 12, 13]

        self.target_op_host = "Random"
        self.target_op_hosts = ["Auth", "Database", "Front"]

        self.target_prefix = None  
        self.target_host = None    
        self.target_ip = None     

    def train(self, results: Results):
        pass

    def _pick_target_prefix(self):
        """Pick a target prefix once per episode."""
        if self.target_prefix is not None:
            return self.target_prefix

        if self.target_op_host == "Random":
            self.target_prefix = random.choice(self.target_op_hosts)
        else:
            self.target_prefix = self.target_op_host
        return self.target_prefix

    def _resolve_target_host_from_obs(self, observation):
        prefix = self._pick_target_prefix()

        # observation keys include 'success' plus hostnames
        candidate_hosts = [
            h for h in observation.keys()
            if h != "success" and isinstance(h, str) and h.startswith(prefix)
        ]
        if not candidate_hosts:
            return None

        return sorted(candidate_hosts)[0]

    def _get_any_ip_from_obs(self, observation):
        host_entries = [v for k, v in observation.items() if k != "success"]
        ips = []
        for v in host_entries:
            try:
                ips.append(v["Interface"][0]["IP Address"])
            except Exception:
                continue
        return random.choice(ips) if ips else None

    def _pick_start_host_and_ip(self, observation):
        """
        Choose a reasonable 'starting' host visible in the observation.
        Prefer any User host, otherwise fall back to the first host entry with an IP.
        """
        # Prefer user hosts if present
        user_keys = [k for k in observation.keys() if k != "success" and isinstance(k, str) and k.startswith("User")]
        candidate_keys = user_keys if user_keys else [k for k in observation.keys() if k != "success" and isinstance(k, str)]

        for k in sorted(candidate_keys):
            v = observation.get(k, {})
            try:
                ip = v["Interface"][0]["IP Address"]
                subnet = v["Interface"][0].get("Subnet", None)
                return k, ip, subnet
            except Exception:
                continue

        return None, None, None

    def get_action(self, observation, action_space):
        session = 0

        # Pick prefix once
        self._pick_target_prefix()

        while True:
            if observation.get("success") == TrinaryEnum.TRUE:
                self.action += 1 if self.action < 14 else 0
            else:
                self.action = self.jumps[self.action]

            # Only cache non-target-dependent actions
            if self.action in self.action_history and self.action not in (11, 12, 13, 14):
                return self.action_history[self.action]

            # Discover Remote Systems (User subnet)
            if self.action == 0:
                start_host, start_ip, start_subnet = self._pick_start_host_and_ip(observation)
                if start_ip is None or start_subnet is None:
                    # Not enough info yet; just wait a tick
                    return Sleep()

                self.initial_ip = start_ip
                self.last_subnet = start_subnet
                action = DiscoverRemoteSystems(session=session, agent="Red", subnet=self.last_subnet)


            # Discover Network Services on a new found host
            elif self.action == 1:
                hosts = [value for key, value in observation.items() if key != "success"]
                get_ip = lambda x: x["Interface"][0]["IP Address"]
                interfaces = [get_ip(x) for x in hosts if get_ip(x) != self.initial_ip]
                self.last_ip_address = random.choice(interfaces)
                action = DiscoverNetworkServices(session=session, agent="Red", ip_address=self.last_ip_address)

            # Exploit the chosen IP
            elif self.action == 2:
                action = ExploitRemoteService(session=session, agent="Red", ip_address=self.last_ip_address)

            # PrivEsc on current host
            elif self.action == 3:
                hostname = [
                    value for key, value in observation.items()
                    if key != "success" and "System info" in value
                ][0]["System info"]["Hostname"]
                action = PrivilegeEscalate(agent="Red", hostname=hostname, session=session)

            # Discover Network Services - Enterprise host
            elif self.action == 4:
                self.enterprise_host = [x for x in observation if isinstance(x, str) and "Enterprise" in x][0]
                self.last_ip_address = observation[self.enterprise_host]["Interface"][0]["IP Address"]
                action = DiscoverNetworkServices(session=session, agent="Red", ip_address=self.last_ip_address)

            # Exploit Enterprise host
            elif self.action == 5:
                self.target_ip_address = [
                    value for key, value in observation.items() if key != "success"
                ][0]["Interface"][0]["IP Address"]
                action = ExploitRemoteService(session=session, agent="Red", ip_address=self.target_ip_address)

            # PrivEsc on Enterprise host
            elif self.action == 6:
                hostname = [
                    value for key, value in observation.items()
                    if key != "success" and "System info" in value
                ][0]["System info"]["Hostname"]
                action = PrivilegeEscalate(agent="Red", hostname=hostname, session=session)

            # Discover Remote Systems in Enterprise subnet
            elif self.action == 7:
                self.last_subnet = observation[self.enterprise_host]["Interface"][0]["Subnet"]
                action = DiscoverRemoteSystems(subnet=self.last_subnet, agent="Red", session=session)

            # Discover Network Services - pick a host entry
            elif self.action == 8:
                info = [value for key, value in observation.items() if key != "success"]
                if len(info) < 3:
                    # Not enough info returned yet; rescan
                    self.action = 7
                    continue
                self.target_ip_address = info[2]["Interface"][0]["IP Address"]
                action = DiscoverNetworkServices(session=session, agent="Red", ip_address=self.target_ip_address)

            # Exploit another discovered host
            elif self.action == 9:
                ip = self._get_any_ip_from_obs(observation)
                if ip is None:
                    self.action = 7
                    continue
                self.target_ip_address = ip
                action = ExploitRemoteService(session=session, agent="Red", ip_address=self.target_ip_address)

            # PrivEsc after exploit
            elif self.action == 10:
                hostname = [
                    value for key, value in observation.items()
                    if key != "success" and "System info" in value
                ][0]["System info"]["Hostname"]
                action = PrivilegeEscalate(agent="Red", hostname=hostname, session=session)

            # Discover services on Auth/Database/Front
            elif self.action == 11:
                resolved = self._resolve_target_host_from_obs(observation)
                if resolved is None:
                    #we haven't observed the target host yet.
                    self.action = 7
                    continue

                self.target_host = resolved
                self.target_ip = observation[self.target_host]["Interface"][0]["IP Address"]
                action = DiscoverNetworkServices(session=session, agent="Red", ip_address=self.target_ip)

            # Exploit host (use stored target_ip)
            elif self.action == 12:
                if not self.target_ip:
                    self.action = 11
                    continue
                action = ExploitRemoteService(agent="Red", session=session, ip_address=self.target_ip)

            # PrivEsc host (use resolved hostname, not prefix)
            elif self.action == 13:
                if not self.target_host:
                    self.action = 11
                    continue
                action = PrivilegeEscalate(agent="Red", hostname=self.target_host, session=session)

            # Impact host
            elif self.action == 14:
                if not self.target_host:
                    self.action = 11
                    continue
                action = Impact(agent="Red", session=session, hostname=self.target_host)

            else:
                action = Sleep()

            # Cache only stable actions
            if self.action not in self.action_history and self.action not in (11, 12, 13, 14):
                self.action_history[self.action] = action

            return action

    def end_episode(self):
        self.action = 0
        self.target_ip_address = None
        self.last_subnet = None
        self.last_ip_address = None
        self.action_history = {}

        self.target_prefix = None
        self.target_host = None
        self.target_ip = None

    def set_initial_values(self, action_space, observation):
        pass


class B_lineAgent_Resilience_A(B_lineAgent_Resilience):
    def __init__(self):
        super().__init__()
        self.target_op_hosts = ["Front", "Database", "Auth"]


class B_lineAgent_Resilience_I(B_lineAgent_Resilience):
    def __init__(self):
        super().__init__()
        self.target_op_hosts = ["Auth", "Front"]


class B_lineAgent_Resilience_C(B_lineAgent_Resilience):
    def __init__(self):
        super().__init__()
        self.target_op_hosts = ["Auth", "Database"]
