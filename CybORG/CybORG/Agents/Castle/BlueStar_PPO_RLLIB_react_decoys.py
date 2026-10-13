import warnings

from CybORG.Agents.SimpleAgents.BaseAgent import BaseAgent

warnings.filterwarnings("ignore", category=DeprecationWarning)

from ray.rllib.models import ModelCatalog
from ray.rllib.policy import Policy

from CybORG.Agents.Castle.action_mask_model import TorchActionMaskModel

ModelCatalog.register_custom_model("my_model", TorchActionMaskModel)

import copy

from CybORG.Shared.Actions import *

from CybORG.Agents.Castle.agent_utils import *

USE_ALL_DECOYS = False
ASSETS_ONLY = False


class BlueStarDecoysAgent(BaseAgent):
    def __init__(self, cyborg_env=None, model=None):

        model_path = model
        print("\nLoading Serializable blue agent model from ", model_path)
        self.model = SerializablePolicy.from_checkpoint(model_path)
        self.cyborg_env = cyborg_env
        self.decoy_db = 199
        self.step = 1

    def train(self, results):
        pass

    def get_action(self, observation, action_space=None):

        user_hosts, escalated_hosts = get_compromised_hosts(
            self.cyborg_env, observation
        )

        if len(user_hosts) != 0 or len(escalated_hosts) != 0:  # recover
            possible_reactions = get_possible_reactions(action_space, [Restore])
            filtered_reactions = filter_reactions(
                user_hosts, escalated_hosts, possible_reactions, action_space
            )
            action_mask = filter_action_mask(action_space, filtered_reactions)
        else:
            indexes_to_deploy = []
            deployed, available = deployed_decoys(
                action_space,
                self.decoy_history_init,
                self.decoy_history,
                assets_only=ASSETS_ONLY,
            )

            for host in available:
                if USE_ALL_DECOYS:  # deploy all available decoys
                    indexes_to_deploy += available[host]
                elif len(deployed[host]) == 0:  # ensure at least one decoy per host
                    indexes_to_deploy += available[host]

            if len(indexes_to_deploy) > 0:  # harden
                action_mask = filter_action_mask(action_space, indexes_to_deploy)
            else:  # try any valid action
                action_mask = valid_action_mask(action_space, self.decoy_history)

        observation = {
            "observations": observation,
            "action_mask": action_mask,
        }

        action, _, _ = self.model.compute_single_action(obs=observation)

        self.decoy_history = update_decoy_history(
            action_space, action, self.decoy_history_init, self.decoy_history
        )

        return action

    def end_episode(self):
        pass

    def set_initial_values(self, action_space=None, observation=None):
        decoy_history = reset_decoy_history(action_space)
        self.decoy_history_init = prune_decoys_on_used_ports(
            self.cyborg_env, action_space, observation, decoy_history
        )
        self.decoy_history = copy.deepcopy(self.decoy_history_init)

        for (
            hostname,
            _,
        ) in self.cyborg_env.environment_controller.state.hosts.items():
            print_valid_decoys(action_space, self.decoy_history, hostname)


class SerializablePolicy:
    def __init__(self, policy):
        self.policy = policy

    def compute_single_action(self, obs):
        return self.policy.compute_single_action(obs)

    @staticmethod
    def from_checkpoint(checkpoint_dir):
        p = Policy.from_checkpoint(checkpoint_dir)
        return SerializablePolicy(p)

    @staticmethod
    def from_state(state):
        p = Policy.from_state(state)
        return SerializablePolicy(p)

    def get_state(self):
        return self.policy.get_state()

    def __reduce__(self):
        return (SerializablePolicy.from_state, (self.get_state(),))
