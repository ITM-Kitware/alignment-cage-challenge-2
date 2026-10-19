import warnings

from CybORG.Agents.SimpleAgents.BaseAgent import BaseAgent

warnings.filterwarnings("ignore", category=DeprecationWarning)

from ray.rllib.models import ModelCatalog
from ray.rllib.policy import Policy

from CybORG.Agents.Cadet.action_mask_model import TorchActionMaskModel

ModelCatalog.register_custom_model("my_model", TorchActionMaskModel)

from CybORG.Shared.Actions import *

from CybORG.Agents.Cadet.agent_utils import *


class BlueStarReactAgent(BaseAgent):
    def __init__(self, cyborg_env=None, model=None):

        model_path = model
        print("\nLoading Serializable blue agent model from ", model_path)
        self.model = SerializablePolicy.from_checkpoint(model_path)
        self.cyborg_env = cyborg_env

    def train(self, results):
        pass

    def get_action(self, observation, action_space=None):
        user_hosts, escalated_hosts = get_compromised_hosts(
            self.cyborg_env, observation
        )

        if len(user_hosts) == 0 and len(escalated_hosts) == 0:
            # action_mask = valid_action_mask(
            action_mask = valid_action_mask_v2(
                action_space,
                ["Sleep", "Remove", "Decoy"],
            )
        else:
            possible_reactions = get_possible_reactions(action_space, [Restore])
            filtered_reactions = filter_reactions(
                user_hosts, escalated_hosts, possible_reactions, action_space
            )
            action_mask = filter_action_mask(action_space, filtered_reactions)

        observation = {
            "observations": observation,
            "action_mask": action_mask,
        }

        action, state_out, extra = self.model.compute_single_action(obs=observation)

        return action

    def end_episode(self):
        pass

    def set_initial_values(self, action_space=None, observation=None):
        pass


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
