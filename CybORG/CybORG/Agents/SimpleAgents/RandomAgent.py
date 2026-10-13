from CybORG.Agents.SimpleAgents.BaseAgent import BaseAgent
from CybORG.Shared.Actions import Restore, Remove, Monitor, Analyse
import random


class RandomAgent(BaseAgent):
    def train(self):
        pass

    def end_episode(self):
        pass

    def set_initial_values(self, action_space, observation):
        pass

    def __init__(self):
        pass

    def get_action(self, observation, action_space):
        hosts = [h for h, v in action_space['hostname'].items() if v]
        session = list(action_space['session'].keys())[0]

        random_action = random.randint(0, 3)
        hostname = random.choice(hosts)

        if random_action == 0:
            return Analyse(agent='Blue', hostname=hostname, session=session)
        elif random_action == 1:
            return Restore(agent='Blue', hostname=hostname, session=session)
        elif random_action == 2:
            return Remove(agent='Blue', hostname=hostname, session=session)
        else:
            return Monitor(agent='Blue', session=session)
