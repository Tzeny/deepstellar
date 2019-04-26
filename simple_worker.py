from pysc2.env import sc2_env
from pysc2.lib import actions, features, units

from absl import app
import time
import gc

import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.autograd as autograd
import torch.optim as optim

torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(1337)


import numpy as np

np.random.seed(1337)

import pandas as pd

from models import DeepStellar

from env_wrapper import *


# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')



action_dict = {
    'select_point_act': 4,
    'select_add': 2,
    'control_group_act': 5,
    'control_group_id': 10,
    'select_unit_act': 4,
    'select_unit_id': 500,
    'select_worker': 4,
    'unload_id': 500,
    'build_queue_id': 10,
    'queued': 2
}

epsilon = 1e-7

class SimpleWorker(object):
    def __init__(self, map_name, test_mode=True, visualize=False):
        if  map_name == "Simple64":
            players = [sc2_env.Bot(sc2_env.Race.random,sc2_env.Difficulty.very_easy)
                    ]
        else:
            players = None

        env_seed = 1337 if test_mode else None

        self.env_config = dict(
            map_name=map_name,
            players=players,
            agent_interface_format=features.AgentInterfaceFormat(
                feature_dimensions=features.Dimensions(screen=84, minimap=64),
                use_feature_units=True),
            step_mul=16,
            game_steps_per_episode=0,
            visualize=visualize,
            random_seed=env_seed
        )

        self.action_list = [i for i in range(len(actions.FUNCTIONS))]
        self.number_of_actions = len(self.action_list)
        self.number_of_continous = 5

        self.screen_size = 84
        self.minimap_size = 64

        self.deep_stellar = DeepStellar(
            self.screen_size,
            self.screen_size,
            17,
            self.minimap_size,
            self.minimap_size,
            7,
            61,
            self.number_of_actions,
            self.number_of_continous
        )
        if device.type != 'cpu':
            self.deep_stellar = self.deep_stellar.cuda()

        self.deep_stellar.train(False)
        self.optimizer = optim.Adam(self.deep_stellar.parameters(), lr=1e-3)

        self.num_forward_steps = 10

    def run_n_times(self, max_number_of_episodes=10):
        episode_counter = 0

        data = {
            'action_entropy': [],
            'action_log': [],
            'continous_log': [],
            'value': [],
            'predicted_value': [],
            'reward': [], 
        }

        gamma = 0.99
        tau = 1.00
        entropy_weight = 1e-3
        max_grad_norm = 10
        epsilon = 1.

        with sc2_env.SC2Env(**self.env_config) as env:
            obs = env.reset()[0]

            episode_done = True
            episode_length = 0

            while True:
                data = {
                    'action_entropy': [],
                    'action_log': [],
                    'continous_log': [],
                    'value': [],
                    'predicted_value': [],
                    'reward': [], 
                }
                gc.collect()

                # take num_forward_steps simulation steps
                for step in range(self.num_forward_steps):
                    continous, action, value = self.run_model_on_observation(obs)

                    # since everything is batch sized we only care about the first element
                    # we'll use epsilon greedy for the continous outputs
                    continous = continous.clamp(0,1)[0]
                    sample_from_normal = np.random.rand(continous.shape[0])
                    continous_random = torch.rand(continous.shape).to(device)

                    for i in range(continous.shape[0]):
                        if sample_from_normal[i] > epsilon:
                            continous_random[i] = continous[i]

                    continous = continous_random

                    action_id = action[0].multinomial(1)
                    value = value[0]

                    sc2_action = postprocess_action(action_id, continous, self.screen_size, self.minimap_size)

                    # take step
                    obs = env.step([sc2_action])[0]
                    episode_done = obs.last()

                    # record outcome
                    action_entropy = F.softmax(action, dim=1) * F.log_softmax(action, dim=1).sum()  

                    action_log =  torch.log(action[0].gather(0, Variable(action_id)))
                    continous_log = torch.log(torch.clamp(continous, min=1e-12))

                    data['action_entropy'].append(action_entropy)
                    data['action_log'].append(action_log)
                    data['continous_log'].append(continous_log)
                    data['predicted_value'].append(value)
                    data['reward'].append(obs.reward)

                    episode_length += 1

                    if episode_done:
                        episode_counter += 1
                        episode_length = 0
                        obs = env.reset()[0]

                        epsilon -= 0.1
                        break

                # estimate reward based on policy
                reward = torch.zeros(1,1).to(device)
                if not episode_done: 
                    # if we are not done yet, bootstrap value from our latest estimate
                    _, _, value = self.run_model_on_observation(obs)
                    reward = value[0]

                reward = Variable(reward)
                data['predicted_value'].append(reward)

                policy_loss = 0.
                value_loss = 0.
                gae = torch.zeros(1,1).to(device)

                # go backwards in time from our latest step
                for i in reversed(range(len(data['reward']))):
                    reward = gamma * reward + data['reward'][i]
                    data['value'].append(reward)

                    advantage = reward - data['predicted_value'][i]

                    value_loss += 0.5 * advantage.pow(2)

                    tderr = data['reward'][i] + gamma * data['predicted_value'][i+1] - data['predicted_value'][i]
                    gae =  gae * gamma * tau + tderr

                    # action loss
                    policy_loss += -(data['action_log'][i] * Variable(gae) + entropy_weight * data['action_entropy'][i]).sum()
                    # continous loss
                    policy_loss += -(data['continous_log'][i] * Variable(gae)).sum()

                self.optimizer.zero_grad()

                self.deep_stellar.train(True)

                loss = policy_loss + 0.5 * value_loss
                loss.backward()

                # prevent gradient explosion
                torch.nn.utils.clip_grad_norm_(self.deep_stellar.parameters(), max_grad_norm)

                self.optimizer.step()

                if episode_counter > max_number_of_episodes:
                    break

                

                

    def run_model_on_observation(self, obs):
        self.deep_stellar.train(False)

        screen_ft = Variable(
            torch.Tensor(np.expand_dims(obs.observation['feature_screen'],0)).to(device))
        minimap_ft = Variable(
            torch.Tensor(np.expand_dims(obs.observation['feature_minimap'],0)).to(device))
        numerical_obs =  Variable(
            torch.Tensor(np.expand_dims(get_numerical_numpy(obs),0)).to(device))
        available_actions_obs = torch.Tensor(
            np.expand_dims(get_available_actions_numpy(obs),0)).to(device)

        continous, action, value = self.deep_stellar.get_prediction(
            screen_ft,
            minimap_ft,
            numerical_obs,
            available_actions_obs,
        )

        return continous, action, value




def main(unused_argv):
    map_name = "MoveToBeacon" # "CollectMineralShards" "Simple64" MoveToBeacon

    worker = SimpleWorker(map_name,visualize=True)

    worker.run_n_times(20)


if __name__ == "__main__":
    app.run(main)
