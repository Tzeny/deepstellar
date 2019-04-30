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

        self.screen_size = 84
        self.minimap_size = 84

        self.env_config = dict(
            map_name=map_name,
            players=players,
            agent_interface_format=features.AgentInterfaceFormat(
                feature_dimensions=features.Dimensions(screen=self.screen_size, minimap=self.minimap_size),
                use_feature_units=True),
            step_mul=16,
            game_steps_per_episode=0,
            visualize=visualize,
            random_seed=env_seed
        )

        self.action_list = [i for i in range(len(actions.FUNCTIONS))]
        self.number_of_actions = len(self.action_list)
        self.number_of_continous = 5

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

        self.num_forward_steps = 20

    def run_n_times(self, max_number_of_episodes=100):
        episode_counter = 0

        data = {
            'action_entropy': [],
            'action_log': [],
            'continous_log': [],
            'value': [],
            'predicted_value': [],
            'reward': [], 
        }

        gamma = 0.95
        entropy_weight = 1e-4
        max_grad_norm = 0.5

        with sc2_env.SC2Env(**self.env_config) as env:
            obs = env.reset()[0]

            episode_done = True
            episode_length = 0

            while True:
                data = {
                    'screen_0_entropy': [],
                    'screen_0_log': [],  
                    'screen_1_entropy': [],
                    'screen_1_log': [],                   
                    'minimap_0_entropy': [],
                    'minimap_0_log': [],                     
                    'action_entropy': [],
                    'action_log': [],
                    'value': [],
                    'predicted_value': [],
                    'reward': [], 
                }
                gc.collect()

                # take num_forward_steps simulation steps
                for _ in range(self.num_forward_steps):
                    # (
                    #     screen_0, 
                    #     screen_1, 
                    #     minimap_0, 
                    #     first_arg, 
                    #     action, 
                    #     value
                    # ),
                    # (
                    #     selected_screen_0_t,
                    #     selected_screen_1_t,
                    #     selected_minimap_0_t,
                    #     selected_action_t,
                    #     value,
                    # ),
                    model_out = self.run_model_on_observation(obs)

                    # we'll use the weights for screen and minimap to set the loss and entropy
                    # to 0 for cases where we choose an action not directly impacting them
                    sc2_action, screen_0_weight, screen_1_weight, minimap_0_weight = postprocess_action(model_out[2], self.screen_size, self.minimap_size)

                    # take step
                    obs = env.step([sc2_action])[0]
                    episode_done = obs.last()

                    screen_0_prob = F.softmax(model_out[0][0], dim=1)
                    screen_1_prob = F.softmax(model_out[0][1], dim=1)
                    minimap_0_prob = F.softmax(model_out[0][2], dim=1)
                    action_0_prob = F.softmax(model_out[0][4], dim=1)

                    # record outcome
                    screen_0_entropy = (screen_0_prob * screen_0_prob.clamp(min=1e-12).log()).sum().reshape(1) * screen_0_weight
                    screen_1_entropy = (screen_1_prob * screen_1_prob.clamp(min=1e-12).log()).sum().reshape(1) * screen_1_weight
                    minimap_0_entropy = (minimap_0_prob * minimap_0_prob.clamp(min=1e-12).log()).sum().reshape(1) * minimap_0_weight
                    action_entropy = (action_0_prob * action_0_prob.clamp(min=1e-12).log()).sum().reshape(1)

                    screen_0_log =  torch.log(screen_0_prob[0].gather(0, Variable(model_out[1][0]))) * screen_0_weight
                    screen_1_log =  torch.log(screen_1_prob[0].gather(0, Variable(model_out[1][1]))) * screen_1_weight
                    minimap_0_log =  torch.log(minimap_0_prob[0].gather(0, Variable(model_out[1][2]))) * minimap_0_weight
                    action_log =  torch.log(action_0_prob[0].gather(0, Variable(model_out[1][4])))

                    data['screen_0_entropy'].append(screen_0_entropy)
                    data['screen_0_log'].append(screen_0_log)
                    data['screen_1_entropy'].append(screen_1_entropy)
                    data['screen_1_log'].append(screen_1_log)
                    data['minimap_0_entropy'].append(minimap_0_entropy)
                    data['minimap_0_log'].append(minimap_0_log)                                        
                    data['action_entropy'].append(action_entropy)
                    data['action_log'].append(action_log)
                    data['predicted_value'].append(model_out[0][5])
                    data['reward'].append(obs.reward)

                    episode_length += 1

                    if episode_done:
                        episode_counter += 1
                        episode_length = 0
                        obs = env.reset()[0]
                        break

                # estimate reward based on policy
                reward = torch.zeros(1,1).to(device)
                if not episode_done: 
                    # if we are not done yet, bootstrap value from our latest estimate
                    model_out = self.run_model_on_observation(obs)
                    reward = model_out[0][5]

                reward = Variable(reward)
                data['predicted_value'].append(reward)

                policy_loss = 0.
                value_loss = 0.
                gae = torch.zeros(1,1).to(device)

                # go backwards in time from our latest step
                reward_count = len(data['reward'])
                # gae_ts = torch.zeros(1, 1).to(device)
                for i in reversed(range(reward_count)):
                    reward = gamma * reward + data['reward'][i]
                    data['value'].append(reward)

                    # tderr_ts = data['reward'][i] + gamma * data['predicted_value'][i+1] - data['predicted_value'][i]

                data['value'].reverse()

                values = torch.cat(data['value']).squeeze()
                predicted_values = torch.cat(data['predicted_value'][:-1]).squeeze()

                advantages = predicted_values - values

                entropy = (torch.cat(data['screen_0_entropy']).sum() +
                    torch.cat(data['screen_1_entropy']).sum() +
                    torch.cat(data['minimap_0_entropy']).sum() +
                    torch.cat(data['action_entropy']).sum())
                action_gain = ((torch.cat(data['screen_0_log']) * advantages).mean() + 
                    (torch.cat(data['screen_1_log']) * advantages).mean() + 
                    (torch.cat(data['minimap_0_log']) * advantages).mean() + 
                    (torch.cat(data['action_log']) * advantages).mean())

                value_loss = advantages.pow(2).mean()
                total_loss = value_loss - action_gain - entropy_weight * entropy

                # if np.sum(np.array(data['reward'])) > 0:
                #     print('hooray')

                self.optimizer.zero_grad()

                self.deep_stellar.train(True)

                print(total_loss)
                total_loss.backward()

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

        screen_0, screen_1, minimap_0, first_arg, action, value = self.deep_stellar(
            screen_ft,
            minimap_ft,
            numerical_obs,
        )

        selected_action_t = (F.softmax(action) * available_actions_obs)[0].multinomial(1)
        selected_action = selected_action_t.cpu().detach().numpy()[0]

        selected_screen_0_t = F.softmax(screen_0)[0].multinomial(1)
        selected_screen_0 = selected_screen_0_t.cpu().detach().numpy()[0]
        # transform the selection into a (x, y) tuple
        selected_screen_0 = (selected_screen_0 // self.screen_size, selected_screen_0 % self.screen_size)

        selected_screen_1_t = F.softmax(screen_1)[0].multinomial(1)
        selected_screen_1 = selected_screen_1_t.cpu().detach().numpy()[0]
        selected_screen_1 = (selected_screen_1 // self.screen_size, selected_screen_1 % self.screen_size)

        selected_minimap_0_t = F.softmax(minimap_0)[0].multinomial(1)
        selected_minimap_0 = selected_minimap_0_t.cpu().detach().numpy()[0]
        selected_minimap_0 = (selected_minimap_0 // self.minimap_size, selected_minimap_0 % self.minimap_size)

        # position 0 - model output
        # position 1 - processed output
        return (
            (
                screen_0, 
                screen_1, 
                minimap_0, 
                first_arg, 
                action, 
                value
            ),
            (
                selected_screen_0_t,
                selected_screen_1_t,
                selected_minimap_0_t,
                first_arg,
                selected_action_t,
                value,
            ),
            (
                selected_screen_0, 
                selected_screen_1, 
                selected_minimap_0, 
                torch.clamp(F.relu(first_arg[0]), 0, 1).cpu().detach().numpy()[0], 
                selected_action, 
                value[0].cpu().detach().numpy()[0]
            )
        )


def main(unused_argv):
    map_name = "MoveToBeacon" # "CollectMineralShards" "Simple64" MoveToBeacon

    worker = SimpleWorker(map_name,visualize=True)

    worker.run_n_times()


if __name__ == "__main__":
    app.run(main)
