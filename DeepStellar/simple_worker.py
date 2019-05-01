from pysc2.env import sc2_env
from pysc2.lib import actions, features, units

import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.autograd as autograd
import torch.optim as optim

from tensorboardX import SummaryWriter

from datetime import datetime
import os
from pathlib import PosixPath

from absl import app
import time
import gc

torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(1337)


import numpy as np

np.random.seed(1337)

import pandas as pd

from models import DeepStellar

from env_wrapper import *


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')



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

        self.map_name = map_name

        env_seed = 1337 if test_mode else None

        self.screen_size = 84
        self.minimap_size = 84

        self.env_config = dict(
            map_name=self.map_name,
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
        self.optimizer = optim.Adam(self.deep_stellar.parameters(), lr=1e-4)

        self.num_forward_steps = 20

    def run_n_times(self, max_number_of_episodes=10000):
        episode_counter = 0
        global_step = 0

        gamma = 0.95
        tau = 1.
        entropy_weight = 1e-3
        max_grad_norm = 0.5

        base_model_dir = PosixPath('/', 'sc2ai-models')
        now_str =  datetime.now().strftime("%m.%d.%Y-%H:%M:%S")+'_'+self.map_name+'_1e-4_3steps'

        model_output_dir = base_model_dir/now_str
        model_logs_output_dir = model_output_dir/'logs'

        os.makedirs(str(model_logs_output_dir))
        print(f'Created {model_logs_output_dir}')

        writer = SummaryWriter(str(model_logs_output_dir))

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
                score = 0
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
                    score += np.sum(obs.reward)
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

                    global_step += 1

                    if episode_done:
                        writer.add_scalar('episode_score/total', np.sum(obs.observation['score_cumulative'][0]), episode_counter)
                        torch.save(self.deep_stellar, model_output_dir/f'_episode_{episode_counter}')

                        episode_counter += 1
                        episode_length = 0
                        obs = env.reset()[0]

                        writer.file_writer.flush()
                        break

                # estimate reward based on policy
                reward = torch.zeros(1,1).to(device)
                if not episode_done: 
                    # if we are not done yet, bootstrap value from our latest estimate
                    model_out = self.run_model_on_observation(obs)
                    reward = model_out[0][5]

                reward = Variable(reward)
                data['predicted_value'].append(reward)

                policy_loss_vb = 0.
                value_loss_vb = 0.

                # go backwards in time from our latest step
                reward_count = len(data['reward'])
                gae_ts = torch.zeros(1, 1).to(device)
                for i in reversed(range(reward_count)):
                    reward = gamma * reward + data['reward'][i]
                    data['value'].append(reward)

                    advantage_vb = reward - data['predicted_value'][i]
                    value_loss_vb += 0.5 * advantage_vb.pow(2)

                    tderr_ts = data['reward'][i] + gamma * data['predicted_value'][i+1] - data['predicted_value'][i]
                    gae_ts = gae_ts * gamma * tau + tderr_ts

                    policy_log_for_action_vb =  data['screen_0_log'][i] + data['screen_1_log'][i] + data['minimap_0_log'][i] + data['action_log'][i]
                    policy_loss_vb += -(policy_log_for_action_vb * Variable(gae_ts) + 
                        entropy_weight * (
                            data['screen_0_entropy'][i]+
                            data['screen_1_entropy'][i]+
                            data['minimap_0_entropy'][i]+
                            data['action_entropy'][i]
                        )
                    )


                data['value'].reverse()

                # values = torch.cat(data['value']).squeeze()
                # predicted_values = torch.cat(data['predicted_value'][:-1]).squeeze()

                # advantages = predicted_values - values

                # entropy = (torch.cat(data['screen_0_entropy']).sum() +
                #     torch.cat(data['screen_1_entropy']).sum() +
                #     torch.cat(data['minimap_0_entropy']).sum() +
                #     torch.cat(data['action_entropy']).sum())
                # action_gain = ((torch.cat(data['screen_0_log']) * advantages).mean() + 
                #     (torch.cat(data['screen_1_log']) * advantages).mean() + 
                #     (torch.cat(data['minimap_0_log']) * advantages).mean() + 
                #     (torch.cat(data['action_log']) * advantages).mean())

                # value_loss = advantages.pow(2).mean()
                # total_loss = value_loss - action_gain - entropy_weight * entropy

                # if np.sum(np.array(data['reward'])) > 0:
                #     print('hooray')

                total_loss = policy_loss_vb + 0.5 * value_loss_vb

                self.optimizer.zero_grad()

                self.deep_stellar.train(True)

                print(total_loss)
                total_loss.backward()

                # prevent gradient explosion
                torch.nn.utils.clip_grad_norm_(self.deep_stellar.parameters(), max_grad_norm)

                self.optimizer.step()

                writer.add_scalar('step_loss/policy_loss', policy_loss_vb, global_step)
                writer.add_scalar('step_loss/value_loss', value_loss_vb, global_step)
                writer.add_scalar('step_loss/total_loss', total_loss, global_step)
                writer.add_scalar('step_score/total', score, global_step)

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
    map_name = "CollectMineralShards" # "CollectMineralShards" "Simple64" MoveToBeacon

    worker = SimpleWorker(map_name,visualize=False)

    worker.run_n_times()


if __name__ == "__main__":
    app.run(main)
