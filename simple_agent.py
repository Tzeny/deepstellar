from pysc2.env import sc2_env
from pysc2.lib import actions, features, units
from absl import app
import time

import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.autograd as autograd
import torch.optim as optim

import numpy as np
import pandas as pd

from models import DeepStellar

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

class SimpleAgent(object):
    def __init__(self):
        self.reward = 0
        self.episodes = 0
        self.steps = 0
        self.obs_spec = None
        self.action_spec = None

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

        self.train_interval = 10

        dtypes = [
            ('step', np.int32),
            ('predicted_V', np.float32),
            ('actual_V', np.float32),
            ('error', np.float32),

            ('state_screen', np.uint8, (17, self.screen_size, self.screen_size)),
            ('state_minimap', np.uint8, (7, self.minimap_size, self.minimap_size)),
            ('state_numerical', np.uint8, (61)),
            ('state_available_actions', np.uint8, (self.number_of_actions)),
            ('action_chosen', np.int16),

            ('continous_chosen', np.float32, (self.number_of_continous)),
            ('reward', np.float32)
        ]

        self.step_recordings = np.empty(self.train_interval, dtype=dtypes)

        print(' --- Model parameter #: ', sum(p.numel() for p in self.deep_stellar.parameters()))

    def setup(self, obs_spec, action_spec):
        self.obs_spec = obs_spec
        self.action_spec = action_spec

    def reset(self):
        self.episodes += 1

    def postprocess_action(self, action_id, p_array):
        act_args = []
        for arg in actions.FUNCTIONS[action_id].args:
            # use the same output for screen and minimap moves
            if arg.name in ('screen'):
                x = p_array[0]*(self.screen_size-1)
                y = p_array[1]*(self.screen_size-1)

                # if x >= arg.sizes[0]:
                #     x = arg.sizes[0] - epsilon
                # if y >= arg.sizes[1]:
                #     y = arg.sizes[1] - epsilon

                act_args.append([int(x), int(y)])
            elif arg.name in ('minimap'):
                x = p_array[0]*(self.minimap_size-1)
                y = p_array[1]*(self.minimap_size-1)

                # if x >= arg.sizes[0]:
                #     x = arg.sizes[0] - epsilon
                # if y >= arg.sizes[1]:
                #     y = arg.sizes[1] - epsilon

                act_args.append([int(x), int(y)])
            elif arg.name in ('screen2'):
                x = p_array[2]*(self.screen_size-1)
                y = p_array[3]*(self.screen_size-1)

                if x >= arg.sizes[0]:
                    x = arg.sizes[0] - epsilon
                if y >= arg.sizes[1]:
                    y = arg.sizes[1] - epsilon

                act_args.append([int(x), int(y)])
            elif arg.name in action_dict:
                k = p_array[4] * (action_dict[arg.name] - 1)

                if k >= arg.sizes[0]:
                    k = arg.sizes[0] - epsilon

                act_args.append([int(k)])
            else:
                raise ValueError(arg.name)

        # print(act_args)
                
        return actions.FunctionCall(action_id, act_args)

    def step(self, obs):
        # prepare observations

        if obs.observation['multi_select'].shape[0] == 0:
            obs.observation['multi_select'] = np.zeros((1,7))
        if obs.observation['cargo'].shape[0] == 0:
            obs.observation['cargo'] = np.zeros((1,7))
        if obs.observation['build_queue'].shape[0] == 0:
            obs.observation['build_queue'] = np.zeros((1,7))
        if obs.observation['alerts'].shape[0] == 0:
            obs.observation['alerts'] = np.zeros((2))
        elif obs.observation['alerts'].shape[0] == 1:
            obs.observation['alerts'] = np.array([ obs.observation['alerts'][0], 0])

        numerical_observations = np.concatenate((
            obs.observation['player'],
            obs.observation['control_groups'].reshape((20)),
            obs.observation['single_select'].reshape((7)),
            obs.observation['multi_select'].mean(axis=0).reshape((7)),
            obs.observation['cargo'].mean(axis=0).reshape((7)),
            obs.observation['build_queue'].mean(axis=0).reshape((7)),
            obs.observation['alerts'],
        ))

        available_actions = np.zeros((self.number_of_actions))

        for i in obs.observation['available_actions']:
            available_actions[i] = 1

        # predict

        continous, action, value = self.deep_stellar.get_prediction(
            torch.Tensor(np.expand_dims(obs.observation['feature_screen'],0)).to(device),
            torch.Tensor(np.expand_dims(obs.observation['feature_minimap'],0)).to(device),
            torch.Tensor(np.expand_dims(numerical_observations,0)).to(device),
            torch.Tensor(np.expand_dims(available_actions,0)).to(device),
        )

        continous_distribution = continous.cpu().detach().numpy()

        # sample from a normal distribution centered around our desired coordinates
        # for i in range(continous[0].shape[0]):
        #     continous[0][i] = torch.distributions.Normal(continous[0][i], 0.15).sample()
        continous = continous.clamp(0,1)


        #action_cpu = action[0].cpu().detach().numpy()
        #action_id = np.argmax(action_cpu)
        # action *= torch.Tensor(available_actions).to(device)

        # action_id = -1
        # while action_id not in obs.observation['available_actions']:

        action_id = action[0].multinomial(1).cpu().detach().numpy()[0]
        # action_id = np.random.multinomial(1, action_distribution)[0]
        continous_cpu = continous[0].cpu().detach().numpy()
        value_cpu = value[0].cpu().detach().numpy()[0]

        # print(np.mean(action_cpu))
        # print(np.mean(continous_cpu))

        # ret = actions.FunctionCall(actions.FUNCTIONS.no_op.id, [])
        ret = self.postprocess_action(action_id, continous_cpu)

        index = self.steps % self.train_interval
        self.step_recordings[index]['step'] = self.steps
        self.step_recordings[index]['predicted_V'] = value_cpu

        self.step_recordings[index]['state_screen'] = obs.observation['feature_screen']
        self.step_recordings[index]['state_minimap'] = obs.observation['feature_minimap']
        self.step_recordings[index]['state_numerical'] = numerical_observations
        self.step_recordings[index]['state_available_actions'] = available_actions

        self.step_recordings[index]['action_chosen'] = action_id
        self.step_recordings[index]['continous_chosen'] = continous_cpu

        if obs.reward != 0:
            print(obs.reward)

        if self.steps > 0:
            self.step_recordings[index-1]['reward'] = obs.reward

            # we do this so that we can use all the values in step_recordings
            if self.steps % self.train_interval == 0:
                self.update_actual_state_values(value_cpu, 0.95)

                # train
                self.reflect()

        self.steps += 1
        self.reward += obs.reward

        return ret

    def reflect(self):
        state_values_true = torch.Tensor(self.step_recordings['actual_V']).to(device)

        state_screen = Variable(torch.Tensor(self.step_recordings['state_screen']).to(device))
        state_minimap = Variable(torch.Tensor(self.step_recordings['state_minimap']).to(device))
        state_numerical = Variable(torch.Tensor(self.step_recordings['state_numerical']).to(device))
        state_available_actions = Variable(torch.Tensor(self.step_recordings['state_available_actions']).to(device))

        self.deep_stellar.train(False)

        continous_probs, policy_probs, value_ests = self.deep_stellar(
            state_screen,
            state_minimap,
            state_numerical,
        )

        actions_taken = Variable(torch.LongTensor(self.step_recordings['action_chosen']).view(-1,1).to(device))
        actions_taken_log_probs = F.log_softmax(policy_probs, dim=1).gather(1, actions_taken)

        # also the TD error
        value_ests = value_ests.squeeze() # flatten 
        advantages = state_values_true - value_ests

        # entropy will have a negative positive value, but it is substracted from the loss so everythin is ok
        entropy = F.softmax(policy_probs, dim=1) * F.log_softmax(policy_probs, dim=1)
        entropy = entropy.sum() 

        action_gain = (actions_taken_log_probs * advantages).mean()
        continous_action_gain = (continous_probs.log().t() * advantages).mean()

        value_loss = advantages.pow(2).mean()
        total_loss = value_loss - (action_gain + continous_action_gain) - 1e-4 * entropy

        # take optimizer step
        self.deep_stellar.train(True)

        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm(self.deep_stellar.parameters(), 0.5)
        self.optimizer.step()

        self.deep_stellar.train(False)

    def update_actual_state_values(self, latest_value, gamma):
        """
        Calculate actual_V for all elements in self.step_recordings except the last one
        """

        # the reward for the current state
        next_value = latest_value

        # since we reversed, this goes from the newsest state, back in time to the oldest
        for i in range(self.step_recordings.shape[0] - 2, -1, -1):
            current_value = self.step_recordings['reward'][i] + next_value * gamma
            self.step_recordings['actual_V'][i] = current_value

            next_value = current_value


def main(unused_argv):
    agent = SimpleAgent()

    map_name = "CollectMineralShards" # "CollectMineralShards" "Simple64" MoveToBeacon

    if  map_name == "Simple64":
        players = [sc2_env.Agent(sc2_env.Race.terran),
                    sc2_env.Bot(sc2_env.Race.random,
                                sc2_env.Difficulty.very_easy)
                ]
    else:
        players = [sc2_env.Agent(sc2_env.Race.terran)]

    try:
        while True:
            with sc2_env.SC2Env(
                    map_name=map_name,
                    players=players,
                    agent_interface_format=features.AgentInterfaceFormat(
                        feature_dimensions=features.Dimensions(screen=84, minimap=64),
                        use_feature_units=True),
                    step_mul=16,
                    game_steps_per_episode=0,
                    visualize=True) as env:
                agent.setup(env.observation_spec(), env.action_spec())

                timesteps = env.reset()
                agent.reset()

                while True:
                    step_actions = [agent.step(timesteps[0])]
                    if timesteps[0].last():
                        break
                    timesteps = env.step(step_actions)
                
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    app.run(main)
