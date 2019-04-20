from pysc2.env import sc2_env
from pysc2.lib import actions, features, units
from absl import app
import time

import torch

import numpy as np

from models import DeepStellar

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

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

class SimpleAgent(object):
    def __init__(self):
        self.reward = 0
        self.episodes = 0
        self.steps = 0
        self.obs_spec = None
        self.action_spec = None

        self.action_list = [i for i in range(len(actions.FUNCTIONS))]
        self.number_of_actions = len(self.action_list)

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
            self.number_of_actions
        ).cuda()

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
                act_args.append([int(p_array[0]*(self.screen_size-1)), int(p_array[1]*(self.screen_size-1))])
            elif arg.name in ('minimap'):
                act_args.append([int(p_array[0]*(self.minimap_size-1)), int(p_array[1]*(self.minimap_size-1))])
            elif arg.name in ('screen2'):
                act_args.append([int(p_array[2]*(self.screen_size-1)), int(p_array[3]*(self.screen_size-1))])
            elif arg.name in action_dict:
                act_args.append(p_array[4] * (action_dict[arg.name] - 1))
            else:
                raise ValueError(arg.name)
                
        return actions.FunctionCall(action_id, act_args)

    def step(self, obs):
        # ret = actions.FunctionCall(actions.FUNCTIONS.no_op.id, [])  # no-op

        a = time.time()

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

        b = time.time()

        continous, action, value = self.deep_stellar(
            torch.Tensor(np.expand_dims(obs.observation['feature_screen'],0)).to(device),
            torch.Tensor(np.expand_dims(obs.observation['feature_minimap'],0)).to(device),
            torch.Tensor(np.expand_dims(numerical_observations,0)).to(device),
            torch.Tensor(np.expand_dims(available_actions,0)).to(device),
        )

        c = time.time()

        # print('Step 1: ', b-a)
        # print('Step 2: ', c-b)

        self.steps += 1
        self.reward += obs.reward

        action_cpu = action[0].cpu().detach().numpy()
        action_id = np.argmax(action_cpu)

        continous = continous[0].cpu().detach().numpy()

        ret = self.postprocess_action(action_id, continous)

        return ret

def main(unused_argv):
    agent = SimpleAgent()
    try:
        while True:
            with sc2_env.SC2Env(
                    map_name="Simple64",
                    players=[sc2_env.Agent(sc2_env.Race.terran),
                             sc2_env.Bot(sc2_env.Race.random,
                                         sc2_env.Difficulty.very_easy)
                            ],
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
