from pysc2.env import sc2_env
from pysc2.lib import actions, features, units
from absl import app
import time

import torch

import numpy as np

from models import DeepStellar

no_args =[ 0, 9, 63,118,120,123,126,129,131,133,136,139,186,199,202,221,223,225,246,531,534]
one_args =[ 1, 6, 7, 8, 10, 11, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 62, 66, 71, 73, 75, 77, 92, 94, 96, 98,103,104,105,106,107,108,109,110,111,112,113,114,115,116,117,119,121,122,124,125,127,128,130,132,134,135,137,138,140,141,142,143,144,145,146,147,148,149,150,151,152,153,154,155,156,157,158,159,160,161,162,163,164,165,166,167,168,169,170,171,172,173,174,175,191,197,201,211,226,228,234,235,236,237,238,244,248,249,250,251,252,253,254,255,256,257,258,259,260,261,262,263,269,270,271,272,273,274,281,282,283,284,285,286,294,295,296,297,298,299,300,301,302,303,304,306,307,308,309,310,311,312,313,317,318,319,320,321,322,323,324,325,326,327,328,329,330,351,352,353,354,355,356,357,358,359,360,361,362,363,364,365,366,367,368,369,370,371,372,373,374,375,376,377,378,379,380,381,382,383,384,385,386,387,388,389,390,391,392,393,394,395,396,397,398,399,400,401,402,403,404,405,406,407,408,409,410,411,412,413,414,415,416,417,418,419,420,421,422,423,424,425,426,427,428,429,430,431,432,433,434,435,436,437,438,439,440,441,442,443,444,445,446,447,448,449,450,453,454,455,456,457,458,459,460,461,462,463,464,465,466,467,468,469,470,471,472,473,474,475,476,477,478,479,480,481,482,483,484,485,486,487,488,489,490,491,492,493,494,495,496,497,498,499,500,501,502,503,504,511,512,513,514,515,535,536,537,538,539,540,541,546]
two_args = [ 2, 4, 5, 12, 13, 14, 15, 16, 17, 18, 19, 20, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 64, 65, 67, 68, 69, 70, 72, 74, 76, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 93, 95, 97, 99,100,101,102,176,177,178,179,180,181,182,183,184,185,187,188,189,190,192,193,194,195,196,198,200,203,204,205,206,207,208,209,210,212,213,214,215,216,217,218,219,220,222,224,227,229,230,231,232,233,239,240,241,242,243,245,247,264,265,266,267,268,275,276,277,278,279,280,287,288,289,290,291,292,293,305,314,315,316,331,332,333,334,335,336,337,338,339,340,341,342,343,344,345,346,347,348,349,350,451,452,505,506,507,508,509,510,516,517,518,519,520,521,522,523,524,525,526,527,528,529,530,532,533,542,543,544,545,547,548]

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class SimpleAgent(object):
    def __init__(self):
        self.reward = 0
        self.episodes = 0
        self.steps = 0
        self.obs_spec = None
        self.action_spec = None

        self.action_list = [i for i in range(len(actions.FUNCTIONS))]
        self.number_of_actions = len(self.action_list)

        self.deep_stellar = DeepStellar(84,84,17,64,64,7,61,self.number_of_actions).cuda()

        print(' --- Model parameter #: ', sum(p.numel() for p in self.deep_stellar.parameters()))

    def setup(self, obs_spec, action_spec):
        self.obs_spec = obs_spec
        self.action_spec = action_spec

    def reset(self):
        self.episodes += 1

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

        action = action[0].cpu().detach().numpy()
        action_id = np.argmax(action)

        continous = continous[0].cpu().detach().numpy()
        x = round(continous[0])
        y = round(continous[1])
        queued = round(continous[2])

        if action_id in no_args:
            ret = actions.FunctionCall(action_id, [])
        elif action_id in one_args:
            ret = actions.FunctionCall(action_id, [[queued]])
        elif action_id in two_args:
            ret = actions.FunctionCall(action_id, [[queued], [x,y]])
        else:
            pass

        return ret


def main(unused_argv):
    agent = SimpleAgent()
    try:
        while True:
            with sc2_env.SC2Env(
                    map_name="CollectMineralShards",
                    players=[sc2_env.Agent(sc2_env.Race.protoss),
                            #  sc2_env.Bot(sc2_env.Race.random,
                            #              sc2_env.Difficulty.very_easy)
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
