from pysc2.agents import base_agent
from pysc2.lib import actions
from pysc2.lib import features

import numpy as np

import time
import pickle

# Functions
_BUILD_SUPPLYDEPOT = actions.FUNCTIONS.Build_SupplyDepot_screen.id
_BUILD_BARRACKS = actions.FUNCTIONS.Build_Barracks_screen.id
_NOOP = actions.FUNCTIONS.no_op.id
_SELECT_POINT = actions.FUNCTIONS.select_point.id

# Features
_PLAYER_RELATIVE = features.SCREEN_FEATURES.player_relative.index
_UNIT_TYPE = features.SCREEN_FEATURES.unit_type.index

# Unit IDs
_TERRAN_COMMANDCENTER = 18
_TERRAN_SCV = 45

# Parameters
_PLAYER_SELF = 1
_NOT_QUEUED = [0]
_QUEUED = [1]

def select_scv(self, obs):
    unit_type = obs.observation["feature_screen"][_UNIT_TYPE]
    unit_y, unit_x = (unit_type == _TERRAN_SCV).nonzero()
    
    target = [unit_x[0], unit_y[0]]
    
    return actions.FunctionCall(_SELECT_POINT, [_NOT_QUEUED, target])

def get_location_near_point(point, distance_x, distance_y):
    ret = np.zeros((2))

    ret[0] = point[0] + np.random.randint(-distance_y, distance_y)
    ret[1] = point[1] + np.random.randint(-distance_x, distance_x)

    return ret

def build_supply_depot(self, obs):
    if _BUILD_SUPPLYDEPOT in obs.observation["available_actions"]:
        #unit_type = obs.observation["feature_screen"][_UNIT_TYPE]
        #unit_y, unit_x = (unit_type == _TERRAN_COMMANDCENTER).nonzero()
        
        target = (25, 25)
        
        return actions.FunctionCall(_BUILD_SUPPLYDEPOT, [_NOT_QUEUED, target])
    else:
        print(obs.observation["available_actions"])
        return False

# step : action
build_order = [select_scv, build_supply_depot]

class SimpleAgent(object):
  def __init__(self):
    self.reward = 0
    self.episodes = 0
    self.steps = 0
    self.obs_spec = None
    self.action_spec = None

    self.build_order_index = 0
    self.start_position = np.zeros((2))

  def setup(self, obs_spec, action_spec):
    self.obs_spec = obs_spec
    self.action_spec = action_spec

  def reset(self):
    self.episodes += 1

  def step(self, obs):
    ret = actions.FunctionCall(_NOOP, []) #no-op

    if self.steps == 1:
        self.start_position = (obs.observation["feature_minimap"][_PLAYER_RELATIVE] == _PLAYER_SELF).nonzero()    
        self.start_position = np.array([self.start_position[1].mean(), self.start_position[0].mean()]) # y,x

        print('Our starting position is {0}'.format(self.start_position))
    elif self.build_order_index < len(build_order):
        #try to execute our next build order
        ret = build_order[self.build_order_index](self, obs)

        # if the order fails, do nothing for now
        if ret == False:
            print('Step {0}, fail'.format(self.steps))
            ret = ret = actions.FunctionCall(_NOOP, []) #no-op
        else:
            print('Step {0}, sucess'.format(self.steps))
            self.build_order_index += 1

    # with open('temp/step_{0}.pkl'.format(self.steps), 'wb') as f:
    #     pickle.dump(obs, f)

    self.steps += 1
    self.reward += obs.reward

    return ret