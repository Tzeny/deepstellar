import numpy as np
from pysc2.lib import actions

sc2_number_of_actions = len(actions.FUNCTIONS)

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

def get_numerical_numpy(obs):
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

    return numerical_observations

def get_available_actions_numpy(obs):
    available_actions = np.zeros((sc2_number_of_actions))

    for i in obs.observation['available_actions']:
        available_actions[i] = 1

    return available_actions

def postprocess_action(action_id, p_array, screen_size, minimap_size):
    action_id = action_id.cpu().detach().numpy()[0]
    p_array = p_array.cpu().detach().numpy()

    act_args = []
    for arg in actions.FUNCTIONS[action_id].args:
        # use the same output for screen and minimap moves
        if arg.name in ('screen'):
            x = p_array[0]*(screen_size-1)
            y = p_array[1]*(screen_size-1)

            # if x >= arg.sizes[0]:
            #     x = arg.sizes[0] - epsilon
            # if y >= arg.sizes[1]:
            #     y = arg.sizes[1] - epsilon

            act_args.append([int(x), int(y)])
        elif arg.name in ('minimap'):
            x = p_array[0]*(minimap_size-1)
            y = p_array[1]*(minimap_size-1)

            # if x >= arg.sizes[0]:
            #     x = arg.sizes[0] - epsilon
            # if y >= arg.sizes[1]:
            #     y = arg.sizes[1] - epsilon

            act_args.append([int(x), int(y)])
        elif arg.name in ('screen2'):
            x = p_array[2]*(screen_size-1)
            y = p_array[3]*(screen_size-1)

            # if x >= arg.sizes[0]:
            #     x = arg.sizes[0] - epsilon
            # if y >= arg.sizes[1]:
            #     y = arg.sizes[1] - epsilon

            act_args.append([int(x), int(y)])
        elif arg.name in action_dict:
            k = p_array[4] * (action_dict[arg.name] - 1)

            # if k >= arg.sizes[0]:
            #     k = arg.sizes[0] - epsilon

            act_args.append([int(k)])
        else:
            raise ValueError(arg.name)

    return actions.FunctionCall(action_id, act_args)