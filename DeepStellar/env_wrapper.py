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

def postprocess_action(model_out, screen_size, minimap_size):
    # (selected_screen_0,   selected_screen_1,  selected_minimap_0, F.relu(first_arg[0]),   selected_action,    value[0])
    action_id = model_out[4]

    act_args = []

    screen_0_weight = 0
    screen_1_weight = 0
    minimap_0_weight = 0

    for arg in actions.FUNCTIONS[action_id].args:
        # use the same output for screen and minimap moves
        if arg.name in ('screen'):
            x = model_out[0][0]
            y = model_out[0][1]
            screen_0_weight = 1

            act_args.append([int(x), int(y)])
        elif arg.name in ('screen2'):
            x = model_out[1][0]
            y = model_out[1][1]
            screen_1_weight = 1

            act_args.append([int(x), int(y)])
        elif arg.name in ('minimap'):
            x = model_out[2][0]
            y = model_out[2][1]
            minimap_0_weight = 1

            act_args.append([int(x), int(y)])
        elif arg.name in action_dict:
            # k = model_out[3] * (action_dict[arg.name] - 1)
            k = 0

            act_args.append([int(k)])
        else:
            raise ValueError(arg.name)

    return actions.FunctionCall(action_id, act_args), screen_0_weight, screen_1_weight, minimap_0_weight