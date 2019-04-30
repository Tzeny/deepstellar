import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


import math

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class DeepStellar(torch.nn.Module):
    def __init__(self, 
        screen_width, 
        screen_height, 
        screen_features, 
        minimap_width, 
        minimap_height, 
        minimap_features,
        numerical_features,
        action_space_len,
        continous_len):
        super().__init__()

        padding_size = 1
        conv_features = 64
        dense_features = 256

        self.screen_cnn = nn.Sequential(
            nn.Conv2d(screen_features, conv_features, kernel_size=3, stride=1, padding=padding_size),
            nn.ReLU(inplace=True),
            nn.Conv2d(conv_features, conv_features, kernel_size=3, stride=1, padding=padding_size),
            nn.ReLU(inplace=True),
        )

        self.minimap_cnn = nn.Sequential(
            nn.Conv2d(minimap_features, conv_features, kernel_size=3, stride=1, padding=padding_size),
            nn.ReLU(inplace=True),
            nn.Conv2d(conv_features, conv_features, kernel_size=3, stride=1, padding=padding_size),
            nn.ReLU(inplace=True),
            nn.ZeroPad2d((screen_width - minimap_width)//2)
        )

        self.numerical_cnn = nn.Sequential(
            nn.Conv2d(1, conv_features, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(conv_features, conv_features, kernel_size=3, stride=1, padding=padding_size),
            nn.ReLU(inplace=True),
        )

        # concat screen_cnn, minimap_cnn and repeated numerical features here
        self.post_concat_cnn = nn.Sequential(
            nn.Conv2d(conv_features * 3, conv_features, kernel_size=3, stride=1, padding=padding_size),
            nn.ReLU(inplace=True),
            nn.Conv2d(conv_features, conv_features, kernel_size=3, stride=1, padding=padding_size),
            nn.ReLU(inplace=True)
        )
        self.concat_avg = torch.nn.AvgPool2d((screen_width, screen_height))

        # actor head
        self.screen_0_cnn = nn.Sequential(
            nn.Conv2d(conv_features, 1, kernel_size=3, stride=1, padding=padding_size),
            nn.ReLU(inplace=True),
            Flatten()
        )

        self.screen_1_cnn = nn.Sequential(
            nn.Conv2d(conv_features, 1, kernel_size=3, stride=1, padding=padding_size),
            nn.ReLU(inplace=True),
            Flatten()
        )

        self.minimap_0_cnn = nn.Sequential(
            nn.Conv2d(conv_features, 1, kernel_size=3, stride=1, padding=padding_size), # this is 84x84, gotta fix it
            nn.ReLU(inplace=True),
            Flatten()
        )

        self.actor_dense = torch.nn.Linear(conv_features, dense_features)
        self.action_space_output = torch.nn.Linear(dense_features, action_space_len)
        self.first_argument_output = torch.nn.Linear(dense_features, 1)

        # critic head
        self.critic_output = nn.Sequential(
            nn.Linear(conv_features, dense_features),
            nn.ReLU(inplace=True),
            nn.Linear(dense_features, 1),
        )

        self.total_params = sum(p.numel() for p in self.parameters())
        print(f'DeepStellar initialized, number of parameters: {self.total_params}')

    def forward(self, batch_screen, batch_minimap, batch_numerical):
        # x = (batch, smth)

        # visual features
        x_screen = self.screen_cnn(batch_screen)

        x_minimap = self.minimap_cnn(batch_minimap)

        x_numerical = Variable(
            batch_numerical.data.repeat(
                1,
                math.ceil(x_screen.shape[2] * x_screen.shape[3] / batch_numerical.shape[1])
            ).resize_(
                x_minimap.shape[2], x_minimap.shape[3]
            )
        )
        x_numerical = x_numerical.unsqueeze(0).unsqueeze(0) # (1, 1, 84, 84)

        x_numerical = self.numerical_cnn(x_numerical)

        # concatenate
        x_concat = torch.cat((x_screen, x_minimap, x_numerical), dim=1)
        x_concat = self.post_concat_cnn(x_concat)

        x_concat_avg = self.concat_avg(x_concat).view(x_concat.shape[0], -1)

        # actor
        x_screen_0 = self.screen_0_cnn(x_concat)
        x_screen_1 = self.screen_1_cnn(x_concat)
        x_minimap = self.minimap_0_cnn(x_concat)

        x_actor = F.relu(self.actor_dense(x_concat_avg))
        x_action = self.action_space_output(x_actor)
        x_first_arg = self.first_argument_output(x_actor)

        # critic
        x_value = self.critic_output(x_concat_avg)

        return x_screen_0, x_screen_1, x_minimap, x_first_arg, x_action, x_value


    def get_prediction(self, batch_screen, batch_minimap, batch_numerical, available_actions):
        """
        Set the model in predicting mode and get a prediction
        """

        self.training = False # changes the behaviour of certain layers, like BN and Dropout
        # continous, policy, value = self.forward(batch_screen, batch_minimap, batch_numerical)
        # policy = F.softmax(policy) * available_actions
        screen_0, screen_1, minimap, first_arg, action, value = self.forward(batch_screen, batch_minimap, batch_numerical)

        action = F.softmax(action) * available_actions
        

        return continous, action, value

    def loss(self):
        self.train()

        policy_loss = 0 # will be comprised of policy_action_loss and policy_continous_loss
        value_loss = 0
        

