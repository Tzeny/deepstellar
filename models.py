import torch
import torch.nn.functional as F

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class DeepStellar(torch.nn.Module):
    def __init__(self, 
        screen_width, 
        screen_height, 
        screen_features, 
        minimap_width, 
        minimap_height, 
        minimap_features,
        numerical_features,
        action_space):
        super().__init__()

        padding_size = 1
        conv_features = 64
        dense_features = 256

        # visual features
        self.s_conv1 = torch.nn.Conv2d(screen_features, conv_features, (3,3), padding=padding_size)
        self.s_conv2 = torch.nn.Conv2d(conv_features, conv_features, (3,3), padding=padding_size)

        self.m_conv1 = torch.nn.Conv2d(minimap_features, conv_features, (3,3), padding=padding_size)
        self.m_conv2 = torch.nn.Conv2d(conv_features, conv_features, (3,3), padding=padding_size)
        self.m_pad = torch.nn.ZeroPad2d((screen_width - minimap_width)//2)

        self.f_conv1 = torch.nn.Conv2d(conv_features*2, conv_features*2, (3,3), padding=padding_size)
        self.f_conv2 = torch.nn.Conv2d(conv_features*2, conv_features*2, (3,3), padding=padding_size)

        self.f_pool = torch.nn.AvgPool2d((screen_width, screen_height))

        # numerical features
        self.n_dense1 = torch.nn.Linear(numerical_features, dense_features)
        self.n_dense2 = torch.nn.Linear(dense_features, dense_features)

        self.n_dense3 = torch.nn.Linear(dense_features, dense_features * 2)

        # outputs
        self.continous_dense = torch.nn.Linear(dense_features * 2 + conv_features*2, dense_features)
        self.continous_output = torch.nn.Linear(dense_features, 5)

        self.action_space_dense = torch.nn.Linear(dense_features * 2 + conv_features*2, dense_features)
        self.action_space_output = torch.nn.Linear(dense_features, action_space)

        self.value_dense = torch.nn.Linear(dense_features * 2 + conv_features*2, dense_features)
        self.value_output = torch.nn.Linear(dense_features, 1)

    def forward(self, batch_screen, batch_minimap, batch_numerical, available_actions):
        # x = (batch, smth)

        # visual features
        x_screen = F.relu(self.s_conv1(batch_screen))
        x_screen = F.relu(self.s_conv2(x_screen))

        x_minimap = F.relu(self.m_conv1(batch_minimap))
        x_minimap = F.relu(self.m_conv2(x_minimap))
        x_minimap = self.m_pad(x_minimap)

        x_feature = torch.cat((x_screen, x_minimap), dim=1)
        x_feature = F.relu(self.f_conv1(x_feature))
        x_feature = F.relu(self.f_conv2(x_feature))
        x_feature = self.f_pool(x_feature).view(x_feature.shape[0], -1) #pool + flatten

        # numerical features
        x_numerical = F.relu(self.n_dense1(batch_numerical))
        x_numerical = F.relu(self.n_dense2(x_numerical))
        x_numerical = F.relu(self.n_dense3(x_numerical))

        # concatenate
        x_feature = torch.cat((x_feature, x_numerical), dim=1)

        # outputs
        x_continous = F.relu(self.continous_dense(x_feature))
        x_continous = F.relu(self.continous_output(x_continous))

        x_policy = F.relu(self.action_space_dense(x_feature))
        x_policy = self.action_space_output(x_policy)
        x_policy = x_policy * available_actions
        # x_policy = F.softmax(x_policy) # we'll apply this in the loss function

        x_value = F.relu(self.value_dense(x_feature))
        x_value = self.value_output(x_value)

        return x_continous, x_policy, x_value


    def get_prediction(self, batch_screen, batch_minimap, batch_numerical, available_actions):
        """
        Set the model in predicting mode and get a prediction
        """

        self.training = False # changes the behaviour of certain layers, like BN and Dropout
        continous, policy, value = self.forward(batch_screen, batch_minimap, batch_numerical, available_actions)

        return continous, F.softmax(policy), value

    # def get_log_probs(self, state):
    #     body_output = self.get_body_output(state)
    #     logprobs = F.log_softmax(self.policy(body_output), dim=-1)
    #     return logprobs 

    # def calc_loss(self, states, actions, rewards, advantages, beta=0.001):
    #     actions_t = torch.LongTensor(actions).to(self.network.device)
    #     rewards_t = torch.FloatTensor(rewards).to(self.network.device)
    #     advantages_t = torch.FloatTensor(advantages).to(self.network.device)
        
    #     continouss, policys, values = self.forward(states)

    #     log_probs = 
    #     log_prob_actions = advantages_t * log_probs[range(len(actions)), actions]
    #     policy_loss = -log_prob_actions.mean()
        
    #     action_probs, values = self.network.predict(states)
    #     entropy_loss = -self.beta * (action_probs * log_probs).sum(dim=1).mean()
        
    #     value_loss = self.zeta * nn.MSELoss()(values.squeeze(-1), rewards_t)
        
    #     # Append values
    #     self.policy_loss.append(policy_loss)
    #     self.value_loss.append(value_loss)
    #     self.entropy_loss.append(entropy_loss)
        
    #     return policy_loss, entropy_loss, value_loss



        

