import torch as th
import torch.nn as nn
import torch.nn.functional as F
import math


class LIOAgent(nn.Module):
    def __init__(self, input_shape, args):#input_shape = obs
        super(LIOAgent, self).__init__()
        self.args = args
        self.n_agents = args.n_agents
        self.n_actions = args.n_actions
        self.input_shape = input_shape

        #********************************************* actor ****************************************************************#
        #不知道卷积具体多大
        self.conv_to_fc_actor = nn.Sequential(
                nn.Conv2d(3, args.conv_out, args.conv_kernel, args.conv_stride),
                nn.LeakyReLU(),
                nn.Flatten(),
                nn.Linear(args.conv_out * (args.obs_dims[0] - args.conv_kernel + 1) * (
                        args.obs_dims[1] - args.conv_kernel + 1), args.obs_dim_net),
                nn.LeakyReLU()
            )

        self.fc1_actor = nn.Linear(args.obs_dim_net, args.rnn_hidden_dim)

        self.fc2_actor = nn.Linear(input_shape, args.rnn_hidden_dim)

        self.rnn_actor = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc3_actor = nn.Linear(args.rnn_hidden_dim, self.n_actions)

        #********************************************* reward *************************************************************#
        
        self.conv_to_fc_reward = nn.Sequential(
                nn.Conv2d(3, args.conv_out, args.conv_kernel, args.conv_stride),
                nn.LeakyReLU(),
                nn.Flatten(),
                nn.Linear(args.conv_out * (args.obs_dims[0] - args.conv_kernel + 1) * (
                        args.obs_dims[1] - args.conv_kernel + 1), args.obs_dim_net),
                nn.LeakyReLU()
            )
        
        self.fc1_reward = nn.Linear(args.obs_dim_net + self.n_agents - 1, args.rnn_hidden_dim)

        self.fc2_reward = nn.Linear(input_shape + self.n_agents - 1, args.rnn_hidden_dim)
        
        self.rnn_reward = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc3_reward = nn.Linear(args.rnn_hidden_dim, self.n_agents-1)

        #********************************************** value ******************************************************************#

        self.conv_to_fc_value = nn.Sequential(
                nn.Conv2d(3, args.conv_out, args.conv_kernel, args.conv_stride),
                nn.LeakyReLU(),
                nn.Flatten(),
                nn.Linear(args.conv_out * (args.obs_dims[0] - args.conv_kernel + 1) * (
                        args.obs_dims[1] - args.conv_kernel + 1), args.obs_dim_net),
                nn.LeakyReLU()
            )
        
        self.fc1_value = nn.Linear(args.obs_dim_net, args.rnn_hidden_dim)
        
        self.fc2_value = nn.Linear(input_shape, args.rnn_hidden_dim)
        
        self.rnn_value = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc3_value = nn.Linear(args.rnn_hidden_dim, 1)

    def parameters_actor(self):
        params = []
        for n, p in self.named_parameters():
            if 'actor' in n:
                params.append(p)

        return params

    def parameters_reward(self):
        params = []
        for n, p in self.named_parameters():
            if 'reward' in n:
                params.append(p)

        return params

    def parameters_value(self):
        params = []
        for n, p in self.named_parameters():
            if 'value' in n:
                params.append(p)

        return params

    def init_hidden(self): #不知道hidden的具体尺寸
        h_actor = self.fc1_actor.weight.new_zeros(1, self.n_agents, 1, self.args.rnn_hidden_dim).detach()
        h_reward = self.fc1_reward.weight.new_zeros(1, self.n_agents, 1, self.args.rnn_hidden_dim).detach()
        h_value = self.fc1_value.weight.new_zeros(1, self.n_agents, 1, self.args.rnn_hidden_dim).detach()
        return h_actor, h_reward, h_value

    def forward_actor(self, inputs, hidden_state=None):
        if self.args.rgb_input:
            inputs = self.conv_to_fc_actor(inputs)
        inputs = inputs.reshape(-1, self.n_agents, 1, self.input_shape)
        if self.args.rgb_input:
            x = self.fc1_actor(inputs)
        else: 
            x = self.fc2_actor(inputs)
        if hidden_state is not None:
            hidden_state = hidden_state.reshape(-1, self.n_agents, 1, self.args.rnn_hidden_dim)
        x = self.rnn_actor(x, hidden_state)
        action = F.softmax(self.fc3_actor(x), dim=-1)

        return action, x

    def forward_reward(self, inputs, actions, hidden_state=None):
        if self.args.rgb_input:
            inputs = self.conv_to_fc_reward(inputs)
        inputs = inputs.reshape(-1, self.n_agents, 1, self.input_shape)
        actions = actions.reshape(-1, self.n_agents, 1, self.n_actions)
        x = th.cat([inputs, actions], dim=-1)
        if self.args.rgb_input:
            x = self.fc1_reward(x)
        else:
            x = self.fc2_reward(x)
        if hidden_state is not None:
            hidden_state = hidden_state.reshape(-1, self.n_agents, 1, self.args.rnn_hidden_dim)
        x = self.rnn_reward(x, hidden_state)
        reward = th.sigmoid(self.fc3_reward(x))

        return reward, x
    
    def forward_value(self, inputs, hidden_state=None):
        if self.args.rgb_input:
            inputs = self.conv_to_fc_value(inputs)
        inputs = inputs.reshape(-1, self.n_agents, 1, self.input_shape)
        if self.args.rgb_input:
            x = self.fc1_value(inputs)
        else:
            x = self.fc2_value(inputs)
        if hidden_state is not None:
            hidden_state = hidden_state.reshape(-1, self.n_agents, 1, self.args.rnn_hidden_dim)
        x = self.rnn_value(x, hidden_state)
        value = self.fc3_value(x)

        return value, x
    

