import torch
import torch.nn as nn
import torch.nn.functional as F


# args env 主要用了num actions确定动作数量，args alg确定参数
class Actor(nn.Module):
    def __init__(self, input_shape, scheme, args_alg):
        super(Actor, self).__init__()
        self.n_actions = scheme.get("avail_actions")['vshape'][0]
        self.actor_h1 = args_alg.get('actor_h1')
        self.actor_h2 = args_alg.get('actor_h2')

        self.fc1_actor = nn.Sequential(
                nn.Linear(input_shape, self.actor_h1), 
                nn.ReLU()
            )  
        self.fc2_actor = nn.Sequential(
                nn.Linear(self.actor_h1, self.actor_h2),
                nn.ReLU()
            )
        self.fc3_actor = nn.Linear(self.actor_h2, self.n_actions)

    def forward(self, inputs):
        x = self.fc1_actor(inputs)
        x = self.fc2_actor(x)
        action = self.fc3_actor(x)

        return action
    

class ActorConv(nn.Module):
    def __init__(self, input_shape, scheme, args_alg):
        super(ActorConv, self).__init__()
        self.n_actions = scheme.get("avail_actions")['vshape'][0]
        self.obs_height = scheme.get("obs_dims")['vshape'][0]
        self.obs_width = scheme.get("obs_dims")['vshape'][1]
        # self.obs_height = args_alg.obs_height
        # self.obs_width = args_alg.obs_width

        self.n_filters = args_alg.get('n_filters')
        self.kernel = args_alg.get('kernel')
        self.stride = args_alg.get('stride')
        self.actor_h1 = args_alg.get('actor_h1')
        self.actor_h2 = args_alg.get('actor_h2')

        self.conv_to_fc_actor = nn.Sequential(
                nn.Conv2d(3, self.n_filters, self.kernel, self.stride),
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(self.n_filters * (self.obs_height - self.kernel[0] + 1) * (
                        self.obs_width - self.kernel[1] + 1), self.actor_h1),
                nn.ReLU()
            )
        self.fc2_actor = nn.Sequential(
                nn.Linear(self.actor_h1, self.actor_h2),
                nn.ReLU()
            )
        self.fc3_actor = nn.Linear(self.actor_h2, self.n_actions)

    def forward(self, inputs):
        x = self.conv_to_fc_actor(inputs)   # [1,64]
        x = self.fc2_actor(x)               # [1,64]
        action = self.fc3_actor(x)          # [1,64]

        return action  # [1,9]
    
class Incentive(nn.Module):
    def __init__(self, input_shape, scheme, args_alg, num_agents):
        super(Incentive, self).__init__()
        self.n_actions = scheme.get("avail_actions")['vshape'][0]
        self.other_action_1hot = args_alg.get("other_action_1hot")

        self.inc_h1 = args_alg.get('inc_h1')
        self.inc_h2 = args_alg.get('inc_h2')

        if self.other_action_1hot:
            other_action_shape = (num_agents-1)*self.n_actions
        else:
            other_action_shape = num_agents-1

        self.fc_reward = nn.Sequential(
            nn.Linear(input_shape, self.inc_h1),
            nn.ReLU()
            )
        self.fc1_reward = nn.Sequential(
            nn.Linear(self.inc_h1 + other_action_shape, self.inc_h2),
            nn.ReLU()
            )
        self.fc2_reward = nn.Linear(self.inc_h2, num_agents)

    def forward(self, inputs, actions):
        x = self.fc_reward(inputs)
        x = torch.cat([x, actions], dim=-1)
        x = self.fc1_reward(x)
        x = self.fc2_reward(x)
        inc_reward = torch.sigmoid(x)
 
        return inc_reward
    

class IncentiveConv(nn.Module):
    def __init__(self, input_shape, scheme, args_alg, num_agents):
        super(IncentiveConv, self).__init__()

        self.n_actions = scheme.get("avail_actions")['vshape'][0]
        self.other_action_1hot = args_alg.get("other_action_1hot")

        self.n_filters = args_alg.get('n_filters')
        self.kernel = args_alg.get('kernel')
        self.stride = args_alg.get('stride')
        self.inc_h1 = args_alg.get('inc_h1')
        self.inc_h2 = args_alg.get('inc_h2')
        

        if self.other_action_1hot:
            other_action_shape = (num_agents-1)*self.n_actions
        else:
            other_action_shape = num_agents-1

        self.obs_height = scheme.get("obs_dims")['vshape'][0]
        self.obs_width = scheme.get("obs_dims")['vshape'][1]


        self.conv_to_fc_reward = nn.Sequential(
                nn.Conv2d(3, self.n_filters, self.kernel, self.stride),
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(self.n_filters * (self.obs_height - self.kernel[0] + 1) * (
                        self.obs_width - self.kernel[1] + 1), self.inc_h1),
                nn.ReLU()
            )
        self.fc1_reward = nn.Sequential(
            nn.Linear(self.inc_h1 + other_action_shape, self.inc_h2),
            nn.ReLU()
            )
        self.fc2_reward = nn.Linear(self.inc_h2, num_agents)


    def forward(self, inputs, actions):
        x = self.conv_to_fc_reward(inputs)
        x = torch.cat([x, actions], dim=-1)
        x = self.fc1_reward(x)
        x = self.fc2_reward(x)
        inc_reward = torch.sigmoid(x)
 
        return inc_reward
