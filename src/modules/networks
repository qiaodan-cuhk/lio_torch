import torch
import torch.nn as nn
import torch.nn.functional as F

class Actor(nn.Module):
    def __init__(self, input_shape, args_env, args_alg):
        super(Actor, self).__init__()
        self.fc1_actor = nn.Sequential(
                nn.Linear(input_shape, args_alg.actor_h1), 
                nn.ReLU()
            )  
        self.fc2_actor = nn.Sequential(
                nn.Linear(args_alg.actor_h1, args_alg.actor_h2),
                nn.ReLU()
            )
        self.fc3_actor = nn.Linear(args_alg.actor_h2, args_env.num_actions)

    def forward(self, inputs):
        x = self.fc1_actor(inputs)
        x = self.fc2_actor(x)
        action = self.fc3_actor(x)

        return action
    

class ActorConv(nn.Module):
    def __init__(self, input_shape, args_env, args_alg):
        super(ActorConv, self).__init__()
        self.conv_to_fc_actor = nn.Sequential(
                nn.Conv2d(3, args_alg.n_filters, args_alg.kernel, args_alg.stride),
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(args_alg.n_filters * (args_env.obs_height - args_alg.kernel[0] + 1) * (
                        args_env.obs_width - args_alg.kernel[1] + 1), args_alg.actor_h1),
                nn.ReLU()
            )
        self.fc2_actor = nn.Sequential(
                nn.Linear(args_alg.actor_h1, args_alg.actor_h2),
                nn.ReLU()
            )
        self.fc3_actor = nn.Linear(args_alg.actor_h2, args_env.num_actions)

    def forward(self, inputs):
        x = self.conv_to_fc_actor(inputs)
        x = self.fc2_actor(x)
        action = self.fc3_actor(x)

        return action
    
class Critic(nn.Module):
    def __init__(self, input_shape, args_env, args_alg):
        super(Critic, self).__init__()
        self.fc1_value = nn.Sequential(
                nn.Linear(input_shape, args_alg.critic_h1), 
                nn.ReLU()
            )  
        self.fc2_value = nn.Sequential(
                nn.Linear(args_alg.critic_h1, args_alg.critic_h2),
                nn.ReLU()
            )
        self.fc3_value = nn.Linear(args_alg.critic_h2, 1)

    def forward(self, inputs):
        x = self.fc1_value(inputs)
        x = self.fc2_value(x)
        value = self.fc3_value(x)

        return value
    
class CriticConv(nn.Module):
    def __init__(self, input_shape, args_env, args_alg):
        super(CriticConv, self).__init__()
        self.conv_to_fc_value = nn.Sequential(
                nn.Conv2d(3, args_alg.n_filters, args_alg.kernel, args_alg.stride),
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(args_alg.n_filters * (args_env.obs_height - args_alg.kernel[0] + 1) * (
                        args_env.obs_width - args_alg.kernel[1] + 1), args_alg.critic_h1),
                nn.ReLU()
            )
        self.fc2_value = nn.Sequential(
                nn.Linear(args_alg.critic_h1, args_alg.critic_h2),
                nn.ReLU()
            )
        self.fc3_value = nn.Linear(args_alg.critic_h2, 1)
    
    def forward(self, inputs):
        x = self.conv_to_fc_value(inputs)
        x = self.fc2_value(x)
        value = self.fc3_value(x)

        return value
    
class Incentive(nn.Module):
    def __init__(self, input_shape, args_env, args_alg):
        super(Incentive, self).__init()
        self.fc_reward = nn.Sequential(
            nn.Linear(input_shape, args_alg.inc_h1),
            nn.ReLU()
            )
        self.fc1_reward = nn.Sequential(
            nn.Linear(args_alg.inc_h1 + args_env.num_agents - 1, args_alg.inc_h2),
            nn.ReLU()
            )
        self.fc2_reward = nn.Linear(args_alg.inc_h2, args_env.n_agents)

    def forward(self, inputs, actions):
        x = self.fc_reward(inputs)
        x = torch.cat([x, actions], dim=-1)
        x = self.fc1_reward(x)
        x = self.fc2_reward(x)
        inc_reward = torch.sigmoid(x)
 
        return inc_reward
    
class IncentiveConv(nn.Module):
    def __init__(self, input_shape, args_env, args_alg):
        super(IncentiveConv, self).__init()
        self.conv_to_fc_reward = nn.Sequential(
                nn.Conv2d(3, args_alg.n_filters, args_alg.kernel, args_alg.stride),
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(args_alg.n_filters * (args_env.obs_height - args_alg.kernel[0] + 1) * (
                        args_env.obs_width - args_alg.kernel[1] + 1), args_alg.inc_h1),
                nn.ReLU()
            )
        self.fc1_reward = nn.Sequential(
            nn.Linear(args_alg.inc_h1 + args_env.num_agents - 1, args_alg.inc_h2),
            nn.ReLU()
            )
        self.fc2_reward = nn.Linear(args_alg.inc_h2, args_env.num_agents)

    def forward(self, inputs, actions):
        x = self.conv_to_fc_reward(inputs)
        x = torch.cat([x, actions], dim=-1)
        x = self.fc1_reward(x)
        x = self.fc2_reward(x)
        inc_reward = torch.sigmoid(x)
 
        return inc_reward
