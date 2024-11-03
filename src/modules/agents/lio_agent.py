import torch as th
import torch.nn as nn
import torch.nn.functional as F
import math

""" 需要新增一个 prime policy 和 self.policy gradient 存储 grad """
class LIOAgent(nn.Module):
    def __init__(self, input_shape, agent_id, args_env, args_alg):
        super(LIOAgent, self).__init__()
        self.args_env = args_env
        self.args_alg = args_alg
        self.n_agents = args_env.num_agents
        self.n_actions = args_env.num_actions
        self.agent_id = agent_id
        self.input_shape = input_shape

        #********************************************* actor ****************************************************************#
        #不知道卷积具体多大, 参数从 alg/lio.yaml 中获取

        self.conv_to_fc_actor = nn.Sequential(
                nn.Conv2d(3, args_alg.n_filters, args_alg.kernel, args_alg.stride),
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(args_alg.n_filters * (args_env.obs_height - args_alg.kernel[0] + 1) * (
                        args_env.obs_width - args_alg.kernel[1] + 1), args_alg.n_h1),
                nn.ReLU()
            )

        self.fc1_actor = nn.Sequential(
                nn.Linear(input_shape, args_alg.n_h1), 
                nn.ReLU()
            )   

        self.fc2_actor = nn.Sequential(
                nn.Linear(args_alg.n_h1, args_alg.n_h2),
                nn.ReLU()
            )
        
        self.fc3_actor = nn.Linear(args_alg.n_h2, self.n_actions)

        #********************************************* reward *************************************************************#
        
        self.conv_to_fc_reward = nn.Sequential(
                nn.Conv2d(3, args_alg.n_filters, args_alg.kernel, args_alg.stride),
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(args_alg.n_filters * (args_env.obs_height - args_alg.kernel[0] + 1) * (
                        args_env.obs_width - args_alg.kernel[1] + 1), args_alg.n_h1),
                nn.ReLU()
            )
        
        self.fc1_reward = nn.Sequential(
                nn.Linear(args_env.n_h1 + self.n_agents - 1, args_env.n_h2),
                nn.ReLU()
            )

        self.fc2_reward = nn.Sequential(
            nn.Linear(input_shape + self.n_agents - 1, args_env.n_h1),
            nn.ReLU()
            )
        
        self.fc3_reward = nn.Sequential(
            nn.Linear(args_alg.n_h1, args_alg.n_h2),
            nn.ReLU()
            )

        self.fc4_reward = nn.Linear(args_alg.n_h2, self.n_agents-1)

        #********************************************** value ******************************************************************#

        self.conv_to_fc_value = nn.Sequential(
                nn.Conv2d(3, args_alg.n_filters, args_alg.kernel, args_alg.stride),
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(args_alg.n_filters * (args_env.obs_height - args_alg.kernel[0] + 1) * (
                        args_env.obs_width - args_alg.kernel[1] + 1), args_alg.n_h1),
                nn.ReLU()
            )

        self.fc1_value = nn.Sequential(
                nn.Linear(input_shape, args_alg.n_h1), 
                nn.ReLU()
            )   

        self.fc2_value = nn.Sequential(
                nn.Linear(args_alg.n_h1, args_alg.n_h2),
                nn.ReLU()
            )
        
        self.fc3_value = nn.Linear(args_alg.n_h2, 1)

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

    def forward_actor(self, inputs):
        if self.args.rgb_input:
            x = self.conv_to_fc_actor(inputs)
        else:
            x = self.fc1_actor(inputs)
        x = self.fc2_actor(x)
        action = self.fc3_actor(x)

        return action  # logits rather than probs
    
    # 用于 prime policy 采样
    def forward_actor_prime(self, inputs):
        if self.args.rgb_input:
            # x = self.conv_to_fc_actor(inputs)
        else:
            # x = self.fc1_actor(inputs)
        # x = self.fc2_actor(x)
        # action = self.fc3_actor(x)

        return action  # logits rather than probs
    


    def forward_reward(self, inputs, actions):
        if self.args.rgb_input:
            x = self.conv_to_fc_reward(inputs)
            x = th.cat([x, actions], dim=-1)
            x = self.fc1_reward(x)
        else:
            x = th.cat([x, actions], dim=-1)
            x = self.fc2_reward(x)
            x = self.fc3_reward(x)
        reward = th.sigmoid(self.fc4_reward(x))

        return reward
    
    def forward_value(self, inputs):
        if self.args.rgb_input:
            x = self.conv_to_fc_value(inputs)
        else:
            x = self.fc1_value(inputs)
        x = self.fc2_value(x)
        value = self.fc3_value(x)

        return value

    def initialize_reg_coeff(self):
        """初始化正则化系数"""
        if isinstance(self.config.lio.reg_coeff, float):
            return self.config.lio.reg_coeff
        else:
            if self.config.lio.reg_coeff == 'linear':
                self.reg_coeff_step = 1.0 / self.config.alg.n_episodes
            elif self.config.lio.reg_coeff == 'adaptive':
                return 0.0  # 适应性正则化系数初始为0
            return 0.0


    def update_reg_coeff(self, performance, prev_reward_env):
        """更新正则化系数，用于 incentive reward 的gain作为loss """
        if self.config.lio.reg_coeff == 'adaptive':
            sign = 1 if performance > prev_reward_env else -1
            self.reg_coeff = max(0, min(1.0, self.reg_coeff + sign * self.reg_coeff_step))
        elif self.config.lio.reg_coeff == 'linear':
            self.reg_coeff = min(1.0, self.reg_coeff + self.reg_coeff_step)

    



