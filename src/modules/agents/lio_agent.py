import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math


class LIOAgent(nn.Module):
    def __init__(self, input_shape, agent_id, args_env, args_alg):
        super(LIOAgent, self).__init__()
        self.agent_id = agent_id

        self.actor_gradient = None
        # Default is allow the agent to give rewards
        self.can_give = True

        if args_env.rgb_input:
            from networks import ActorConv, CriticConv, IncentiveConv

            self.actor = ActorConv(input_shape, args_env, args_alg)
            self.actor_prime = ActorConv(input_shape, args_env, args_alg)
            self.critic = CriticConv(input_shape, args_env, args_alg)
            self.inc = IncentiveConv(input_shape, args_env, args_alg)

        else:
            from networks import Actor, Critic, Incentive

            self.actor = Actor(input_shape, args_env, args_alg)
            self.actor_prime = Actor(input_shape, args_env, args_alg)
            self.critic = Critic(input_shape, args_env, args_alg)
            self.inc = Incentive(input_shape, args_env, args_alg)


        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=0.001)
        self.actor_prime_optimizer = optim.Adam(self.actor_prime.parameters(), lr=0.001)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=0.001)
        self.inc_optimizer = optim.Adam(self.inc.parameters(), lr=0.001)


        # 考虑是否添加 other id list
        # self.list_other_id = list(range(0, self.n_agents))
        # del self.list_other_id[self.agent_id]


    def forward_actor(self, inputs):
        action = self.actor.forward(inputs)

        return action
    
    # 用于 prime policy 采样
    def forward_actor_prime(self, inputs):
        action_prime = self.actor_prime.forward(inputs)

        return action_prime

    def forward_incentive(self, inputs, actions):
        inc_reward = self.inc.forward(inputs, actions)
 
        return inc_reward
    
    def forward_value(self, inputs):
        value = self.critic.forward(inputs)

        return value
    
    def set_can_give(self, can_give):
        self.can_give = can_give



    """ 考虑 self.step update """
    def initialize_reg_coeff(self):
        """初始化正则化系数"""
        if isinstance(self.self.args_alg.reg_coeff, float):
            return self.self.args_alg.reg_coeff
        else:
            if self.self.args_alg.reg_coeff == 'linear':
                self.reg_coeff_step = 1.0 / self.config.alg.n_episodes
            elif self.self.args_alg.reg_coeff == 'adaptive':
                return 0.0  # 适应性正则化系数初始为0
            return 0.0


    def update_reg_coeff(self, performance, prev_reward_env):
        """更新正则化系数，用于 incentive reward 的gain作为loss """
        if self.self.args_alg.reg_coeff == 'adaptive':
            sign = 1 if performance > prev_reward_env else -1
            self.reg_coeff = max(0, min(1.0, self.reg_coeff + sign * self.reg_coeff_step))
        elif self.self.args_alg.reg_coeff == 'linear':
            self.reg_coeff = min(1.0, self.reg_coeff + self.reg_coeff_step)


"""以下仅作为参考，改写完以后，把对应的网络文件，存储到src/modules/networks文件夹下，在本py文件开始位置import"""

class CriticNetwork(nn.Module):
    def __init__(self, input_shape, args_env, args_alg):
        super(CriticNetwork, self).__init__()
        self.args_env = args_env
        self.args_alg = args_alg

        # 定义卷积层和全连接层
        self.conv_to_fc_value = nn.Sequential(
            nn.Conv2d(3, self.args_alg.n_filters, self.args_alg.kernel, self.args_alg.stride),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(self.args_alg.n_filters * (self.args_env.obs_height - self.args_alg.kernel[0] + 1) * (
                    self.args_env.obs_width - self.args_alg.kernel[1] + 1), self.args_alg.critic_h1),
            nn.ReLU()
        )
        
        self.fc1_value = nn.Linear(input_shape, self.args_alg.critic_h1)
        self.fc2_value = nn.Linear(self.args_alg.critic_h1, self.args_alg.critic_h2)
        self.fc3_value = nn.Linear(self.args_alg.critic_h2, 1)

    def forward(self, inputs):
        if self.args_env.rgb_input:
            x = self.conv_to_fc_value(inputs)
        else:
            x = self.fc1_value(inputs)
        
        x = self.fc2_value(x)
        value = self.fc3_value(x)

        return value
    

# 实例化及优化的例子
# 实例化 Critic 网络
# self.critic = CriticNetwork(input_shape, args_env, args_alg)

# # 定义优化器，注意参数是哪个网络的
# self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=0.001)
