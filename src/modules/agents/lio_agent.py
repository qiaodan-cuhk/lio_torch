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
            from lio_torch.src.modules.networks import ActorConv, CriticConv, IncentiveConv

            self.actor = ActorConv(input_shape, args_env, args_alg)
            self.actor_prime = ActorConv(input_shape, args_env, args_alg)
            self.critic = CriticConv(input_shape, args_env, args_alg)
            self.inc = IncentiveConv(input_shape, args_env, args_alg)

        else:
            from lio_torch.src.modules.networks import Actor, Critic, Incentive

            self.actor = Actor(input_shape, args_env, args_alg)
            self.actor_prime = Actor(input_shape, args_env, args_alg)
            self.critic = Critic(input_shape, args_env, args_alg)
            self.inc = Incentive(input_shape, args_env, args_alg)


        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=args_alg.lr_actor)
        self.actor_prime_optimizer = optim.Adam(self.actor_prime.parameters(), lr=args_alg.lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=args_alg.lr_v)
        self.inc_optimizer = optim.Adam(self.inc.parameters(), lr=args_alg.lr_reward)


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


