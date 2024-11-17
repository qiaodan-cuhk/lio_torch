import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math

from ..networks import ActorConv, IncentiveConv, Actor, Incentive

# agent 只用来定义 actor 的网络结构，输入 obs 输出 n_actions 的NN forward

"""这里 args env和alg考虑替换成scheme和group"""
class LIOAgent(nn.Module):
    def __init__(self, input_shape, agent_id, scheme, args):
        super(LIOAgent, self).__init__()
        self.args_env = args.env_args
        self.args_alg = args.alg_args
        self.n_agents = self.args_env.get('num_agents', 0) # args_env.num_agents
        self.n_actions = scheme.get("avail_actions")['vshape'][0]  # args_env.num_actions

        self.agent_id = agent_id
        self.alg_name = args.name    # "lio"
        self.agent_name = args.agent # "lio"
        self.r_multiplier = self.args_alg.get('r_multiplier')
        # Default is allow the agent to give rewards
        self.can_give = True
        self.rgb_input = args.rgb_input   # self.args_alg.get("rgb_input")
        
        self.separate_cost_optimizer = self.args_alg.get('separate_cost_optimizer')
        self.include_cost_in_chain_rule = self.args_alg.get('include_cost_in_chain_rule')
        assert not (self.separate_cost_optimizer and self.include_cost_in_chain_rule)

        # 不知道干什么的
        # self.image_obs = isinstance(self.dim_obs, list)
        # self.actor_gradient = None

        # other_actions_shape = scheme.actions*(self.n_agents-1)  # flattened 1hot other actions 
        if self.rgb_input:
            self.actor = ActorConv(input_shape, scheme, self.args_alg)
            self.actor_prime = ActorConv(input_shape, scheme, self.args_alg)
            self.inc = IncentiveConv(input_shape, scheme, self.args_alg, self.n_agents)
        else:
            self.actor = Actor(input_shape, scheme, self.args_alg)
            self.actor_prime = Actor(input_shape, scheme, self.args_alg)
            self.inc = Incentive(input_shape, scheme, self.args_alg, self.n_agents)

    def forward_actor(self, inputs):
        action = self.actor.forward(inputs)   # input = [bs, 3, height, width]
        return action
    
    # 用于 prime policy 采样
    def forward_actor_prime(self, inputs):
        action_prime = self.actor_prime.forward(inputs)
        return action_prime

    def forward_incentive(self, inputs, other_actions_1hot):
        inc_reward = self.inc.forward(inputs, other_actions_1hot)
        return inc_reward
    
    
    def set_can_give(self, can_give):
        self.can_give = can_give

    def cuda(self):
        self.actor.cuda()  # 将 actor 网络移动到 GPU
        self.actor_prime.cuda()  # 将 actor_prime 网络移动到 GPU
        self.inc.cuda()


    """ 考虑 self.step update """
    def initialize_reg_coeff(self):
        """初始化正则化系数"""
        if isinstance(self.args_alg.reg_coeff, float):
            return self.args_alg.reg_coeff
        else:
            if self.args_alg.reg_coeff == 'linear':
                self.reg_coeff_step = 1.0 / self.args_env.t_max   # 这里本来是除以episode，后续考虑是否加一个counter/参数
                return 0.0
            elif self.args_alg.reg_coeff == 'adaptive':
                self.reg_coeff_step = 1.0 / self.args_env.t_max   # 考虑一下adaptive应该是什么
                return 0.0  # 适应性正则化系数初始为0
            return 0.0

    def update_reg_coeff(self, performance, prev_reward_env):
        """更新正则化系数，用于 incentive reward 的gain作为loss """
        if self.args_alg.reg_coeff == 'adaptive':
            sign = 1 if performance > prev_reward_env else -1
            self.reg_coeff = max(0, min(1.0, self.reg_coeff + sign * self.reg_coeff_step))
        elif self.args_alg.reg_coeff == 'linear':
            self.reg_coeff = min(1.0, self.reg_coeff + self.reg_coeff_step)

            