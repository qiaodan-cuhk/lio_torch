import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math

from ..networks import ActorConv, IncentiveConv, Actor, Incentive

# agent 只用来定义 actor 的网络结构，输入 obs 输出 n_actions 的NN forward

class LIOAgent(nn.Module):
    def __init__(self, input_shape, agent_id, args_env, args_alg):
        super(LIOAgent, self).__init__()
        self.args_env = args_env
        self.args_alg = args_alg
        self.n_agents = args_env.num_agents
        self.n_actions = args_env.num_actions
        self.agent_id = agent_id

        self.alg_name = self.args_alg.name  # "lio"
        self.image_obs = isinstance(self.dim_obs, list)
        self.agent_name = self.args_alg.agent
        self.r_multiplier = self.args_alg.r_multiplier


        self.actor_gradient = None
        # Default is allow the agent to give rewards
        self.can_give = True


        # self.l_action = l_action
        # self.dim_obs = dim_obs
        # self.l_action_for_r = l_action_for_r if l_action_for_r else l_action

        # 考虑是否添加 other id list
        # self.list_other_id = list(range(0, self.n_agents))
        # del self.list_other_id[self.agent_id]

        self.entropy_coeff = self.args_alg.entropy_coeff
        self.gamma = self.args_alg.gamma_env
        
        self.lr_actor = self.args_alg.lr_actor
        self.lr_inc = self.args_alg.lr_inc
        self.lr_v = self.args_alg.lr_v
        self.lr_cost = self.args_alg.lr_cost # 不知道啥用

        self.reg = self.args_alg.reg  # l1, l2
        # self.reg_coeff = tf.placeholder(tf.float32, None, 'reg_coeff')
        # self.reg_coeff = th.tensor(self.args_alg.reg_coeff, dtype=th.float32)  # 将 reg_coeff 转换为 PyTorch 张量
        self.reg_coeff = self.initialize_reg_coeff()
        self.tau = self.args_alg.tau

        # 不知道干什么的
        self.separate_cost_optimizer = self.args_alg.separate_cost_optimizer
        self.include_cost_in_chain_rule = self.args_alg.include_cost_in_chain_rule
        assert not (self.separate_cost_optimizer and self.include_cost_in_chain_rule)

        if args_env.rgb_input:
            self.actor = ActorConv(input_shape, args_env, args_alg)
            self.actor_prime = ActorConv(input_shape, args_env, args_alg)
            self.inc = IncentiveConv(input_shape, args_env, args_alg)
        else:
            self.actor = Actor(input_shape, args_env, args_alg)
            self.actor_prime = Actor(input_shape, args_env, args_alg)
            self.inc = Incentive(input_shape, args_env, args_alg)

        """lio代码里想要把激励限制在[0, r_multiplier]范围内，所以激励输出时先过一个sigmoid，
        在之后调用的时候再乘一个r_multiplier的超参数，比如ER和cleanup里的r_multiplier是2.0，ipd里面是3.0"""

    """ self.mac.params_env/inc 调用网络参数 """
    def parameters_env(self):
        params = []
        if self.args.rgb_input:
            params += list(self.actor.parameters())

        for n, p in self.named_parameters():
            if 'env' in n:
                params.append(p)

        return params
    

    def parameters_env_prime(self):
        params = []
        if self.args.rgb_input:
            params += list(self.actor_prime.parameters())

        for n, p in self.named_parameters():
            if 'env' in n:
                params.append(p)

        return params

    def parameters_inc(self):
        params = []
        if self.args.rgb_input:
            params += list(self.inc.parameters())

        for n, p in self.named_parameters():
            if 'inc' in n:
                params.append(p)
        return params
    

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
    
    
    def set_can_give(self, can_give):
        self.can_give = can_give


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

            