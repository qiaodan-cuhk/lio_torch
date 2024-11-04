import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math

class LIOAgent(nn.Module):
    def __init__(self, input_shape, agent_id, config):
        super(LIOAgent, self).__init__()
        self.config = config
        self.n_agents = config.env.num_agents
        self.n_actions = config.env.num_actions

#     def __init__(self, input_shape, agent_id, args_env, args_alg):
#         super(LIOAgent, self).__init__()
#         self.args_env = args_env
#         self.args_alg = args_alg
#         self.n_agents = args_env.num_agents
#         self.n_actions = args_env.num_actions

        self.agent_id = agent_id
        self.input_shape = input_shape

        self.optimizer = optim.Adam(self.parameters(), lr=0.001)
        self.actor_gradient = None
        # Default is allow the agent to give rewards
        self.can_give = True

        # 考虑是否添加 other id list
        # self.list_other_id = list(range(0, self.n_agents))
        # del self.list_other_id[self.agent_id]
        
        """To YLJN：
        这里的NN参数需要重新设置，不能所有网络都是同样的n_h1/n_h1；
        改成 critic NN 参数/actor NN 参数（两个actor共享一套参数）/reward NN一个参数"""

        # Critic
        self.conv_to_fc_value = nn.Sequential(
                nn.Conv2d(3, config.lio.n_filters, config.lio.kernel, config.lio.stride),
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(config.lio.n_filters * (config.env.obs_height - config.lio.kernel[0] + 1) * (
                        config.env.obs_width - config.lio.kernel[1] + 1), config.lio.critic_h1),
                nn.ReLU()
            )
        #   nn.Conv2d(3, args_alg.n_filters, args_alg.kernel, args_alg.stride),

        self.fc1_value = nn.Sequential(
                nn.Linear(input_shape, config.lio.critic_h1), 
                nn.ReLU()
            )   
#                         nn.Linear(input_shape, args_alg.n_h1), 

        self.fc2_value = nn.Sequential(
                nn.Linear(config.lio.critic_h1, config.lio.critic_h2),
                nn.ReLU()
            )
        
        self.fc3_value = nn.Linear(config.lio.critic_h2, 1)

        # Actor Network

        self.conv_to_fc_actor = nn.Sequential(
                nn.Conv2d(3, config.lio.n_filters, config.lio.kernel, config.lio.stride),
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(config.lio.n_filters * (config.env.obs_height - config.lio.kernel[0] + 1) * (
                        config.env.obs_width - config.lio.kernel[1] + 1), config.lio.actor_h1),
                nn.ReLU()
            )
        
#                         nn.Linear(args_alg.n_h1, args_alg.n_h2),
#                 nn.ReLU()
#             )
        
#         self.fc3_value = nn.Linear(args_alg.n_h2, 1)

#         # Actor Network
#         # 不知道卷积具体多大, 参数从 alg/lio.yaml 中获取

#         self.conv_to_fc_actor = nn.Sequential(
#                 nn.Conv2d(3, args_alg.n_filters, args_alg.kernel, args_alg.stride),
        
        

        self.fc1_actor = nn.Sequential(
                nn.Linear(input_shape, config.lio.actor_h1), 
                nn.ReLU()
            )   
#       nn.Linear(input_shape, args_alg.n_h1), 

        self.fc2_actor = nn.Sequential(
                nn.Linear(config.lio.actor_h1, config.lio.actor_h2),
                nn.ReLU()
            )
        
        self.fc3_actor = nn.Linear(config.lio.actor_h2, self.n_actions)


        # Prime Actor Network
        self.conv_to_fc_actor_prime = nn.Sequential(
                nn.Conv2d(3, config.lio.n_filters, config.lio.kernel, config.lio.stride),

                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(config.lio.n_filters * (config.env.obs_height - config.lio.kernel[0] + 1) * (
                        config.env.obs_width - config.lio.kernel[1] + 1), config.lio.actor_h1),
                nn.ReLU()
            )
        
#                         nn.Linear(args_alg.n_h1, args_alg.n_h2),
#                 nn.ReLU()
#             )
        
#         self.fc3_actor = nn.Linear(args_alg.n_h2, self.n_actions)


#         # Prime Actor Network
#         self.conv_to_fc_actor_prime = nn.Sequential(
#                 nn.Conv2d(3, args_alg.n_filters, args_alg.kernel, args_alg.stride),

        self.fc1_actor_prime = nn.Sequential(
                nn.Linear(input_shape, config.lio.actor_h1), 
                nn.ReLU()
            )   
#   nn.Linear(input_shape, args_alg.n_h1), 

        self.fc2_actor_prime = nn.Sequential(
                nn.Linear(config.lio.actor_h1, config.lio.actor_h2),
                nn.ReLU()
            )
        
        self.fc3_actor_prime = nn.Linear(config.lio.actor_h2, self.n_actions)

        # Incentivize Model
        """比如这里我简单改成了inc_h1，并且修改了网络逻辑保证fc1输入维度一致
            问题是：为什么reward最后linear一层的输出，还要过一个sigmoid？"""

        """lio代码里想要把激励限制在[0, r_multiplier]范围内，所以激励输出时先过一个sigmoid，在之后调用的时候再乘一个r_multiplier的超参数，
            比如ER和cleanup里的r_multiplier是2.0，ipd里面是3.0"""
        
        self.conv_to_fc_reward = nn.Sequential(
                nn.Conv2d(3, config.lio.n_filters, config.lio.kernel, config.lio.stride),
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(config.lio.n_filters * (config.env.obs_height - config.lio.kernel[0] + 1) * (
                        config.env.obs_width - config.lio.kernel[1] + 1), config.lio.inc_h1),
                nn.ReLU()
            )
        
#                         nn.Linear(args_alg.n_h1, args_alg.n_h2),
#                 nn.ReLU()
#             )
        
#         self.fc3_actor_prime = nn.Linear(args_alg.n_h2, self.n_actions)

#         # Incentivize Model
#         """比如这里我简单改成了inc_h1，并且修改了网络逻辑保证fc1输入维度一致
#             问题是：为什么reward最后linear一层的输出，还要过一个sigmoid？"""
#         self.conv_to_fc_reward = nn.Sequential(
#                 nn.Conv2d(3, args_alg.n_filters, args_alg.kernel, args_alg.stride),
#                 nn.ReLU(),
#                 nn.Flatten(),
#                 nn.Linear(args_alg.n_filters * (args_env.obs_height - args_alg.kernel[0] + 1) * (
#                         args_env.obs_width - args_alg.kernel[1] + 1), args_alg.inc_h1),

        self.fc_reward = nn.Sequential(
            nn.Linear(input_shape, config.lio.inc_h1),
            nn.ReLU()
            )
  
#   nn.Linear(input_shape, args_env.inc_h1),

        self.fc1_reward = nn.Sequential(
            nn.Linear(config.lio.inc_h1 + self.n_agents - 1, config.lio.inc_h2),
            nn.ReLU()
            )
#   nn.Linear(args_env.inc_h1 + self.n_agents - 1, args_env.inc_h2),

        self.fc2_reward = nn.Linear(config.lio.inc_h2, self.n_agents)

#         self.fc2_reward = nn.Linear(args_alg.inc_h2, self.n_agents-1)



    def forward_actor(self, inputs):
        if self.config.env.rgb_input:
            
        '''这里rgb_input应该是放在env（lio_cu）里面的吧？我暂时没有加进去'''
        
            x = self.conv_to_fc_actor(inputs)
        else:
            x = self.fc1_actor(inputs)
        x = self.fc2_actor(x)
        action = self.fc3_actor(x)

        return action  # logits rather than probs
    
    # 用于 prime policy 采样
    def forward_actor_prime(self, inputs):
        if self.config.env.rgb_input:
#         if self.args.rgb_input:
            x = self.conv_to_fc_actor_prime(inputs)
        else:
            x = self.fc1_actor_prime(inputs)
        x = self.fc2_actor_prime(x)
        action_prime = self.fc3_actor_prime(x)

        return action_prime  # logits rather than probs
    

    def forward_incentive(self, inputs, actions):
        if self.args.rgb_input:
            x = self.conv_to_fc_reward(inputs)
        else:
            x = self.fc_reward(inputs)

        x = th.cat([x, actions], dim=-1)
        x = self.fc1_reward(x)
        x = self.fc2_reward(x)
        inc_reward = th.sigmoid(x)
 
        return inc_reward
    
    def forward_value(self, inputs):
        if self.config.env.rgb_input:
            x = self.conv_to_fc_value(inputs)
        else:
            x = self.fc1_value(inputs)
        x = self.fc2_value(x)
        value = self.fc3_value(x)

        return value
    
    def set_can_give(self, can_give):
        self.can_give = can_give


    """ 考虑 self.step update """
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


