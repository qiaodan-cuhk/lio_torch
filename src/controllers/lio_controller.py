from modules.agents import REGISTRY as agent_REGISTRY
from components.action_selectors import REGISTRY as action_REGISTRY
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

# mac = basicmac，会build agents，mac.agents = RNNAgent，其实相当于actor，输入obs输出n_actions的拟合
# critics/ac 等等才是critic，输入obs输出一个value 1
# learner里self.mac=mac 是actor，self.critic=Critic 是critic网络

# lio homo的处理方式是，设定了homo agent，只用于输出env和inc动作，同时留了三个接口params/params env/inc用于分别存储全部&两个网络的参数
# 在homo mac中，build agents得到self.agent = [homo homo homo]，使用 self.param_env/inc调用 self.agents.params env/inc; 使用self.param 调用 self.agents.parameters 这里可能是直接网络的parameters，而不是某个agent里定义的函数了
# mac只用于输出动作，所以不考虑critic；在learner里才会考虑self.critic = / self.mac 作为actor/ target mac
# 由于homo是QMIX结构，不是a2c，直接用agent拟合q value

# 先 forward 得到所有动作的 output，以所有agent的list形式输出，然后用select action调用output和selector输出最终动作
# 这里也不需要forward critic，critic的td value集成在了learner的critic train中，同时step和得到value？因为只有update时候用得到critic


# This multi-agent controller DO NOT shares parameters between agents
class LIOMAC(nn.Module):
    def __init__(self, scheme, args_env, args_alg):
        super(LIOMAC, self).__init__()
        self.num_agents = args_env.num_agents
        self.args_env = args_env
        self.args_alg = args_alg
        input_shape = self._get_input_shape(scheme)

        self.l_actions = args_env.actions_nums ????

        # 创建 self.agents = list[]
        self._build_agents(input_shape)

        # This handles the special case of two asymmetric agents,
        # one of which is the reward-giver and the other is the recipient
        if args_alg.lio_asymmetric:
            assert args_alg.n_agents == 2
            for agent_id in range(args_alg.n_agents):
                self.agents[agent_id].set_can_give(agent_id != args_alg.idx_recipient)

        self.agent_output_type = args_alg.agent_output_type
        self.action_selector = action_REGISTRY[args_alg.action_selector](self.args_alg)

        self.mask = th.ones(self.num_agents, self.num_agents, dtype=th.bool, device=self.args.device)
        self.mask.fill_diagonal_(False)
        self.mask_ = self.mask.unsqueeze(0)  # [1,n,n]

        # 没使用RNN，所以不用考虑hidden states和init hidden
        # 激励函数的mask，禁止自己给自己奖励，创建一个diag为0的单位矩阵，我们考虑使用原始lio的方法，可以不用这个
        self.inc_mask_actions = (1 - th.eye(self.n_agents)).reshape(1, self.n_agents, self.n_agents, 1).to(self.args.device)

    """Build inputs的逻辑要改一下，是先build完，然后依次每个agent调用自己的；
        还是直接传入自己的ep batch，在里面build
        forward actor 考虑加id属性，因为在learner里每个agent都要用他人的logits计算 loss
         
          select 用于 runner，forward用于learner """
    
    def select_actions_env(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False):
        # Only select actions for the selected batch elements in bs
        avail_actions = ep_batch["avail_actions"][:, t_ep]

        agent_outputs = self.forward_actors(ep_batch, t_ep, test_mode=test_mode) 
        # [bs, n_aegnts, -1] logits可以选择

        masks = avail_actions[bs] # [bs,n,num_action]
        chosen_actions = self.action_selector.select_action(agent_outputs[bs], masks , t_env, test_mode=test_mode)
        chosen_actions = chosen_actions.unsqueeze(-1)

        return chosen_actions
        # [bs,n,1]


    def select_actions_env_prime(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False):
        # 使用 prime policy 采样动作
        avail_actions = ep_batch["avail_actions"][:, t_ep]
        agent_outputs = self.forward_actors_prime(ep_batch, t_ep, test_mode=test_mode) 
        # [bs,n,-1]，可能是logits，也可能是softmax分布，看参数设置

        masks = avail_actions[bs] # [bs,n,num_action]
        chosen_actions = self.action_selector.select_action(agent_outputs[bs], masks , t_env, test_mode=test_mode)
        chosen_actions = chosen_actions.unsqueeze(-1)

        return chosen_actions
        # [bs,n,1]


    # 所有agent的incentive reward选择，返回的是list_rewards=[n_agents,n-1]代表每个agent的激励选择，
    # 以及 total_reward，表示每个agent叠加到一起收到的总激励奖励。
    def select_actions_inc(self, list_actions, ep_batch, t_ep,
                           t_env,  bs=slice(None), test_mode=False,
                           agent_pos_replay = None):   

        agent_inputs_inc = self._build_inputs(ep_batch, t_ep)  # [n,bs,...]

        ?
        # 重新shape action lists,检查数据维度!
        actions_for_agents = list_actions.unsqueeze(1).expand(ep_batch.batch_size, self.num_agents, self.num_agents)
        actions_for_agents = actions_for_agents.masked_select(self.mask_).view(ep_batch.batch_size, self.num_agents, self.num_agents-1).transpose(0, 1)
        # [n,bs,n-1]

        # list_rewards, total_reward_given_to_each_agent = self.forward_incentive(ep_batch, t_ep, list_actions, test_mode=test_mode) # [bs,n,num_action]
        list_rewards = []
        total_reward_given_to_each_agent = np.zeros(self.num_agents)

        for i in range(self.num_agents):
            agent = self.agents[i]

            other_actions_1hot = self.get_action_others_1hot(actions_for_agents, i, self.l_actions)

            if agent.can_give:
                reward = agent.forward_incentive(agent_inputs_inc[i], other_actions_1hot)
            else:
                reward = np.zeros(self.args_alg.n_agents)
            reward[agent.agent_id] = 0
            # [n_agent, 1] 表示当前agent i给所有agent 0-n 的reward，自己给自己的是 0

            total_reward_given_to_each_agent += reward
            # 表示了每个agent收到了总共多少reward
            list_rewards.append(reward)
            # 表示所有agent的inc情况，用于runner里的reward buffer save和learner里的计算

        return total_reward_given_to_each_agent, list_rewards
        

    """ policy sampling """
    # mac.forward_actors 调用每个LIO.forward_actor，返回 logits 用于计算loss
    def forward_actors(self, ep_batch, t, test_mode=False, learning_mode=False):
        self.agent_inputs = self._build_inputs(ep_batch, t)  # [n,bs,...]
        avail_actions = ep_batch["avail_actions"][:, t]

        agent_outs = th.concat([self.agents[i].forward_actor(self.agent_inputs[i]) for i in range(self.num_agents)], dim=-1) 
        # 这里是actor网络输出的 # [bs, n_agents, num_actions, 1]

        # Softmax the agent outputs if they're policy logits
        if self.agent_output_type == "pi_logits":
            if getattr(self.args, "mask_before_softmax", True):
                # Make the logits for unavailable actions very negative to minimise their affect on the softmax
                reshaped_avail_actions = avail_actions.reshape(ep_batch.batch_size * self.n_agents, -1)
                agent_outs[reshaped_avail_actions == 0] = -1e10
            agent_outs = th.nn.functional.softmax(agent_outs, dim=-1)

        return agent_outs.view(ep_batch.batch_size, self.num_agents, -1) 
        # [bs,n,num_action] logits，如果pi logits那么返回的是一个softmax后的分布，可以直接sample
    
    """ prime policy sampling """
    def forward_actors_prime(self, ep_batch, t, test_mode=False, learning_mode=False):

        self.agent_inputs = self._build_inputs(ep_batch, t)  # [n,bs,...]
        avail_actions = ep_batch["avail_actions"][:, t]

        agent_outs = th.concat([self.agents[i].forward_actor_prime(self.agent_inputs[i]) for i in range(self.num_agents)], dim=-1)  # 这里是actor网络输出的 [bs, n_actions, 1]
        # [bs, n_agents, num_actions, 1]

        # Softmax the agent outputs if they're policy logits
        if self.agent_output_type == "pi_logits":

            if getattr(self.args, "mask_before_softmax", True):
                # Make the logits for unavailable actions very negative to minimise their affect on the softmax
                reshaped_avail_actions = avail_actions.reshape(ep_batch.batch_size * self.n_agents, -1)
                agent_outs[reshaped_avail_actions == 0] = -1e10

            agent_outs = th.nn.functional.softmax(agent_outs, dim=-1)

        return agent_outs.view(ep_batch.batch_size, self.num_agents, -1) # [bs,n,num_action]
    

    """ TO DO LIST """
    # load save cuda 的逻辑要按照list来改
    def load_state(self, other_mac):
        self.agents.load_state_dict(other_mac.agent.state_dict())

    def cuda(self):
        self.agents.cuda()

    def save_models(self, path):
        th.save(self.agents.state_dict(), "{}/agent.th".format(path))

    def load_models(self, path):
        self.agents.load_state_dict(th.load("{}/agent.th".format(path), map_location=lambda storage, loc: storage))


    # 构建list agents
    def _build_agents(self, input_shape):
        # 传统 mac 用了 param sharing，只需要一个agent过多次；lio mac 用 ind learning，维护 list agents
        # self.agents = agent_REGISTRY[self.args.agent](input_shape, agent_id=i, args_env=self.args_env, args_alg=self.args_alg) 
        self.agents = [agent_REGISTRY[self.args.agent](input_shape, agent_id=i, args_env=self.args_env, args_alg=self.args_alg) for i in range(self.num_agents)]
        # id 目前没有参与计算，只是用来添加id属性给每个agent

    """这里要考虑检查lio源代码，actor和inc的input是不是一样的，有没有inc多了一部分other actions
        但是本代码选择手动定义了1hot other actions处理方法，所以似乎这里也用不到了，就是返回obs即可"""
    def _build_inputs(self, batch, t):
        # Assumes homogenous agents with flat observations.
        # Other MACs might want to e.g. delegate building inputs to each agent
        bs = batch.batch_size
        inputs = []
        if self.args.rgb_input:
            data = batch['obs'][:, t]
            data = data.reshape((self.num_agents,bs, 3, self.args.height, self.args.width))  
            return data # [n,bs,...]
        else:
            inputs.append(batch["obs"][:, t])  # b1av, [bs,t,n,..] ==> [bs,n,...]


        # """以下为 homo 代码，考虑额外的输入信息"""
        # if self.args.obs_last_action:
        #     if t == 0:
        #         inputs.append(th.zeros_like(batch["actions_onehot"][:, t]))
        #     else:
        #         inputs.append(batch["actions_onehot"][:, t - 1])
        # if self.args.obs_agent_id:
        #     inputs.append(th.eye(self.n_agents, device=batch.device).unsqueeze(0).expand(bs, -1, -1))

        # if self.args.obs_reward:
        #     if t == 0:
        #         inputs.append(th.zeros_like(batch["reward"][:, t]))
        #     else:
        #         rewards = batch["reward"][:, t - 1].clone()  # [bs,n]
        #         inputs.append(th.sign(rewards))

        # if self.args.obs_inc_reward:
        #     if t == 0:
        #         inputs.append(th.zeros_like(batch["reward"][:, t]))
        #     else:
        #         actions_inc = batch["actions_inc"][:, t - 1]  # [bs,n,n,1]
        #         actions_inc_masked = actions_inc * self.inc_mask_actions
        #         receive_value = th.stack([
        #             th.sum(actions_inc_masked[:, :, i] == 1, dim=(1, 2)) \
        #             - th.sum(actions_inc_masked[:, :, i] == 2, dim=(1, 2))
        #             for i in range(self.n_agents)], dim=-1)
        #         inputs.append(th.sign(receive_value.float()))

        # if self.args.obs_others_last_action:
        #     bs = batch["actions_onehot"][:, t].shape[0]
        #     if t == 0:
        #         inputs.append(th.zeros_like(
        #             batch["actions_onehot"][:, t].repeat(1, self.n_agents, 1).reshape(bs, self.n_agents, -1)))
        #     else:
        #         inputs.append(
        #             batch["actions_onehot"][:, t - 1].repeat(1, self.n_agents, 1).reshape(bs, self.n_agents, -1))
                
        # if self.args.obs_distance:
        #     agent_pos = batch["agent_pos"][:, t]  # [bs,n,2]
        #     agent_distance = 1.0 - (agent_pos.unsqueeze(2) - agent_pos.unsqueeze(1)).norm(dim=-1) / (
        #         np.linalg.norm(self.args.state_dims))  # [bs,n,n]
        #     inputs.append(agent_distance)

        # if self.args.obs_agent_pos:
        #     agent_pos = batch["agent_pos"][:, t] / np.linalg.norm(self.args.state_dims)  # [bs,n,2]
        #     inputs.append(agent_pos)


        inputs = th.cat([x.transpose(0, 1) for x in inputs], dim=-1)
        return inputs # [n,bs,-1]
    

    def get_action_others_1hot(action_all, agent_id, l_action):
        action_all = list(action_all)
        del action_all[agent_id]
        num_others = len(action_all)
        actions_1hot = np.zeros([num_others, l_action], dtype=int)
        actions_1hot[np.arange(num_others), action_all] = 1
        # [[0, 1, 0],  # agent 1 选择了动作 1
        #  [0, 0, 1]]  # agent 2 选择了动作 2

        return actions_1hot.flatten()
        # flattened_actions_1hot = [0, 1, 0, 0, 0, 1]  
        

    def _get_input_shape(self, scheme):
        if self.args.rgb_input:
            input_shape = 1 # 无意义，整小点省的浪费空间
        else:
            input_shape = scheme["obs"]["vshape"]

        # """ homo 额外信息，同样的逻辑，单独定义了other actions 1hot所以不需要这里处理了 """
        # if self.args.obs_last_action:
        #     input_shape += scheme["actions_onehot"]["vshape"][0]
        # if self.args.obs_agent_id:
        #     input_shape += self.n_agents
        # if self.args.obs_reward:
        #     input_shape += 1
        # if self.args.obs_inc_reward:
        #     input_shape += 1
        # if self.args.obs_others_last_action:
        #     input_shape += scheme["actions_onehot"]["vshape"][0] * self.n_agents
        # if self.args.obs_distance:
        #     input_shape += self.n_agents
        # if self.args.obs_agent_pos:
        #     input_shape += 2

        return input_shape
    
    """这个似乎没用到？"""
    def expand_rewards_batch(self, rewards):
        bs, _, _ = rewards.size()
        rewards_full = th.zeros(bs, self.num_agents, self.num_agents, device=self.args.device)
 
        rewards_full[:, self.mask] = rewards.view(bs, -1)
        
        return rewards_full


    


