from modules.agents import REGISTRY as agent_REGISTRY
from components.action_selectors import REGISTRY as action_REGISTRY
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

# This multi-agent controller DO NOT shares parameters between agents
class LIOMAC(nn.Module):
    def __init__(self, scheme, groups, args):
        super(LIOMAC, self).__init__()
        self.scheme = scheme
        self.args = args
        self.args_env = args.env_args
        self.args_alg = args.alg_args
        self.num_agents = self.args_env.get('num_agents', 0)
        if self.num_agents == 2:
            self.args_alg["lio_asymmetric"] = True

        self.rgb_input = args.rgb_input  # self.args_alg.get("rgb_input")
        if self.rgb_input:
            self.obs_height = scheme.get("obs_dims")['vshape'][0]
            self.obs_width = scheme.get("obs_dims")['vshape'][1]

        input_shape = self._get_input_shape(scheme) 

        self.l_actions = scheme.get('avail_actions', 0)['vshape'][0]
        self._build_agents(input_shape)
        # 这里要考虑 agent 的Conv kernel长度，用view size还是scheme里的obs shape？


        # This handles the special case of two asymmetric agents,
        # one of which is the reward-giver and the other is the recipient
        if self.args_alg.get('lio_asymmetric', False):
            assert self.num_agents == 2
            for agent_id in range(self.num_agents):
                self.agents[agent_id].set_can_give(agent_id != self.args_alg.get('idx_recipient'))

        self.agent_output_type = self.args.agent_output_type
        self.action_selector = action_REGISTRY[self.args.action_selector](self.args)

        self.mask = th.ones(self.num_agents, self.num_agents, dtype=th.bool, device=self.args.device)
        self.mask.fill_diagonal_(False)
        self.mask_ = self.mask.unsqueeze(0)  # [1,n,n]  

        # 没使用RNN，所以不用考虑hidden states和init hidden
        # 激励函数的mask，禁止自己给自己奖励，创建一个diag为0的单位矩阵，我们考虑使用原始lio的方法，可以不用这个
        self.inc_mask_actions = (1 - th.eye(self.num_agents)).reshape(1, self.num_agents, self.num_agents, 1).to(self.args.device)
    
    def select_actions_env(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False):
        # Only select actions for the selected batch elements in bs
        avail_actions = ep_batch["avail_actions"][:, t_ep]
        agent_outputs = self.forward_actors(ep_batch, t_ep, test_mode=test_mode) 
        # [bs, n_agents, n_actions] 输出为logits (q)或者softmax (pi_logits)

        masks = avail_actions[bs] # [bs,n,num_action]
        chosen_actions = self.action_selector.select_action(agent_outputs[bs], masks , t_env, test_mode=test_mode)  # [bs,n]
        chosen_actions = chosen_actions.unsqueeze(-1) # [bs,n,1]

        return chosen_actions
        


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
        actions_for_agents = list_actions.squeeze(-1)   # 1,2,1 -> 1,2  [bs, n agents]
        actions_for_agents = actions_for_agents.unsqueeze(1).expand(ep_batch.batch_size, self.num_agents, self.num_agents)
        actions_for_agents = actions_for_agents.masked_select(self.mask_).view(ep_batch.batch_size, self.num_agents, self.num_agents-1).transpose(0, 1)
        # [n,bs,n-1] 代表每个agent，bs数据，拿掉自己的动作后其他agent的动作列表

        list_rewards = []
        total_reward_given_to_each_agent = th.zeros((ep_batch.batch_size, self.num_agents), device=self.args.device)

        all_other_actions_1hot = self.get_action_others_1hot(actions_for_agents, self.l_actions)
        # [n_agent, bs, n_agent-1, n_actions]

        for i in range(self.num_agents):
            agent = self.agents[i]
            other_actions_1hot = all_other_actions_1hot[i].view(ep_batch.batch_size,-1)   # [bs, n_agent-1, n_actions]

            if agent.can_give:
                reward = agent.forward_incentive(agent_inputs_inc[i], other_actions_1hot)  # sigmoid output 
                reward[-1][agent.agent_id] = 0
            else:
                reward = th.zeros((ep_batch.batch_size, self.num_agents), device=self.args.device) 
            # [bs, n_agents] 表示当前agent i给所有agent 0-n 的reward，自己给自己的是 0

            total_reward_given_to_each_agent += reward   # 表示了每个agent收到了总共多少reward
            list_rewards.append(reward)   # 表示所有agent的inc情况，用于runner里的reward buffer save和learner里的计算

        return total_reward_given_to_each_agent, list_rewards
        

    """ policy sampling """
    # mac.forward_actors 调用每个LIO.forward_actor，返回 logits 用于计算loss
    def forward_actors(self, ep_batch, t, test_mode=False, learning_mode=False):
        self.agent_inputs = self._build_inputs(ep_batch, t)  # [n,bs,...]
        avail_actions = ep_batch["avail_actions"][:, t]

        agent_outs = th.cat([self.agents[i].forward_actor(self.agent_inputs[i]) for i in range(self.num_agents)], dim=0) # 把两个 [1,9] 拼成 [n_agents, 9]
        # 这里是actor网络输出的 [1,18]
        # # [bs, n_agents, num_actions, 1]

        # Softmax the agent outputs if they're policy logits
        if self.agent_output_type == "pi_logits":
            # 如果在softmax之前mask，那么就先把logits做mask，再softmax，这个output在select actions时也会被再次mask一次；
            # 如果不用，那么就是先softmax，再去select actions里mask掉，categorical采样时可能不是sum 1的概率
            if getattr(self.args, "mask_before_softmax", True):
                # Make the logits for unavailable actions very negative to minimise their affect on the softmax
                reshaped_avail_actions = avail_actions.reshape(ep_batch.batch_size * self.num_agents, -1)
                agent_outs[reshaped_avail_actions == 0] = -1e10
            agent_outs = th.nn.functional.softmax(agent_outs, dim=-1)

        return agent_outs.view(ep_batch.batch_size, self.num_agents, -1) 
        # [bs,n,num_action] 
        # 如果用 q，返回的就是 Q value logits，如果pi logits那么返回的是一个softmax后的分布，可以直接sample
        
    
    """ prime policy sampling """
    def forward_actors_prime(self, ep_batch, t, test_mode=False, learning_mode=False):

        self.agent_inputs = self._build_inputs(ep_batch, t)  # [n,bs,...]
        avail_actions = ep_batch["avail_actions"][:, t]

        agent_outs = th.cat([self.agents[i].forward_actor_prime(self.agent_inputs[i]) for i in range(self.num_agents)], dim=-1)  # 这里是actor网络输出的 [bs, n_actions, 1]
        # [bs, n_agents, num_actions, 1]

        # Softmax the agent outputs if they're policy logits
        if self.agent_output_type == "pi_logits":

            if getattr(self.args, "mask_before_softmax", True):
                # Make the logits for unavailable actions very negative to minimise their affect on the softmax
                reshaped_avail_actions = avail_actions.reshape(ep_batch.batch_size * self.n_agents, -1)
                agent_outs[reshaped_avail_actions == 0] = -1e10

            agent_outs = th.nn.functional.softmax(agent_outs, dim=-1)

        return agent_outs.view(ep_batch.batch_size, self.num_agents, -1) # [bs,n,num_action]
    

    def load_state(self, other_mac):
        self.agents.load_state_dict(other_mac.agent.state_dict())

    def cuda(self):
        for agent in self.agents:
            agent.cuda()

    def save_models(self, path):
        # th.save(self.agents.state_dict(), "{}/agent.th".format(path))
        agents_dict = {}
        for i, agent in enumerate(self.agents):
            agents_dict.update({
                f"agent_{i}_actor": agent.actor.state_dict(),
                f"agent_{i}_actor_prime": agent.actor_prime.state_dict(),
                f"agent_{i}_inc": agent.inc.state_dict()
            })
        th.save(agents_dict, f"{path}/agents.th")


    def load_models(self, path):
        # self.agents.load_state_dict(th.load("{}/agent.th".format(path), map_location=lambda storage, loc: storage))
        agents_dict = th.load(f"{path}/agents.th", map_location=self.args.device)
        for i, agent in enumerate(self.agents):
            agent.actor.load_state_dict(agents_dict[f"agent_{i}_actor"])
            agent.actor_prime.load_state_dict(agents_dict[f"agent_{i}_actor_prime"])
            agent.inc.load_state_dict(agents_dict[f"agent_{i}_inc"])


    # ind learning, list agents; param sharing, one agent
    def _build_agents(self, input_shape):
        # self.agents = agent_REGISTRY[self.args.agent](input_shape, agent_id=i, args_env=self.args_env, args_alg=self.args_alg) 
        self.agents = [agent_REGISTRY[self.args.agent](input_shape, agent_id=i, scheme=self.scheme, args=self.args) for i in range(self.num_agents)]



    """这里要考虑检查lio源代码，actor和inc的input是不是一样的，有没有inc多了一部分other actions
        但是本代码选择手动定义了1hot other actions处理方法，所以似乎这里也用不到了，就是返回obs即可"""
    def _build_inputs(self, batch, t):
        # Assumes homogenous agents with flat observations.
        # Other MACs might want to e.g. delegate building inputs to each agent
        bs = batch.batch_size
        inputs = []

        if self.rgb_input:
            data = batch['obs'][:, t]   # batch["obs"][:,t].shape = [1(bs), 2(agents), 3,9,9 (obs shape)]
            data = data.reshape((self.num_agents, bs, 3, self.obs_height, self.obs_width))  
            return data # [n,bs,...]
        else:
            inputs.append(batch["obs"][:, t])  
            # b1av, [bs,t,n,..] ==> [bs,n,...]
            

        """以下为 homo 代码，考虑额外的输入信息"""
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
        return inputs # [n,bs,-1]   [2, 1, 3, 9, 9]
    

    def get_action_others_1hot(self, action_all, l_action):
        
        actions_1hot = F.one_hot(action_all, l_action)  # [n, bs, n-1] -> [n, bs, n-1, n_actions]

        # action_all = list(action_all)
        # del action_all[agent_id]
        # num_others = len(action_all)
        # actions_1hot = np.zeros([num_others, l_action], dtype=int)
        # actions_1hot[np.arange(num_others), action_all] = 1
        # [[0, 1, 0],  # agent 1 选择了动作 1
        #  [0, 0, 1]]  # agent 2 选择了动作 2

        return actions_1hot
        # flattened_actions_1hot = [0, 1, 0, 0, 0, 1]  
        

    def _get_input_shape(self, scheme):
        if self.rgb_input:
            input_shape = None 
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


    


