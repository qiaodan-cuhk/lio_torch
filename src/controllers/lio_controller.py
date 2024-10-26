from modules.agents import REGISTRY as agent_REGISTRY
from components.action_selectors import REGISTRY as action_REGISTRY
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt


# This multi-agent controller shares parameters between agents
class LIOMAC(nn.Module):
    def __init__(self, scheme, args_env, args_alg):
        super(LIOMAC, self).__init__()
        self.num_agents = args_env.num_agents
        self.args_env = args_env
        self.args_alg = args_alg
        input_shape = self._get_input_shape(scheme)
        self._build_agents(input_shape)
        self.agent_output_type = args_alg.agent_output_type

        self.action_selector = action_REGISTRY[args_alg.action_selector](self.args_alg)

        self.mask = th.ones(self.num_agents, self.num_agents, dtype=th.bool, device=self.args.device)
        self.mask.fill_diagonal_(False)
        self.mask_ = self.mask.unsqueeze(0)  # [1,n,n]

    def select_actions_actor(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False):
        # Only select actions for the selected batch elements in bs
        avail_actions = ep_batch["avail_actions"][:, t_ep]
        policies = self.forward_actor(ep_batch, t_ep, test_mode=test_mode) # [bs,n,num_action]
        masks = avail_actions[bs] # [bs,n,num_action]
        chosen_actions = self.action_selector.select_action(policies[bs], masks , t_env, test_mode=test_mode)
        return chosen_actions
        # [bs,n]

    def expand_rewards_batch(self, rewards):
        bs, _, _ = rewards.size()
        rewards_full = th.zeros(bs, self.num_agents, self.num_agents, device=self.args.device)
 
        rewards_full[:, self.mask] = rewards.view(bs, -1)
        
        return rewards_full

    def parameters_actor(self):
        return [agent.parameters_actor() for agent in self.agents]
 
    def parameters_value(self):
        return [agent.parameters_value() for agent in self.agents]

    def parameters_reward(self):
        return [agent.parameters_reward() for agent in self.agents]
    
    def forward_actor(self, ep_batch, t, test_mode=False, learning_mode=False):
        self.agent_inputs = self._build_inputs(ep_batch, t)  # [n,bs,...]
        policies = th.concat([self.agents[i].forward_actor(self.agent_inputs[i]) for i in range(self.num_agents)], dim=-1)
        return policies.view(ep_batch.batch_size, self.num_agents, -1) # [bs,n,num_action]
    
    def forward_value(self, ep_batch, t, test_mode=False, learning_mode=False):
        self.agent_inputs = self._build_inputs(ep_batch, t)  # [n,bs,...]
        values = th.concat([self.agents[i].forward_value(self.agent_inputs[i]) for i in range(self.num_agents)], dim=-1)
        return values.view(ep_batch.batch_size, self.num_agents) # [bs,n]
    
    def forward_reward(self, ep_batch, t, actions, test_mode=False, learning_mode=False):
        actions_for_agents = actions.unsqueeze(1).expand(ep_batch.batch_size, self.num_agents, self.num_agents)
        actions_for_agents = actions_for_agents.masked_select(self.mask_).view(ep_batch.batch_size, self.num_agents, self.num_agents-1).transpose(0, 1)
        # [n,bs,n-1]

        rewards = th.concat([self.agenst[i].forward_reward(self.agent_inputs[i], actions_for_agents[i]).unsqueeze(1) for i in range], dim=1)
        # [bs,n,n-1]
        rewards = self.expand_rewards_batch(rewards) # [bs,n,n]

        return rewards
    
    def _build_agents(self, input_shape):
        self.agents = [agent_REGISTRY[self.args.agent](input_shape, agent_id=i, self.args_env, self.args_alg) for i in range(self.num_agents)]

    def _get_input_shape(self, scheme):
        if self.args.rgb_input:
            input_shape = 1 # 无意义，整小点省的浪费空间
        else:
            input_shape = scheme["obs"]["vshape"]

        return input_shape

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

        inputs = th.cat([x.transpose(0, 1) for x in inputs], dim=-1)
        return inputs # [n,bs,-1]
