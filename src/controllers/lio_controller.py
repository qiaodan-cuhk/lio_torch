from modules.agents import REGISTRY as agent_REGISTRY
from components.action_selectors import REGISTRY as action_REGISTRY
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt


# This multi-agent controller shares parameters between agents
class LIOMAC(nn.Module):
    def __init__(self, scheme, groups, args):
        super(LIOMAC, self).__init__()
        self.n_agents = args.n_agents
        self.args = args
        input_shape = self._get_input_shape(scheme)
        self._build_agents(input_shape)
        self.agent_output_type = args.agent_output_type

        self.action_selector = action_REGISTRY[args.action_selector](args)

        self.h_env = None
        self.h_inc = None

        self.extra_return_env = None
        self.extra_return_inc = None

        self.inc_mask_actions = (1 - th.eye(self.n_agents)).reshape(1, self.n_agents, self.n_agents, 1).to(self.args.device)

    def select_actions_env(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False):
        # Only select actions for the selected batch elements in bs
        avail_actions = ep_batch["avail_actions"][:, t_ep]
        q_env = self.forward_env(ep_batch, t_ep, test_mode=test_mode)
        masks = avail_actions[bs] # [bs,n,n_env]
        chosen_actions = self.action_selector.select_action(q_env[bs], masks , t_env, test_mode=test_mode)
        chosen_actions = chosen_actions.unsqueeze(-1)

        return chosen_actions
        # [bs,n,1]

    def select_actions_inc(self, chosen_actions, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False, agent_pos_replay = None):
        
