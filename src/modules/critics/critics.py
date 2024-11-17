import torch as th
import torch.nn as nn
import torch.nn.functional as F

# critics定义  critc 网络
class Critic(nn.Module):
    # def __init__(self, input_shape, args_env, args_alg):
    def __init__(self, scheme, args):
        super(Critic, self).__init__()

        self.args = args
        self.n_agents = args.n_agents
        self.critic_h1 = args.alg_args.get('critic_h1')
        self.critic_h2 = args.alg_args.get('critic_h2')
        

        input_shape = self._get_input_shape(scheme)

        """同样的问题，input shape在scheme里是 3，9，9"""
        input_shape = input_shape[1]

        self.output_type = args.agent_output_type   # "q" / "v"

        # Set up network layers
        self.fc1_value = nn.Sequential(
                nn.Linear(input_shape, self.critic_h1), 
                nn.ReLU()
            )  
        self.fc2_value = nn.Sequential(
                nn.Linear(self.critic_h1, self.critic_h2),
                nn.ReLU()
            )
        self.fc3_value = nn.Linear(self.critic_h2, 1)

    def forward(self, batch, t=None):
        
        inputs, bs, max_t = self._build_inputs(batch, t=t)

        x = self.fc1_value(inputs)
        x = self.fc2_value(x)
        value = self.fc3_value(x)

        return value

    def _build_inputs(self, batch, t=None):
        bs = batch.batch_size
        max_t = batch.max_seq_length if t is None else 1
        ts = slice(None) if t is None else slice(t, t+1)
        inputs = []
        # observations
        inputs.append(batch["obs"][:, ts])

        inputs.append(th.eye(self.n_agents, device=batch.device).unsqueeze(0).unsqueeze(0).expand(bs, max_t, -1, -1))

        inputs = th.cat(inputs, dim=-1)
        return inputs, bs, max_t



    def _get_input_shape(self, scheme):
        
        # observation
        if self.args.obs_individual_obs:
            input_shape += scheme["obs"]["vshape"]
        else:
            # state
            input_shape = scheme["state"]["vshape"]

        # actions
        if self.args.obs_other_actions:
            input_shape += scheme["actions_onehot"]["vshape"][0] * self.n_agents

        # last action
        if self.args.obs_last_action:
            input_shape += scheme["actions_onehot"]["vshape"][0] * self.n_agents

        # agent id
        if self.args.obs_agent_id:
            input_shape += self.n_agents

        return input_shape
    


class CriticConv(nn.Module):
    def __init__(self, scheme, args):
        super(CriticConv, self).__init__()

        self.args = args
        self.n_agents = args.n_agents

        self.obs_height = scheme.get("view_size")['vshape'][0]
        self.obs_width = scheme.get("view_size")['vshape'][0]
        self.n_filters = args.alg_args.get('n_filters')
        self.kernel = args.alg_args.get('kernel')
        self.stride = args.alg_args.get('stride')
        self.critic_h1 = args.alg_args.get('critic_h1')
        self.critic_h2 = args.alg_args.get('critic_h2')

        input_shape = self._get_input_shape(scheme)
        self.output_type = args.agent_output_type   # "q" / "v"

        # Set up network layers
        self.conv_to_fc_value = nn.Sequential(
                nn.Conv2d(3, self.n_filters, self.kernel, self.stride),
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(self.n_filters * (self.obs_height - self.kernel[0] + 1) * (
                        self.obs_width - self.kernel[1] + 1), self.critic_h1),
                nn.ReLU()
            )
        self.fc2_value = nn.Sequential(
                nn.Linear(self.critic_h1, self.critic_h2),
                nn.ReLU()
            )
        self.fc3_value = nn.Linear(self.critic_h2, 1)
    
    
    def forward(self, batch, t=None):

        inputs, bs, max_t = self._build_inputs(batch, t=t)

        x = self.conv_to_fc_value(inputs)
        x = self.fc2_value(x)
        value = self.fc3_value(x)

        return value
    
    # independet learning 先不考虑 COMA 的其他agent动作作为输入
    def _build_inputs(self, batch, t=None):
        bs = batch.batch_size
        max_t = batch.max_seq_length if t is None else 1
        ts = slice(None) if t is None else slice(t, t+1)
        inputs = []
        # observations
        inputs.append(batch["obs"][:, ts])

        inputs.append(th.eye(self.n_agents, device=batch.device).unsqueeze(0).unsqueeze(0).expand(bs, max_t, -1, -1))

        inputs = th.cat(inputs, dim=-1)
        return inputs, bs, max_t


    def _get_input_shape(self, scheme):
        
        # observation
        if self.args.obs_individual_obs:
            input_shape += scheme["obs"]["vshape"]
        else:
            # state
            input_shape = scheme["state"]["vshape"]

        # actions
        if self.args.obs_other_actions:
            input_shape += scheme["actions_onehot"]["vshape"][0] * self.n_agents

        # last action
        if self.args.obs_last_action:
            input_shape += scheme["actions_onehot"]["vshape"][0] * self.n_agents

        # agent id
        if self.args.obs_agent_id:
            input_shape += self.n_agents

        return input_shape