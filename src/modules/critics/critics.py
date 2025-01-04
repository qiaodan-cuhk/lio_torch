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


    def forward(self, inputs, t=None):        
        # inputs, bs, max_t = self._build_inputs(batch, t=t)

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
        # batch['obs']  16, 101, 2, 3, 9, 9; 上面相当于对第二个维度 101 timesteps 做slice，得到16 1 2 3 9 9

        inputs.append(th.eye(self.n_agents, device=batch.device).unsqueeze(0).unsqueeze(0).expand(bs, max_t, -1, -1))

        inputs = th.cat(inputs, dim=-1)
        return inputs, bs, max_t
    
    def _build_inputs_next(self, batch, t=None):
        bs = batch.batch_size
        max_t = batch.max_seq_length if t is None else 1
        ts = slice(None) if t is None else slice(t, t+1)
        inputs = []

        # observations next time
        inputs.append(batch["obs_next"][:, ts])  
        
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

        self.obs_height = scheme.get("obs_dims")['vshape'][0]
        self.obs_width = scheme.get("obs_dims")['vshape'][1]
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
    
    
    def forward(self, inputs, t=None):
        # inputs, bs, max_t = self._build_inputs(batch, t=t)   # 1616,3,9,9
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

        if self.args.rgb_input:
            data = batch['obs'][:, ts]   # 16, 101, 2, 3, 9, 9 
            data = data.view(-1, self.n_agents, 3, 9, 9)  # [16*101, 2, 3, 9, 9]
            inputs.append(data)
            # data = data.permute(1,0,2,3,4)  # 2, 101*16, 399
            # data = data.reshape((self.num_agents, bs, 3, self.obs_height, self.obs_width))  
        else:
            inputs.append(batch["obs"][:, t])  
            # b1av, [bs,t,n,..] ==> [bs,n,...]

        inputs = th.cat([x.transpose(0, 1) for x in inputs], dim=-1)
 
        # 这里应该是为了增加diag（n agents）标识id
        # inputs.append(th.eye(self.n_agents, device=batch.device).unsqueeze(0).unsqueeze(0).expand(bs, max_t, -1, -1))

        return inputs, bs, max_t
    
    
    def _build_inputs_next(self, batch, t=None):
        bs = batch.batch_size
        max_t = batch.max_seq_length if t is None else 1
        ts = slice(None) if t is None else slice(t, t+1)
        inputs = []

        # observations next time
        inputs.append(batch["obs_next"][:, ts])  
        
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

    # def get_next_v(self, batch, t=None):
    #     """获取下一时刻的V值"""
    #     # 使用batch中的next_obs计算next_v
    #     inputs, bs, max_t = self._build_next_inputs(batch, t=t)
        
    #     x = self.fc1_value(inputs)
    #     x = self.fc2_value(x) 
    #     next_v = self.fc3_value(x)
        
    #     return next_v

    