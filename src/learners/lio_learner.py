#创建新的策略网络&赋值

import copy
from components.episode_buffer import EpisodeBatch
import torch as th
from torch.optim import RMSprop, Adam
import numpy as np

import sys
import os
# 添加全局路径
sys.path.append('/home/qiaodan/Projects/lio_torch/src')

from modules.critics import REGISTRY as critic_resigtry

# logger 要增加一些测量incentivize的metric

class LIOLearner:
    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.n_agents = args.n_agents
        self.n_actions = args.n_actions
        self.logger = logger
        self.scheme = scheme

        """ actor 输入 obs 输出 n actions；inc输入obs+other actions，输出n_agents个激励，然后mask自己
            其中 inc 输入的 other actions 是一个 bs, n-1 的量，默认id顺序，剔除自己"""
        # each agent one opt
        self.mac = mac  # 包括 actor/prime actor/inc NN
        self.agents = mac.agents  # 假设 mac 是 NonSharedMAC，[lio_1, lio_2, lio_3]
        self.actor_params = [list(agent.actor.parameters()) for agent in self.agents]  
        self.actor_optimizers = [Adam(params=params, lr=args.lr_actor) for params in self.actor_params] 

        self.actor_prime_params = [list(agent.actor_prime.parameters()) for agent in self.agents]  
        self.actor_prime_optimizers = [Adam(params=params, lr=args.lr_actor) for params in self.actor_prime_params]  

        self.inc_params = [list(agent.inc.parameters()) for agent in self.agents]  
        self.inc_optimizers = [Adam(params=params, lr=args.lr_inc) for params in self.inc_params] 

        # 假设 self.critic 是一个包含多个独立 critic 网络的列表
        if args.rgb_input:
            critic_type = "ac_conv"
        else:
            critic_type = "ac"
        self.critics = [critic_resigtry[critic_type](scheme, args) for i in range(self.n_agents)] # args.critic_type = rgb/not
        self.target_critics = copy.deepcopy(self.critics)
        self.critic_params = [list(critic.parameters()) for critic in self.critics]  # 每个 agent 的 critic 参数
        self.critic_optimizers = [Adam(params=params, lr=args.lr_v) for params in self.critic_params]  # 每个 agent 的 critic 优化器

        self.log_stats_t = -self.args.learner_log_interval - 1

        self.gamma = args.gamma   # 也可以 gamma_env/inc
        self.gamma_env = args.gamma_env
        self.gamma_inc = args.gamma_inc

        self.critic_training_steps = 0
        self.last_target_update_step = 0
        


    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):

        # 把main actor参数复制给prime保证一致
        self.update_prime_from_policy()
        # for i, agent in enumerate(self.agents):
        #     # 获取当前 agent 的 actor 和 actor_prime
        #     actor_params = agent.actor.state_dict()  # 获取 actor 的参数
        #     agent.actor_prime.load_state_dict(actor_params)  # 将参数加载到 actor_prime

        # Get the relevant quantities
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        
        mask = mask.repeat(1, 1, self.n_agents)
        critic_mask = mask.clone()
        bs = batch.batch_size

        # ************************************************ mac out *****************************************************
        mac_out_prime = []
        # self.mac.init_hidden(batch.batch_size)  # RNN agent 才用

        # 检查数据格式
        for t in range(batch.max_seq_length):
            # 确认返回的是logits还是softmax
            actor_logits_prime = self.mac.forward_actors_prime(batch, t=t)
            # inc_logits = self.mac.forward_inc(batch, t=t)
            # q_env_t, q_inc_t, extra_return = self.mac.forward(batch, t=t)
            mac_out_prime.append(actor_logits_prime)  # [t, bs,n,a_env] # 101 list torch[16,5,9]
            # inc_out.append(inc_logits)  # [bs,n,n,a_inc]

        mac_out_prime = th.stack(mac_out_prime, dim=1)  # [bs,t,n,a_env]  
        pi_prime = mac_out_prime[:, :-1] # 16, 100, 2, 9

        """ Update value network """
        V_t, r2_val, critic_train_stats_log, V_td_error = self._train_critic(batch, rewards, bs, 1) 

        actions = actions[:, :-1]  # [16, 100, 2, 1]
        # V_t_next = V_t_next.detach()  # 需要detach，作为一个参考值

        # Calculate policy grad with mask，都用prime policy的参数，这里计算entropy表示prime pi必然是一个softmax分布
        mask_prime = mask.unsqueeze(-1).expand(-1,-1,-1,9)   # [16, 100, 2, 9]
        pi_prime[mask_prime == 0] = 1.0

        # 处理动作
        # actions_1hot = util.process_actions(buf.action, self.l_action)

        """梯度从prime算"""
        actor_prime_losses = []  # 用于存储每个代理的损失

        for i, agent in enumerate(self.agents):
            # 计算当前代理的 mask
            agent_mask_i = mask_prime[:, :, i, 0]  # mask 的形状是 [bs, seq_len, n_agents, 1]
            agent_pi = pi_prime[:, :, i, :]
            agent_actions = actions[:, :, i, 0]

            # gather操作获取实际动作的概率
            pi_taken_prime_i = th.gather(
                agent_pi,           # [16, 100, 9]
                dim=-1,            # 在动作维度上gather
                index=agent_actions.unsqueeze(-1)  # [16, 100, 1]
            ).squeeze(-1)         # [16, 100]
            
            # pi_taken_prime_i = th.gather(pi_prime[i], dim=-1, index=actions[..., i]).squeeze(3)  # 所有agent的prime
            log_pi_prime_taken_i = th.log(pi_taken_prime_i + 1e-10)
            entropy_prime_i = -th.sum(agent_pi * th.log(agent_pi + 1e-10), dim=-1)

            # 计算当前代理的损失
            # 需要将 V_td_error 的维度调整为 [batch_size]，以便与其他项相乘
            current_V_td_error = V_td_error[i].detach()  # 选择当前代理的 TD 误差，维度为 [batch_size]，断开梯度防止回流critic

            actor_prime_loss_i = (
                -(
                    (current_V_td_error * log_pi_prime_taken_i + self.args.alg_args["entropy_coeff"] * entropy_prime_i) * agent_mask_i
                ).sum()
                / agent_mask_i.sum()
            )

            actor_prime_losses.append(actor_prime_loss_i)


        """更新 prime actor, actor还是原来的参数"""
        # 对每个代理的损失进行反向传播,存储梯度用不上了 (old)
        # 还是要存梯度用于手动更新策略
        total_actor_loss = sum(actor_prime_losses)
        for optimizer in self.actor_prime_optimizers:
            optimizer.zero_grad()
        total_actor_loss.backward()
        for optimizer in self.actor_prime_optimizers:
            optimizer.step()

        # 存储梯度
        self.actor_gradients = []
        for agent_params in self.actor_prime_params:
            agent_gradients = []
            for param in agent_params:
                if param.grad is not None:
                    agent_gradients.append(param.grad.clone())
                else:
                    agent_gradients.append(None)
            self.actor_gradients.append(agent_gradients)


        """ 更新 critic target network """
        if (
            self.args.target_update_interval > 1
            and (self.critic_training_steps - self.last_target_update_step)
            / self.args.target_update_interval
            >= 1.0
        ):
            self._update_targets_hard()
            self.last_target_update_step = self.critic_training_steps
        elif self.args.target_update_interval <= 1.0:
            self._update_targets_soft(self.args.tau)


        """ Logging """
        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            ts_logged = len(critic_train_stats_log["critic_loss"])
            for key in [
                "critic_loss",
                "critic_grad_norm",
                "td_error_abs",
                "q_taken_mean",
                "target_mean",
            ]:
                self.logger.log_stat(
                    key, sum(critic_train_stats_log[key]) / ts_logged, t_env
                )

            self.log_stats_t = t_env

       

    def train_reward(self, buffer, new_buffer, t_env):
        """训练激励函数"""

        # 1. 创建并更新所有agent的new policy
        # 注意原始代码的写法是每个agent都维护一整个n agents的list policy new
        for id, agent in enumerate(self.agents): 
            agent.list_policy_new = [0 for x in range(self.n_agents)]
            for other_id, other_agent in enumerate(self.agents):
                if other_id == id:
                    continue

                new_policy = copy.deepcopy(self.agents[other_id].actor)  # mac.agents.actor
                # 使用存储的梯度手动更新参数
                for param, grad in zip(new_policy.parameters(), self.actor_gradients[other_id]):
                    if grad is not None:
                        # param.data.add_(-self.args.lr_actor * grad)
                        # 创建新参数,保持计算图连接
                        param.data = param.data - self.args.lr_actor * grad
            
                agent.list_policy_new[other_id] = new_policy
        
        
        inc_loss_list = []
        # 这层for循环对应要更新的inc网络的agent
        for id, agent in enumerate(self.agents):
            assert id == agent.agent_id

            # # 更新正则化系数
            # agent.update_reg_coeff(self, performance, prev_reward_env)  # 更新 agent.reg_coeff

            if agent.can_give:
                buf_self_new = new_buffer[id]
                new_obs = buf_self_new["obs"][:, :-1, id]
                new_obs_next = buf_self_new["obs"][:, 1:, id]
                new_v = self.critics[id](new_obs)
                new_v_next = self.target_critics[id](new_obs_next)

                effect_ratio = self.args.alg_args["incentive_ratio"]
                cost_ratio = self.args.alg_args["incentive_cost"]
                # incentive values
                inc_rewards_list = buf_self_new["give_other_rewards_list"][:, :-1]  # [bs, t-1, n, n]
                recieved_rewards = buf_self_new["recieved_rewards"][:, :-1].squeeze(-1)  # [bs,t-1,n,1]
                rewards = buf_self_new["reward"][:, :-1]
                r_new_buffer = rewards + recieved_rewards * effect_ratio
                if self.args.alg_args["include_cost_in_chain_rule"]:
                    r_new_buffer -= cost_ratio * inc_rewards_list.sum(dim=-1)
                td_error_self_new = r_new_buffer + self.gamma * new_v_next.detach() - new_v

                # 存储对每个其他智能体的reward loss
                list_reward_loss = []  
                # 计算当前agent对于其他每个收到该agent的奖励的策略的影响
                for other_id, other_agent in enumerate(self.agents):
                    if agent.agent_id == other_agent.agent_id and not agent.include_cost_in_chain_rule:
                        continue
                    
                    # 计算接收者的td loss，用new buffer数据
                    """这里要注意self buffer的数据格式，保证能取到其他agent的观测，以及other actions的one-hot是从谁的视角出发的"""
                    other_obs = buf_self_new["obs"][:, :-1, other_id]  # [bs, t, obs_dim]
                    other_actions = buf_self_new["actions"][:, :-1, other_id]  # [bs, t, 1]
                    
                    # 正在更新的agent存储的其他agent new policy
                    actor_outs = agent.list_policy_new[other_id](other_obs)  # [bs, t, n_actions] 
                    actor_probs = th.nn.functional.softmax(actor_outs, dim=-1)

                    # 3. 计算选中动作的对数概率
                    chosen_action_probs = th.gather(
                        actor_probs,
                        dim=-1,
                        index=other_actions
                    ).squeeze(-1)  # [bs, t]
                    log_probs_i = th.log(chosen_action_probs + 1e-10)
                    loss_inc_i = -(log_probs_i * td_error_self_new.detach()).mean()  #这里为什么detach td error，是作为scalar不反传梯度吗？
                    list_reward_loss.append(loss_inc_i)

                # 6. 根据配置计算最终的loss
                if agent.include_cost_in_chain_rule:
                    inc_loss_i = sum(list_reward_loss)
                else:
                    # 创建掩码排除自己
                    reverse_mask = th.ones(self.n_agents)
                    reverse_mask[id] = 0
                    
                    # 计算 l1/l2 的 total give out reward
                    if agent.separate_cost_optimizer or agent.reg == 'l1':
                        # batch_size = agent.bs  """?????"""
                        batch_size = new_buffer["obs"].shape[0]
                        gamma_prod = th.cumprod(
                            th.ones(batch_size) * self.gamma, 
                            dim=0
                        )
                        given_each_step = th.sum(
                            th.abs(inc_rewards_list.sum(dim=-1) * reverse_mask), 
                            dim=1
                        )
                        total_given = th.sum(
                            given_each_step * gamma_prod/self.gamma
                        )
                    elif agent.reg == 'l2':
                        total_given = th.sum(
                            th.square(inc_rewards_list.sum(dim=-1) * reverse_mask)
                        )

                    if agent.separate_cost_optimizer:
                        inc_loss_i = th.sum(th.stack(list_reward_loss))
                    else:
                        inc_loss_i = (th.sum(th.stack(list_reward_loss)) + 
                                    agent.reg_coeff * total_given)
                
                inc_loss_list.append(inc_loss_i)


        """更新 inc net """
        total_inc_loss = sum(inc_loss_list)
        for optimizer_inc in self.inc_optimizers:
            optimizer_inc.zero_grad()
        total_inc_loss.backward()
        for optimizer_inc in self.inc_optimizers:
            optimizer_inc.step()


        # 8. 在所有agent的inc网络更新结束后，更新prime到actor策略网络；如果agent不能give，那么不更新inc直接覆盖actor网络
        self.update_policy_from_prime()


    def _train_critic(self, batch, rewards, bs, t):

        running_log = {
            "critic_loss": [],
            "critic_grad_norm": [],
            "td_error_abs": [],
            "target_mean": [],
            "q_taken_mean": [],
        }

        # 计算 V t
        # 构建所有agent的观测输入
        build_obs, _, max_t = self.critics[0]._build_inputs(batch)  # [agent_nums, bs*max_step, 3, 9, 9]

        # 每个critic前向计算得到对应agent的V值
        V_t_list = []
        for i in range(self.n_agents):
            V_i = self.critics[i](build_obs[i]).reshape(bs, max_t, 1)  # [bs, 101, 1]   # input 1616, 3, 9, 9
            V_t_list.append(V_i)

        # 拼接所有agent的V值
        V_t = th.cat(V_t_list, dim=-1)  # [bs, max_t, n_agents]
        V_t = V_t.permute(0,2,1) # [bs, n_agents, max_t]

        # V_t next，这里其实可以复用 V，串一个step就好了，按照episode 101 steps存的buffer
        # V_t_next = V_t.clone()[1,:]

        V_t_target = []
        # forward get value作为TD error的标量, 不需要 grad 做推理
        with th.no_grad():
            for i in range(self.n_agents):
                V_t_target_i = self.target_critics[i](build_obs[i]).reshape(bs, max_t, 1)   # [bs, 101, 1]   
                V_t_target.append(V_t_target_i)

        # 拼接所有agent的V值
        V_t_target = th.cat(V_t_target, dim=-1)  # [bs, n_agents]
        V_t_target = V_t_target.permute(0,2,1).detach() # [bs, n_agents, max_t]

        # inc ratio
        effect_ratio = self.args.alg_args["incentive_ratio"]
        cost_ratio = self.args.alg_args["incentive_cost"]

        # incentive values
        inc_rewards_list = batch["give_other_rewards_list"][:, :-1]  # [bs, t-1, n, n]
        recieved_rewards = batch["recieved_rewards"][:, :-1].squeeze(-1)  # [bs,t-1,n,1]

        # cal loss
        r2_val = rewards + recieved_rewards * effect_ratio
        if self.args.alg_args["include_cost_in_chain_rule"]:
            inc_loss = cost_ratio * inc_rewards_list.sum(dim=-1)
            r2_val -= inc_loss
        
        # 我认为应该断开env+inc reward的grad，因为本环节仅更新critic，不用保留到其他inc的梯度，V' target也是
        # v_td_error = r2_val.permute(0,2,1) + self.gamma * V_t_target[:,:,1:] - V_t[:,:,:-1]

        # 这里没选择 n-step TD error 写法，而是采用的 V target next time
        # target_returns = self.nstep_returns(r2_val, mask, target_vals, self.args.q_nstep)

        # 平方loss，没有考虑mask
        # 正确的写法：分别计算每个agent的td_error和loss

        loss = 0
        v_td_error = []
        for id, agent in enumerate(self.agents):
            # 为每个agent独立计算td_error
            # 这里使用V向后串一个位置的值作为V t+1
            agent_td_error = (r2_val.permute(0,2,1)[:,id] + 
                            self.gamma * V_t_target[:,id,1:] - 
                            V_t[:,id,:-1])  # shape: [16, 100]
            v_td_error.append(agent_td_error)
            
            # 计算当前agent的loss
            loss_i = (agent_td_error ** 2).sum()
            loss += loss_i
            
            # 更新当前agent的critic
            self.critic_optimizers[id].zero_grad()

        
        loss.backward()  # 不需要retain_graph

        grad_norms = []  # 新增：用于存储每个agent的grad_norm
        # 新增：在循环外计算所有agent的grad_norm
        for id in range(len(self.agents)):
            grad_norm = th.nn.utils.clip_grad_norm_(self.critic_params[id], self.args.grad_norm_clip)
            self.critic_optimizers[id].step()
            grad_norms.append(grad_norm)

        self.critic_training_steps += 1
        
        logging_td_error = th.stack(v_td_error, dim=1)
        running_log["critic_loss"].append(loss.item())
        running_log["critic_grad_norm"].append(sum(grad_norms) / len(grad_norms))
        # mask_elems = mask_t.sum().item()
        running_log["td_error_abs"].append(logging_td_error.abs().sum().item())  # 没考虑 mas
        # running_log["q_taken_mean"].append((q_taken * mask_t).sum().item() / mask_elems)
        # running_log["target_mean"].append((targets_t * mask_t).sum().item() / mask_elems)

        return V_t, r2_val, running_log, v_td_error
    



    def update_policy_from_prime(self):
        # 把actor参数复制给prime保证一致
        for i, agent in enumerate(self.agents):
            # 获取当前 agent 的 actor 和 actor_prime
            actor_prime_params = agent.actor_prime.state_dict()  # 获取 actor_prime 的参数
            agent.actor.load_state_dict(actor_prime_params)  # 将参数加载到 actor

    def update_prime_from_policy(self):
        # 把main actor参数复制给prime保证一致
        for i, agent in enumerate(self.agents):
            # 获取当前 agent 的 actor 和 actor_prime
            actor_params = agent.actor.state_dict()  # 获取 actor 的参数
            agent.actor_prime.load_state_dict(actor_params)  # 将参数加载到 actor_prime


    """也考虑是不是 target critic """

    def _update_targets_hard(self):
        for critic, target_critic in zip(self.critics, self.target_critics):
            target_critic.load_state_dict(critic.state_dict())

    def _update_targets_soft(self, tau):
        for i in range(len(self.target_critics)):
            target_params = self.target_critics[i].parameters()
            params = self.critics[i].parameters()
            
            for target_param, param in zip(target_params, params):
                target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)
                

    def cuda(self):
        self.mac.cuda()  
        for critic in self.critics:
            critic.to(self.args.device) 
        for target_critic in self.target_critics:
            target_critic.to(self.args.device) 


    def save_models(self, path):
        self.mac.save_models(path)
        critics_dict = {f"critic_{i}": critic.state_dict() for i, critic in enumerate(self.critics)}
        th.save(critics_dict, f"{path}/critics.th")
        
        actor_optimizers_dict = {f"actor_opt_{i}": opt.state_dict() for i, opt in enumerate(self.actor_optimizers)}
        th.save(actor_optimizers_dict, f"{path}/actor_opt.th")
        inc_optimizers_dict = {f"inc_opt_{i}": opt.state_dict() for i, opt in enumerate(self.inc_optimizers)}
        th.save(inc_optimizers_dict, f"{path}/inc_opt.th")
        critic_optimizers_dict = {f"critic_opt_{i}": opt.state_dict() for i, opt in enumerate(self.critic_optimizers)}
        th.save(critic_optimizers_dict, f"{path}/critic_opt.th")


    def load_models(self, path):
        self.mac.load_models(path)
        critics_dict = th.load(f"{path}/critics.th", map_location=self.args.device)
        for i, critic in enumerate(self.critics):
            critic.load_state_dict(critics_dict[f"critic_{i}"])
        self.target_critics = copy.deepcopy(self.critics)

        actor_opt_dict = th.load(f"{path}/actor_opt.th", map_location=self.args.device)
        for i, opt in enumerate(self.actor_optimizers):
            opt.load_state_dict(actor_opt_dict[f"actor_opt_{i}"])
        inc_opt_dict = th.load(f"{path}/inc_opt.th", map_location=self.args.device)
        for i, opt in enumerate(self.inc_optimizers):
            opt.load_state_dict(inc_opt_dict[f"inc_opt_{i}"])
        critic_opt_dict = th.load(f"{path}/critic_opt.th", map_location=self.args.device)
        for i, opt in enumerate(self.critic_optimizers):
            opt.load_state_dict(critic_opt_dict[f"critic_opt_{i}"])
        


    def process_actions(self, actions, l_action):

        n_steps = len(actions)
        actions_1hot = np.zeros([n_steps, l_action], dtype=int)
        actions_1hot[np.arange(n_steps), actions] = 1

        return actions_1hot
    
    def update_coeff(self, eval_reward, previous_eval_reward):
        for agent in self.agents:
            agent.update_reg_coeff(eval_reward, previous_eval_reward)



    # def _build_critic_inputs(self, batch):
    #     """构建critic网络的输入
    #     Args:
    #         batch: EpisodeBatch对象
    #     Returns:
    #         obs_batch: [bs, n_agents, obs_dim] 所有agent的观测
    #     """
    #     # 从batch中获取观测数据
    #     obs = batch["obs"]  # [bs, t, n_agents, obs_dim]
        
    #     # 处理维度,确保形状正确
    #     bs = obs.shape[0]
    #     obs_dim = obs.shape[-1] 
        
    #     # 重塑为所需形状
    #     obs_batch = obs.view(bs, self.n_agents, -1)
        
    #     return obs_batch