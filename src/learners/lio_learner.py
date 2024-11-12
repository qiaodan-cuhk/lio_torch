import copy
from components.episode_buffer import EpisodeBatch
import torch as th
from torch.optim import RMSprop, Adam

from ..modules.critics import REGISTRY as critic_resigtry

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
        self.critics = [critic_resigtry[args.critic_type](scheme, args) for i in range(self.n_agents)] # args.critic_type = rgb/not
        self.target_critics = copy.deepcopy(self.critics)
        self.critic_params = [list(critic.parameters()) for critic in self.critics]  # 每个 agent 的 critic 参数
        self.critic_optimizers = [Adam(params=params, lr=args.lr_v) for params in self.critic_params]  # 每个 agent 的 critic 优化器

        self.log_stats_t = -self.args.learner_log_interval - 1

        self.gamma = 

        


    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):

        # 把main actor参数复制给prime保证一致
        for i, agent in enumerate(self.agents):
            # 获取当前 agent 的 actor 和 actor_prime
            actor_params = agent.actor.state_dict()  # 获取 actor 的参数
            agent.actor_prime.load_state_dict(actor_params)  # 将参数加载到 actor_prime

        # Get the relevant quantities
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        
        mask = mask.repeat(1, 1, self.n_agents)
        critic_mask = mask.clone()

        # bs = ep_batch.batch_size

        # ************************************************ mac out *****************************************************
        mac_out_prime = []

        # self.mac.init_hidden(batch.batch_size)  # RNN agent 才用

        # 检查数据格式
        for t in range(batch.max_seq_length):
            # 确认返回的是logits还是softmax
            actor_logits_prime = self.mac.forward_actor_prime(batch, t=t)
            # inc_logits = self.mac.forward_inc(batch, t=t)
            # q_env_t, q_inc_t, extra_return = self.mac.forward(batch, t=t)
            mac_out_prime.append(actor_logits_prime)  # [t, bs,n,a_env]
            # inc_out.append(inc_logits)  # [bs,n,n,a_inc]

        mac_out_prime = th.stack(mac_out_prime, dim=1)  # [bs,t,n,a_env]
        pi_prime = mac_out_prime

        """ Update value network """
        V_t, V_t_next, r2_val, critic_train_stats_log = self._train_critic(batch, rewards, terminated, actions, avail_actions,
                                                critic_mask, bs, max_t)

        actions = actions[:, :-1]
        V_t_next = V_t_next.detach()  # 需要detach，作为一个参考值

        # Calculate policy grad with mask，都用prime policy的参数，这里计算entropy表示prime pi必然是一个softmax分布
        pi_prime[mask == 0] = 1.0
        pi_taken_prime = th.gather(pi_prime, dim=3, index=actions).squeeze(3)  # 所有agent的prime

        V_td_error = r2_val +  self.gamma*V_t_next - V_t

        # 处理动作
        # actions_1hot = util.process_actions(buf.action, self.l_action)

        """梯度从prime算"""
        actor_prime_losses = []  # 用于存储每个代理的损失

        for i, agent in enumerate(self.agents):
            # 计算当前代理的 mask
            agent_mask_i = mask[:, :, i]  # 假设 mask 的形状是 [bs, seq_len, n_agents]
            pi_taken_prime_i = th.gather(pi_prime[i], dim=3, index=actions).squeeze(3)  # 所有agent的prime
            log_pi_prime_taken_i = th.log(pi_taken_prime_i + 1e-10)
            entropy_prime_i = -th.sum(pi_prime[i] * th.log(pi_prime[i] + 1e-10), dim=-1)

            # 计算当前代理的损失
            # 需要将 V_td_error 的维度调整为 [batch_size]，以便与其他项相乘
            current_V_td_error = V_td_error[i]  # 选择当前代理的 TD 误差，维度为 [batch_size]

            actor_prime_loss_i = (
                -(
                    (current_V_td_error * log_pi_prime_taken_i + self.args.entropy_coef * entropy_prime_i) * agent_mask_i
                ).sum()
                / agent_mask_i.sum()
            )

            actor_prime_losses.append(actor_prime_loss_i)


        """更新 prime actor, actor还是原来的参数"""
        # 对每个代理的损失进行反向传播,存储梯度用不上了
        for i, agent in enumerate(self.agents):
            self.actor_prime_optimizers[i].zero_grad()  # 清除之前的梯度
            actor_prime_losses[i].backward()  # 反向传播
            self.actor_prime_optimizers[i].step()  # 更新参数

        """ 更新 critic target network """
        if (
            self.args.target_update_interval_or_tau > 1
            and (self.critic_training_steps - self.last_target_update_step)
            / self.args.target_update_interval_or_tau
            >= 1.0
        ):
            self._update_targets_hard()
            self.last_target_update_step = self.critic_training_steps
        elif self.args.target_update_interval_or_tau <= 1.0:
            self._update_targets_soft(self.args.target_update_interval_or_tau)


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

            self.logger.log_stat("actor_prime_loss", actor_prime_loss.ite(), t_env,)

            self.logger.log_stat(
                "advantage_mean",
                (advantages * mask).sum().item() / mask.sum().item(),
                t_env,
            )
            self.logger.log_stat("pg_loss", pg_loss.item(), t_env)
            self.logger.log_stat("agent_grad_norm", grad_norm.item(), t_env)
            self.logger.log_stat(
                "pi_max",
                (pi.max(dim=-1)[0] * mask).sum().item() / mask.sum().item(),
                t_env,
            )

            # self.logger.log_stat(critic_train_stats_log, t_env)

            self.log_stats_t = t_env

       

    def train_reward(self, buffer, new_buffer, t_env):
        """训练激励函数"""

        for id, agent in enumerate(self.agents):
            
            assert id == agent.agent_id

            # 把 old traj和 new traj 的数据提出来
            buf_self = buffer[id]
            buf_self_new = new_buffer[id]

            # 更新正则化系数
            agent.update_reg_coeff(self, performance, prev_reward_env).  # 更新 agent.reg_coeff

            list_reward_loss = []  # 用于表示当前agent对于所有其他agent奖励的loss，包括自己但是mask为0

            if agent.can_give:
                for inc_id, inc_to_agent in enumerate(self.agents):
                    if agent.agent_id == inc_to_agent.agent_id and not agent.include_cost_in_chain_rule:
                        # In this case, cost for giving is not accounted in chain rule, so the agent can skip over itself
                        continue
                    


                    # 用 old buffer 轨迹观测数据
                    # 这个其实是存在每个agent自己的agent.v_new里用于计算loss了，改一下逻辑，提到外面循环
                    # other_v_next = inc_to_agent.mac.critic()
                    # other_v = inc_to_agent.mac.critic()

                    """考虑是否要加 ijk 那个逻辑"""
                    inc_to_agent.action_others = util.get_action_others_1hot_batch(
                    buf_other.action_all, other_id, agent.l_action_for_r)

                    
                    if self.include_cost_in_chain_rule:
                        new_total_reward = new_buffer.reward + new_buffer.from_others - new_buffer.give_out_list
                    else:
                        new_total_reward = new_buffer.reward

                    #计算每个agent的时候，用自己的loss，new buffer V loss & reward
                    ? 检查命名buffer
                    self_new_obs_next = buf_self_new["next_obs"]
                    self_new_obs = buf_self_new["obs"]
                    new_self_v_next = self.critics[id].forward(self_new_obs_next)
                    new_self_v = self.critics[id].forward(self_new_obs)
                    new_td_error = new_total_reward + new_self_v_next*self.gamma - new_self_v


                    other_policy_new = self.agents[inc_id].actor_prime

                    agent.list_policy_new[inc_to_agent.agent_id] = other_policy_new


                    """在learner里，调用forward actor返回的是所有agent，所有动作的logits，考虑一下怎么 logits[i] 来只用某一个agent的，或者添加id"""

                    other_action_new = new_buffer[inc_id]["actions"]
                    other_obs_new = new_buffer[inc_id]["obs"]?

                    actor_logits_new = other_policy_new.forward(other_obs_new)
                    actor_probs_new = th.nn.functional.softmax(actor_logits_new)
                    # 把其他agent的选择动作 new buffer 处理成 1hot 用于乘法
                    taken_action_1hot_new = util.process_actions(other_action_new, self.l_action) ？
                    log_probs_taken = th.log(
                        th.sum(actor_probs_new * taken_action_1hot_new, dim=1) + 1e-15)  
                    # action_taken [bs, 1hot len actions]，两个相乘得到选择的动作的prob
                    loss_term = -th.sum(log_probs_taken * new_td_error)  
                    list_reward_loss.append(loss_term) 


                    if agent.include_cost_in_chain_rule:
                        agent.reward_loss = th.sum(th.stack(list_reward_loss)) 
                    else:  # directly minimize given rewards
                        reverse_1hot = 1 - th.nn.functional.one_hot(self.agent_id, num_classes=self.n_agents).float()

                        if self.separate_cost_optimizer or self.reg == 'l1':
                            # 创建一个全为1的张量，大小与批次相同
                            self.ones = th.ones(batch_size)  # 假设 batch_size 是当前批次的大小
                            self.gamma_prod = th.cumprod(self.ones * self.gamma, dim=0)  # 计算折扣因子的累积乘积
                            given_each_step = th.sum(th.abs(self.reward_function * reverse_1hot), dim=1)  # 计算每一步的奖励
                            total_given = th.sum(given_each_step * (self.gamma_prod / self.gamma))  # 计算总的给定奖励
                        elif self.reg == 'l2':
                            total_given = th.sum(th.square(self.reward_function * reverse_1hot))  # 计算平方和

                        if self.separate_cost_optimizer:
                            reward_loss = th.sum(th.stack(list_reward_loss))  # 直接求和
                        else:
                            reward_loss = th.sum(th.stack(list_reward_loss)) + self.reg_coeff * total_given  # 结合正则化项

                self.inc_optimizers[id].zero_grad()
                reward_loss.backward()
                self.inc_optimizers[id].step()

        # 用更新后的prime去覆盖 policy
        self.update_policy_from_prime()

    # 这里要改成 list 形式的
    def _train_critic(self, batch, rewards, terminated, actions, avail_actions, mask, bs, max_t):
        
        # forward get value, 不需要 grad 做推理
        # with th.no_grad():
        V_t = self.critics(batch, t)
        V_t[:, t] = V_t.view(bs, self.n_agents, self.n_actions)

        V_t_next = self.critics(batch, t+1)  # 输入 next obs
        V_t_next[:, t+1] = V_t_next.view(bs, self.n_agents, self.n_actions)

        V_t_target_next = self.target_critics(batch, t+1) # 输入 next obs
        V_t_target_next[:, t+1] = V_t_target_next.view(bs, self.n_agents, self.n_actions).detach()

        effect_ratio = self.args.incentive_ratio
        cost_ratio = self.args.incentive_cost

        # incentive values
        inc_rewards_list = batch["actions_inc_list"][:, :-1]  # [bs,t-1,n,n,1]
        recieved_rewards = batch["recieved_rewards"][:, :-1]  # [bs,t-1,n,n,1]

        r2_val = rewards + recieved_rewards
        if self.include_cost_in_chain_rule:
            inc_loss = cost_ratio * inc_rewards_list.squeeze(:)
            r2_val -= inc_loss
            
        v_td_error = r2_val + self.gamma * V_t_target_next - V_t


        running_log = {
            "critic_loss": [],
            "critic_grad_norm": [],
            "td_error_abs": [],
            "target_mean": [],
            "q_taken_mean": [],
        }

        # 平方loss，没有考虑mask
        loss = (v_td_error ** 2).sum()
        self.critic_optimizer.zero_grad()
        loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.critic_params, self.args.grad_norm_clip)
        self.critic_optimizer.step()
        self.critic_training_steps += 1

        running_log["critic_loss"].append(loss.item())
        running_log["critic_grad_norm"].append(grad_norm)
        # mask_elems = mask_t.sum().item()
        running_log["td_error_abs"].append(v_td_error.abs().sum().item())  # 没考虑 mas
        # running_log["q_taken_mean"].append((q_taken * mask_t).sum().item() / mask_elems)
        # running_log["target_mean"].append((targets_t * mask_t).sum().item() / mask_elems)

        return V_t, V_t_next, r2_val, running_log
    



    def update_policy_from_prime(self):
        """用 prime policy 更新主策略"""
        for agent in self.agents:
            agent.policy = agent.policy_prime

    def update_prime_from_policy(self):
        """用主策略更新 prime policy """
        for agent in self.agents:
            agent.policy = agent.policy_prime


 

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
        self.critics.cuda()
        self.target_critics.cuda()
        self.target_mac.cuda()


    def save_models(self, path):
        self.mac.save_models(path)
        th.save(self.critics.state_dict(), "{}/critic.th".format(path))

        th.save(self.actor_optimizer.state_dict(), "{}/opt_actor.th".format(path))
        th.save(self.critic_optimizer.state_dict(), "{}/opt_critic.th".format(path))
        th.save(self.inc_optimizer.state_dict(), "{}/opt_inc.th".format(path))


    def load_models(self, path):
        self.mac.load_models(path)
        self.critics.load_state_dict(
            th.load("{}/critic.th".format(path), map_location=lambda storage, loc: storage))
        # Not quite right but I don't want to save target networks
        self.target_mac.load_models(path)
        self.target_critics.load_state_dict(self.critics.state_dict())

        self.actor_optimizer.load_state_dict(
            th.load("{}/opt_actor.th".format(path), map_location=lambda storage, loc: storage))
        self.critic_optimizer.load_state_dict(
            th.load("{}/opt_critic.th".format(path), map_location=lambda storage, loc: storage))
        self.critic_optimizer.load_state_dict(
            th.load("{}/opt_critic.th".format(path), map_location=lambda storage, loc: storage))

