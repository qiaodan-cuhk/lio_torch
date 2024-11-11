import copy
from components.episode_buffer import EpisodeBatch
import torch as th
from torch.optim import RMSprop, Adam

from pyclustering.cluster.xmeans import xmeans
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer


from ..modules.critics import REGISTRY as critic_resigtry

# logger 要增加一些测量incentivize的metric

class LIOLearner:
    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.n_agents = args.n_agents
        self.n_actions = args.n_actions
        self.logger = logger
        self.scheme = scheme


        # 这里有个问题是，optimizer到底是直接优化整个list LIO agents，还是每个agent都有一个optimizer？
        # 看上去 critic 和 actor 更新符合前者，而 inc 的优化更像后者，要考虑一致
        self.mac = mac  # 包括 actor/prime actor/inc NN
        self.actor_params = list(mac.actor.parameters())   # parameters_actor
        self.actor_optimizer = Adam(params=self.actor_params, lr=args.lr_actor)
        
        self.actor_prime_params = list(mac.actor_prime.parameters())
        self.actor_prime_optimizer = Adam(params=self.actor_prime_params, lr=args.lr_actor)
        
        self.inc_params = list(mac.inc.parameters())
        self.inc_optimizer = Adam(self.inc_params, lr=args.lr_inc)

        # a little wasteful to deepcopy (e.g. duplicates action selector), but should work for any MAC
        # self.target_mac = copy.deepcopy(mac)

        self.critic = critic_resigtry[args.critic_type](scheme, args)  # args.critic_type = rgb/not
        self.target_critic = copy.deepcopy(self.critic)

        self.critic_params = list(self.critic.parameters())
        self.critic_optimizer = Adam(self.critic_params, lr=args.lr_v)

        self.log_stats_t = -self.args.learner_log_interval - 1

        # 用于计算 reward 更新
        self.policy_new = PolicyNewCNN if self.image_obs else PolicyNewMLP

        self.list_agents = [Lio * n] = self.mac.agent ？
        self.list_policy_new = [0 for x in range(self.n_agents)]
        


    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):

        # 把main actor参数复制给prime保证一致
        self.policy_prime.load_state_dict(self.policy.state_dict()) 

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
        # inc 有一个单独的critic吗？似乎没有，完全由同一个critic驱动的？LIO？
        mac_out_prime = []
        inc_out = []

        self.mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):

            actor_logits_prime = self.mac.forward_actor_prime(batch, t=t)
            inc_logits = self.mac.forward_inc(batch, t=t)
            # q_env_t, q_inc_t, extra_return = self.mac.forward(batch, t=t)
            mac_out_prime.append(actor_logits_prime)  # [bs,n,a_env]
            inc_out.append(inc_logits)  # [bs,n,n,a_inc]

        mac_out_prime = th.stack(mac_out_prime, dim=1)  # [bs,t,n,a_env]
        inc_out = th.stack(inc_out, dim=1)  # [bs,t,n,n,a_inc]

        pi_prime = mac_out_prime

        # avail_inc_actions = th.ones_like(q_inc) # [bs,t,n,n,a_inc]

        """ Update value network """
        V_t, V_t_target, V_t_next, V_t_target_next, r2_val, critic_train_stats_log = self._train_critic(batch, rewards, terminated, actions, avail_actions,
                                                critic_mask, bs, max_t)

        actions = actions[:, :-1]
        q_vals = q_vals.detach()

        # Calculate policy grad with mask，都用prime policy的参数
        pi_prime[mask == 0] = 1.0
        pi_taken_prime = th.gather(pi_prime, dim=3, index=actions).squeeze(3)
        log_pi_prime_taken = th.log(pi_taken_prime + 1e-10)
        entropy_prime = -th.sum(pi_prime * th.log(pi + 1e-10), dim=-1)

        V_td_error = r2_val +  self.gamma*V_t_next - V_t

        # 处理动作
        # actions_1hot = util.process_actions(buf.action, self.l_action)

        """梯度从prime算"""
        actor_prime_loss = (
            -(
                (V_td_error * log_pi_prime_taken + self.args.entropy_coef * entropy_prime) * mask
            ).sum()
            / mask.sum()
        )

        """更新 prime actor, actor还是原来的参数"""
        # 更新 prime 网络的参数，存储梯度
        self.actor_prime_optimizer.zero_grad()
        actor_prime_loss.backward()
        self.policy_grads = [p.grad.detach() for p in self.actor_prime.parameters()]  # detach 以避免梯度累积


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
            ts_logged = len(critic_train_stats["critic_loss"])
            for key in [
                "critic_loss",
                "critic_grad_norm",
                "td_error_abs",
                "q_taken_mean",
                "target_mean",
            ]:
                self.logger.log_stat(
                    key, sum(critic_train_stats[key]) / ts_logged, t_env
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
            self.log_stats_t = t_env

       

    def train_reward(self, buffer, new_buffer, t_env):
        """训练激励函数"""




        for agent in self.list_agents:
            
            # 把 old traj和 new traj 的数据提出来
            buf_self = buffer[agent.agent_id]
            buf_self_new = new_buffer[agent.agent_id]

            agent.reg_coeff = self.agent.update_reg_coeff(self, performance, prev_reward_env).

            list_reward_loss = []  #用于表示当前agent对于所有其他agent奖励的loss
            agent.list_policy_new = [0 for x in range(self.n_agents)]

            if agent.can_give:
                
                for inc_to_agent in self.list_of_all_agents:
                    
                    # 其他agent的buffer提出来
                    other_agent_it = inc_to_agent.agent_id
                    other_buff = buffer[]
                    other_buff_new = new_buffer[]

                    # 用 old buffer 轨迹观测数据
                    # 这个其实是存在每个agent自己的agent.v_new里用于计算loss了，改一下逻辑，提到外面循环
                    other_v_next = inc_to_agent.mac.critic()
                    other_v = inc_to_agent.mac.critic()

                    """考虑是否要加 ijk 那个逻辑"""
                    inc_to_agent.action_others = util.get_action_others_1hot_batch(
                    buf_other.action_all, other_id, agent.l_action_for_r)

                    other_policy_new = list_policy_new[inc.id]
                    other_action_new = new_buffer
                    other_obs_new = new_buffer
                    # 要用这个计算 log probs prime，使用new buffer的obs和action
                    # 还要把自己的 new buffer obs和action输入进整个list_policy_new

                    if self.include_cost_in_chain_rule:
                        new_total_reward = new_buffer.reward + new_buffer.from_others - new_buffer.give_out_list
                    else:
                        new_total_reward = new_buffer.reward

                    #计算每个agent的时候，用自己的loss，new buffer V loss & reward
                    new_td_error = new_total_reward + new_self_v_next*gamma - new_self_v


                    """一些需要确认的变量"""


                    # 忽略对于自身的奖励
                    if agent.agent_id == inc_to_agent.agent_id and not agent.include_cost_in_chain_rule:
                        continue
                        
                    # 假设 agent.policy_params 是一个包含所有参数的列表
                    for param, grad in zip(inc_to_agent.policy_params, inc_to_agent.policy_grads):
                        param.data -= agent.lr_actor * grad  # 直接更新参数

                    # 创建新的策略实例
                    other_policy_new = inc_to_agent.policy_new(
                        inc_to_agent.dim_obs, inc_to_agent.l_action, inc_to_agent.agent_name).load_dict(inc_to_agent.policy_params)  # 这里不需要传递参数字典
                    self.list_policy_new[inc_to_agent.agent_id] = other_policy_new

                    # 这里有没有可能直接用self.actor prime代替？tf为了创建网络所以不得不这么搞


                    # 直接复制 prime policy 的参数
                    """这里逻辑找一下"""
                    other_policy_new = agent.policy_new(
                        inc_to_agent.policy_prime.state_dict(),  # 复制 prime policy 的参数
                        inc_to_agent.dim_obs,
                        inc_to_agent.l_action,
                        inc_to_agent.agent_name
                    )
                    agent.list_policy_new[inc_to_agent.agent_id] = other_policy_new


                    # 这里要注意输入的数据是什么，other actions是什么，new buffer的数据
                    log_probs_taken = th.log(
                        th.sum(other_policy_new.probs * other_policy_new.action_taken, dim=1) + 1e-15)  # 加上小常数以避免 log(0)
                    loss_term = -th.sum(log_probs_taken * new_td_error)  # new_td_error 是用新轨迹的obs计算的
                    list_reward_loss.append(loss_term)  # 第 agent 个对于所有 inc_to_agent 的loss list

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


                # 这里要考虑更新用agent还是self
                self.inc_optimizer.zero_grad()
                reward_loss.backward()
                self.inc_optimizer.step()


        # 更新目标网络
        self.update_target_network()

        # 用更新后的prime去覆盖 policy
        self.update_policy_from_prime()

    """gpt的写法"""
    def train_and_create_reward_op(self, list_buf, list_buf_new, epsilon, reg_coeff=1e-3):
        buf_self = list_buf[self.agent_id]
        buf_self_new = list_buf_new[self.agent_id]
        n_steps = len(buf_self.obs)
        ones = torch.ones(n_steps)  # 使用 PyTorch 的张量

        # 初始化奖励损失列表
        list_reward_loss = []
        self.list_policy_new = [0 for _ in range(self.n_agents)]
        
        # 处理其他代理的数据
        for agent in self.list_of_agents:
            other_id = agent.agent_id
            if other_id == self.agent_id:
                continue
            buf_other = list_buf[other_id]

            # 计算 V 值
            v_next = agent.v(buf_other.obs_next).detach()  # detach 防止梯度累积
            v = agent.v(buf_other.obs).detach()

            actions_other_1hot = util.process_actions(buf_other.action, self.l_action)

            # 计算其他代理的策略更新
            other_policy_params_new = {}
            for grad, var in zip(agent.policy_grads, agent.policy_params):
                other_policy_params_new[var] = var - agent.lr_actor * grad
            other_policy_new = agent.policy_new(
                other_policy_params_new, agent.dim_obs, agent.l_action,
                agent.agent_name)
            self.list_policy_new[agent.agent_id] = other_policy_new

            log_probs_taken = torch.log(
                torch.sum(other_policy_new.probs * other_policy_new.action_taken, dim=1))
            loss_term = -torch.sum(log_probs_taken * self.v_td_error)
            list_reward_loss.append(loss_term)

        # 处理自身代理的数据
        if self.include_cost_in_chain_rule:
            action_self_1hot = util.process_actions(buf_self.action, self.l_action)
            v_next = self.v(buf_self.obs_next).detach()
            v = self.v(buf_self.obs).detach()
            action_self_1hot_new = util.process_actions(buf_self_new.action, self.l_action)
            self_policy_new = self.list_policy_new[self.agent_id]

            # 计算总奖励
            total_reward = buf_self_new.reward + buf_self_new.r_from_others - buf_self_new.r_given
            self.v_td_error = total_reward + self.gamma * v_next - v

            feed = {
                self.action_taken: action_self_1hot,
                self.r_ext: buf_self.reward,
                self.epsilon: epsilon,
                self.v_next_ph: v_next,
                self.v_ph: v,
                self_policy_new.obs: buf_self_new.obs,
                self_policy_new.action_taken: action_self_1hot_new,
                self.obs: buf_self.obs,
                self.action_others: util.get_action_others_1hot_batch(
                    buf_self.action_all, self.agent_id, self.l_action_for_r),
                self.ones: ones
            }
        else:
            self.v_td_error = buf_self_new.reward  # 直接使用奖励
            feed = {
                self.obs: buf_self.obs,
                self.action_others: util.get_action_others_1hot_batch(
                    buf_self.action_all, self.agent_id, self.l_action_for_r),
                self.ones: ones
            }

        # 计算损失
        if self.include_cost_in_chain_rule:
            self.reward_loss = torch.sum(torch.stack(list_reward_loss))
        else:
            self.reward_loss = torch.sum(torch.stack(list_reward_loss))

        # 反向传播
        self.optimizer.zero_grad()  # 清空梯度
        self.reward_loss.backward()  # 反向传播
        self.optimizer.step()  # 更新参数




    def _train_critic(self, batch, rewards, terminated, actions, avail_actions, mask, bs, max_t):
        
        # Optimise critic
        # with th.no_grad():
        V_t = self.critic(batch, t)
        V_t[:, t] = V_t.view(bs, self.n_agents, self.n_actions)

        V_t_target = self.target_critic(batch, t)
        V_t_target[:, t] = V_t_target.view(bs, self.n_agents, self.n_actions).detach()

        V_t_next = self.critic(batch, t+1)  # 输入 next obs
        V_t_next[:, t+1] = V_t_next.view(bs, self.n_agents, self.n_actions)

        V_t_target_next = self.target_critic(batch, t+1) # 输入 next obs
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

        return V_t, V_t_target, V_t_next, V_t_target_next, r2_val, running_log
    



    def update_policy_from_prime(self):
        """用 prime policy 更新主策略"""
        for agent in self.list_agents:
            agent.policy = agent.policy_prime


        # 如果你想手动使用这些梯度进行更新
        # self.actor_prime_optimizer.zero_grad()
        # for p, grad in zip(self.actor_params, self.policy_grads):
        #     p.grad = grad  # 手动设置参数的梯度
        # self.actor_prime_optimizer.step()  # 使用优化器更新prime actor

        # # 计算超梯度
        # hypergradients = []
        # for p, grad in zip(agent.policy.parameters(), agent.policy_grads):
        #     # 计算超梯度（例如，损失函数对学习率的导数）
        #     hypergrad = compute_hypergradient(p, grad)  # 你需要定义这个函数
        #     hypergradients.append(hypergrad)

        # # 更新超参数（例如，学习率）
        # for i, param in enumerate(agent.hyperparameters):
        #     param.data -= learning_rate * hypergradients[i]  # 使用超梯度更新超参数



    """也考虑是不是 target critic """
    def _update_targets(self):
        self.target_mac.load_state(self.mac)
        self.logger.console_logger.info("Updated target network")

    def _update_targets_hard(self):
        self.target_critic.load_state_dict(self.critic.state_dict())

    def _update_targets_soft(self, tau):
        for target_param, param in zip(
            self.target_critic.parameters(), self.critic.parameters()
        ):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


    def cuda(self):
        self.mac.cuda()
        self.critic.cuda()
        self.target_critic.cuda()
        self.target_mac.cuda()

    def save_models(self, path):
        self.mac.save_models(path)
        th.save(self.critic.state_dict(), "{}/critic.th".format(path))

        th.save(self.actor_optimizer.state_dict(), "{}/opt_actor.th".format(path))
        th.save(self.critic_optimizer.state_dict(), "{}/opt_critic.th".format(path))
        th.save(self.inc_optimizer.state_dict(), "{}/opt_inc.th".format(path))

    def load_models(self, path):
        self.mac.load_models(path)
        self.critic.load_state_dict(
            th.load("{}/critic.th".format(path), map_location=lambda storage, loc: storage))
        # Not quite right but I don't want to save target networks
        self.target_mac.load_models(path)
        self.target_critic.load_state_dict(self.critic.state_dict())

        self.actor_optimizer.load_state_dict(
            th.load("{}/opt_actor.th".format(path), map_location=lambda storage, loc: storage))
        self.critic_optimizer.load_state_dict(
            th.load("{}/opt_critic.th".format(path), map_location=lambda storage, loc: storage))
        self.critic_optimizer.load_state_dict(
            th.load("{}/opt_critic.th".format(path), map_location=lambda storage, loc: storage))

