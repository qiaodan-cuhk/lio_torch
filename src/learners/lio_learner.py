import copy
from components.episode_buffer import EpisodeBatch
import torch as th
from torch.optim import RMSprop, Adam

from pyclustering.cluster.xmeans import xmeans
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer

# from ..modules.critics import CriticConv,Critic
from ..modules.critics import REGISTRY as critic_resigtry

# logger 要增加一些测量incentivize的metric

class LIOLearner:
    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.n_agents = args.n_agents
        self.n_actions = args.n_actions
        self.logger = logger
        self.scheme = scheme

        self.mac = mac  # 包括 actor/prime actor/inc NN
        self.actor_params = list(mac.actor.parameters())
        self.actor_optimiser = Adam(params=self.actor_params, lr=args.lr_actor)
        # prime actor 网络只用来推理和存储，不用opt更新
        
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
        


    def train(self, ep_batch, t_env):

        # Get the relevant quantities
        bs = batch.batch_size
        max_t = batch.max_seq_length
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"][:, :-1]

        critic_mask = mask.clone()

        mask = mask.repeat(1, 1, self.n_agents).view(-1)



        # bs = ep_batch.batch_size

        """ Update value network """
        q_vals, critic_train_stats = self._train_critic(batch, rewards, terminated, actions, avail_actions,
                                                critic_mask, bs, max_t)

        # state = ep_batch.obs  # 获取当前状态
        # next_state = ep_batch.obs_next  # 获取下一个状态
        # n_steps = ep_batch.n_steps  # 获取步数
        # rewards = ep_batch.   # 奖励

        # v_next = self.mac.forward_value(self, next_state, t_env, test_mode=False, learning_mode=False)
        # v = self.mac.forward_value(self, state, t_env, test_mode=False, learning_mode=False)
        # """这里要加一个target mac clone"""
        # v_target = self.clone_mac

        # # 计算优势
        # if self.include_cost_in_chain_rule:
        #     total_reward = [buf.reward[idx] + buf.r_from_others[idx] - buf.r_given[idx] for idx in range(n_steps)]
        # else:
        #     total_reward = [buf.reward[idx] + buf.r_from_others[idx] for idx in range(n_steps)]
        
        # # 计算 TD 误差
        # td_error = th.tensor(total_reward, dtype=th.float32) + self.gamma * v_target - v

        # # 更新价值网络
        # agent.critic_optimizer.zero_grad()
        # td_error.backward()  # 注意这里是 backward 而不是 backwards
        # agent.critic_optimizer.step()


        """更新 prime actor"""
        # 把main actor参数复制给prime保证一致
        self.policy_prime.load_state_dict(self.policy.state_dict()) 

         # 处理动作
        actions_1hot = util.process_actions(buf.action, self.l_action)

        # 准备输入数据
        obs = th.tensor(buf.obs, dtype=th.float32)
        action_taken = th.tensor(actions_1hot, dtype=th.float32)
        r_ext = th.tensor(buf.reward, dtype=th.float32)
        r_from_others = th.tensor(buf.r_from_others, dtype=th.float32)

        """这里要考虑inc rewards是否用于critic更新"""
        r2_val = r_ext + r_from_others
        if self.include_cost_in_chain_rule:
            r_given = th.tensor(buf.r_given, dtype=th.float32)
            r2_val -= r_given

        # 计算 TD 错误
        v_td_error = r2_val + self.gamma * v_next - v

        # 计算策略损失，用的都是prime policy
        log_probs_taken = th.log(th.sum(self.policy_prime(obs) * action_taken, dim=1) + 1e-15)
        entropy = -th.sum(self.policy_prime(obs) * th.log(self.policy_prime(obs) + 1e-15))
        policy_loss = -th.sum(log_probs_taken * v_td_error)
        loss = policy_loss - self.reg_coeff * entropy

        # 更新 prime 网络的参数，存储梯度
        agent.prime_optimizer.zero_grad()
        loss.backward()
        agent.prime_optimizer.step()


        agent.policy_grads = [p.grad.detach() for p in agent.policy_prime.parameters()]  # detach 以避免梯度累积

        # 如果你想手动使用这些梯度进行更新
        # agent.policy_optimizer.zero_grad()
        # for p, grad in zip(agent.policy.parameters(), agent.policy_grads):
        #     p.grad = grad  # 手动设置参数的梯度
        # agent.policy_optimizer.step()  # 使用优化器更新参数

        # 假设你已经计算了策略梯度并存储在 agent.policy_grads 中

        # # 计算超梯度
        # hypergradients = []
        # for p, grad in zip(agent.policy.parameters(), agent.policy_grads):
        #     # 计算超梯度（例如，损失函数对学习率的导数）
        #     hypergrad = compute_hypergradient(p, grad)  # 你需要定义这个函数
        #     hypergradients.append(hypergrad)

        # # 更新超参数（例如，学习率）
        # for i, param in enumerate(agent.hyperparameters):
        #     param.data -= learning_rate * hypergradients[i]  # 使用超梯度更新超参数


    def train_reward(self, buffer, new_buffer, t_env):
        """训练激励函数"""

        for agent in self.list_agents:

            agent.reg_coeff = self.agent.update_reg_coeff(self, performance, prev_reward_env).

            list_reward_loss = []  #用于表示当前agent对于所有其他agent奖励的loss
            agent.list_policy_new = [0 for x in range(self.n_agents)]

            if agent.can_give:
                
                for inc_to_agent in self.list_of_all_agents:

                    # 忽略对于自身的奖励
                    if agent.agent_id == inc_to_agent.agent_id and not agent.include_cost_in_chain_rule:
                        continue


                    # 直接复制 prime policy 的参数
                    """这里逻辑找一下"""
                    other_policy_new = agent.policy_new(
                        inc_to_agent.policy_prime.state_dict(),  # 复制 prime policy 的参数
                        inc_to_agent.dim_obs,
                        inc_to_agent.l_action,
                        inc_to_agent.agent_name
                    )
                    agent.list_policy_new[inc_to_agent.agent_id] = other_policy_new

                    log_probs_taken = th.log(
                        th.sum(other_policy_new.probs * other_policy_new.action_taken, dim=1) + 1e-15)  # 加上小常数以避免 log(0)
                    loss_term = -th.sum(log_probs_taken * self.v_td_error)  # v_td_error 是用前面的critic update计算的
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
                            self.reward_loss = th.sum(th.stack(list_reward_loss))  # 直接求和
                        else:
                            agent.reward_loss = th.sum(th.stack(list_reward_loss)) + self.reg_coeff * total_given  # 结合正则化项

            agent.inc_optimizer.zero_grad()
            agent.reward_loss.backward()
            agent.inc_optimizer.step()


        # 更新目标网络
        self.update_target_network()

        # 用更新后的prime去覆盖 policy
        self.update_policy_from_prime()



    """"""

    def _train_critic(self, batch, rewards, terminated, actions, avail_actions, mask, bs, max_t):
        # Optimise critic
        target_q_vals = self.target_critic(batch)[:, :]
        targets_taken = th.gather(target_q_vals, dim=3, index=actions).squeeze(3)

        # Calculate td-lambda targets
        targets = build_td_lambda_targets(rewards, terminated, mask, targets_taken, self.n_agents, self.args.gamma, self.args.td_lambda)

        q_vals = th.zeros_like(target_q_vals)[:, :-1]

        running_log = {
            "critic_loss": [],
            "critic_grad_norm": [],
            "td_error_abs": [],
            "target_mean": [],
            "q_taken_mean": [],
        }

        for t in reversed(range(rewards.size(1))):
            mask_t = mask[:, t].expand(-1, self.n_agents)
            if mask_t.sum() == 0:
                continue

            q_t = self.critic(batch, t)
            q_vals[:, t] = q_t.view(bs, self.n_agents, self.n_actions)
            q_taken = th.gather(q_t, dim=3, index=actions[:, t:t+1]).squeeze(3).squeeze(1)
            targets_t = targets[:, t]

            td_error = (q_taken - targets_t.detach())

            # 0-out the targets that came from padded data
            masked_td_error = td_error * mask_t

            # Normal L2 loss, take mean over actual data
            loss = (masked_td_error ** 2).sum() / mask_t.sum()
            self.critic_optimiser.zero_grad()
            loss.backward()
            grad_norm = th.nn.utils.clip_grad_norm_(self.critic_params, self.args.grad_norm_clip)
            self.critic_optimiser.step()
            self.critic_training_steps += 1

            running_log["critic_loss"].append(loss.item())
            running_log["critic_grad_norm"].append(grad_norm)
            mask_elems = mask_t.sum().item()
            running_log["td_error_abs"].append((masked_td_error.abs().sum().item() / mask_elems))
            running_log["q_taken_mean"].append((q_taken * mask_t).sum().item() / mask_elems)
            running_log["target_mean"].append((targets_t * mask_t).sum().item() / mask_elems)

        return q_vals, running_log



    def update_policy_from_prime(self):
        """用 prime policy 更新主策略"""
        for agent in self.list_agents:
            agent.policy = agent.policy_prime



    """也考虑是不是 target critic """
    def _update_targets(self):
        self.target_mac.load_state(self.mac)
        self.logger.console_logger.info("Updated target network")

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

