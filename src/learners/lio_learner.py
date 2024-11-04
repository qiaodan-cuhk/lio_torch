import copy
from components.episode_buffer import EpisodeBatch
import torch as th
from torch.optim import RMSprop, Adam

from pyclustering.cluster.xmeans import xmeans
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer


# logger 要增加一些测量incentivize的metric

class LIOLearner:
    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.mac = mac
        self.logger = logger

    
    def create_agents(self):
        """创建和初始化代理"""
        from lio_ac import LIO
        for agent_id in range(self.env.n_agents):
            agent = LIO(self.config.lio, self.env.dim_obs, self.env.l_action,
                        self.config.nn, 'agent_%d' % agent_id,
                        self.config.env.r_multiplier, self.env.n_agents,
                        agent_id, self.env.l_action_for_r)
            self.list_agents.append(agent)
            self.optimizers.append(optim.Adam(agent.parameters(), lr=self.config.lio.learning_rate))
            agent.receive_list_of_agents(self.list_agents)
            agent.create_policy_gradient_op()
            agent.create_update_op()
            if self.config.lio.use_actor_critic:
                agent.create_critic_train_op()
            agent.create_reward_train_op()

    def train(self, buffers, t_env):
        """ 训练过程，包括更新策略和奖励训练 """
        for idx, agent in enumerate(self.list_agents):
            self.optimizers[idx].zero_grad()  # 清空梯度
            agent.update(sess, list_buffers[idx], epsilon)  # 更新策略
            self.optimizers[idx].step()  # 更新参数

        for agent in self.list_agents:
            if agent.can_give:
                agent.train_reward(sess, list_buffers, list_buffers_new, epsilon, self.reg_coeff)

        for agent in self.list_agents:
            agent.update_main(sess)

        # train 里面不更新step，而是把梯度存进 self.policy_grads
        self.policy_grads = loss.backwards()
        # self.policy 不用step，prime policy去step
        self.prime_policy.optimizer.step()

        # 会用于更新 reward inc learning，detach 掉
        self.policy_grad.detach()

        
    def train_reward(self, buffer, new_buffer, t_env):
        """训练奖励函数"""

        self.reg_coeff = update(t_env)

        if agent.can_give:


        for agent in self.list_agents:
            if agent.agent_id == self.agent_id:
                continue
            buf_other = buffer[agent.agent_id]
            n_steps = len(buf_other.obs)

            # 使用PyTorch计算v_next和v
            v_next = agent.v(buf_other.obs_next).detach().numpy()
            v = agent.v(buf_other.obs).detach().numpy()

            actions_other_1hot = util.process_actions(buf_other.action, agent.l_action)
            feed = {
                'obs': buf_other.obs,
                'action_taken': actions_other_1hot,
                'r_ext': buf_other.reward,
                'epsilon': epsilon,
                'v_next': v_next,
                'v': v
            }

            buf_other_new = list_buf_new[agent.agent_id]
            actions_other_1hot_new = util.process_actions(buf_other_new.action, agent.l_action)
            feed['action_others'] = util.get_action_others_1hot_batch(buf_other.action_all, agent.agent_id, agent.l_action_for_r)

            # 处理self agent的情况
            if self.include_cost_in_chain_rule:
                action_self_1hot = util.process_actions(buf_self.action, self.l_action)
                feed['action_taken'] = action_self_1hot
                feed['r_ext'] = buf_self.reward
                v_next = self.v(buf_self.obs_next).detach().numpy()
                v = self.v(buf_self.obs).detach().numpy()
                feed['v_next'] = v_next
                feed['v'] = v

            # 计算奖励
            feed['obs'] = buf_self.obs
            feed['action_others'] = util.get_action_others_1hot_batch(buf_self.action_all, self.agent_id, self.l_action_for_r)
            feed['ones'] = np.ones(n_steps)

            # 计算总奖励
            total_reward = buf_self_new.reward + self.gamma * v_next - v
            feed['v_td_error'] = total_reward

            if not (self.include_cost_in_chain_rule or self.separate_cost_optimizer):
                feed['reg_coeff'] = reg_coeff

            # 使用PyTorch的优化器更新奖励
            self.reward_op(feed)

        # 更新目标网络
        self.update_target_network()

        # 用更新后的prime去覆盖 policy
        self.update_policy_from_prime()

    def update_policy_from_prime(self):
        """用 prime policy 更新主策略"""
        for agent in self.list_agents:
            agent.policy = agent.policy_prime

    def set_can_give(self, can_give):
        """设置代理是否可以给予奖励"""
        self.can_give = can_give


    
    """"""
    def _update_targets(self):
        self.target_mac.load_state(self.mac)
        self.logger.console_logger.info("Updated target network")

    def cuda(self):
        self.mac.cuda()
        self.target_mac.cuda()

    def save_models(self, path):
        self.mac.save_models(path)
        th.save(self.optimiser_env.state_dict(), "{}/opt_env.th".format(path))
        th.save(self.optimiser_inc.state_dict(), "{}/opt_inc.th".format(path))

    def load_models(self, path):
        self.mac.load_models(path)
        # Not quite right but I don't want to save target networks
        self.target_mac.load_models(path)
        self.optimiser_env.load_state_dict(
            th.load("{}/opt_env.th".format(path), map_location=lambda storage, loc: storage))
        self.optimiser_inc.load_state_dict(
            th.load("{}/opt_inc.th".format(path), map_location=lambda storage, loc: storage))

