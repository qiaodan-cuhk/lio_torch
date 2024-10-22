import copy
from components.episode_buffer import EpisodeBatch
import torch as th
from torch.optim import RMSprop, Adam

from pyclustering.cluster.xmeans import xmeans
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer



class LIOLearner:
    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.mac = mac
        self.logger = logger

    # copy from LIO ac，需要调整
    def update(self, sess, buf, epsilon):
        sess.run(self.list_copy_main_to_prime_ops)

        batch_size = len(buf.obs)
        # Update value network
        feed = {self.obs: buf.obs_next}
        v_target_next, v_next = sess.run([self.v_target, self.v],
                                         feed_dict=feed)
        v_target_next = np.reshape(v_target_next, [batch_size])
        v_next = np.reshape(v_next, [batch_size])
        n_steps = len(buf.obs)
        if self.include_cost_in_chain_rule:
            total_reward = [buf.reward[idx] + buf.r_from_others[idx]
                            - buf.r_given[idx] for idx in range(n_steps)]
        else:
            total_reward = [buf.reward[idx] + buf.r_from_others[idx]
                            for idx in range(n_steps)]
        feed = {self.obs: buf.obs,
                self.v_target_next: v_target_next,
                self.total_reward: total_reward}
        _, v = sess.run([self.v_op, self.v], feed_dict=feed)
        v = np.reshape(v, [batch_size])

        # Update prime policy network
        actions_1hot = util.process_actions(buf.action, self.l_action)
        feed = {self.obs: buf.obs,
                self.action_taken: actions_1hot,
                self.r_ext: buf.reward,
                self.epsilon: epsilon}
        feed[self.r_from_others] = buf.r_from_others
        if self.include_cost_in_chain_rule:
            feed[self.r_given] = buf.r_given
        feed[self.v_next_ph] = v_next
        feed[self.v_ph] = v
        _ = sess.run(self.policy_op_prime, feed_dict=feed)

        # Update target network
        sess.run(self.list_update_v_ops)
    

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

