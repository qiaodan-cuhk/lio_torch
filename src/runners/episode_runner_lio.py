from envs import REGISTRY as env_REGISTRY
from functools import partial
from components.episode_buffer import EpisodeBatch
import numpy as np
import torch as th


class EpisodeRunner_LIO:

    def __init__(self, args, logger):
        self.args = args
        self.logger = logger
        self.batch_size = self.args.batch_size_run
        assert self.batch_size == 1

        self.env = env_REGISTRY[self.args.env](**self.args.env_args)
        self.episode_limit = self.env.episode_limit
        self.t = 0

        self.t_env = 0

        self.train_returns = []
        self.test_returns = []
        self.train_stats = {}
        self.test_stats = {}

        # Log the first run
        self.log_train_stats_t = -1000000

    def setup(self, scheme, groups, preprocess, mac):
        self.new_batch = partial(EpisodeBatch, scheme, groups, self.batch_size, self.episode_limit + 1,
                                 preprocess=preprocess, device=self.args.device)
        self.mac = mac

    def get_env_info(self):
        return self.env.get_env_info()

    def save_replay(self):
        self.env.save_replay()

    def close_env(self):
        self.env.close()

    def reset(self):
        self.batch = self.new_batch()
        self.env.reset()
        self.t = 0

    # 添加 prime 参数控制用哪个策略网络采样动作
    def run(self, test_mode=False, prime=False):
        self.reset()

        terminated = False
        episode_return = 0

        """没用RNN agent，不需要init hidden state"""
        # self.mac.init_hidden(batch_size=self.batch_size)

        if self.args.mac == "separate_mac":
            self.mac.init_latent(batch_size=self.batch_size)

        while not terminated:

            # transition 之前的信息
            pre_transition_data = {
                "state": [self.env.get_state()],   # [3, 48, 18]
                "avail_actions": [self.env.get_avail_actions()],  # [2,9]
                "obs": [self.env.get_obs()],   # [2 agents, 3, 9, 9]
                "agent_pos": [self.env.get_agent_pos()],   # [2, 2]
                "agent_orientation": [self.env.get_agent_orientation()],   #  [2, 2]
            }

            self.batch.update(pre_transition_data, ts=self.t)

            # Pass the entire batch of experiences up till now to the agents
            # Receive the actions for each agent at this timestep in a batch of size 1
            if 'lio' in self.args.name:
                if prime:
                    actions = self.mac.select_actions_env_prime(self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode)
                else:
                    actions = self.mac.select_actions_env(self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode)
            else:
                actions = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode)

            actions_env = actions % self.args.n_actions
            reward, terminated, env_info = self.env.step(actions_env[0])
            episode_return += reward

            post_transition_data = {
                "actions": actions,   # 1,2,1
                "reward": [(reward,)],
                "terminated": [(terminated != env_info.get("episode_limit", False),)],
                "clean_num": [(env_info["clean_num"],)],
                "apple_den": [(env_info["apple_den"],)],
            }
            self.batch.update(post_transition_data, ts=self.t)


            if 'lio' in self.args.name:
                recieved_rewards, give_rewards_list = self.mac.select_actions_inc(actions_env, self.batch, t_ep=self.t,
                                                        t_env=self.t_env, test_mode=test_mode,
                                                        agent_pos_replay = self.env.get_agent_pos())
                # recieved代表每个人接收到的总奖励，give代表每个agent具体给了谁多少奖励
                # 将 give_rewards_list 转换为 [bs, n_agents, n_agents-1] 形式
                give_rewards_tensor = th.stack(give_rewards_list, dim=1)  # 将列表转换为tensor, bs, n_agents send, n_agents reciever

                incentivize_data = {
                    "recieved_rewards": recieved_rewards,
                    "give_other_rewards_list": give_rewards_tensor,
                }
                self.batch.update(incentivize_data, ts=self.t)

            self.t += 1

        # print("reward type:", type(reward), "shape:", np.array(reward).shape if hasattr(reward, 'shape') else None)
        # print("clean_num type:", type(env_info["clean_num"]), "shape:", np.array(env_info["clean_num"]).shape if hasattr(env_info["clean_num"], 'shape') else None)
        # print("apple_den type:", type(env_info["apple_den"]), "shape:", np.array(env_info["apple_den"]).shape if hasattr(env_info["apple_den"], 'shape') else None)
        
        last_data = {
            "state": [self.env.get_state()],
            "avail_actions": [self.env.get_avail_actions()],
            "obs": [self.env.get_obs()],
            "agent_pos": [self.env.get_agent_pos()],
            "agent_orientation": [self.env.get_agent_orientation()],
        }
        self.batch.update(last_data, ts=self.t)

        # Select actions in the last stored state

        # if 'lio' in self.args.name:
        #     if prime:
        #         actions = self.mac.select_actions_env_prime(self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode)
        #     else:
        #         actions = self.mac.select_actions_env(self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode)
        #     # actions = self.mac.select_actions_env(self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode)
        #     recieved_rewards, give_rewards_list = self.mac.select_actions_inc(actions, self.batch, t_ep=self.t,
        #                                                 t_env=self.t_env, test_mode=test_mode,
        #                                                 agent_pos_replay = self.env.get_agent_pos())
        #     self.batch.update({ "recieved_rewards": recieved_rewards, "give_other_rewards_list": give_rewards_list,}, ts=self.t)
        # else:
        #     actions = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env,test_mode=test_mode)

        # self.batch.update({ "actions": actions,}, ts=self.t)
        


        cur_stats = self.test_stats if test_mode else self.train_stats
        cur_returns = self.test_returns if test_mode else self.train_returns
        log_prefix = "test_" if test_mode else ""
        cur_stats.update({k: cur_stats.get(k, 0) + env_info.get(k, 0) for k in set(cur_stats) | set(env_info)})
        cur_stats["n_episodes"] = 1 + cur_stats.get("n_episodes", 0)
        cur_stats["ep_length"] = self.t + cur_stats.get("ep_length", 0)

        if not test_mode:
            self.t_env += self.t

        cur_returns.append(episode_return)

        if test_mode and (len(self.test_returns) == self.args.test_nepisode):
            self._log(cur_returns, cur_stats, log_prefix)
        elif self.t_env - self.log_train_stats_t >= self.args.runner_log_interval:
            self._log(cur_returns, cur_stats, log_prefix)
            if hasattr(self.mac.action_selector, "epsilon"):
                self.logger.log_stat("epsilon", self.mac.action_selector.epsilon, self.t_env)
            self.log_train_stats_t = self.t_env

        return self.batch

    def _log(self, returns, stats, prefix):
        self.logger.log_stat(prefix + "return_mean", np.mean(returns), self.t_env)
        self.logger.log_stat(prefix + "return_std", np.std(returns), self.t_env)
        returns.clear()

        for k, v in stats.items():
            #if k != "n_episodes" and k!="clean_num":
            if k not in {"n_episodes", "clean_num", "apple_den","agent_pos","agent_orientation"}:
                self.logger.log_stat(prefix + k + "_mean" , v/stats["n_episodes"], self.t_env)
        stats.clear()
