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

