from .critics import Critic, CriticConv


REGISTRY = {}

REGISTRY["ac"] = Critic
REGISTRY["ac_conv"] = CriticConv
