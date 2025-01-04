from .homophily_learner import HomophilyLearner
from .lio_learner_old import LIOLearner_old
from .lio_learner import LIOLearner


REGISTRY = {}

REGISTRY["homophily_learner"] =HomophilyLearner
REGISTRY["lio_learner"] =LIOLearner
REGISTRY["lio_learner_old"] =LIOLearner_old

# LIOLearner 采用 lio 的逻辑，保存参数，创建新的网络，用来维持hypergradient
# LIOLearner_old 采用最初的逻辑，直接用step更新后的prime网络计算loss，可能会丢失hyper