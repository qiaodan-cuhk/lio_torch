from .homophily_learner import HomophilyLearner
from .lio_learner import LIOLearner

REGISTRY = {}

REGISTRY["homophily_learner"] =HomophilyLearner
REGISTRY["lio_learner"] =LIOLearner