REGISTRY = {}

from .homophily_agent import HomophilyAgent
from .lio_agent import LIOAgent

REGISTRY["homophily"] = HomophilyAgent
REGISTRY["lio"] = LIOAgent