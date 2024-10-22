REGISTRY = {}

from .homophily_controller import HomophilyMAC
from .lio_controller import LIOMAC

REGISTRY["homophily_mac"] = HomophilyMAC
REGISTRY["lio_mac"] = LIOMAC