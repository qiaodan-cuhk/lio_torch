REGISTRY = {}

from .episode_runner import EpisodeRunner
from .episode_runner_lio import EpisodeRunner_LIO

REGISTRY["episode"] = EpisodeRunner
REGISTRY["episode_lio"] = EpisodeRunner_LIO
