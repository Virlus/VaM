"""Joint action chunk dispatch for the I2RT execution loop."""

from .action_manager import BaseChunkManager
from .joint_action_manager import JointActionManager


def load_chunk_manager(config, **kwargs):
    config = dict(config)
    kind = config.pop("type", "joint_action_manager")
    if kind != "joint_action_manager":
        raise ValueError(f"Unsupported I2RT chunk manager: {kind}")
    return JointActionManager(**{**kwargs, **config})


__all__ = ["BaseChunkManager", "JointActionManager", "load_chunk_manager"]
