"""VAM policy adapter, loaded only when inference is requested."""

from .base import BasePolicy
from .vam_policy import VAMPolicy

POLICY_CLASSES = {"VAMPolicy": VAMPolicy}
__all__ = ["BasePolicy", "VAMPolicy", "POLICY_CLASSES"]
