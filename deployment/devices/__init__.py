"""I2RT bimanual robot and its observation cameras."""

from .base import BaseDevice
from .camera import OpenCVCameraDevice
from .robot import BaseRobot, I2RTRobot

DEVICE_CLASSES = {"OpenCVCameraDevice": OpenCVCameraDevice, "I2RTRobot": I2RTRobot}
__all__ = ["BaseDevice", "BaseRobot", "I2RTRobot", "OpenCVCameraDevice", "DEVICE_CLASSES"]
