#!/usr/bin/env python3
"""
Manager Package - Contains different types of device managers.

This package provides various manager implementations for handling different
types of device data aggregation and processing.

Author: Jun Lv
"""

from .base_device_manager import BaseDeviceManager
from .synchronized_device_manager import SynchronizedDeviceManager
from .base_policy_runner import BasePolicyRunner

__all__ = ["BaseDeviceManager", "SynchronizedDeviceManager", "BasePolicyRunner"]

# Registry for dynamic loading
MANAGER_CLASSES = {
    "base": BaseDeviceManager,
    "synchronized": SynchronizedDeviceManager,
    "base_policy_runner": BasePolicyRunner,
}
