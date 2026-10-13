#!/usr/bin/env python3
"""
Configuration Parser - Generic device configuration parsing utilities.

This module provides centralized configuration parsing logic that can be used
by both device_starter.py and device_manager.py to ensure consistency.

Author: Jun Lv
"""

import yaml
from typing import Dict, List, Any, Mapping, Optional, Union, cast

from omegaconf import DictConfig, OmegaConf
from utils.logger_config import logger


ConfigInput = Union[Dict[str, Any], DictConfig]


def ensure_config_dict(config: ConfigInput) -> Dict[str, Any]:
    """Convert Hydra DictConfig or plain mapping into a standard dictionary."""
    if isinstance(config, DictConfig):
        return cast(Dict[str, Any], OmegaConf.to_container(config, resolve=True))
    if isinstance(config, Mapping):
        return dict(config)


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to configuration YAML file

    Returns:
        Configuration dictionary

    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If YAML parsing fails
    """
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        logger.info(f"Configuration loaded from {config_path}")
        return config
    except FileNotFoundError:
        logger.error(f"Configuration file {config_path} not found")
        raise
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML configuration: {e}")
        raise


def parse_device_configs(config: ConfigInput) -> List[Dict[str, Any]]:
    """
    Parse device configurations and create device dictionaries.

    This is a generic parser that works with any device class defined in the config.
    The device class is specified directly in the config (e.g., 'BaseDevice', 'SimRobot').

    Args:
        config: Configuration dictionary loaded from YAML

    Returns:
        List of device dictionaries with standard format
    """
    config_dict = ensure_config_dict(config)

    devices: List[Dict[str, Any]] = []
    device_section = config_dict.get("devices", [])

    if isinstance(device_section, dict):
        device_list = device_section.get("devices", [])
    else:
        device_list = device_section

    if not isinstance(device_list, list):
        logger.error("Device configuration is not a list; received type %s", type(device_list))
        return devices

    logger.info(f"Found {len(device_list)} devices in configuration")

    # Parse all devices - each device has its class specified
    for i, device_config in enumerate(device_list):
        logger.info(f"Processing device {i}: {device_config}")

        # Create standard device info dictionary
        device_info = {
            "device_class": device_config["class"],
            "device_id": device_config["device_id"],
            "config": dict(device_config),
            # Additional fields can be added by the caller as needed
        }

        devices.append(device_info)
        logger.info(f"Added device: {device_config['class']}:{device_config['device_id']}")

    logger.info(f"Total devices parsed: {len(devices)}")
    return devices


def parse_device_configs_with_fields(
    config: ConfigInput, additional_fields: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
    """
    Parse device configurations with additional fields.

    Args:
        config: Configuration dictionary loaded from YAML
        additional_fields: Dictionary of additional fields to add to each device info

    Returns:
        List of device dictionaries with standard format plus additional fields
    """
    devices = parse_device_configs(config)

    if additional_fields:
        for device in devices:
            device.update(additional_fields)

    return devices
