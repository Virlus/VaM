"""
VAM inference for offline DAgger warmup data collection and Online DAgger/Learn from Experience.
Writes joint action chunks to policy SHM.

This module:
1. Reads observations from summary SHM (via BaseConsumer)
2. Runs VAM to compute action chunking
3. Writes action chunking into policy SHM for ActionExecutor to consume

Policy SHM is connect-only, created by ActionExecutor.

Author: Han Xue, Wang Zheng
"""

import sys

sys.path.append("..")  # To allow imports from parent directory
from typing import Any, Dict

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from consumer.policy_connector import PolicyConnector
from utils.logger_config import logger

from typing import cast
from omegaconf import DictConfig, OmegaConf
import hydra

# import debugpy; debugpy.connect(("0.0.0.0", 5678))


def _flatten_config(cfg: DictConfig) -> Dict[str, Any]:
    return cast(Dict[str, Any], OmegaConf.to_container(cfg, resolve=True))


@hydra.main(config_path="../configs", config_name="rollout", version_base="1.3")
def main(cfg: DictConfig) -> None:
    merged_cfg = _flatten_config(cfg=cfg)

    # For Hydra mode, policy config is nested at top-level in the selected config

    policy_cfg = merged_cfg.get("policy", {})

    policy_class = policy_cfg.get("class", "VAMPolicy")
    policy_params = policy_cfg.get("params", {})
    fps = policy_cfg.get("fps", 50.0)
    control_fps = policy_cfg.get("control_fps", 50.0)
    enable_value_prediction = policy_cfg.get("enable_value_prediction", False)
    value_prediction_query_fps = policy_cfg.get("value_prediction_query_fps", fps)
    history_with_value = policy_cfg.get("history_with_value", False)
    chunk_length = policy_cfg.get("chunk_length", 10)

    obs_cfg = policy_cfg.get("obs", {})
    obs_devices = obs_cfg.get("devices")

    controls_cfg = policy_cfg.get("controls", [])
    reset_cfg = policy_cfg.get("reset", {})
    halt_and_home_cfg = policy_cfg.get("halt_and_home", {})

    master_device = policy_cfg.get("master_device", None)
    online_training_config = (
        policy_cfg.get("online_training")
        or merged_cfg.get("online_training")
        or merged_cfg.get("augmental_data_upload", {}).get("online_training")
    )

    # Get config file path from Hydra for rollout saving
    # Hydra stores the config-name in HydraConfig
    config_file_path = merged_cfg.get("_resolved_config_path")
    try:
        from hydra.core.hydra_config import HydraConfig

        if not config_file_path and HydraConfig.initialized():
            config_name = HydraConfig.get().job.config_name
            logger.info(f"Hydra config name: {config_name}")
            # config_name may or may not include .yaml extension
            # Remove .yaml if present to avoid double extension
            if config_name.endswith(".yaml"):
                config_filename = config_name
            else:
                config_filename = f"{config_name}.yaml"
            # Construct full path to config file
            config_file_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs", config_filename
            )
            logger.info(f"Config file path: {config_file_path}")
            if not os.path.exists(config_file_path):
                logger.warning(f"Config file not found at: {config_file_path}")
                config_file_path = None
            else:
                logger.info(f"Config file found: {config_file_path}")
        elif not config_file_path:
            logger.warning("HydraConfig not initialized, cannot get config file path")
    except Exception as e:
        logger.warning(f"Could not get config file path from Hydra: {e}")

    connector = PolicyConnector(
        summary_shm_name=merged_cfg.get("summary_shm", "device_summary_data"),
        policy_class=policy_class,
        policy_params=policy_params,
        obs_devices=obs_devices,
        controls=controls_cfg,
        reset_cfg=reset_cfg,
        halt_and_home_cfg=halt_and_home_cfg,
        fps=fps,
        control_fps=control_fps,
        enable_value_prediction=enable_value_prediction,
        value_prediction_query_fps=value_prediction_query_fps,
        history_with_value=history_with_value,
        chunk_length=chunk_length,
        master_device=master_device,
        infer_rollout_config=policy_cfg.get("infer_rollout"),
        config_file_path=config_file_path,
        runtime_config=merged_cfg.get("runtime", {}),
        online_training_config=online_training_config,
    )

    connector.run()


if __name__ == "__main__":
    main()
