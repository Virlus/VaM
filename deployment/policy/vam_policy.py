"""
VAM Policy Integration for Dual-Arm Robot Control
The policy must be trained in the robot coordinate system.
Author: Wang Zheng
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from policy.base import BasePolicy
import numpy as np
import copy
from typing import Dict, List, Any, Optional, Tuple

try:
    import torch
except ImportError:
    print("torch not found")
    pass

from utils.logger_config import logger
from policy.vam_reload_utils import clear_accelerator_caches, resolve_checkpoint_norm_stats
import time
from utils.latency_tracker import LatencyTracker


class VAMPolicy(BasePolicy):
    def __init__(
        self,
        pi_config_name: str,
        ckpt_dir: str,
        task: str = "",
        device: str = "cuda",
        obs_horizon: int = 1,
        chunk_length: int = 100,
        downsample: int = 1,
        obs_time_interval: float = 0.04,
        execution_fps: Optional[float] = None,
        enable_rtc: bool = False,
        cfg_guidance_weight: float = 1.0,
        enable_human_rtc: bool = False,
        human_rtc_action_shm: Optional[str] = None,
        human_rtc_control_flag_shm: Optional[str] = None,
        human_rtc_prefix_attention_horizon: int = 20,
        human_rtc_velocity_decay_steps: int = 20,
        human_rtc_max_velocity: float = 2.0,
    ) -> None:
        super().__init__()
        self.pi_config_name = pi_config_name
        self.ckpt_dir = ckpt_dir
        self.task = task
        self.device = device
        self.obs_horizon = obs_horizon
        self.chunk_length = chunk_length
        self.downsample = downsample
        self.obs_time_interval = obs_time_interval
        self.execution_fps = execution_fps
        self.enable_rtc = enable_rtc
        self.cfg_guidance_weight = cfg_guidance_weight
        if enable_human_rtc and not enable_rtc:
            raise ValueError("enable_human_rtc requires enable_rtc")
        self.enable_human_rtc = enable_human_rtc
        self.human_rtc_action_shm = human_rtc_action_shm
        self.human_rtc_control_flag_shm = human_rtc_control_flag_shm
        self.human_rtc_prefix_attention_horizon = max(1, int(human_rtc_prefix_attention_horizon))
        self.human_rtc_velocity_decay_steps = max(1, int(human_rtc_velocity_decay_steps))
        self.human_rtc_max_velocity = float(human_rtc_max_velocity)
        self._rtc_prefix_attention_horizon_override: Optional[int] = None

        self._latency_tracker: LatencyTracker = LatencyTracker(maxlen=100)
        self._policy = self.load_vam_policy(self.pi_config_name, self.ckpt_dir, self.device)

    def reset(self) -> None:
        """
        Reset the policy's internal state for a new episode.

        This clears the cached previous actions and state used by RTC,
        ensuring the policy doesn't use stale data from the previous episode.
        """
        if hasattr(self._policy, "_prev_actions"):
            self._policy._prev_actions = None
            logger.info("VAMPolicy: Cleared _prev_actions for RTC reset")
        if hasattr(self._policy, "_prev_state_abs"):
            self._policy._prev_state_abs = None
            logger.info("VAMPolicy: Cleared _prev_state_abs for RTC reset")
        self._rtc_prefix_attention_horizon_override = None

    def seed_human_rtc_prefix(self, action_prefix: np.ndarray) -> None:
        """Seed the next RTC inference with a future-facing human command prefix."""
        prefix = np.asarray(action_prefix, dtype=np.float64)
        if prefix.ndim != 2 or prefix.shape[0] != self.chunk_length:
            raise ValueError(f"Human RTC prefix must have shape ({self.chunk_length}, action_dim), got {prefix.shape}")
        self._policy._prev_actions = prefix.copy()
        self._rtc_prefix_attention_horizon_override = self.human_rtc_prefix_attention_horizon

    def load_vam_policy(self, pi_config_name: str, ckpt_dir: str, device="cuda"):
        try:
            from vam.training import config as _config
            from vam.policies import policy_config as _policy_config
            from vam.shared import download as _download, normalize as _normalize
        except ImportError as exc:
            raise RuntimeError(
                f"VAM import failed with Python {sys.executable}: {exc}. "
                "Launch policy inference from the repository root with "
                "../.venv/bin/python consumer/vam_policy_infer.py"
            ) from exc

        self.pi_config = _config.get_config(pi_config_name)
        checkpoint_dir = _download.maybe_download(str(ckpt_dir))
        data_config = self.pi_config.data.create_base_config(self.pi_config.assets_dirs, self.pi_config.model)
        norm_path = resolve_checkpoint_norm_stats(checkpoint_dir, data_config.asset_id)
        norm_stats = _normalize.load(norm_path.parent)
        print(f"Loading policy from {ckpt_dir}")
        policy = _policy_config.create_trained_policy(
            self.pi_config,
            checkpoint_dir,
            sample_kwargs={"cfg_guidance_weight": self.cfg_guidance_weight},
            norm_stats=norm_stats,
        )
        return policy

    def reload_checkpoint(
        self,
        ckpt_dir: str,
        *,
        policy_id: Optional[int] = None,
        checkpoint_id: Optional[str] = None,
    ) -> None:
        old_policy = self._policy
        self._policy = None
        del old_policy
        clear_accelerator_caches()

        self._policy = self.load_vam_policy(self.pi_config_name, ckpt_dir, self.device)
        self.ckpt_dir = ckpt_dir
        self.policy_id = policy_id
        self.checkpoint_id = checkpoint_id
        self.policy_loaded_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        self.reset()
        logger.info(f"VAMPolicy: reloaded checkpoint={checkpoint_id}, policy_id={policy_id}, ckpt_dir={ckpt_dir}")

    def _convert_obs_to_pi(
        self, obs_batch_robot: Dict[str, np.ndarray], for_action_policy: bool = True
    ) -> Dict[str, np.ndarray]:
        def squeeze_single_batch(value: np.ndarray) -> np.ndarray:
            return value.squeeze(0) if value.shape[0] == 1 else value

        model_config = getattr(self.pi_config, "model", None)
        action_observation_horizon = int(getattr(model_config, "action_observation_horizon", None))
        value_observation_horizon = int(getattr(model_config, "value_observation_horizon", None))
        use_action_observation_history = for_action_policy and (action_observation_horizon > 1)
        use_value_observation_history = not for_action_policy and (value_observation_horizon > 1)

        # Process robot observation
        state = obs_batch_robot["robot_joint_pos"].astype(np.float32)
        ego_image = obs_batch_robot["camera0_rgb"]
        left_wrist_image = obs_batch_robot["camera1_rgb"]
        right_wrist_image = obs_batch_robot["camera2_rgb"]

        repacked_data = {  # Keep the observation horizon dimension if > 1
            "observation/base_image": squeeze_single_batch(ego_image),
            "observation/left_wrist_image": squeeze_single_batch(left_wrist_image),
            "observation/right_wrist_image": squeeze_single_batch(right_wrist_image),
            "observation/state": squeeze_single_batch(state),
            "prompt": self.task,
            "advantage": np.array([1.0], dtype=np.float32),  # For Pi0.6 inference
        }
        if for_action_policy:
            value_prediction = obs_batch_robot["value_prediction"]
            repacked_data["value_prediction"] = squeeze_single_batch(value_prediction)

        # Process 'pad' indicators. shape: (horizon,)
        if use_action_observation_history or use_value_observation_history:
            # These source keys are internal names produced by sensor_data_to_umi_data().
            # The target keys are the VAM data-transform contract used by AsyncSequenceAlohaInputs.
            state_is_pad = obs_batch_robot["robot_joint_pos_is_pad"]
            ego_image_is_pad = obs_batch_robot["camera0_rgb_is_pad"]
            left_wrist_image_is_pad = obs_batch_robot["camera1_rgb_is_pad"]
            right_wrist_image_is_pad = obs_batch_robot["camera2_rgb_is_pad"]
            mask_key_map = {
                "observation/base_image_is_pad": squeeze_single_batch(ego_image_is_pad),
                "observation/left_wrist_image_is_pad": squeeze_single_batch(left_wrist_image_is_pad),
                "observation/right_wrist_image_is_pad": squeeze_single_batch(right_wrist_image_is_pad),
                "observation/state_is_pad": squeeze_single_batch(state_is_pad),
            }
            if for_action_policy:
                value_prediction_is_pad = obs_batch_robot["value_prediction_is_pad"]
                mask_key_map["value_prediction_is_pad"] = squeeze_single_batch(value_prediction_is_pad)

            # Make is_pads: Here all False means the corresponding observation is not 'paded', and won't be masked in the 'input mask' of the VAM model
            for target_key, value in mask_key_map.items():
                if value is None:
                    logger.warning(
                        f"we use observation history, but {target_key} is not found in the observation dictionary"
                    )
                else:
                    repacked_data[target_key] = squeeze_single_batch(value)
        repacked_data = {k: torch.from_numpy(v) if isinstance(v, np.ndarray) else v for k, v in repacked_data.items()}
        return repacked_data

    def predict(
        self,
        observation,
        action_configs: List[Dict[str, Any]],
        chunk_length: int = 1,
        action_index: Optional[int] = None,
    ) -> Dict[str, np.ndarray]:
        # 1. Organize observation data for the policy
        obs_batch_robot = self.sensor_data_to_umi_data(observation, for_action_policy=True)
        obs_batch_for_policy = self._convert_obs_to_pi(obs_batch_robot, for_action_policy=True)
        prefix_attention_horizon_override = self._rtc_prefix_attention_horizon_override
        self._rtc_prefix_attention_horizon_override = None

        # Prepare rtc_config if RTC is enabled and action_index is provided
        rtc_config: Optional[Dict[str, float | int]] = None
        if self.enable_rtc and action_index is not None:
            # Compute inference_delay dynamically from tracked latencies
            default_delay_steps = 10
            if self.execution_fps is not None and self.execution_fps > 0:
                inference_delay = self._latency_tracker.compute_inference_delay_steps(
                    execution_fps=self.execution_fps, use_percentile=0.95, default_delay_steps=default_delay_steps
                )
            else:
                inference_delay = default_delay_steps
            # logger.info(f"Inference delay: {inference_delay}")
            # Compute prefix_attention_horizon based on inference_delay and action_index
            prefix_attention_horizon = (
                prefix_attention_horizon_override
                if prefix_attention_horizon_override is not None
                else chunk_length - inference_delay - action_index
            )
            prefix_attention_horizon = max(
                1,
                min(int(prefix_attention_horizon), chunk_length - inference_delay - action_index),
            )

            rtc_config = {
                "inference_delay": inference_delay,
                "prefix_attention_horizon": prefix_attention_horizon,
                "max_guidance_weight": 10.0,
            }

        with torch.no_grad():
            inference_delay = int(rtc_config.get("inference_delay", 0)) if rtc_config is not None else 0
            if (
                self.enable_rtc
                and action_index is not None
                and rtc_config is not None
                and (action_index + inference_delay < chunk_length)
            ):
                model_outputs = self._policy.infer(
                    obs_batch_for_policy, time_base=action_index, enable_rtc=True, rtc_config=rtc_config
                )
                robot_actions = model_outputs["actions"]
            else:
                if (
                    self.enable_rtc
                    and action_index is not None
                    and rtc_config is not None
                    and (action_index + inference_delay >= chunk_length)
                ):
                    # For RTC, we need: action_index + inference_delay < chunk_length.
                    # A practical lower bound for policy fps is:
                    #   fps > execution_fps / (chunk_length - M - inference_delay)
                    # where M is a safety margin in steps.
                    rtc_chunk_length: int = chunk_length
                    recommended_fps_range_str: str = ""
                    if self.execution_fps is not None and self.execution_fps > 0:
                        exec_fps: float = float(self.execution_fps)
                        # Use inference_delay as the minimum safety margin:
                        # For the very first inference, no action chunk is being executed yet,
                        # so the effective "execution consumed during inference" can be smaller.
                        # Using m_min=inference_delay provides a conservative lower-bound.
                        m_min: int = max(int(inference_delay), 0)
                        m_max: int = max(chunk_length - inference_delay - 1, m_min)

                        def _min_required_policy_fps(margin_steps: int) -> Optional[float]:
                            denom: int = rtc_chunk_length - int(margin_steps) - int(inference_delay)
                            if denom <= 0:
                                return None
                            return exec_fps / float(denom)

                        fps_lb_m2 = _min_required_policy_fps(m_min)
                        fps_lb_mmax = _min_required_policy_fps(m_max)
                        if fps_lb_m2 is not None and fps_lb_mmax is not None:
                            recommended_fps_range_str = (
                                f", recommended policy.fps lower-bound for M={m_min}..{m_max}: "
                                f"{fps_lb_m2:.3f}..{fps_lb_mmax:.3f} Hz "
                                f"(execution_fps={exec_fps:.1f} Hz, chunk_length={rtc_chunk_length}, inference_delay={inference_delay})"
                            )
                        else:
                            recommended_fps_range_str = (
                                f", cannot compute recommended fps (execution_fps={exec_fps:.1f} Hz, "
                                f"chunk_length={rtc_chunk_length}, inference_delay={inference_delay}, M in [{m_min},{m_max}])"
                            )
                    logger.warning(
                        f"FPS seems too low for RTC, fallback to normal infer(): "
                        f"action_index={action_index}, inference_delay={inference_delay}, "
                        f"action_index+inference_delay={action_index + inference_delay} (>=chunk_length)"
                        f"{recommended_fps_range_str}."
                    )
                model_outputs = self._policy.infer(obs_batch_for_policy)
                robot_actions = model_outputs["actions"]  # absolute joint positions

        return {action_configs[0]["device_name"]: robot_actions.astype(np.float64)}

    def predict_value(self, observation: Dict[str, np.ndarray], predicted_action: np.ndarray) -> Dict[str, np.ndarray]:
        obs_batch_robot = self.sensor_data_to_umi_data(observation, for_action_policy=False)
        obs_batch_for_policy = self._convert_obs_to_pi(obs_batch_robot, for_action_policy=False)
        processed_action = torch.from_numpy(predicted_action).unsqueeze(0)

        with torch.no_grad():
            outputs = self._policy.infer_value(obs_batch_for_policy, processed_action)
            value = outputs["value"]
        return {"value": value.astype(np.float64)}

    def sensor_data_to_umi_data(
        self, obs_dict: Dict[str, Any], for_action_policy: bool = True
    ) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
        # Validate input format based on configured obs_horizon
        # # 'use_history' is for DP obs, the other two is for VAM obs preparation, so we expect that they cannot be true at the same time
        model_config = getattr(self.pi_config, "model", None)
        action_observation_horizon = int(getattr(model_config, "action_observation_horizon", 1))
        value_observation_horizon = int(getattr(model_config, "value_observation_horizon", 1))
        use_history = self.obs_horizon > 1 and action_observation_horizon == 1 and value_observation_horizon == 1

        use_action_observation_history = for_action_policy
        use_value_observation_history = not for_action_policy

        # Input shape check first.
        if use_history:
            # History mode: each value is a list of obs_horizon arrays
            for key, value in obs_dict.items():
                assert isinstance(value, list), f"Expected list for key '{key}', got {type(value)}"
                assert len(value) == self.obs_horizon, f"Expected {self.obs_horizon} items, got {len(value)}"
        elif use_action_observation_history:
            for key, value in obs_dict.items():
                assert isinstance(value, list), f"Expected list for key '{key}', got {type(value)}"
                assert len(value) >= action_observation_horizon, (
                    f"Expected {action_observation_horizon} items, got {len(value)}"
                )
                obs_dict[key] = value[-action_observation_horizon:]
        elif use_value_observation_history:
            for key, value in obs_dict.items():
                assert isinstance(value, list), f"Expected list for key '{key}', got {type(value)}"
                assert len(value) >= value_observation_horizon, (
                    f"Expected {value_observation_horizon} items, got {len(value)}"
                )
                obs_dict[key] = value[-value_observation_horizon:]
        else:
            raise ValueError(
                f"Invalid observation history mode: use_history={use_history}, use_action_observation_history={use_action_observation_history}, use_value_observation_history={use_value_observation_history}"
            )

        batch_data: Dict[str, np.ndarray] = {}

        # Process camera observations
        cameras = [name for name in obs_dict.keys() if "camera" in name.lower() and not name.endswith("_is_pad")]
        for idx, camera_name in enumerate(cameras):
            if use_history:
                # Stack all observations in history
                raw_images = obs_dict[camera_name]
                stacked_images = np.stack(raw_images)
                batch_data[f"camera{idx}_rgb"] = (stacked_images / 255.0).transpose(
                    0, 3, 1, 2
                )  # (obs_horizon, C, H, W)
            elif use_action_observation_history or use_value_observation_history:
                # Stack all observations in history
                raw_images = obs_dict[camera_name]
                stacked_images = np.stack(raw_images)
                batch_data[f"camera{idx}_rgb"] = (stacked_images / 255.0).transpose(
                    0, 3, 1, 2
                )  # (obs_horizon, C, H, W)
                pad_key = f"{camera_name}_is_pad"
                if pad_key in obs_dict:
                    batch_data[f"camera{idx}_rgb_is_pad"] = np.asarray(obs_dict[pad_key], dtype=bool)
                else:
                    logger.warning(f"camera{idx}_rgb_is_pad not found in obs_dict. ")
            else:
                raw_image = obs_dict[camera_name]
                stacked_images = np.stack(raw_image)[None, :]
                batch_data[f"camera{idx}_rgb"] = (stacked_images / 255.0).transpose(0, 3, 1, 2)  # (1, C, H, W)

        # Process robot observation
        robot_device = next(
            iter([key for key in obs_dict.keys() if "robot" in key.lower() and not key.endswith("_is_pad")]), None
        )
        if robot_device is None:
            raise KeyError("Robot observation data not found in observation dictionary")

        if use_history:
            robot_obs_list = obs_dict[robot_device]
            robot_obs_array = np.stack(robot_obs_list)  # (obs_horizon, obs_dim)
        elif use_action_observation_history or use_value_observation_history:
            robot_obs_list = obs_dict[robot_device]
            robot_obs_array = np.stack(robot_obs_list)
        else:
            robot_obs = obs_dict[robot_device]
            assert isinstance(robot_obs, np.ndarray)
            robot_obs_array = robot_obs if robot_obs.ndim > 1 else robot_obs[None, :]

        # Validate robot observation
        obs_dim = robot_obs_array.shape[-1]
        expected_obs_dim = 14
        if obs_dim != expected_obs_dim:
            raise ValueError(
                f"Robot observation dimension {obs_dim} does not match expected "
                f"{expected_obs_dim} for {self.num_robots} robot(s)"
            )

        batch_data["robot_joint_pos"] = robot_obs_array
        if use_action_observation_history or use_value_observation_history:
            robot_pad_key = f"{robot_device}_is_pad"
            if robot_pad_key in obs_dict:
                batch_data["robot_joint_pos_is_pad"] = np.asarray(obs_dict[robot_pad_key], dtype=bool)
            else:
                logger.warning("robot_joint_pos_is_pad not found in obs_dict. ")

        # Process value prediction
        value_keys = next(
            iter([key for key in obs_dict.keys() if "value_prediction" in key.lower() and not key.endswith("_is_pad")]),
            None,
        )
        if value_keys is not None:
            value_obs_list = obs_dict[value_keys]
            value_obs_array = np.stack(value_obs_list)
            batch_data[value_keys] = value_obs_array
            value_pad_key = f"{value_keys}_is_pad"
            if value_pad_key in obs_dict:
                batch_data[value_pad_key] = np.asarray(obs_dict[value_pad_key], dtype=bool)
            else:
                logger.warning(f"{value_pad_key} not found in obs_dict. ")

        return batch_data

    def warm_up(
        self,
        action_configs: List[Dict[str, Any]],
        num_observations: int = 10,
        sample_observation: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Warm up the model by running inference on observations.
        If a sample observation is provided (from PolicyConnector), use it to
        align warm-up input dimensions with the real inference path.
        First 5 observations use standard inference (no RTC), next 5 use RTC.

        The first call of each inference mode (no RTC and with RTC) is excluded
        from the latency tracker to avoid counting JIT compilation time.

        Args:
            action_configs: Action configuration list for predict method
            num_observations: Total number of warm-up observations (default: 10)
            sample_observation: Optional observation batch to mirror real data
        """
        logger.info(f"Starting warm-up with {num_observations} observations")

        if sample_observation is not None:
            observation = sample_observation
            logger.info("Warm-up uses provided sample observation from PolicyConnector")
        else:
            observation = {
                "OpenCVCameraDevice_0": [
                    np.random.randint(0, 255, (360, 640, 3), dtype=np.uint8) for _ in range(self.obs_horizon)
                ],
                "OpenCVCameraDevice_1": [
                    np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8) for _ in range(self.obs_horizon)
                ],
                "OpenCVCameraDevice_2": [
                    np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8) for _ in range(self.obs_horizon)
                ],
                "I2RTRobot_1": [np.random.rand(14).astype(np.float64) for _ in range(self.obs_horizon)],
            }

        # First 5 observations: without RTC (action_index=None)
        # The first call (i=0) is excluded from latency tracking to skip JIT overhead
        logger.info("Warming up with standard inference (no RTC)...")
        for i in range(min(5, num_observations)):
            start_time = time.time()
            try:
                _ = self.predict(observation, action_configs, chunk_length=self.chunk_length, action_index=None)
                inference_time = time.time() - start_time
                # Skip the first call (i=0) for latency tracking to exclude JIT compilation time
                if i > 0:
                    self._latency_tracker.add(inference_time)
                logger.info(
                    f"Warm-up observation {i + 1}/5 (no RTC): {inference_time * 1000:.2f} ms"
                    + (" [excluded from latency tracker]" if i == 0 else "")
                )
            except Exception as e:
                logger.error(f"Warm-up observation {i + 1}/5 failed: {e}")

        # Next 5 observations: with RTC (action_index=0) if RTC is enabled
        # The first RTC call (i=5) is excluded from latency tracking to skip JIT overhead
        if self.enable_rtc:
            logger.info("Warming up with RTC...")
            first_rtc_call = True
            for i in range(5, min(10, num_observations)):
                start_time = time.time()
                try:
                    # Use a valid action_index (0 is safe as it's the first index)
                    _ = self.predict(observation, action_configs, chunk_length=self.chunk_length, action_index=0)
                    inference_time = time.time() - start_time
                    # Skip the first RTC call for latency tracking to exclude JIT compilation time
                    if not first_rtc_call:
                        self._latency_tracker.add(inference_time)
                    logger.info(
                        f"Warm-up observation {i + 1}/5 (RTC): {inference_time * 1000:.2f} ms"
                        + (" [excluded from latency tracker]" if first_rtc_call else "")
                    )
                    first_rtc_call = False
                except Exception as e:
                    logger.error(f"Warm-up observation {i + 1}/10 failed: {e}")

        logger.info("Warming up value prediction...")
        action_dim = action_configs[0].get("action_dim")
        if action_dim is None:
            logger.warning("Value warm-up skipped: action_configs[0]['action_dim'] is missing")
        else:
            predicted_action = np.zeros((self.chunk_length, action_dim), dtype=np.float64)
            for i in range(min(5, num_observations)):
                start_time = time.time()
                try:
                    _ = self.predict_value(copy.deepcopy(observation), predicted_action)
                    inference_time = time.time() - start_time
                    logger.info(f"Warm-up value prediction {i + 1}/5: {inference_time * 1000:.2f} ms")
                except Exception as e:
                    logger.error(f"Warm-up value prediction {i + 1}/5 failed: {e}")

        # Log latency tracker statistics after warm-up
        if len(self._latency_tracker) > 0:
            logger.info(
                f"Warm-up latency statistics: count={len(self._latency_tracker)}, "
                f"mean={self._latency_tracker.mean() * 1000:.2f} ms, "
                f"p95={self._latency_tracker.p95() * 1000:.2f} ms, "
                f"max={self._latency_tracker.max() * 1000:.2f} ms"
            )
            if self.execution_fps is not None and self.execution_fps > 0:
                computed_delay = self._latency_tracker.compute_inference_delay_steps(
                    execution_fps=self.execution_fps, use_percentile=0.95, default_delay_steps=6
                )
                logger.info(
                    f"Computed inference_delay based on p95 latency: {computed_delay} steps "
                    f"(execution_fps={self.execution_fps:.1f} Hz)"
                )

        logger.info("Warm-up completed")
