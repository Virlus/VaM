from omegaconf import DictConfig

"""
Device Starter - Starts device processes and the Device Manager.

This module provides centralized management for running multiple devices simultaneously
using separate processes for isolation and reliability, plus the new Device Manager
for data aggregation.

Author: Jun Lv
"""
import os
import sys
import time
import signal
import multiprocessing as mp
from typing import Dict, List, Any, Optional
from pathlib import Path

sys.path.append(os.path.dirname(__file__))
from devices import DEVICE_CLASSES
from manager import MANAGER_CLASSES
from utils.config_parser import ensure_config_dict, parse_device_configs_with_fields, load_config
from utils.logger_config import logger


class DeviceStarter:
    """
    Device starter that handles multiple device processes and the Device Manager.

    Features:
    - Load configuration from YAML file
    - Start/stop multiple devices in separate processes
    - Start/stop the Device Manager for data aggregation
    - Monitor process health and restart on failure
    - Graceful shutdown handling
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        *,
        config_path: Optional[str] = None,
        enable_device_manager: bool = True,
    ) -> None:
        """
        Initialize the device starter.

        Args:
            config_path: Path to configuration YAML file
            enable_device_manager: Whether to start the Device Manager for data aggregation
        """
        if config is None:
            resolved_path = config_path or "config.yaml"
            logger.info(f"Loading configuration from {resolved_path}")
            self.config_path = resolved_path
            self.config = load_config(resolved_path)
        else:
            self.config_path = config_path
            self.config = ensure_config_dict(config)
        self.devices: List[Dict[str, Any]] = []
        self.running = False
        self.enable_device_manager = enable_device_manager
        self.device_manager_process = None
        self.policy_runner_process = None
        runtime_cfg = self.config.get("runtime", {})
        self.camera_overlay_enabled_default = runtime_cfg.get("camera_overlay_enabled", False)
        logger.info("DeviceStarter initialized")
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _parse_device_configs(self) -> List[Dict[str, Any]]:
        """Parse device configurations and create device dictionaries."""
        additional_fields = {"process": None, "restart_count": 0, "last_restart": None, "ready": False}
        return parse_device_configs_with_fields(self.config, additional_fields)

    def _start_generic_device_process(self, device_info: Dict[str, Any]) -> Optional[mp.Process]:
        """Start a device process using dynamic import based on device class."""
        try:
            device_class = DEVICE_CLASSES.get(device_info["device_class"])
            if not device_class:
                logger.error(f"Unknown device class: {device_info['device_class']}")
                return None

            def run_device() -> None:
                try:
                    from devices import DEVICE_CLASSES

                    device_class = DEVICE_CLASSES.get(device_info["device_class"])
                    if not device_class:
                        raise ValueError(f"Unknown device class: {device_info['device_class']}")
                    device_config = device_info["config"].copy()
                    device_config.pop("class", None)
                    import inspect

                    sig = inspect.signature(device_class.__init__)
                    valid_params = list(sig.parameters.keys())
                    filtered_config = {k: v for k, v in device_config.items() if k in valid_params}
                    device = device_class(**filtered_config)
                    logger.info(f"Starting {device_info['device_class']} {device_info['config']['device_id']}")
                    device.start_server()
                except Exception as e:
                    logger.error(f"{device_info['device_class']} {device_info['config']['device_id']} error: {e}")
                    raise

            process = mp.Process(
                target=run_device, name=f"{device_info['device_class']}-{device_info['config']['device_id']}"
            )
            return process
        except Exception as e:
            logger.error(f"Failed to create device process for {device_info['device_class']}: {e}")
            return None

    def _start_device_manager_process(self) -> Optional[mp.Process]:
        """Start the Device Manager process for data aggregation."""
        try:

            def run_device_manager() -> None:
                try:
                    device_manager_config = self.config.get("device_manager", {})
                    manager_type = device_manager_config.get("type", "base")
                    manager_class = MANAGER_CLASSES.get(manager_type)
                    if not manager_class:
                        raise ValueError(f"Unknown device manager type: {manager_type}")
                    manager_params = device_manager_config.get("params", {}) or {}
                    manager_params = ensure_config_dict(manager_params)
                    if hasattr(self, "config_path") and self.config_path and (not manager_params.get("config_path")):
                        manager_params.setdefault("config_path", self.config_path)
                    else:
                        manager_params.setdefault("config", self.config)
                    manager = manager_class(**manager_params)
                    logger.info(f"Device Manager ({manager_type}) started for data aggregation")
                    manager.start()
                except Exception as e:
                    logger.error(f"Device Manager error: {e}")
                    raise

            process = mp.Process(target=run_device_manager, name="DeviceManager")
            return process
        except Exception as e:
            logger.error(f"Failed to create Device Manager process: {e}")
            return None

    def _start_policy_runner_process(self) -> Optional[mp.Process]:
        """Start the Policy Runner process for action dispatching."""
        try:

            def run_policy_runner() -> None:
                try:
                    policy_runner_config = self.config.get("action_executor", {})
                    if not policy_runner_config:
                        logger.warning("No action_executor configuration found, skipping PolicyRunner")
                        return
                    runner_type = policy_runner_config.get("type", "base_policy_runner")
                    manager_class = MANAGER_CLASSES.get(runner_type)
                    if not manager_class:
                        raise ValueError(f"Unknown policy runner type: {runner_type}")
                    valid_params = {
                        "config_path",
                        "robot_control_shms",
                        "execution_fps",
                        "chunk_length",
                        "chunk_manager",
                        "config",
                        "device_delays",
                        "device_dimension_delays",
                        "signal_files",
                    }
                    runner_config = {}
                    for key, value in policy_runner_config.items():
                        if key in valid_params:
                            runner_config[key] = value
                    if hasattr(self, "config_path") and self.config_path and (not runner_config.get("config_path")):
                        runner_config.setdefault("config_path", self.config_path)
                    else:
                        runner_config.setdefault("config", self.config)
                    logger.debug(f"Policy runner config keys: {list(runner_config.keys())}")
                    runner = manager_class(**runner_config)
                    logger.info(f"Policy Runner ({runner_type}) started for action dispatching")
                    runner.start()
                except Exception as e:
                    logger.error(f"Policy Runner error: {e}")
                    raise

            process = mp.Process(target=run_policy_runner, name="PolicyRunner")
            return process
        except Exception as e:
            logger.error(f"Failed to create Policy Runner process: {e}")
            return None

    def _start_device_process(self, device_info: Dict[str, Any]) -> bool:
        """Start a single device process."""
        try:
            process = self._start_generic_device_process(device_info)
            if not process:
                return False
            process.start()
            device_info["process"] = process
            device_info["last_restart"] = time.time()
            device_info["ready"] = False
            logger.info(f"Started {device_info['device_class']} device {device_info['device_id']} (PID: {process.pid})")
            return True
        except Exception as e:
            logger.error(f"Failed to start {device_info['device_class']} device {device_info['device_id']}: {e}")
            return False

    @staticmethod
    def _device_shm_name(device_info: Dict[str, Any]) -> str:
        return f"{device_info['device_class']}_{device_info['device_id']}_data"

    def _device_is_ready(self, device_info: Dict[str, Any]) -> bool:
        """Check that the current child is alive and created its state SHM."""
        process = device_info.get("process")
        if process is None or not process.is_alive():
            return False
        shm_name = self._device_shm_name(device_info)
        if sys.platform.startswith("linux"):
            shm_path = Path("/dev/shm") / shm_name
            if not shm_path.exists():
                return False
            last_restart = device_info.get("last_restart") or 0.0
            return shm_path.stat().st_ctime >= last_restart - 0.5
        try:
            from utils.shm_utils import connect_to_shared_memory

            channel = connect_to_shared_memory(shm_name, read_only=True)
            channel.close()
            return True
        except (FileNotFoundError, OSError):
            return False

    def _wait_for_devices_ready(
        self, devices: Optional[List[Dict[str, Any]]] = None, timeout_s: Optional[float] = None
    ) -> bool:
        """Wait until every device process has published its state SHM."""
        target_devices = self.devices if devices is None else devices
        global_settings = self.config.get("global_settings", {})
        if timeout_s is None:
            configured_timeout = global_settings.get("device_startup_timeout_s")
            if configured_timeout is None:
                configured_timeout = max(15.0, float(global_settings.get("device_init_delay", 5.0)))
            timeout_s = float(configured_timeout)
        pending = list(target_devices)
        deadline = time.monotonic() + timeout_s
        while pending and time.monotonic() < deadline:
            for device_info in list(pending):
                process = device_info.get("process")
                if process is None or not process.is_alive():
                    exit_code = None if process is None else process.exitcode
                    logger.error(
                        f"Device exited before becoming ready: {device_info['device_class']}:{device_info['device_id']} exit_code={exit_code}"
                    )
                    return False
                if self._device_is_ready(device_info):
                    device_info["ready"] = True
                    pending.remove(device_info)
                    logger.info(
                        f"Device ready: {device_info['device_class']}:{device_info['device_id']} SHM={self._device_shm_name(device_info)}"
                    )
            if pending:
                time.sleep(0.1)
        if pending:
            pending_names = [f"{device['device_class']}:{device['device_id']}" for device in pending]
            logger.error(f"Timed out after {timeout_s:.1f}s waiting for devices: {pending_names}")
            return False
        return True

    def _terminate_started_device_processes(self) -> None:
        """Stop children after a startup failure without starting other services."""
        for device_info in self.devices:
            process = device_info.get("process")
            if process is not None and process.is_alive():
                process.terminate()
        for device_info in self.devices:
            process = device_info.get("process")
            if process is None:
                continue
            process.join(timeout=5.0)
            if process.is_alive():
                process.kill()
                process.join(timeout=2.0)
            device_info["ready"] = False

    def _monitor_devices(self) -> None:
        """Monitor device processes and restart failed ones if configured."""
        global_settings = self.config.get("global_settings", {})
        restart_on_failure = global_settings.get("restart_on_failure", True)
        max_restart_attempts = global_settings.get("max_restart_attempts", 3)
        restart_delay = global_settings.get("restart_delay", 2.0)
        logger.info("Starting device monitoring...")
        try:
            while self.running:
                for device_info in self.devices:
                    if device_info["process"] and (not device_info["process"].is_alive()):
                        logger.warning(f"Device {device_info['device_class']}:{device_info['device_id']} process died")
                        if restart_on_failure and device_info["restart_count"] < max_restart_attempts:
                            logger.info(
                                f"Restarting device {device_info['device_class']}:{device_info['device_id']} (attempt {device_info['restart_count'] + 1}/{max_restart_attempts})"
                            )
                            time.sleep(restart_delay)
                            device_info["restart_count"] += 1
                            started = self._start_device_process(device_info)
                            if started and (not self._wait_for_devices_ready([device_info])):
                                logger.error(
                                    f"Restarted device did not become ready: {device_info['device_class']}:{device_info['device_id']}"
                                )
                        else:
                            logger.error(
                                f"Device {device_info['device_class']}:{device_info['device_id']} exceeded max restart attempts or restart disabled"
                            )
                time.sleep(1.0)
        except KeyboardInterrupt:
            logger.info("Device monitoring interrupted")
            self.running = False

    def start_all(self) -> None:
        """Start all enabled devices and the Device Manager."""
        if self.running:
            logger.warning("Device starter is already running")
            return
        logger.info("Starting Device Starter")
        self.devices = self._parse_device_configs()
        if not self.devices:
            logger.warning("No enabled devices found in configuration")
            return
        global_settings = self.config.get("global_settings", {})
        i2rt_startup_stagger_s = float(global_settings.get("i2rt_startup_stagger_s", 1.0))
        last_i2rt_launch = None
        self.running = True
        launched_count = 0
        for device_info in self.devices:
            if device_info["device_class"] == "I2RTRobot" and last_i2rt_launch is not None:
                remaining = i2rt_startup_stagger_s - (time.monotonic() - last_i2rt_launch)
                if remaining > 0:
                    logger.info(f"Staggering I2RT startup by {remaining:.2f}s")
                    time.sleep(remaining)
            if self._start_device_process(device_info):
                launched_count += 1
                if device_info["device_class"] == "I2RTRobot":
                    last_i2rt_launch = time.monotonic()
        if launched_count != len(self.devices):
            logger.error(f"Only launched {launched_count}/{len(self.devices)} device processes")
            self._terminate_started_device_processes()
            self.running = False
            return False
        logger.info("Waiting for device processes to publish their state SHMs...")
        if not self._wait_for_devices_ready():
            self._terminate_started_device_processes()
            self.running = False
            return False
        ready_count = sum((bool(device.get("ready")) for device in self.devices))
        if self.enable_device_manager:
            logger.info("Starting Device Manager for data aggregation...")
            self.device_manager_process = self._start_device_manager_process()
            if self.device_manager_process:
                self.device_manager_process.start()
                logger.info(f"Started Device Manager (PID: {self.device_manager_process.pid})")
            else:
                logger.error("Failed to start Device Manager")
                self._terminate_started_device_processes()
                self.running = False
                return False
        action_executor_config = self.config.get("action_executor", {})
        if action_executor_config and action_executor_config.get("type") not in [None, "null", "disabled"]:
            logger.info("Starting Policy Runner for action dispatching...")
            self.policy_runner_process = self._start_policy_runner_process()
            if self.policy_runner_process:
                self.policy_runner_process.start()
                logger.info(f"Started Policy Runner (PID: {self.policy_runner_process.pid})")
            else:
                logger.error("Failed to start Policy Runner")
        self._write_robot_home_index()
        logger.info(f"Successfully initialized {ready_count}/{len(self.devices)} devices")
        return True

    def stop_all(self) -> None:
        """Stop all running devices and the Device Manager.

        On Ctrl+C, child processes already receive SIGINT and start their own
        graceful shutdown (go-rest, etc.).  We first wait for them to finish
        before resorting to SIGTERM/SIGKILL.
        """
        if not self.running:
            return
        logger.info("Stopping all devices and Device Manager...")
        self.running = False
        self._create_reset_signal()
        graceful_timeout = 10.0
        logger.info(f"Waiting up to {graceful_timeout}s for device processes to shut down gracefully...")
        for device_info in self.devices:
            if device_info.get("process"):
                try:
                    device_info["process"].join(timeout=graceful_timeout)
                except (AssertionError, ValueError):
                    pass
        for device_info in self.devices:
            if device_info.get("process"):
                try:
                    if device_info["process"].is_alive():
                        logger.info(
                            f"Sending SIGTERM to {device_info['device_class']} device {device_info['device_id']}"
                        )
                        device_info["process"].terminate()
                except (AssertionError, ValueError):
                    pass
        for device_info in self.devices:
            if device_info.get("process"):
                try:
                    device_info["process"].join(timeout=5.0)
                    if device_info["process"].is_alive():
                        logger.warning(f"Force killing {device_info['device_class']} device {device_info['device_id']}")
                        device_info["process"].kill()
                except (AssertionError, ValueError):
                    pass
        if self.device_manager_process:
            try:
                if self.device_manager_process.is_alive():
                    logger.info("Stopping Device Manager")
                    self.device_manager_process.terminate()
            except (AssertionError, ValueError):
                pass
            try:
                self.device_manager_process.join(timeout=5.0)
                if self.device_manager_process.is_alive():
                    logger.warning("Force killing Device Manager")
                    self.device_manager_process.kill()
            except (AssertionError, ValueError):
                pass
        if self.policy_runner_process:
            try:
                if self.policy_runner_process.is_alive():
                    logger.info("Stopping Policy Runner")
                    self.policy_runner_process.terminate()
            except (AssertionError, ValueError):
                pass
            try:
                self.policy_runner_process.join(timeout=5.0)
                if self.policy_runner_process.is_alive():
                    logger.warning("Force killing Policy Runner")
                    self.policy_runner_process.kill()
            except (AssertionError, ValueError):
                pass
        logger.info("All devices and Device Manager stopped")

    def _create_reset_signal(self) -> None:
        """Create reset signal file to notify PolicyConnector to reconnect SHMs."""
        try:
            signal_files = self.config.get("runtime", {}).get("signal_files", {})
            reset_signal_path = signal_files.get("reset_signal")
            assert reset_signal_path is not None, "reset_signal_path is required in signal_files"
            reset_signal_file = Path(reset_signal_path)
            reset_signal_file.parent.mkdir(parents=True, exist_ok=True)
            reset_signal_file.touch()
            logger.info(f"DeviceStarter: Created reset signal file at {reset_signal_path}")
        except Exception as e:
            logger.warning(f"DeviceStarter: Failed to create reset signal file: {e}")

    def _write_robot_home_index(self) -> None:
        """Write robot home index to shared file for PolicyConnector to read."""
        try:
            robot_cfg = self.config.get("robot", {})
            home_index = robot_cfg.get("home", 0) if isinstance(robot_cfg, dict) else 0
            signal_files = self.config.get("runtime", {}).get("signal_files", {})
            home_index_path = signal_files.get("robot_home_index")
            assert home_index_path is not None, "home_index_path is required in signal_files"
            home_index_file = Path(home_index_path)
            home_index_file.parent.mkdir(parents=True, exist_ok=True)
            home_index_file.write_text(str(home_index))
            logger.info(f"DeviceStarter: Wrote robot home index {home_index} to {home_index_path}")
        except Exception as e:
            logger.warning(f"DeviceStarter: Failed to write robot home index: {e}")

    def _signal_handler(self, signum: int, frame: Any) -> None:
        """Handle shutdown signals."""
        logger.info(f"Received signal {signum}, shutting down...")
        self.stop_all()
        sys.exit(0)

    def get_status(self) -> Dict[str, Any]:
        """Get status of all managed devices and Device Manager."""
        all_devices = self._parse_device_configs()
        status = {
            "running": self.running,
            "total_devices": len(all_devices),
            "enable_device_manager": self.enable_device_manager,
            "device_manager_running": False,
            "device_manager_pid": None,
            "devices": [],
        }
        if self.device_manager_process:
            try:
                status["device_manager_running"] = self.device_manager_process.is_alive()
                status["device_manager_pid"] = self.device_manager_process.pid
            except (AssertionError, ValueError):
                status["device_manager_running"] = False
                status["device_manager_pid"] = None
        for device_info in all_devices:
            device_status = {
                "type": device_info["device_class"],
                "id": device_info["device_id"],
                "configured": True,
                "running": False,
                "pid": None,
                "restart_count": 0,
            }
            if self.running and self.devices:
                for running_device in self.devices:
                    if (
                        running_device["device_class"] == device_info["device_class"]
                        and running_device["device_id"] == device_info["device_id"]
                    ):
                        device_status.update(
                            {
                                "running": running_device["process"].is_alive() if running_device["process"] else False,
                                "pid": running_device["process"].pid if running_device["process"] else None,
                                "restart_count": running_device["restart_count"],
                            }
                        )
                        break
            status["devices"].append(device_status)
        return status


def _flatten_config(cfg: "DictConfig") -> Dict[str, Any]:
    """Convert DictConfig to plain dict for DeviceStarter."""
    from omegaconf import OmegaConf

    return OmegaConf.to_container(cfg, resolve=True)


def main() -> None:
    """Main function to run the device starter using Hydra."""
    import hydra
    from omegaconf import DictConfig
    import signal
    import sys

    @hydra.main(config_path="configs", config_name="config", version_base="1.3")
    def hydra_main(cfg: DictConfig) -> None:
        merged_cfg = _flatten_config(cfg)
        runtime_cfg = merged_cfg.get("runtime", {})
        if "devices" in merged_cfg and isinstance(merged_cfg["devices"], dict):
            merged_cfg["devices"] = merged_cfg["devices"].get("devices", [])
        if "device_manager" in merged_cfg and isinstance(merged_cfg["device_manager"], dict):
            if "type" not in merged_cfg["device_manager"] and "device_manager" in merged_cfg["device_manager"]:
                merged_cfg["device_manager"] = merged_cfg["device_manager"]["device_manager"]
        if "action_executor" in merged_cfg and isinstance(merged_cfg["action_executor"], dict):
            if "type" not in merged_cfg["action_executor"] and "action_executor" in merged_cfg["action_executor"]:
                merged_cfg["action_executor"] = merged_cfg["action_executor"]["action_executor"]
        starter = DeviceStarter(
            config=merged_cfg, config_path=None, enable_device_manager=runtime_cfg.get("enable_device_manager", True)
        )

        def _shutdown(signum: int, frame: Any) -> None:
            logger.info(f"Received signal {signum}, stopping DeviceStarter")
            starter.stop_all()
            sys.exit(0)

        signal.signal(signal.SIGINT, _shutdown)
        signal.signal(signal.SIGTERM, _shutdown)
        try:
            starter.start_all()
            starter._monitor_devices()
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        finally:
            starter.stop_all()

    hydra_main()


if __name__ == "__main__":
    main()
