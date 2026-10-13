"""Collection episode signals shared by recorded and LeRobot recorders."""

import os
import time
from pathlib import Path
from typing import Any

import numpy as np

from utils.logger_config import logger


class CollectionSignals:
    def _parse_robot_identity(self, device_key: str, device_info: dict[str, Any]) -> tuple[str, str] | None:
        """Parse robot identity tuple `(device_name, device_id)` from device metadata."""
        identity_candidates: list[str] = []
        device_name = device_info.get("name")
        if isinstance(device_name, str) and device_name:
            identity_candidates.append(device_name)
        identity_candidates.append(device_key)

        for candidate in identity_candidates:
            normalized = candidate
            for suffix in ("_data", "_control", "_control_flag"):
                normalized = normalized.removesuffix(suffix)
            parts = normalized.rsplit("_", 1)
            if len(parts) == 2 and parts[0] and parts[1].isdigit():
                return parts[0], parts[1]

        device_id = device_info.get("id")
        if isinstance(device_id, (int, np.integer)):
            return "I2RTRobot", str(int(device_id))
        return None

    def _get_i2rt_robot_identities(self) -> list[tuple[str, str]]:
        """Return unique I2RT robot identities discovered from connected devices."""
        identities: list[tuple[str, str]] = []
        seen = set()
        for device_key, device_info in self.devices.items():
            device_type = str(device_info.get("type", ""))
            if device_type != "I2RTRobot" and "i2rtrobot" not in device_key.lower():
                continue
            identity = self._parse_robot_identity(device_key, device_info)
            if identity is None or identity in seen:
                continue
            seen.add(identity)
            identities.append(identity)
        return identities

    def _write_robot_signal(self, command_pattern: str, payload: str, command_name: str) -> int:
        """Write command files for all detected I2RT robots."""
        robot_identities = self._get_i2rt_robot_identities()
        if not robot_identities:
            logger.warning(f"No I2RT robot devices found, skip {command_name} signal")
            return 0

        sent_count = 0
        for robot_name, robot_id in robot_identities:
            signal_file = Path(command_pattern.format(device_name=robot_name, device_id=robot_id))
            try:
                signal_file.parent.mkdir(parents=True, exist_ok=True)
                signal_file.write_text(payload)
                sent_count += 1
                logger.info(f"Sent {command_name} signal to {robot_name}_{robot_id}")
            except Exception as e:  # noqa: BLE001
                logger.error(f"Failed to send {command_name} signal to {robot_name}_{robot_id}: {e}")
        return sent_count

    def _signal_i2rt_home(self) -> int:
        """Signal all active I2RT robots to move to home pose."""
        return self._write_robot_signal(
            command_pattern=self.robot_home_command_pattern,
            payload="0",
            command_name="home",
        )

    def _signal_i2rt_teleop_start(self) -> int:
        """Signal all active I2RT robots to enter teleoperation-ready state."""
        return self._write_robot_signal(
            command_pattern=self.robot_teleop_start_command_pattern,
            payload="1",
            command_name="teleop-start",
        )

    def _signal_i2rt_teleop_discard(self) -> int:
        """Signal all active I2RT robots to exit teleoperation-ready state."""
        return self._write_robot_signal(
            command_pattern=self.robot_teleop_discard_command_pattern,
            payload="1",
            command_name="teleop-discard",
        )

    def _signal_and_wait_for_robot_halt(self) -> None:
        """Signal the robot to halt and wait for the robot to report its "do halt state"."""
        try:
            for halt_device in self.halt_and_home_cfg:
                device_name = halt_device.get("device_name", "")
                device_id = halt_device.get("device_id", 0)
                robot_halt_signal = Path(
                    self.robot_halt_command_pattern.format(device_name=device_name, device_id=device_id)
                )
                robot_halt_signal.parent.mkdir(parents=True, exist_ok=True)
                robot_halt_signal.touch()
                logger.info(f"PolicyConnector: Signaled robot to halt, device: {device_name}_{device_id}")
        except Exception as e:  # noqa: BLE001
            logger.error(f"PolicyConnector: Failed to signal robot halt: {e}")
            return

        all_robot_halt_reply_files_exist = False

        while not all_robot_halt_reply_files_exist:
            all_robot_halt_reply_files_exist_list = []
            for reply_device in self.halt_and_home_cfg:
                device_name = reply_device.get("device_name", "")
                device_id = reply_device.get("device_id", 0)
                robot_halt_reply_file = Path(
                    self.robot_halt_reply_pattern.format(device_name=device_name, device_id=device_id)
                )
                if robot_halt_reply_file.exists():
                    logger.info(f"PolicyConnector: {robot_halt_reply_file} exists")
                    now_robot_halt_reply_file_exists = True
                else:
                    logger.info(f"PolicyConnector: {robot_halt_reply_file} does not exist")
                    now_robot_halt_reply_file_exists = False
                all_robot_halt_reply_files_exist_list.append(now_robot_halt_reply_file_exists)

            all_robot_halt_reply_files_exist = all(all_robot_halt_reply_files_exist_list)

            if all_robot_halt_reply_files_exist:
                break
            else:
                time.sleep(0.1)

        logger.info("PolicyConnector: Robot reported its 'do halt state'")

        # consume the robot_halt_reply_file
        for reply_device in self.halt_and_home_cfg:
            device_name = reply_device.get("device_name", "")
            device_id = reply_device.get("device_id", 0)
            robot_halt_reply_file = Path(
                self.robot_halt_reply_pattern.format(device_name=device_name, device_id=device_id)
            )
            robot_halt_reply_file.unlink()
            logger.info(f"PolicyConnector: Consumed robot halt reply file for {device_name}_{device_id}")

    def _setup_keyboard_listener(self) -> None:
        """Setup keyboard listener for manual episode discard using 'd' key."""
        try:

            def on_press(key: Any) -> None:
                try:
                    if hasattr(key, "char") and key.char == "d" and self.episode_running:
                        self._discard_keyboard_pressed = True
                        logger.info("'D' pressed - discarding episode and resetting to home")
                except AttributeError:
                    pass

            from pynput import keyboard as pynput_keyboard

            self.keyboard_listener = pynput_keyboard.Listener(on_press=on_press)
            self.keyboard_listener.start()
            logger.info("Keyboard listener started. Press 'D' to discard current episode and reset to home.")
        except Exception as e:  # noqa: BLE001
            logger.error(f"Failed to setup keyboard listener: {e}")
            self.keyboard_listener = None

    def _stop_keyboard_listener(self) -> None:
        """Stop the keyboard listener."""
        if self.keyboard_listener is not None:
            try:
                self.keyboard_listener.stop()
                self.keyboard_listener = None
                logger.info("Keyboard listener stopped")
            except Exception as e:  # noqa: BLE001
                logger.warning(f"Failed to stop keyboard listener: {e}")

    def _wait_for_episode_start_signal(self) -> None:
        """Wait for episode start signal file to be created by leader robot."""
        while self.running and not os.path.exists(self._episode_start_signal_path):
            time.sleep(0.01)
        # Clear the signal file after detection
        self._clear_episode_signal("start")

    def _check_episode_end_signal(self) -> bool:
        """Check if episode end signal file exists."""
        if os.path.exists(self._episode_end_signal_path):
            self._clear_episode_signal("end")
            return True
        return False

    def _clear_episode_signal(self, signal_type: str) -> None:
        """Clear the episode signal file after processing."""
        signal_path = self.episode_signal_pattern.format(signal=signal_type)
        try:
            if os.path.exists(signal_path):
                os.remove(signal_path)
                logger.debug(f"Cleared episode {signal_type} signal")
        except Exception as e:  # noqa: BLE001
            logger.warning(f"Failed to clear episode {signal_type} signal: {e}")
