#!/usr/bin/env python3
"""
Bilateral control coordinator.

Launches 4 RPC server processes (2 followers, 2 leaders) and runs a control
loop that cycles through three states via the teaching-handle button:

    IDLE -> LEADER_TO_FOLLOWER -> FOLLOWER_AUTONOMOUS -> IDLE -> ...

IDLE:                No commands sent. Leaders are backdrivable.
LEADER_TO_FOLLOWER:  Leader positions are relayed to followers (teleoperation).
FOLLOWER_AUTONOMOUS: Followers execute a joint-0 oscillation; leaders track.
"""

import os
import subprocess
import sys
import time
from enum import Enum

import numpy as np
import portal

# import debugpy; debugpy.connect(("0.0.0.0", 5678))

current_file_path = os.path.dirname(os.path.abspath(__file__))

FOLLOWER_R_PORT = 1234
FOLLOWER_L_PORT = 1235
LEADER_R_PORT = 1236
LEADER_L_PORT = 1237

LEADER_DEFAULT_KP = np.array([250.0, 200.0, 200.0, 100.0, 25.0, 25.0])
LEADER_DEFAULT_KD = np.array([5.0, 5.0, 5.0, 1.5, 1.5, 1.5])
FOLLOWER_DEFAULT_KP = np.array([250.0, 200.0, 200.0, 100.0, 25.0, 25.0, 25.0])
FOLLOWER_DEFAULT_KD = np.array([5.0, 5.0, 5.0, 1.5, 1.5, 1.5, 1.5])
LEADER_ZERO_KP = np.zeros(6)
LEADER_ZERO_KD = np.zeros(6)
FOLLOWER_ZERO_KP = np.zeros(7)
FOLLOWER_ZERO_KD = np.zeros(7)

OSCILLATION_AMP_DEG = 5.0
OSCILLATION_FREQ_HZ = 0.5
SLOW_MOVE_STEPS = 100
SLOW_MOVE_DT = 0.03
SERVER_STARTUP_WAIT = 3.0


class Mode(Enum):
    IDLE = 0
    LEADER_TO_FOLLOWER = 1
    FOLLOWER_AUTONOMOUS = 2


# ---------------------------------------------------------------------------
# RPC client wrappers
# ---------------------------------------------------------------------------

class FollowerClient:
    """RPC client for a follower robot (ServerRobot on the other end)."""

    def __init__(self, port: int, host: str = "127.0.0.1"):
        self._client = portal.Client(f"{host}:{port}")

    def get_info(self) -> np.ndarray:
        return self._client.get_info().result()

    def command_joint_pos(self, joint_pos: np.ndarray) -> None:
        self._client.command_joint_pos(joint_pos)

    def update_kp_kd(self, kp: np.ndarray, kd: np.ndarray) -> None:
        self._client.update_kp_kd(kp, kd)


class LeaderClient:
    """RPC client for a leader robot (LeaderServerRobot on the other end)."""

    def __init__(self, port: int, host: str = "127.0.0.1"):
        self._client = portal.Client(f"{host}:{port}")

    def get_info(self):
        return self._client.get_info().result()

    def command_joint_pos(self, joint_pos: np.ndarray) -> None:
        self._client.command_joint_pos(joint_pos)

    def update_kp_kd(self, kp: np.ndarray, kd: np.ndarray) -> None:
        self._client.update_kp_kd(kp, kd)


# ---------------------------------------------------------------------------
# CAN interface checks (reused from bimanual_lead_follower)
# ---------------------------------------------------------------------------

def check_can_interface(interface: str) -> bool:
    try:
        result = subprocess.run(
            ["ip", "link", "show", interface],
            capture_output=True, text=True, check=False,
        )
        if result.returncode != 0:
            return False
        if "state UP" in result.stdout or "state UNKNOWN" in result.stdout:
            return True
        print(f"Warning: CAN interface {interface} exists but is not UP")
        return False
    except Exception as e:
        print(f"Error checking CAN interface {interface}: {e}")
        return False


def check_all_can_interfaces() -> None:
    required = ["can_follower_r", "can_leader_r", "can_follower_l", "can_leader_l"]
    missing = [iface for iface in required if not check_can_interface(iface)]
    if missing:
        raise RuntimeError(f"Missing or unavailable CAN interfaces: {', '.join(missing)}")
    print("All CAN interfaces are available")


# ---------------------------------------------------------------------------
# Subprocess launcher
# ---------------------------------------------------------------------------

def launch_gello_process(can_channel: str, gripper: str, mode: str, server_port: int):
    script_path = os.path.join(current_file_path, "..", "scripts", "minimum_gello.py")
    cmd = [
        "python", os.path.expanduser(script_path),
        "--can_channel", can_channel,
        "--gripper", gripper,
        "--mode", mode,
        "--server_port", str(server_port),
    ]
    print(f"Starting: {' '.join(cmd)}")
    try:
        return subprocess.Popen(cmd)
    except Exception as e:
        print(f"Error starting process for {can_channel}: {e}")
        return None


def terminate_processes(processes: list) -> None:
    for proc in processes:
        try:
            print(f"Terminating process {proc.pid}...")
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                print(f"Force killing process {proc.pid}...")
                proc.kill()
                proc.wait()
        except Exception as e:
            print(f"Error terminating process {proc.pid}: {e}")
    print("All processes terminated")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def slow_move(
    client: FollowerClient,
    target_pos: np.ndarray,
    current_pos: np.ndarray,
) -> None:
    """Linearly interpolate the follower from its current position to *target_pos*."""
    for i in range(SLOW_MOVE_STEPS):
        alpha = i / SLOW_MOVE_STEPS
        interp = current_pos * (1.0 - alpha) + target_pos * alpha
        client.command_joint_pos(interp)
        time.sleep(SLOW_MOVE_DT)


# ---------------------------------------------------------------------------
# Main coordinator
# ---------------------------------------------------------------------------

def main() -> None:
    processes: list = []
    leader_kp_factor = 0.25
    leader_kd_factor = 0.5
    follower_kp_factor = 0.25
    follower_kd_factor = 0.5
    try:
        print("Checking CAN interfaces...")
        check_all_can_interfaces()

        process_configs = [
            {"can_channel": "can_follower_r", "gripper": "linear_4310",
             "mode": "follower_server", "server_port": FOLLOWER_R_PORT},
            {"can_channel": "can_follower_l", "gripper": "linear_4310",
             "mode": "follower_server", "server_port": FOLLOWER_L_PORT},
            {"can_channel": "can_leader_r", "gripper": "yam_teaching_handle",
             "mode": "leader_server", "server_port": LEADER_R_PORT},
            {"can_channel": "can_leader_l", "gripper": "yam_teaching_handle",
             "mode": "leader_server", "server_port": LEADER_L_PORT},
        ]

        print("\nLaunching server processes...")
        for cfg in process_configs:
            proc = launch_gello_process(**cfg)
            if proc is None:
                raise RuntimeError(f"Failed to start process for {cfg['can_channel']}")
            processes.append(proc)
            print(f"  Started PID {proc.pid} for {cfg['can_channel']}")

        print(f"\nWaiting {SERVER_STARTUP_WAIT}s for servers to initialise...")
        time.sleep(SERVER_STARTUP_WAIT)

        follower_r = FollowerClient(FOLLOWER_R_PORT)
        follower_l = FollowerClient(FOLLOWER_L_PORT)
        leader_r = LeaderClient(LEADER_R_PORT)
        leader_l = LeaderClient(LEADER_L_PORT)

        mode = Mode.IDLE
        oscillation_start: float = 0.0
        base_q_r: np.ndarray = np.zeros(0)
        base_q_l: np.ndarray = np.zeros(0)

        print("Bilateral control ready. Press the teaching-handle button to cycle modes.")
        print("  IDLE -> LEADER_TO_FOLLOWER -> FOLLOWER_AUTONOMOUS -> IDLE")
        print("Press Ctrl+C to stop.\n")

        while True:
            for proc in processes:
                if proc.poll() is not None:
                    raise RuntimeError(f"Server process {proc.pid} died unexpectedly")

            leader_r_pos, btn_r = leader_r.get_info()
            leader_l_pos, btn_l = leader_l.get_info()

            button_pressed = btn_r[0] > 0.5 or btn_l[0] > 0.5

            # ---- state transitions ----
            if button_pressed:
                if mode == Mode.IDLE:
                    mode = Mode.LEADER_TO_FOLLOWER
                    leader_r.update_kp_kd(LEADER_ZERO_KP, LEADER_ZERO_KD)
                    leader_l.update_kp_kd(LEADER_ZERO_KP, LEADER_ZERO_KD)
                    print(f"leader_r_pos: {leader_r_pos}")
                    print(f"leader_l_pos: {leader_l_pos}")
                    leader_r.command_joint_pos(leader_r_pos[:6])
                    leader_l.command_joint_pos(leader_l_pos[:6])
                    follower_r.update_kp_kd(FOLLOWER_DEFAULT_KP * follower_kp_factor, FOLLOWER_DEFAULT_KD * follower_kd_factor)
                    follower_l.update_kp_kd(FOLLOWER_DEFAULT_KP * follower_kp_factor, FOLLOWER_DEFAULT_KD * follower_kd_factor)

                    cur_fr = follower_r.get_info()
                    cur_fl = follower_l.get_info()
                    print(f"cur_fr: {cur_fr}")
                    print(f"cur_fl: {cur_fl}")
                    slow_move(follower_r, leader_r_pos, cur_fr)
                    slow_move(follower_l, leader_l_pos, cur_fl)
                    print("[MODE] LEADER_TO_FOLLOWER")

                elif mode == Mode.LEADER_TO_FOLLOWER:
                    mode = Mode.FOLLOWER_AUTONOMOUS
                    leader_r.update_kp_kd(LEADER_DEFAULT_KP * leader_kp_factor, LEADER_DEFAULT_KD * leader_kd_factor)
                    leader_l.update_kp_kd(LEADER_DEFAULT_KP * leader_kp_factor, LEADER_DEFAULT_KD * leader_kd_factor)
                    # follower_r.update_kp_kd(FOLLOWER_ZERO_KP, FOLLOWER_ZERO_KD)
                    # follower_l.update_kp_kd(FOLLOWER_ZERO_KP, FOLLOWER_ZERO_KD)

                    base_q_r = follower_r.get_info()
                    base_q_l = follower_l.get_info()
                    oscillation_start = time.time()
                    print("[MODE] FOLLOWER_AUTONOMOUS")

                elif mode == Mode.FOLLOWER_AUTONOMOUS:
                    mode = Mode.IDLE
                    leader_r.update_kp_kd(LEADER_ZERO_KP, LEADER_ZERO_KD)
                    leader_l.update_kp_kd(LEADER_ZERO_KP, LEADER_ZERO_KD)
                    print("[MODE] IDLE")

                # debounce: wait until button is released
                while True:
                    _, btn_r = leader_r.get_info()
                    _, btn_l = leader_l.get_info()
                    if btn_r[0] <= 0.5 and btn_l[0] <= 0.5:
                        break
                    time.sleep(0.03)

            # ---- mode execution ----
            if mode == Mode.LEADER_TO_FOLLOWER:
                follower_r.command_joint_pos(leader_r_pos)
                follower_l.command_joint_pos(leader_l_pos)

            elif mode == Mode.FOLLOWER_AUTONOMOUS:
                t = time.time() - oscillation_start
                amp = np.deg2rad(OSCILLATION_AMP_DEG)
                offset = amp * np.sin(2.0 * np.pi * OSCILLATION_FREQ_HZ * t)

                target_r = base_q_r.copy()
                target_r[0] += offset
                target_l = base_q_l.copy()
                target_l[0] += offset

                follower_r.command_joint_pos(target_r)
                follower_l.command_joint_pos(target_l)

                f_pos_r = follower_r.get_info()
                f_pos_l = follower_l.get_info()
                leader_r.command_joint_pos(f_pos_r[:6])
                leader_l.command_joint_pos(f_pos_l[:6])

            time.sleep(0.01)

    except KeyboardInterrupt:
        print("\nReceived Ctrl+C, shutting down...")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
    finally:
        terminate_processes(processes)


if __name__ == "__main__":
    main()
