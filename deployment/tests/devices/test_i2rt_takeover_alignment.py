import threading
import time
from unittest.mock import patch

import numpy as np

from devices.robot import I2RTRobot


def test_constructor_initializes_takeover_transition_lock():
    with (
        patch.object(I2RTRobot, "_load_default_bilateral_gains"),
        patch.object(I2RTRobot, "_ensure_rpc_servers_running"),
        patch.object(I2RTRobot, "_connect_rpc_clients"),
        patch.object(I2RTRobot, "stop_server"),
    ):
        robot = I2RTRobot(
            device_id=0,
            mode="leader",
            dual_control_mode=True,
            takeover_alignment={"enabled": True},
            signal_files={},
        )
        assert robot._dagger_mode_transition_lock.acquire(blocking=False)
        robot._dagger_mode_transition_lock.release()
        del robot


def _make_leader(*, alignment_enabled: bool):
    robot = object.__new__(I2RTRobot)
    robot.device_name = "I2RTRobot"
    robot.device_id = 0
    robot.running = False
    robot._i2rt_stopped = True
    robot.data_shape = (14,)
    robot.data_dtype = np.float64
    robot.command_fps = 100.0
    robot.dual_control_mode = True
    robot.is_leader_role = True
    robot.is_follower_role = False
    robot.takeover_alignment_enabled = alignment_enabled
    robot.takeover_alignment_duration_s = 0.0
    robot.takeover_alignment_settle_timeout_s = 0.0
    robot._control_mode = robot._CONTROL_MODE_AUTONOMOUS
    robot._last_button_pressed = False
    robot._last_button_left = 0.0
    robot._last_button_right = 0.0
    robot._button_down_latched = False
    robot._button_release_samples = 0
    robot._button_release_debounce_samples = 3
    robot._dagger_mode_transition_lock = threading.Lock()
    robot._left_client = object()
    robot._right_client = object()
    robot.shared_memory = None
    robot._apply_bilateral_gains_for_mode = lambda: None
    robot.flags = []
    robot.commands = []
    robot._write_leader_control_flag = lambda value, _timestamp_ns: robot.flags.append(float(value))
    robot._command_joint_pos = lambda _client, joints, source: robot.commands.append(
        (source, np.asarray(joints, dtype=np.float64).copy())
    )
    return robot


def test_alignment_disabled_preserves_immediate_binary_handover():
    robot = _make_leader(alignment_enabled=False)
    current = np.arange(14, dtype=np.float64) / 10.0

    def leader_get_info(_client, source):
        if "left" in source:
            return current[:7].copy(), np.array([1.0])
        return current[7:].copy(), np.array([0.0])

    robot._leader_get_info = leader_get_info
    robot._align_leader_for_takeover = lambda _state: (_ for _ in ()).throw(
        AssertionError("disabled alignment must not run")
    )

    state = I2RTRobot.get_robot_state(robot)

    assert robot._control_mode == robot._CONTROL_MODE_TELEOP
    assert robot.flags == [1.0]
    np.testing.assert_allclose(state, current)


def test_leader_state_read_failure_does_not_publish_a_zero_pose():
    robot = _make_leader(alignment_enabled=False)

    def fail_to_read_state(_client, _source):
        raise RuntimeError("leader RPC unavailable")

    robot._leader_get_info = fail_to_read_state

    state = I2RTRobot.get_robot_state(robot)

    assert state is None


def test_unavailable_state_client_does_not_publish_a_zero_pose():
    robot = _make_leader(alignment_enabled=False)
    robot._left_client = None

    state = I2RTRobot.get_robot_state(robot)

    assert state is None


def test_alignment_publishes_half_flag_before_binary_handover():
    robot = _make_leader(alignment_enabled=True)
    current = np.zeros(14, dtype=np.float64)
    current[[6, 13]] = 0.25
    follower = np.ones(14, dtype=np.float64)
    follower[[6, 13]] = 0.75
    aligned = current.copy()
    aligned[:6] = follower[:6]
    aligned[7:13] = follower[7:13]

    def leader_get_info(_client, source):
        use_aligned_state = "takeover_alignment" in source
        state = aligned if use_aligned_state else current
        if "left" in source:
            return state[:7].copy(), np.array([1.0])
        return state[7:].copy(), np.array([0.0])

    robot._leader_get_info = leader_get_info
    robot._read_state_from_the_other_shm = lambda: follower.copy()

    with patch("devices.robot.time.sleep", lambda _seconds: None):
        state = I2RTRobot.get_robot_state(robot)

    assert robot._control_mode == robot._CONTROL_MODE_TELEOP
    assert robot.flags
    assert all(flag == 0.5 for flag in robot.flags[:-1])
    assert robot.flags[-1] == 1.0
    np.testing.assert_allclose(state, aligned)

    left_alignment_commands = [
        command for source, command in robot.commands if source == "command_joint_pos_left_takeover_alignment"
    ]
    right_alignment_commands = [
        command for source, command in robot.commands if source == "command_joint_pos_right_takeover_alignment"
    ]
    np.testing.assert_array_equal(left_alignment_commands[-1], follower[:6])
    np.testing.assert_array_equal(right_alignment_commands[-1], follower[7:13])


def test_alignment_large_joint_error_still_hands_over_to_human():
    robot = _make_leader(alignment_enabled=True)
    current = np.zeros(14, dtype=np.float64)
    follower = np.ones(14, dtype=np.float64)
    robot._read_state_from_the_other_shm = lambda: follower.copy()
    robot._read_leader_joint_state_for_takeover = lambda: current.copy()

    with patch("devices.robot.time.sleep", lambda _seconds: None):
        aligned_state = robot._handle_mode_transition(True, current_leader_state=current)

    np.testing.assert_allclose(aligned_state, current)
    assert robot._control_mode == robot._CONTROL_MODE_TELEOP
    assert 0.5 in robot.flags
    assert 0.0 not in robot.flags


def test_concurrent_state_reader_cannot_overlap_takeover_transition():
    robot = _make_leader(alignment_enabled=True)
    current = np.zeros(14, dtype=np.float64)
    alignment_started = threading.Event()
    finish_alignment = threading.Event()
    second_call_finished = threading.Event()
    alignment_calls = 0

    def leader_get_info(_client, _source):
        return current[:7].copy(), np.array([1.0])

    def align(_state):
        nonlocal alignment_calls
        alignment_calls += 1
        alignment_started.set()
        assert finish_alignment.wait(timeout=1.0)
        return current.copy()

    robot._leader_get_info = leader_get_info
    robot._align_leader_for_takeover = align

    first = threading.Thread(target=I2RTRobot.get_robot_state, args=(robot,))
    second = threading.Thread(
        target=lambda: (
            I2RTRobot.get_robot_state(robot),
            second_call_finished.set(),
        )
    )
    first.start()
    assert alignment_started.wait(timeout=1.0)
    second.start()
    time.sleep(0.05)

    assert not second_call_finished.is_set()
    finish_alignment.set()
    first.join(timeout=1.0)
    second.join(timeout=1.0)

    assert not first.is_alive()
    assert not second.is_alive()
    assert alignment_calls == 1


def test_half_flag_holds_follower_and_ignores_policy_command():
    robot = object.__new__(I2RTRobot)
    robot.device_name = "I2RTRobot"
    robot.device_id = 1
    robot.running = False
    robot._i2rt_stopped = True
    robot.is_leader_role = False
    robot.is_follower_role = True
    robot.dual_control_mode = True
    robot.policy_control_only = False
    robot._control_mode = robot._CONTROL_MODE_AUTONOMOUS
    robot._takeover_hold_joint_pos = None
    robot._latest_joint_state = np.arange(14, dtype=np.float64)
    robot._left_client = object()
    robot._right_client = object()
    robot._apply_bilateral_gains_for_mode = lambda: None
    robot.commands = []
    robot._command_joint_pos = lambda _client, joints, source: robot.commands.append(
        (source, np.asarray(joints, dtype=np.float64).copy())
    )

    robot._set_follower_control_mode_from_flag(0.5)
    robot.commands.clear()
    policy_command = np.full(14, -5.0, dtype=np.float64)
    I2RTRobot.execute_control_command(robot, {"I2RTRobot_1_control": policy_command})

    assert robot._control_mode == robot._CONTROL_MODE_ALIGNMENT
    np.testing.assert_array_equal(robot.commands[0][1], np.arange(7))
    np.testing.assert_array_equal(robot.commands[1][1], np.arange(7, 14))


def test_alignment_waits_full_settle_period_even_at_target():
    robot = _make_leader(alignment_enabled=True)
    robot.takeover_alignment_settle_timeout_s = 0.03
    target = np.ones(14, dtype=np.float64)
    robot._read_state_from_the_other_shm = lambda: target.copy()
    clock = [0.0]
    reads = []

    def measured_state():
        reads.append(clock[0])
        return target.copy()

    robot._read_leader_joint_state_for_takeover = measured_state
    with (
        patch("devices.robot.time.monotonic", lambda: clock[0]),
        patch(
            "devices.robot.time.sleep",
            lambda seconds: clock.__setitem__(0, clock[0] + seconds),
        ),
    ):
        state = robot._handle_mode_transition(True, current_leader_state=target)

    np.testing.assert_allclose(state, target)
    assert reads[-1] - reads[0] >= robot.takeover_alignment_settle_timeout_s
    assert robot._control_mode == robot._CONTROL_MODE_TELEOP
    assert all(flag == 0.5 for flag in robot.flags)


def test_alignment_without_valid_measurement_keeps_policy_control():
    robot = _make_leader(alignment_enabled=True)
    robot._read_state_from_the_other_shm = lambda: np.ones(14)
    robot._read_leader_joint_state_for_takeover = lambda: None
    with patch("devices.robot.time.sleep", lambda _seconds: None):
        state = robot._handle_mode_transition(True, current_leader_state=np.zeros(14))

    assert state is None
    assert robot._control_mode == robot._CONTROL_MODE_AUTONOMOUS
    assert robot.flags[-1] == 0.0
