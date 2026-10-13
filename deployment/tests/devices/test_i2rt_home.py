"""Current I2RT home contract using arm caches; no robot or RPC calls."""

from threading import Lock
from unittest.mock import Mock

import numpy as np
import pytest

from devices.robot import I2RTRobot


def make_robot(follower=True):
    robot = object.__new__(I2RTRobot)
    robot._i2rt_stopped = True
    robot.running = False
    robot.device_name = "I2RTRobot"
    robot.device_id = 1 if follower else 0
    robot.is_follower_role = follower
    robot.is_leader_role = not follower
    robot.dual_control_mode = True
    robot.home_joint_pos = np.array([0, 1.15, 1, -0.45, 0, 0, 0.99] * 2)
    robot._left_client, robot._right_client = object(), object()
    robot._last_trusted_get_info_state_left = np.arange(7) * 0.01
    robot._last_trusted_get_info_state_right = np.arange(7) * 0.02
    robot._trusted_state_lock = Lock()
    robot.random_init_pose_sync = False
    robot._should_randomize_init_pose = Mock(return_value=False)
    robot._invalidate_random_init_pose_target = Mock()
    robot._write_random_init_pose_target = Mock()
    robot._smooth_joint_move = Mock()
    robot._follower_get_info = Mock(side_effect=AssertionError("Trusted cache should avoid RPC"))
    robot._leader_get_info = Mock(side_effect=AssertionError("Trusted cache should avoid RPC"))
    return robot


@pytest.mark.parametrize("follower", [True, False])
def test_home_uses_trusted_arm_states_and_resets_caches(follower):
    robot = make_robot(follower)
    left = robot._last_trusted_get_info_state_left.copy()
    right = robot._last_trusted_get_info_state_right.copy()
    assert robot._go_home()
    args = robot._smooth_joint_move.call_args.kwargs
    width = 7 if follower else 6
    np.testing.assert_array_equal(args["left_current"], left[:width])
    np.testing.assert_array_equal(args["right_current"], right[:width])
    np.testing.assert_array_equal(args["left_target"], robot.home_joint_pos[:width])
    np.testing.assert_array_equal(args["right_target"], robot.home_joint_pos[7 : 7 + width])
    assert robot._last_trusted_get_info_state_left is None
    assert robot._last_trusted_get_info_state_right is None
    assert robot._last_control_flag == 0


def test_home_requires_connected_clients():
    robot = make_robot()
    robot._left_client = None
    assert not robot._go_home()
    robot._smooth_joint_move.assert_not_called()


def test_home_without_target_does_not_move():
    robot = make_robot()
    robot.home_joint_pos = None
    assert robot._go_home()
    robot._smooth_joint_move.assert_not_called()


def test_home_rejects_wrong_joint_count():
    robot = make_robot()
    robot.home_joint_pos = np.zeros(12)
    assert not robot._go_home()
    robot._smooth_joint_move.assert_not_called()
