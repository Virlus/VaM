import types
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from devices.robot import I2RTRobot


def make_random_init_robot(seed=7):
    robot = object.__new__(I2RTRobot)
    robot.device_name = "I2RTRobot"
    robot.device_id = 1
    robot.running = False
    robot._i2rt_stopped = True
    robot.random_init_pose = True
    robot.random_init_pose_params = {
        "position_radius_m": 0.02,
        "rotation_euler_range_deg": np.array([3.0, 4.0, 5.0]),
        "independent_per_arm": True,
        "max_ik_attempts": 3,
        "fallback_to_nominal_home": True,
        "randomize_on_startup": True,
        "randomize_on_episode_reset": True,
    }
    robot._random_init_pose_rng = np.random.default_rng(seed)
    robot._i2rt_kinematics = None
    robot._i2rt_kinematics_backend = None
    return robot


def test_random_pose_offsets_stay_inside_configured_bounds():
    robot = make_random_init_robot()

    for _ in range(1000):
        position_delta, euler_delta_deg = robot._sample_random_init_pose_offset()
        assert np.linalg.norm(position_delta) <= 0.02
        assert np.all(np.abs(euler_delta_deg) <= np.array([3.0, 4.0, 5.0]))


def test_randomized_home_preserves_grippers_and_replaces_arm_joints():
    robot = make_random_init_robot()
    nominal = np.array([0.0, 1.15, 1.0, -0.45, 0.0, 0.0, 0.99] * 2)

    def solve(self, nominal_q, position_delta, euler_delta_deg):
        del position_delta, euler_delta_deg
        return np.asarray(nominal_q) + 0.01

    robot._solve_randomized_home_arm = types.MethodType(solve, robot)
    randomized = robot._generate_randomized_home(nominal)

    assert randomized is not None
    assert np.allclose(randomized[:6], nominal[:6] + 0.01)
    assert np.allclose(randomized[7:13], nominal[7:13] + 0.01)
    assert randomized[6] == nominal[6]
    assert randomized[13] == nominal[13]


def test_randomization_reason_flags_are_respected():
    robot = make_random_init_robot()

    assert robot._should_randomize_init_pose("startup")
    assert robot._should_randomize_init_pose("episode_reset")
    robot.random_init_pose_params["randomize_on_episode_reset"] = False
    assert not robot._should_randomize_init_pose("episode_reset")
    assert robot._should_randomize_init_pose("startup")

    robot.random_init_pose = False
    assert not robot._should_randomize_init_pose("startup")


def test_follower_publishes_target_and_leader_consumes_it():
    with TemporaryDirectory() as temp_dir:
        target_path = Path(temp_dir) / "random_home.json"
        follower = make_random_init_robot()
        follower.is_follower_role = True
        follower.is_leader_role = False
        follower.random_init_pose_sync = True
        follower.random_init_pose_params["sync_target_path"] = str(target_path)
        expected = np.array([0.1, 1.1, 1.0, -0.4, 0.0, 0.0, 0.99] * 2)
        follower._write_random_init_pose_target(expected)

        leader = make_random_init_robot()
        leader.is_follower_role = False
        leader.is_leader_role = True
        leader.random_init_pose_sync = True
        leader.random_init_pose_params["sync_target_path"] = str(target_path)
        leader.random_init_pose_params["sync_wait_before_read_s"] = 0.0
        actual = leader._read_synchronized_home_target()

        assert actual is not None
        assert np.allclose(actual, expected)
