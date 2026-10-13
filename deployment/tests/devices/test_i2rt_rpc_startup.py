"""RPC launcher paths and bimanual startup without spawning hardware processes."""

from unittest.mock import Mock

from devices.robot import I2RTRobot


def make_robot():
    robot = object.__new__(I2RTRobot)
    robot._i2rt_stopped = True
    robot.running = False
    robot.device_name = "I2RTRobot"
    robot.device_id = 0
    robot.left_can_channel, robot.right_can_channel = "left", "right"
    robot.left_server_port, robot.right_server_port = 1234, 1235
    robot.server_startup_wait_s = 0
    robot._server_processes = []
    return robot


def test_startup_keeps_both_owned_processes_and_does_not_duplicate(monkeypatch):
    robot = make_robot()
    processes = [Mock(), Mock()]
    robot._launch_rpc_server_process = Mock(side_effect=processes)
    robot._rpc_endpoints_ready = Mock(return_value=True)
    monkeypatch.setattr("devices.robot.time.sleep", lambda _: None)
    robot._ensure_rpc_servers_running()
    assert robot._server_processes == processes
    assert [c.args for c in robot._launch_rpc_server_process.call_args_list] == [("left", 1234), ("right", 1235)]
    robot._ensure_rpc_servers_running()
    assert robot._launch_rpc_server_process.call_count == 2


def test_rpc_command_preserves_role_gripper_and_terminal_isolation(monkeypatch):
    robot = make_robot()
    robot.server_launch_python = "/venv/bin/python"
    robot.minimum_gello_script = "/vam/packages/i2rt/scripts/minimum_gello.py"
    robot.gripper_type = "linear_4310"
    robot.is_leader_role = False
    popen = Mock()
    monkeypatch.setattr("devices.robot.subprocess.Popen", popen)
    assert robot._launch_rpc_server_process("left", 1234) is popen.return_value
    command = popen.call_args.args[0]
    assert command[:2] == [robot.server_launch_python, robot.minimum_gello_script]
    assert command[command.index("--mode") + 1] == "follower_server"
    assert command[command.index("--gripper") + 1] == "linear_4310"
    assert popen.call_args.kwargs["start_new_session"] is True
