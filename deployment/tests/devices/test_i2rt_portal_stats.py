import threading

from devices.robot import I2RTRobot


class FakePortalClient:
    def __init__(self):
        self.lock = threading.Lock()
        self.connected = True
        self.maxinflight = 16
        self.calls = 0
        self.recvrate = [0, 0.0]
        self.socket = type("FakeSocket", (), {"callbacks_recv": []})()

    def stats(self):
        self.calls += 1
        return {
            "inflight": 3,
            "numsend": 10,
            "numrecv": 9,
            "sendrate": 10.0,
            "recvrate": 9.0,
            "waitmean": 0.02,
        }


def make_robot():
    robot = object.__new__(I2RTRobot)
    robot.device_name = "I2RTRobot"
    robot.device_id = 1
    robot.running = False
    robot._i2rt_stopped = True
    robot.is_leader_role = False
    robot.left_server_host = "127.0.0.1"
    robot.right_server_host = "127.0.0.1"
    robot.left_server_port = 1234
    robot.right_server_port = 1235
    robot._left_client = FakePortalClient()
    robot._right_client = FakePortalClient()
    robot._portal_stats_interval_s = 60.0
    robot._portal_stats_last_log_monotonic = 0.0
    robot._diagnostic_run_id = "test-run"
    return robot


def test_portal_stats_respects_interval_and_force_override():
    """An explicit diagnostic snapshot bypasses the regular log interval."""
    robot = make_robot()

    robot._log_portal_client_stats_if_due()
    robot._log_portal_client_stats_if_due()
    assert robot._left_client.calls == 1
    assert robot._right_client.calls == 1

    robot._log_portal_client_stats_if_due(force=True)
    assert robot._left_client.calls == 2
    assert robot._right_client.calls == 2


def test_disabled_portal_stats_does_not_touch_clients():
    """The baseline path must not call Portal stats or reset its counters."""
    robot = make_robot()
    robot._portal_stats_interval_s = 0.0

    robot._log_portal_client_stats_if_due(force=True)

    assert robot._left_client.calls == 0
    assert robot._right_client.calls == 0


def test_portal_receive_callback_fills_missing_upstream_counter():
    """Repeated setup adds one callback and counts each received response once."""
    robot = make_robot()
    client = robot._left_client

    robot._configure_portal_client_diagnostics(client)
    robot._configure_portal_client_diagnostics(client)
    assert len(client.socket.callbacks_recv) == 1

    client.socket.callbacks_recv[0](b"response")
    assert client.recvrate[0] == 1
