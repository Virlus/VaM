import types

from device_starter import DeviceStarter


class FakeProcess:
    _next_pid = 1000

    def __init__(self, alive=True):
        self.alive = alive
        self.exitcode = None if alive else 1
        self.started = False
        self.terminated = False
        self.pid = FakeProcess._next_pid
        FakeProcess._next_pid += 1

    def start(self):
        self.started = True

    def is_alive(self):
        return self.alive

    def terminate(self):
        self.terminated = True
        self.alive = False
        self.exitcode = -15

    def join(self, timeout=None):
        del timeout

    def kill(self):
        self.alive = False
        self.exitcode = -9


def make_starter(devices, enable_device_manager=True):
    starter = object.__new__(DeviceStarter)
    starter.config = {
        "global_settings": {"i2rt_startup_stagger_s": 0.0},
        "action_executor": {"type": "disabled"},
    }
    starter.devices = devices
    starter.running = False
    starter.enable_visualizers = False
    starter.enable_device_manager = enable_device_manager
    starter.device_manager_process = None
    starter.policy_runner_process = None
    starter._parse_device_configs = types.MethodType(lambda self: devices, starter)
    starter._write_robot_home_index = types.MethodType(lambda self: None, starter)
    return starter


def test_wait_for_devices_ready_fails_fast_when_child_exits():
    device = {
        "device_class": "I2RTRobot",
        "device_id": 1,
        "process": FakeProcess(alive=False),
        "ready": False,
    }
    starter = make_starter([device], enable_device_manager=False)

    assert not starter._wait_for_devices_ready(timeout_s=1.0)
    assert not device["ready"]


def test_start_all_does_not_start_manager_when_device_is_not_ready():
    device = {"device_class": "I2RTRobot", "device_id": 1, "ready": False}
    starter = make_starter([device])
    manager_started = []

    def start_device(self, device_info):
        device_info["process"] = FakeProcess()
        return True

    starter._start_device_process = types.MethodType(start_device, starter)
    starter._wait_for_devices_ready = types.MethodType(lambda self: False, starter)
    starter._start_device_manager_process = types.MethodType(
        lambda self: manager_started.append(True),
        starter,
    )

    assert not starter.start_all()
    assert manager_started == []
    assert device["process"].terminated
    assert not starter.running


def test_start_all_starts_manager_after_all_devices_are_ready():
    device = {"device_class": "I2RTRobot", "device_id": 1, "ready": False}
    starter = make_starter([device])
    manager_process = FakeProcess()

    def start_device(self, device_info):
        device_info["process"] = FakeProcess()
        return True

    def wait_ready(self):
        device["ready"] = True
        return True

    starter._start_device_process = types.MethodType(start_device, starter)
    starter._wait_for_devices_ready = types.MethodType(wait_ready, starter)
    starter._start_device_manager_process = types.MethodType(
        lambda self: manager_process,
        starter,
    )

    assert starter.start_all()
    assert manager_process.started
    assert starter.running
