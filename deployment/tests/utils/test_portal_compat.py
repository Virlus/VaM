import os
import threading
import time

from utils.portal_compat import (
    import_portal_with_wake_pipe_fix,
    install_portal_wake_pipe_fix,
    portal_wake_pipe_stats,
)


def test_wake_pipe_fix_is_idempotent():
    """Installing the process-wide class patch repeatedly is harmless."""
    portal = import_portal_with_wake_pipe_fix()

    assert install_portal_wake_pipe_fix(portal) is False


def test_client_wake_pipe_does_not_accumulate_under_concurrent_rpc():
    """Concurrent callers cannot leave one wake byte behind per request batch."""
    portal = import_portal_with_wake_pipe_fix()
    port = portal.free_port()
    server = portal.Server(port, workers=4)
    server.bind("echo", lambda value: value)
    server.start(block=False)
    client = portal.Client(f"127.0.0.1:{port}", maxinflight=16)
    errors = []

    try:
        assert client.connect(timeout=2.0)
        assert os.get_blocking(client.socket.get_signal) is False
        assert os.get_blocking(client.socket.set_signal) is False

        def caller(offset):
            try:
                for index in range(500):
                    value = offset + index
                    assert client.echo(value).result(timeout=1.0) == value
            except BaseException as error:
                errors.append(error)

        callers = [threading.Thread(target=caller, args=(index * 1000,)) for index in range(4)]
        for caller_thread in callers:
            caller_thread.start()
        for caller_thread in callers:
            caller_thread.join(timeout=10.0)

        time.sleep(0.05)
        assert all(not caller_thread.is_alive() for caller_thread in callers)
        assert not errors
        pipe_stats = portal_wake_pipe_stats(client)
        assert pipe_stats["signal_pending"] == 0
        assert pipe_stats["signal_capacity"] > 0
    finally:
        client.close(timeout=2.0)
        server.close(timeout=2.0)
