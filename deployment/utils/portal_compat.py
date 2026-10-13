"""Compatibility fixes for the Portal RPC client used by I2RT."""

import array
import fcntl
import importlib
import os
import select
import termios
import threading
from types import ModuleType
from typing import Any, Dict

from utils.logger_config import logger


_PATCH_LOCK = threading.Lock()
_PATCH_MARKER = "_vam_wake_pipe_fix"


def _drain_signal_pipe(file_descriptor: int) -> None:
    """Drain all currently queued wake bytes from a non-blocking pipe."""
    while True:
        try:
            if not os.read(file_descriptor, 65536):
                return
        except BlockingIOError:
            return


def install_portal_wake_pipe_fix(portal_module: ModuleType) -> bool:
    """Prevent Portal's client wake pipe from filling under concurrent calls.

    Portal 3.7.3 writes one byte for every queued request but consumes only one
    byte for a potentially larger batch. Eventually the blocking ``os.write``
    in ``ClientSocket.send`` fills the pipe and stops the RPC caller forever.

    Returns:
        True when this call installed the fix, or False when it was installed
        previously.
    """
    client_socket_module = portal_module.client_socket
    client_socket_class = client_socket_module.ClientSocket

    with _PATCH_LOCK:
        if getattr(client_socket_class, _PATCH_MARKER, False):
            return False

        original_init = client_socket_class.__init__

        def init(self: Any, *args: Any, **kwargs: Any) -> None:
            original_init(self, *args, **kwargs)
            os.set_blocking(self.get_signal, False)
            os.set_blocking(self.set_signal, False)

        def send(self: Any, *data: Any, timeout: float | None = None) -> None:
            assert self.running
            if len(self.sendq) > self.options.max_send_queue:
                raise RuntimeError("Too many outgoing messages enqueued")
            self.require_connection(timeout)
            maxsize = self.options.max_msg_size
            self.sendq.append(
                client_socket_module.buffers.SendBuffer(
                    *data,
                    maxsize=maxsize,
                )
            )
            try:
                os.write(self.set_signal, bytes(1))
            except BlockingIOError:
                # A full pipe is already readable, so the socket loop is
                # guaranteed to observe a wakeup for the queued request.
                pass

        def loop(self: Any) -> None:
            recvbuf = client_socket_module.buffers.RecvBuffer(maxsize=self.options.max_msg_size)
            sock = None
            poll = select.poll()
            poll.register(self.get_signal, select.POLLIN)
            isconn = False
            writing = False

            while self.running or (self.sendq and isconn):
                if not isconn:
                    if not self.options.autoconn and not self.wantconn.wait(timeout=0.2):
                        continue
                    sock = self._connect()
                    if not sock:
                        break
                    poll.register(sock, select.POLLIN)
                    self.isconn.set()
                    isconn = True
                    if not self.options.autoconn:
                        self.wantconn.clear()
                    for callback in self.callbacks_conn:
                        callback()

                try:
                    if not writing:
                        ready_descriptors = [descriptor for descriptor, _event in poll.poll(0.2)]
                        if self.get_signal in ready_descriptors:
                            _drain_signal_pipe(self.get_signal)
                            writing = bool(self.sendq)

                    try:
                        recvbuf.recv(sock)
                        if recvbuf.done():
                            if self.recvq.qsize() > self.options.max_recv_queue:
                                raise RuntimeError("Too many incoming messages enqueued")
                            message = recvbuf.result()
                            self.recvq.put(message)
                            for callback in self.callbacks_recv:
                                callback(message)
                            recvbuf = client_socket_module.buffers.RecvBuffer(maxsize=self.options.max_msg_size)
                    except BlockingIOError:
                        pass

                    if self.sendq:
                        try:
                            self.sendq[0].send(sock)
                            if self.sendq[0].done():
                                self.sendq.popleft()
                                if not self.sendq:
                                    writing = False
                        except BlockingIOError:
                            pass
                        except ConnectionResetError:
                            pass

                except OSError as error:
                    detail = type(error).__name__
                    if str(error):
                        detail = f"{detail}: {error}"
                    self._log(f"Connection to server lost ({detail})")
                    self.isconn.clear()
                    isconn = False
                    poll.unregister(sock)
                    sock.close()
                    self.sendq.clear()
                    recvbuf = client_socket_module.buffers.RecvBuffer(maxsize=self.options.max_msg_size)
                    for callback in self.callbacks_disc:
                        callback()

            if sock:
                sock.close()

        client_socket_class.__init__ = init
        client_socket_class.send = send
        client_socket_class._loop = loop
        setattr(client_socket_class, _PATCH_MARKER, True)

    logger.info("Installed Portal client wake-pipe backpressure fix")
    return True


def import_portal_with_wake_pipe_fix() -> ModuleType:
    """Import Portal and install the client wake-pipe fix once."""
    portal_module = importlib.import_module("portal")
    install_portal_wake_pipe_fix(portal_module)
    return portal_module


def portal_wake_pipe_stats(client: Any) -> Dict[str, int]:
    """Return queued-byte and capacity diagnostics for a Portal client pipe."""
    client_socket = getattr(client, "socket", None)
    read_descriptor = getattr(client_socket, "get_signal", None)
    write_descriptor = getattr(client_socket, "set_signal", None)
    if read_descriptor is None or write_descriptor is None:
        return {}

    pending = array.array("i", [0])
    fcntl.ioctl(read_descriptor, termios.FIONREAD, pending, True)
    capacity = fcntl.fcntl(write_descriptor, fcntl.F_GETPIPE_SZ)
    return {
        "signal_pending": int(pending[0]),
        "signal_capacity": int(capacity),
    }
