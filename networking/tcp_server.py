"""
TCP telemetry server — reliable delivery alternative to UDP.

Keeps a persistent connection per client. Sends framed telemetry packets:
  [4-byte length][payload bytes]

Useful when packet loss is unacceptable (e.g., post-flight analysis or
ground-station replay). UDP remains the primary real-time path.
"""

import socket
import struct
import threading
import time
import logging

from networking.packets import pack_telemetry, MAGIC_TELEM
import config

log = logging.getLogger(__name__)


class TcpTelemetryServer(threading.Thread):
    def __init__(self, shared, host: str = config.TELEM_HOST,
                 port: int = 5010, rate_hz: float = 50.0):
        super().__init__(name="TcpServer", daemon=True)
        self.shared    = shared
        self.host      = host
        self.port      = port
        self.interval  = 1.0 / rate_hz
        self._seq      = 0
        self._clients: list[socket.socket] = []
        self._client_lock = threading.Lock()

    def run(self):
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind((self.host, self.port))
        srv.listen(8)
        srv.settimeout(0.5)
        log.info(f"TCP telemetry server listening on {self.host}:{self.port}")

        # Accept loop in background thread
        accept_t = threading.Thread(target=self._accept_loop, args=(srv,), daemon=True)
        accept_t.start()

        # Broadcast loop
        while not self.shared.shutdown.is_set():
            t0 = time.perf_counter()

            state  = self.shared.read_physics()
            sim_t  = self.shared.read_sim_time()
            data   = pack_telemetry(self._seq, state, sim_t)
            frame  = struct.pack('!I', len(data)) + data
            self._seq = (self._seq + 1) & 0xFFFF

            dead = []
            with self._client_lock:
                for c in self._clients:
                    try:
                        c.sendall(frame)
                    except OSError:
                        dead.append(c)
                for c in dead:
                    self._clients.remove(c)
                    log.info("TCP client disconnected")

            elapsed = time.perf_counter() - t0
            sleep_t = self.interval - elapsed
            if sleep_t > 0:
                time.sleep(sleep_t)

        srv.close()

    def _accept_loop(self, srv: socket.socket):
        while not self.shared.shutdown.is_set():
            try:
                conn, addr = srv.accept()
                conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                with self._client_lock:
                    self._clients.append(conn)
                log.info(f"TCP client connected from {addr}")
            except socket.timeout:
                pass
            except OSError:
                break


class TcpGroundStationClient:
    """Blocking TCP client that reads framed telemetry from TcpTelemetryServer."""

    def __init__(self, host: str = config.TELEM_HOST, port: int = 5010):
        self.host = host
        self.port = port
        self._sock: socket.socket | None = None

    def connect(self, timeout: float = 5.0):
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.settimeout(timeout)
        self._sock.connect((self.host, self.port))
        self._sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        log.info(f"TCP connected to {self.host}:{self.port}")

    def read_packet(self) -> bytes | None:
        """Read one framed packet. Returns raw payload bytes."""
        try:
            hdr = self._recvall(4)
            if not hdr:
                return None
            n = struct.unpack('!I', hdr)[0]
            return self._recvall(n)
        except (OSError, struct.error):
            return None

    def _recvall(self, n: int) -> bytes:
        buf = bytearray()
        while len(buf) < n:
            chunk = self._sock.recv(n - len(buf))
            if not chunk:
                raise OSError("connection closed")
            buf.extend(chunk)
        return bytes(buf)

    def close(self):
        if self._sock:
            self._sock.close()
