"""
UDP telemetry server — runs on the "flight computer" side.

Broadcasts the current physics state at 50 Hz to any listening ground station.
Also listens for command packets from the ground station on CMD_PORT.
"""

import socket
import threading
import time
import numpy as np

from networking.packets import pack_telemetry, unpack_command, CMD_SWITCH_PID, CMD_SWITCH_RL
import config


class TelemetryServer(threading.Thread):
    def __init__(self, shared, host: str = config.TELEM_HOST,
                 telem_port: int = config.TELEM_PORT,
                 cmd_port:   int = config.CMD_PORT,
                 rate_hz:    float = 50.0):
        super().__init__(name="TelemServer", daemon=True)
        self.shared      = shared
        self.host        = host
        self.telem_port  = telem_port
        self.cmd_port    = cmd_port
        self.interval    = 1.0 / rate_hz
        self._seq        = 0
        self.packets_sent = 0

        # UDP sockets created in run() so they live on the right thread
        self._tx_sock = None
        self._rx_sock = None

    def run(self):
        # TX: broadcast telemetry
        self._tx_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._tx_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

        # RX: receive commands
        self._rx_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._rx_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._rx_sock.bind((self.host, self.cmd_port))
        self._rx_sock.settimeout(0.001)

        # Start command listener as a sub-thread so TX loop stays tight
        cmd_thread = threading.Thread(target=self._cmd_recv_loop, daemon=True)
        cmd_thread.start()

        shared = self.shared
        while not shared.shutdown.is_set():
            t0 = time.perf_counter()

            state   = shared.read_physics()
            sim_t   = shared.read_sim_time()
            payload = pack_telemetry(self._seq, state, sim_t)
            self._seq = (self._seq + 1) & 0xFFFF

            try:
                self._tx_sock.sendto(payload, (self.host, self.telem_port))
                self.packets_sent += 1
            except OSError:
                pass

            elapsed = time.perf_counter() - t0
            sleep_t = self.interval - elapsed
            if sleep_t > 0:
                time.sleep(sleep_t)

        self._tx_sock.close()
        self._rx_sock.close()

    def _cmd_recv_loop(self):
        """Parse incoming command packets and update shared state."""
        while not self.shared.shutdown.is_set():
            try:
                data, _ = self._rx_sock.recvfrom(64)
                pkt = unpack_command(data)
                if pkt is None:
                    continue
                if pkt.cmd_type == CMD_SWITCH_PID:
                    self.shared.set_controller('PID')
                elif pkt.cmd_type == CMD_SWITCH_RL:
                    self.shared.set_controller('RL')
            except socket.timeout:
                pass
            except OSError:
                break
