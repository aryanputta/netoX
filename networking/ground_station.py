"""
Ground station UDP client.

Receives telemetry from the flight computer at 50 Hz.
Simulates realistic network impairments:
  - Gaussian latency with configurable mean/std
  - Random packet loss
  - Out-of-order detection via sequence number tracking

Also sends command packets back to the flight computer.
"""

import socket
import threading
import time
import random
import numpy as np
from collections import deque

from networking.packets import (
    unpack_telemetry, pack_command,
    CMD_SWITCH_PID, CMD_SWITCH_RL
)
import config


class GroundStationClient(threading.Thread):
    def __init__(
        self,
        shared,
        host:             str   = config.TELEM_HOST,
        telem_port:       int   = config.TELEM_PORT,
        cmd_port:         int   = config.CMD_PORT,
        latency_mean_ms:  float = config.NET_LATENCY_MEAN_MS,
        latency_std_ms:   float = config.NET_LATENCY_STD_MS,
        packet_loss_rate: float = config.NET_LOSS_RATE,
    ):
        super().__init__(name="GroundStation", daemon=True)
        self.shared           = shared
        self.host             = host
        self.telem_port       = telem_port
        self.cmd_port         = cmd_port
        self.latency_mean     = latency_mean_ms / 1000.0
        self.latency_std      = latency_std_ms  / 1000.0
        self.loss_rate        = packet_loss_rate
        self._cmd_seq         = 0

        # Network statistics
        self.packets_received  = 0
        self.packets_dropped   = 0
        self.packets_ooo       = 0    # out-of-order
        self._latency_hist     = deque(maxlen=500)
        self._last_seq         = -1
        self._seq_gaps         = deque(maxlen=100)

    @property
    def mean_latency_ms(self) -> float:
        if not self._latency_hist:
            return 0.0
        return float(np.mean(self._latency_hist)) * 1000.0

    @property
    def drop_rate_pct(self) -> float:
        total = self.packets_received + self.packets_dropped
        return 100.0 * self.packets_dropped / total if total > 0 else 0.0

    def run(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((self.host, self.telem_port))
        sock.settimeout(0.05)

        # Separate socket for sending commands
        self._cmd_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        shared = self.shared
        while not shared.shutdown.is_set():
            try:
                data, _ = sock.recvfrom(256)
            except socket.timeout:
                continue
            except OSError:
                break

            # ── Simulate packet loss ──────────────────────────────────────
            if random.random() < self.loss_rate:
                self.packets_dropped += 1
                shared.record_net_drop()
                continue

            # ── Simulate network latency ──────────────────────────────────
            delay = max(0.0, random.gauss(self.latency_mean, self.latency_std))
            # Spike injection
            if config.COMM_DELAY_SPIKE_T is not None:
                sim_t = shared.read_sim_time()
                if abs(sim_t - config.COMM_DELAY_SPIKE_T) < 0.5:
                    delay += 0.500  # 500 ms spike
            time.sleep(delay)
            self._latency_hist.append(delay)

            # ── Parse packet ───────────────────────────────────────────────
            pkt = unpack_telemetry(data)
            if pkt is None:
                continue

            # ── Sequence number tracking ──────────────────────────────────
            if self._last_seq >= 0:
                expected = (self._last_seq + 1) & 0xFFFF
                if pkt.seq != expected:
                    gap = (pkt.seq - self._last_seq) & 0xFFFF
                    if gap > 1:
                        self._seq_gaps.append(gap)
                    if gap > 32768:   # wrapped backwards → out-of-order
                        self.packets_ooo += 1
                        continue      # discard stale packets

            self._last_seq = pkt.seq
            self.packets_received += 1

            # ── Publish to shared state ───────────────────────────────────
            recv_wall = time.monotonic()
            shared.update_ground_telemetry(pkt, recv_wall, delay)
            shared.record_net_latency(delay * 1000.0)

        sock.close()
        self._cmd_sock.close()

    def send_switch_pid(self):
        self._send_command(CMD_SWITCH_PID)

    def send_switch_rl(self):
        self._send_command(CMD_SWITCH_RL)

    def _send_command(self, cmd_type: int, payload: float = 0.0):
        data = pack_command(self._cmd_seq, cmd_type, payload)
        self._cmd_seq = (self._cmd_seq + 1) & 0xFFFF
        try:
            self._cmd_sock.sendto(data, (self.host, self.cmd_port))
        except OSError:
            pass
