"""
UDP packet definitions using struct.

All packets: big-endian network byte order (!).
Magic bytes identify type and guard against corrupted packets.

Telemetry:  0xAB01  flight computer → ground station   (152 bytes)
Command:    0xAB02  ground station  → flight computer   (11 bytes)
"""

import struct
import time
from dataclasses import dataclass
from typing import Optional
import numpy as np


# ── Magic constants ───────────────────────────────────────────────────────────
MAGIC_TELEM   = 0xAB01
MAGIC_CMD     = 0xAB02

# ── Struct formats ────────────────────────────────────────────────────────────
# Telemetry: magic(H) seq(H) ts_sim_ms(I) pos(3d) quat(4d) vel(3d) omega(3d) fuel(d) ts_wall_ms(I)
_TELEM_FMT  = '!HHI3d4d3d3dddI'
_TELEM_SIZE = struct.calcsize(_TELEM_FMT)   # should be 152 bytes

# Command: magic(H) seq(H) ts_ms(I) type(B) payload(f)
_CMD_FMT    = '!HHIBf'
_CMD_SIZE   = struct.calcsize(_CMD_FMT)

# Command types
CMD_HOLD       = 0
CMD_LAND       = 1
CMD_SWITCH_PID = 2
CMD_SWITCH_RL  = 3
CMD_INJECT_FAIL = 4


@dataclass
class TelemetryPacket:
    seq:         int
    sim_time_ms: int
    position:    np.ndarray   # (3,)
    quaternion:  np.ndarray   # (4,)
    velocity:    np.ndarray   # (3,)
    ang_vel:     np.ndarray   # (3,)
    fuel:        float
    altitude:    float
    wall_time_ms: int


@dataclass
class CommandPacket:
    seq:         int
    ts_ms:       int
    cmd_type:    int
    payload:     float


def pack_telemetry(seq: int, state, sim_time: float) -> bytes:
    """Serialise a PhysicsState into a telemetry UDP packet."""
    pos  = state.position
    q    = state.quaternion
    vel  = state.velocity
    omega = state.ang_vel
    wall_ms = int(time.monotonic() * 1000) & 0xFFFFFFFF
    return struct.pack(
        _TELEM_FMT,
        MAGIC_TELEM, seq & 0xFFFF, int(sim_time * 1000) & 0xFFFFFFFF,
        *pos, *q, *vel, *omega,
        float(state.fuel_mass), float(state.altitude),
        wall_ms,
    )


def unpack_telemetry(data: bytes) -> Optional[TelemetryPacket]:
    if len(data) != _TELEM_SIZE:
        return None
    fields = struct.unpack(_TELEM_FMT, data)
    magic, seq, ts_sim, x, y, z, qw, qx, qy, qz, vx, vy, vz, p, q, r, fuel, alt, ts_wall = fields
    if magic != MAGIC_TELEM:
        return None
    return TelemetryPacket(
        seq          = seq,
        sim_time_ms  = ts_sim,
        position     = np.array([x, y, z]),
        quaternion   = np.array([qw, qx, qy, qz]),
        velocity     = np.array([vx, vy, vz]),
        ang_vel      = np.array([p, q, r]),
        fuel         = fuel,
        altitude     = alt,
        wall_time_ms = ts_wall,
    )


def pack_command(seq: int, cmd_type: int, payload: float = 0.0) -> bytes:
    ts = int(time.monotonic() * 1000) & 0xFFFFFFFF
    return struct.pack(_CMD_FMT, MAGIC_CMD, seq & 0xFFFF, ts, cmd_type, payload)


def unpack_command(data: bytes) -> Optional[CommandPacket]:
    if len(data) != _CMD_SIZE:
        return None
    magic, seq, ts, cmd, payload = struct.unpack(_CMD_FMT, data)
    if magic != MAGIC_CMD:
        return None
    return CommandPacket(seq=seq, ts_ms=ts, cmd_type=cmd, payload=payload)


def packet_sizes():
    return {'telemetry': _TELEM_SIZE, 'command': _CMD_SIZE}
