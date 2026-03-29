"""
CAD-derived vehicle parameters for the autonomous lander.

Bridge between the CAD model (exported as JSON) and the physics simulation.
All physical properties come from the CAD export: mass distribution, inertia
tensor, aerodynamic reference areas, and center-of-gravity offset.

Vehicle: Single-stage rocket lander (Falcon-9 booster style)
  - Cylindrical body with four landing legs
  - Single throttleable main engine
  - Cold-gas RCS thrusters for attitude control
  - CG slightly below geometric center (engine/fuel at bottom)
"""

import json
import numpy as np
from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple

import config


@dataclass
class VehicleParams:
    """Physical parameters derived from the CAD model (immutable after creation)."""

    # ── Mass ──────────────────────────────────────────────────────────────────
    mass_dry:          float   # kg
    mass_fuel_init:    float   # kg
    fuel_burn_rate:    float   # kg/N/s  (= 1 / (Isp * g))

    # ── Geometry ──────────────────────────────────────────────────────────────
    body_radius:       float   # m
    body_height:       float   # m
    cg_offset:         np.ndarray  # (3,) m, body frame

    # ── Inertia (principal + off-diagonal, all non-default first) ────────────
    Ixx:               float   # kg·m²
    Iyy:               float
    Izz:               float
    cd_body:           float   # drag coefficient
    reference_area:    float   # m²
    max_thrust:        float   # N
    min_throttle:      float
    max_rcs_torque:    float   # N·m per axis

    # ── Off-diagonal inertia (default 0 for axisymmetric vehicle) ────────────
    Ixy:               float = 0.0
    Ixz:               float = 0.0
    Iyz:               float = 0.0

    # ── Derived (computed in __post_init__, not passed in) ────────────────────
    inertia_tensor: np.ndarray = field(init=False, repr=False)
    inertia_inv:    np.ndarray = field(init=False, repr=False)

    def __post_init__(self):
        object.__setattr__(self, 'inertia_tensor', np.array([
            [self.Ixx, self.Ixy, self.Ixz],
            [self.Ixy, self.Iyy, self.Iyz],
            [self.Ixz, self.Iyz, self.Izz],
        ]))
        object.__setattr__(self, 'inertia_inv', np.linalg.inv(self.inertia_tensor))
        object.__setattr__(self, 'cg_offset', np.asarray(self.cg_offset, dtype=float))

    @property
    def total_mass(self) -> float:
        return self.mass_dry + self.mass_fuel_init

    def mass_at(self, fuel_remaining: float) -> float:
        return self.mass_dry + max(0.0, fuel_remaining)

    def summary(self) -> str:
        lines = [
            "╔══════════════════════════════════════╗",
            "║       VEHICLE PARAMETERS (CAD)       ║",
            "╠══════════════════════════════════════╣",
            f"║  Dry mass:        {self.mass_dry:7.2f} kg           ║",
            f"║  Fuel (init):     {self.mass_fuel_init:7.2f} kg           ║",
            f"║  Total mass:      {self.total_mass:7.2f} kg           ║",
            f"║  Max thrust:      {self.max_thrust:7.1f} N            ║",
            f"║  T/W (full):      {self.max_thrust/(self.total_mass*config.G):7.3f}              ║",
            f"║  T/W (dry):       {self.max_thrust/(self.mass_dry*config.G):7.3f}              ║",
            f"║  Body R×H:   {self.body_radius:.3f}×{self.body_height:.3f} m       ║",
            f"║  CG offset z:   {self.cg_offset[2]:+.4f} m          ║",
            f"║  Ixx={self.Ixx:.4f}  Iyy={self.Iyy:.4f}  Izz={self.Izz:.4f} ║",
            f"║  Drag Cd:         {self.cd_body:7.3f}              ║",
            "╚══════════════════════════════════════╝",
        ]
        return "\n".join(lines)


def _cylinder_inertia(mass: float, radius: float, height: float) -> Tuple[float, float, float]:
    Ixx = (1.0 / 12.0) * mass * (3.0 * radius**2 + height**2)
    Izz = 0.5 * mass * radius**2
    return Ixx, Ixx, Izz


def load_from_cad(json_path: str = None) -> VehicleParams:
    """Load from CAD JSON export; falls back to config defaults if not found."""
    if json_path is not None:
        p = Path(json_path)
        if p.exists():
            with open(p) as f:
                d = json.load(f)
            return VehicleParams(
                mass_dry=d['mass_dry'],
                mass_fuel_init=d['mass_fuel_init'],
                fuel_burn_rate=d['fuel_burn_rate'],
                body_radius=d['body_radius'],
                body_height=d['body_height'],
                cg_offset=np.array(d['cg_offset']),
                Ixx=d['Ixx'], Iyy=d['Iyy'], Izz=d['Izz'],
                Ixy=d.get('Ixy', 0.0),
                Ixz=d.get('Ixz', 0.0),
                Iyz=d.get('Iyz', 0.0),
                cd_body=d['cd_body'],
                reference_area=d['reference_area'],
                max_thrust=d['max_thrust'],
                min_throttle=d.get('min_throttle', 0.3),
                max_rcs_torque=d['max_rcs_torque'],
            )

    Ixx, Iyy, Izz = _cylinder_inertia(
        config.MASS_DRY, config.BODY_RADIUS, config.BODY_HEIGHT
    )
    return VehicleParams(
        mass_dry=config.MASS_DRY,
        mass_fuel_init=config.MASS_FUEL_INIT,
        fuel_burn_rate=1.0 / (config.SPECIFIC_IMPULSE * config.G),
        body_radius=config.BODY_RADIUS,
        body_height=config.BODY_HEIGHT,
        cg_offset=np.array(config.CG_OFFSET),
        Ixx=Ixx, Iyy=Iyy, Izz=Izz,
        cd_body=config.CD_BODY,
        reference_area=config.REF_AREA,
        max_thrust=config.MAX_THRUST,
        min_throttle=0.30,
        max_rcs_torque=config.MAX_RCS_TORQUE,
    )
