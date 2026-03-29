"""
Global configuration for the autonomous landing system.
All physical constants, tuning parameters, and network settings live here.
"""

# ─── Simulation ───────────────────────────────────────────────────────────────
DT_PHYSICS   = 0.005   # 200 Hz physics integration step (s)
DT_TELEMETRY = 0.020   # 50 Hz telemetry broadcast (s)
DT_VIZ       = 0.100   # 10 Hz visualization refresh (s)
G            = 9.80665  # gravitational acceleration (m/s^2)
RHO_AIR      = 1.225   # air density at sea level (kg/m^3)

# ─── Vehicle (rocket-style lander, derived from CAD) ─────────────────────────
MASS_DRY         = 8.0    # kg (without fuel)
MASS_FUEL_INIT   = 4.0    # kg (initial fuel load)
MAX_THRUST       = 200.0  # N (main engine max thrust)
SPECIFIC_IMPULSE = 200.0  # s (effective Isp)
BODY_RADIUS      = 0.15   # m (cylinder radius)
BODY_HEIGHT      = 0.60   # m (cylinder height)
CD_BODY          = 0.47   # drag coefficient (bluff cylinder)
REF_AREA         = 0.071  # m^2 (pi * r^2)
# Inertia for solid cylinder: Ixx=Iyy=(1/12)*m*(3r^2+h^2), Izz=(1/2)*m*r^2
# With m=MASS_DRY=8 kg, r=0.15, h=0.60:
IXX              = 0.285  # kg*m^2
IYY              = 0.285  # kg*m^2
IZZ              = 0.090  # kg*m^2
MAX_RCS_TORQUE   = 12.0   # N*m (max attitude torque per axis from RCS)
CG_OFFSET        = (0.0, 0.0, -0.05)  # m from geometric center (engine mass at bottom)

# ─── Initial Conditions ───────────────────────────────────────────────────────
INIT_ALTITUDE    = 100.0  # m
INIT_VZ          = -3.0   # m/s (initial descent rate)
INIT_VX          = 1.5    # m/s (small horizontal drift)
INIT_VY          = -1.0   # m/s
INIT_PITCH       = 0.05   # rad (slight initial tilt)
INIT_X           = 5.0    # m (lateral offset from target)
INIT_Y           = -3.0   # m

# ─── Target ───────────────────────────────────────────────────────────────────
TARGET_X = 0.0   # m
TARGET_Y = 0.0   # m
TARGET_Z = 0.0   # m

# ─── PID Gains ────────────────────────────────────────────────────────────────
# Altitude (z) controller → throttle
KP_Z = 4.5;  KI_Z = 0.8;  KD_Z = 3.0
# Horizontal position → desired tilt angle
KP_XY = 1.8; KI_XY = 0.15; KD_XY = 2.2
# Attitude controllers → RCS torques (N*m)
KP_ROLL  = 8.0;  KI_ROLL  = 0.3;  KD_ROLL  = 1.2
KP_PITCH = 8.0;  KI_PITCH = 0.3;  KD_PITCH = 1.2
KP_YAW   = 4.0;  KI_YAW   = 0.1;  KD_YAW   = 0.6
# Integral anti-windup limits
ICLAMP_Z   = 30.0
ICLAMP_XY  = 0.4   # rad (attitude setpoint clamp)
ICLAMP_ATT = 5.0   # N*m

# ─── RL / Neural Network ──────────────────────────────────────────────────────
RL_STATE_DIM    = 14   # [pos_err(3), vel(3), euler(3), omega(3), fuel_frac, altitude]
RL_HIDDEN_1     = 128
RL_HIDDEN_2     = 64
RL_ACTION_DIM   = 4    # [throttle, tau_x, tau_y, tau_z] normalised to [-1,1]
RL_LR           = 3e-4
RL_GAMMA        = 0.99
RL_BC_EPOCHS    = 40
RL_BATCH_SIZE   = 256
RL_REWARD_WEIGHTS = {
    "pos":     3.0,
    "vel":     1.5,
    "att":     1.0,
    "omega":   0.5,
    "effort":  0.05,
    "alive":   0.1,
    "landing": 200.0,
}

# ─── Sensor Noise ─────────────────────────────────────────────────────────────
IMU_ACCEL_NOISE_STD  = 0.05   # m/s^2
IMU_GYRO_NOISE_STD   = 0.002  # rad/s
IMU_ACCEL_BIAS_STD   = 0.01   # m/s^2 (slowly drifting)
GPS_POS_NOISE_STD    = 1.5    # m
GPS_VEL_NOISE_STD    = 0.10   # m/s
GPS_RATE_HZ          = 5.0    # GPS update rate
BARO_ALT_NOISE_STD   = 0.50   # m
BARO_RATE_HZ         = 50.0

# ─── EKF ──────────────────────────────────────────────────────────────────────
EKF_PROCESS_NOISE_POS    = 0.01
EKF_PROCESS_NOISE_VEL    = 0.1
EKF_PROCESS_NOISE_QUAT   = 0.001
EKF_PROCESS_NOISE_BIAS   = 0.001

# ─── Network ──────────────────────────────────────────────────────────────────
TELEM_HOST          = "127.0.0.1"
TELEM_PORT          = 5005
CMD_PORT            = 5006
NET_LATENCY_MEAN_MS = 20.0
NET_LATENCY_STD_MS  = 6.0
NET_LOSS_RATE       = 0.02   # 2% packet loss

# ─── Failures / Disturbances ──────────────────────────────────────────────────
WIND_SPEED_MEAN      = 3.0   # m/s (steady wind)
WIND_SPEED_STD       = 1.5   # m/s (gust std dev)
WIND_DIRECTION       = 0.0   # rad (East)
ENGINE_DEGRADE_START = 50.0  # s (sim time when degradation begins)
ENGINE_DEGRADE_PCT   = 0.0   # 0=disabled, 0.3=30% thrust loss
COMM_DELAY_SPIKE_T   = None  # None or float (sim time to inject 500ms spike)
