# netoX

netoX is a high-fidelity simulation and control environment designed for testing autonomous vertical landing algorithms. By integrating a 6-DOF physics engine with real-world networking constraints, it provides a rigorous testing ground for comparing classical control theory against modern Reinforcement Learning.

System Overview
╔═══════════════════════════════════════════════════════════════════════╗
║                    FLIGHT COMPUTER  (SimulationEngine)                ║
║                                                                       ║
║  ┌─────────────┐   ┌──────────────────┐   ┌─────────────────────┐   ║
║  │  CAD Model  │──▶│  Physics Engine  │──▶│  Sensor Model       │   ║
║  │(lander_cad  │   │  (RK4, 200 Hz)   │   │  IMU + GPS + Baro   │   ║
║  │   .json)    │   │  6-DOF dynamics  │   │  noise + bias drift │   ║
║  └─────────────┘   └──────────────────┘   └──────────┬──────────┘   ║
║    mass, inertia,         ▲                           │              ║
║    drag, CG offset        │ control command           │ noisy sensor ║
║                     ┌─────┴──────┐                    │  readings    ║
║                     │  Control   │                    ▼              ║
║                     │  Thread    │            ┌───────────────┐      ║
║                     │ (200 Hz)   │            │ Extended KF   │      ║
║                     │PID  or  RL │◀───────────│ state estimat.│      ║
║                     └────────────┘            └───────────────┘      ║
╠═══════════════════════════════════════════════════════════════════════╣
║                      NETWORKING  (UDP, 50 Hz)                        ║
║                                                                       ║
║   TelemetryServer ─────────────────────────────▶ GroundStationClient ║
║   (physics state)     UDP port 5005              (latency + loss sim) ║
║                                                                       ║
║   GroundStationClient ◀──────────────────────── TelemetryServer      ║
║   (cmd: switch ctrl)    UDP port 5006            (parses + applies)   ║
╠═══════════════════════════════════════════════════════════════════════╣
║                    GROUND STATION                                     ║
║                                                                       ║
║  ┌───────────────────────────────────────────────────────────────┐   ║
║  │                  Visualization Dashboard                       │   ║
║  │  ┌─────────────┐ ┌───────────────┐ ┌──────────────────────┐  │   ║
║  │  │ 3D Trajectory│ │ Alt + Speed   │ │  Euler Angles        │  │   ║
║  │  │ (body axes)  │ │ time series   │ │  roll pitch yaw      │  │   ║
║  │  └─────────────┘ └───────────────┘ └──────────────────────┘  │   ║
║  │  ┌─────────────┐ ┌───────────────┐ ┌──────────────────────┐  │   ║
║  │  │ Throttle +  │ │ Network Stats │ │  ISE: PID vs RL      │  │   ║
║  │  │ Fuel gauge  │ │ latency hist  │ │  running comparison  │  │   ║
║  │  └─────────────┘ └───────────────┘ └──────────────────────┘  │   ║
║  │  Status bar: T+ alt speed fuel latency controller             │   ║
║  │  Keys: [P]PID [R]RL [F]Fail [W]Wind [Q]Quit                  │   ║
║  └───────────────────────────────────────────────────────────────┘   ║
╠═══════════════════════════════════════════════════════════════════════╣
║                   EVALUATION & METRICS                               ║
║  ISE · ITAE · IAE · settling time · max tilt · touchdown speed       ║
║  touchdown position error · fuel used · control effort               ║
╚═══════════════════════════════════════════════════════════════════════╝
Data Flow
CAD JSON ──▶ VehicleParams
              │
              ├──▶ PhysicsEngine (RK4 @ 200 Hz)
              │         │ PhysicsState [14-dim]
              │         ├──▶ SensorModel → SensorReadings
              │         │         │
              │         │         ▼
              │         │    ExtendedKalmanFilter → estimated state (13-dim)
              │         │
              │         ├──▶ TelemetryServer → UDP 5005 → GroundStationClient
              │         │                                       │
              │         │                             SharedState.ground_telemetry
              │         │
              │         └──▶ ControlThread
              │                   │  reads PhysicsState (or EKF estimate)
              │                   │  PID: cascaded altitude/position/attitude
              │                   │  RL:  NumpyMLP forward pass (~0.05ms)
              │                   └──▶ control [throttle, τx, τy, τz]
              │                             │
              │                             └──▶ SharedState.control
              │                                       │
              └─────────────────────────────────────▶ PhysicsEngine reads
Thread Architecture
Thread	Rate	Role
SimulationEngine	200 Hz	RK4 integration, writes physics state
ControlThread	200 Hz	PID or RL, writes control command
WindModel	20 Hz	Stochastic gust model, writes wind vector
TelemetryServer	50 Hz	UDP broadcast of physics state
GroundStation	async	UDP receive, simulates latency + loss
Dashboard	10 Hz	matplotlib FuncAnimation (main thread)
State Vector (14-dim float64)
Index  Symbol    Unit    Description
─────  ────────  ──────  ────────────────────────────────
 0     x         m       East position (ENU frame)
 1     y         m       North position
 2     z         m       Up position  (z=0 is landing pad)
 3     vx        m/s     East velocity
 4     vy        m/s     North velocity
 5     vz        m/s     Up velocity (negative = descending)
 6     q_w       -       Quaternion scalar (body→world)
 7     q_x       -       Quaternion x
 8     q_y       -       Quaternion y
 9     q_z       -       Quaternion z
10     ωx        rad/s   Body roll rate
11     ωy        rad/s   Body pitch rate
12     ωz        rad/s   Body yaw rate
13     m_fuel    kg      Remaining propellant mass
Control Architecture (Cascaded PID)
  position error          desired attitude
  ┌────────┐  ┌──────────────┐  ┌──────────────┐  ┌───────────┐
  │ Alt PID│─▶│ throttle     │  │attitude error│  │ throttle  │
  │ x PID  │─▶│ roll_des     │─▶│ roll PID     │─▶│ τ_roll    │
  │ y PID  │─▶│ pitch_des    │  │ pitch PID    │  │ τ_pitch   │
  │        │  │ yaw_des=0    │  │ yaw PID      │  │ τ_yaw     │
  └────────┘  └──────────────┘  └──────────────┘  └───────────┘
   50 Hz outer                    200 Hz inner
UDP Packet Format
Telemetry (flight → ground, 152 bytes):

  [magic 2B][seq 2B][ts_sim 4B][pos 24B][quat 32B][vel 24B][omega 24B][fuel 8B][alt 8B][ts_wall 4B]
  magic = 0xAB01, big-endian
Command (ground → flight, 11 bytes):

  [magic 2B][seq 2B][ts 4B][type 1B][payload 4B]
  magic = 0xAB02
  types: 0=HOLD 1=LAND 2=SWITCH_PID 3=SWITCH_RL 4=INJECT_FAIL
Neural Network Policy (Pure NumPy)
Input (14):  pos_error(3) vel(3) euler(3) omega(3) fuel_frac alt_norm
             ──────────────────────────────────────────────────────────
Hidden 1 (128): Linear → Tanh
Hidden 2 (64):  Linear → Tanh
             ──────────────────────────────────────────────────────────
Output (4):  throttle (Sigmoid) │ τ_x τ_y τ_z (Tanh × max_torque)

Parameters:  14×128 + 128×64 + 64×4 = ~11K  (forward pass ~0.05 ms)
Training:

Behavioral Cloning — 200 PID episodes → supervised MSE on (state, action) pairs
REINFORCE — 50 episodes online policy gradient with advantage normalisation
Engineering Tradeoffs
Factor	PID	RL / BC
Interpretability	Full — gains are tunable	Black box
Tuning effort	Manual per disturbance	Automatic via training
Disturbance rejection	Fixed bandwidth	Learned from data
Fuel efficiency	Gravity-FF + proportional	Learns optimal throttle curve
Failure adaptation	Fixed — degrades with fault	May generalise if trained on
augmented failure data
Latency robustness	Outer-loop rate limits impact	Inner loop learns delay
Network latency effect on control performance:

At μ=20ms latency: effective closed-loop bandwidth ≈ 1/(2×0.020+0.010) ≈ 20 Hz
At μ=200ms latency (spike): attitude loop becomes marginally stable
Packet loss >5%: altitude controller develops steady-state offset
CAD parameter sensitivity:

CG offset ±0.05m: attitude coupling with throttle → requires I_att feed-forward
Mass +20%: hover throttle increases → less margin for attitude authority
Ixx increase: lower natural roll frequency → PID gains must be reduced
Docker Deployment
# Local development with visualization
pip install -r requirements.txt
python main.py --controller pid

# Train RL model
python train_rl.py --episodes 500

# Run comparison
python main.py --compare

# Docker headless
docker compose up flight-computer

# Docker training
docker compose --profile training up trainer

# Scale to multiple ground stations
docker compose up --scale ground-station=3
