# RocketSim — Build Plan

A 6-DOF model-rocket flight simulator in Python, built in stages. Each stage ends with passing tests and a short physics note, so every equation can be explained and defended.

**End goal:** a validated simulator that:

- predicts a real rocket's flight (apogee, velocity, drift, landing point) and matches OpenRocket and RocketPy within stated error bounds;
- runs Monte Carlo dispersion analysis (landing zone, apogee spread);
- outputs realistic noisy sensor data (IMU + barometer). A separate flight-computer project (state estimation, apogee detection, parachute deployment) consumes that data.

**Rule for Claude Code:** do ONE stage at a time. Do not start the next stage until every acceptance criterion of the current one passes and the user has signed off.

## 0. Conventions (fixed — do not change without asking)

| Item | Convention |
|---|---|
| Units | SI everywhere internally (m, kg, s, N, Pa, K, rad). Convert only at I/O. |
| Inertial frame | ENU (x = East, y = North, z = Up), origin at the base of the launch rail. Flat, non-rotating Earth. |
| Body frame | x_b along the rocket axis, pointing toward the nose; y_b, z_b complete a right-handed frame. |
| Axial positions | Geometry positions (CG, CP, components) are measured from the nose tip, positive aft. The body-frame vector from the CG to a point at axial position x is [x_cg − x, 0, 0]. |
| Attitude | Unit quaternion q = [w, x, y, z] (scalar first) rotating body → inertial. R(q) maps body vectors to inertial. |
| Launch attitude | Rail elevation θ from horizontal and heading ψ clockwise from North. Initial body x-axis in ENU = [cosθ·sinψ, cosθ·cosψ, sinθ]. |
| Gravity | Constant g = 9.80665 m/s², inertial vector [0, 0, −g]. |
| Angles in config | Degrees in YAML, converted to radians on load. |

6-DOF state vector (13): [r_x, r_y, r_z, v_x, v_y, v_z, q_w, q_x, q_y, q_z, ω_x, ω_y, ω_z] — position and velocity are inertial; ω is angular velocity in the body frame. Mass, CG and inertia are functions of time (motor burn), not state variables.

## 1. Tech stack

- Python 3.11+
- Runtime: numpy, scipy, matplotlib, pyyaml
- Dev: pytest, ruff
- Optional extra `[validation]`: rocketpy, used only as an external reference oracle in the validation tests, never imported by the simulator itself.
- Packaging: pyproject.toml, installed with `pip install -e ".[dev]"`
- CLI: `python -m rocketsim <command> <config.yaml>`

Do not add any other dependency without asking.

## 2. Repository layout

```
rocketsim/
├── CLAUDE.md
├── PLAN.md
├── README.md
├── pyproject.toml
├── rocketsim/
│   ├── __init__.py
│   ├── constants.py        # g0, R_air, gamma, etc.
│   ├── atmosphere.py       # ISA: T, p, rho, speed of sound, viscosity
│   ├── motor.py            # .eng parser, thrust(t), propellant mass/CG/inertia(t)
│   ├── geometry.py         # components (nose, tube, fins, mass items), mass properties
│   ├── frames.py           # quaternion math, R(q), launch attitude
│   ├── wind.py             # constant / power-law / gust models
│   ├── aero/
│   │   ├── barrowman.py    # CN_alpha and CP per component
│   │   ├── drag.py         # Cd(Mach, Re) component buildup + table override
│   │   └── damping.py      # pitch/yaw and roll damping moments
│   ├── dynamics/
│   │   ├── dof1.py         # vertical
│   │   ├── dof3.py         # planar (x, z, pitch)
│   │   └── dof6.py         # full rigid body
│   ├── integrators.py      # fixed-step RK4 + solve_ivp wrapper
│   ├── flight.py           # phase manager: rail → powered → coast → recovery → landed
│   ├── recovery.py         # drogue / main parachutes
│   ├── sensors.py          # IMU + barometer models
│   ├── montecarlo.py       # dispersions, parallel runs, statistics
│   ├── io.py               # YAML config loading/validation, CSV export
│   ├── plots.py
│   └── cli.py
├── configs/
│   └── example_rocket.yaml
├── motors/                 # .eng files from thrustcurve.org (user downloads)
├── reference/
│   ├── sim_1dof.py         # existing standalone 1-DOF starter (test oracle, do not edit)
│   └── openrocket/         # CSVs exported from OpenRocket for validation
├── tests/
├── docs/
│   ├── NOTES_stage1.md … NOTES_stage6.md
│   └── VALIDATION.md
└── outputs/                # gitignored
```

## 3. Key design decisions (with reasons)

- **Phase-based integration.** Flight is split into phases: on rail → powered → coast → drogue → main → landed. Each phase is integrated separately with `scipy.integrate.solve_ivp` (method DOP853, rtol=atol=1e-8) and terminal events end it: rail exit, burnout, apogee (v_z crosses 0 downward), main-deploy altitude, landing (z crosses 0 downward). *Why:* integrators lose accuracy when stepping across discontinuities such as burnout or chute deployment. Events also find apogee exactly instead of to the nearest step.
- **Keep a fixed-step RK4 too.** It serves as a cross-check in tests, and it is what generates fixed-rate sensor output (a real flight computer samples at a fixed rate).
- **Drag table override.** Cd can come from the program's own component-buildup model OR from a user-supplied Cd(Mach) table, e.g. exported from OpenRocket. *Why:* during validation, importing OpenRocket's Cd separates dynamics errors from aerodynamics errors. Debug one thing at a time.
- **Parachute phases use point-mass dynamics.** Under a parachute, attitude doesn't matter for the trajectory; drag acts along the air-relative velocity. This is standard practice and avoids stiff, unstable attitude dynamics.
- **Validity range is explicit.** The aero model targets subsonic flight (Mach < 0.8). The simulator prints a warning if Mach > 0.8 is reached instead of silently producing wrong numbers.
- **Reproducible randomness.** Every random draw (Monte Carlo, sensor noise) uses an explicitly seeded `numpy.random.Generator`. Never the global RNG.
- **Config-driven.** The rocket, motor, launch site, wind and dispersions all live in YAML. No rocket-specific numbers in code.

## 4. Stages

Each stage lists: Tasks → Tests → Acceptance criteria → Physics note topics. Rough effort estimates assume a few evenings per week and are guesses, not commitments.

### Stage 0 — Project skeleton (≈1 evening)

**Tasks**

- Create the layout above, pyproject.toml, .gitignore (outputs/, `__pycache__`, .venv), README with setup and run instructions.
- Copy the existing sim_1dof.py into reference/ unchanged.
- constants.py and atmosphere.py: ISA troposphere and lower stratosphere (0–20 km) — T, p, ρ, speed of sound a = sqrt(γ·R·T), dynamic viscosity (Sutherland's law).
- motor.py: RASP .eng parser (comment lines start with `;`; header line = name, diameter mm, length mm, delays, propellant mass kg, total mass kg, manufacturer; then time thrust pairs). thrust(t) by linear interpolation, zero outside the burn. Propellant mass depletes in proportion to impulse delivered.
- Set up pytest and ruff; add a GitHub Actions workflow if the repo is on GitHub.

**Tests**

- ISA: ρ(0) = 1.225 kg/m³ (±0.001), T(11 km) = 216.65 K, a(0) ≈ 340.3 m/s.
- Motor: total impulse from the parsed curve equals the trapezoidal integral; propellant mass is the full value at t=0 and 0 after burnout; thrust(−1) = thrust(t_burn+1) = 0.
- Parser rejects a malformed file with a clear error message.

**Acceptance:** pytest green, ruff check clean, `pip install -e ".[dev]"` works from scratch.

### Stage 1 — 1-DOF vertical, package-quality (≈1 week)

**Tasks**

- dynamics/dof1.py: state [z, v_z], forces = thrust − drag − weight, constant Cd.
- flight.py (first version): phases on pad → powered → coast → parachute → landed, with events. The rocket stays on the pad until thrust > weight.
- io.py: minimal YAML config (dry mass, diameter, Cd, motor file, chute Cd·A). outputs/<run>/trajectory.csv.
- cli.py: `python -m rocketsim run configs/example_rocket.yaml` → prints summary (apogee, time to apogee, max velocity, max acceleration in g, landing velocity, flight time) and saves plots.

**Tests (physics invariants — these are what make it trustworthy)**

- Vacuum coast: thrust 0, ρ = 0, initial v₀ → apogee = v₀²/(2g) within 1e-6 relative.
- Rocket equation: gravity 0, ρ = 0, constant thrust F with constant mass flow ṁ = F/v_e → Δv = v_e·ln(m₀/m_f) within 1e-6.
- Terminal velocity: under a chute at constant ρ, descent velocity converges to sqrt(2mg / (ρ·CdA)).
- Oracle: same inputs as reference/sim_1dof.py → apogee matches within 0.5%.
- Convergence: tightening tolerances 10× changes apogee by < 0.01%.

**Acceptance:** all tests green; apogee for the example rocket printed; docs/NOTES_stage1.md written.

**Physics note topics:** Newton's 2nd law with variable mass, why RK4/DOP853 beat Euler, ISA derivation, drag equation, terminal velocity.

### Stage 2 — 3-DOF planar flight + real aerodynamics (≈2 weeks)

**Tasks**

- geometry.py: components — nose cone (cone/ogive/elliptical/Von Kármán), body tube, fin set (trapezoidal: root chord, tip chord, semispan, sweep, count, thickness), and point/cylinder mass items. Computes total mass, CG(t) and pitch inertia(t) using the parallel-axis theorem; the motor contributes casing + propellant mass that decreases over time.
- aero/barrowman.py (subsonic, small α), each component's CN_α and CP location:
  - Nose: CN_α = 2; X_CP = 0.666·L (cone), 0.466·L (ogive), 0.5·L (Von Kármán / elliptical — check against the source).
  - Fins: CN_α = [1 + R/(s+R)] · [4N(s/d)²] / [1 + sqrt(1 + (2·l_f/(a+b))²)], where a = root chord, b = tip chord, s = semispan, R = body radius, d = reference diameter, N = fin count, l_f = mid-chord line length.
  - Fin CP: X_F = X_B + m(a+2b)/(3(a+b)) + (1/6)[(a+b) − ab/(a+b)], where X_B = fin leading-edge position and m = sweep distance.
  - Total CP = Σ(CN_α,i · X_i) / Σ CN_α,i.
  - Cite the source in docstrings: Barrowman (1967) and Niskanen, *Development of an Open Source Model Rocket Simulation Software* (OpenRocket technical documentation), ch. 3. Verify each formula against the source before implementing; the formulas above are a guide.
- aero/drag.py: Cd(Mach, Re) component buildup per Niskanen §3.4 (skin friction with Reynolds dependence and compressibility correction, nose pressure drag, fin pressure drag, base drag). Plus table override Cd(Mach) from CSV.
- dynamics/dof3.py: state [x, z, v_x, v_z, θ, θ̇] (pitch θ from horizontal). Aero normal force at the CP produces a pitching moment about the CG. Include a pitch damping moment (Niskanen §3.2.3).
- Launch rail: motion constrained along the rail and no rotation until the rail lug clears the rail end. Report rail-exit velocity, and warn if < 15 m/s (common hobby guideline).
- wind.py: constant wind and power-law profile w(z) = w_ref·(z/z_ref)^(1/7).
- Stability margin (calibers) = (X_CP − X_CG)/d at rail exit and over time. Warn if < 1.0.

**Tests**

- Barrowman: the CP of a reference rocket (e.g. one from OpenRocket's example files) matches OpenRocket's reported CP within 2%.
- Mass properties: hollow tube and solid cylinder inertias match analytic formulas; CG of a two-mass system is correct.
- Sign test: a stable rocket (CP aft of CG) given an initial angle of attack rotates to reduce it, and the oscillation decays.
- Weathercocking: with crosswind, the rocket pitches into the wind.
- With zero wind and a vertical rail, the 3-DOF apogee matches Stage 1 within 0.1% (same Cd table).

**Acceptance:** tests green; summary now also prints rail-exit velocity, stability margin and downrange drift; docs/NOTES_stage2.md.

**Physics note topics:** what CP and CG are and why CP must be aft; Barrowman's assumptions and their limits; where each drag component comes from; weathercocking.

### Stage 3 — Full 6-DOF (≈2–3 weeks; the hardest stage)

**Tasks**

- frames.py: quaternion multiply, conjugate, normalize, R(q), launch attitude from (θ, ψ), Euler angles for plotting only.
- geometry.py: full inertia tensor (axial I_xx, lateral I_yy = I_zz, axisymmetric).
- dynamics/dof6.py, per derivative evaluation:
  - Normalize q before use (protects against numerical drift).
  - Air-relative velocity: v_air = v − w(z) (inertial) → body: v_air_b = Rᵀ·v_air.
  - Total angle of attack α = atan2(sqrt(v_by² + v_bz²), v_bx), Mach = |v_air|/a(z), dynamic pressure q̄ = ½ρ|v_air|².
  - Axial force F_x = −q̄·S·C_A (≈ Cd near α = 0).
  - Normal force magnitude q̄·S·CN_α·sin(α), acting opposite the lateral air-relative velocity: direction −[0, v_by, v_bz]/|[v_by, v_bz]| (guard against division by zero when the lateral velocity is 0).
  - Moment about CG: M = r_cp × F_n, with r_cp = [x_cg − x_cp, 0, 0], plus pitch/yaw damping and roll damping (Niskanen §3.2.3–3.2.4). Optional: fin cant → roll forcing moment.
  - Thrust [F(t), 0, 0] in body (thrust misalignment angle as an optional parameter for Monte Carlo).
  - Translational: m·a = R·(F_aero_b + F_thrust_b) + m·g.
  - Rotational (Euler): I·ω̇ = M − ω × (I·ω). Ignore the variable-mass terms (İ·ω, jet damping) for now; list them as a known simplification in the notes.
  - Quaternion kinematics: q̇ = ½ · q ⊗ [0, ω].
- flight.py: 6-DOF for rail / powered / coast; point-mass for recovery phases. Renormalize q at every phase boundary.
- Plots: 3D trajectory, altitude/velocity/acceleration vs time, α vs time, stability margin vs time, ground track.

**Tests**

- Quaternion: R(q) is orthonormal (det = 1); quaternion norm drift over a full flight < 1e-6.
- Torque-free rigid body: no forces/moments, spinning with off-axis ω → angular momentum (inertial) and rotational kinetic energy conserved within 1e-8 relative.
- Launch attitude: θ = 90° gives body x = [0, 0, 1]; θ = 80°, ψ = 90° gives a body x-axis tilted toward East.
- Reduction test: vertical rail, zero wind → 6-DOF apogee matches 3-DOF and 1-DOF within 0.1%; lateral motion stays < 1e-6 m.
- Planar reduction: with wind in the x-z plane, 6-DOF matches 3-DOF within 0.5%, and y-motion stays ≈ 0.
- Sign and weathercocking tests from Stage 2, repeated in 3D with wind from several headings.

**Acceptance:** tests green; docs/NOTES_stage3.md covering quaternions vs Euler angles (gimbal lock), Euler's rotation equations, and the simplifications made.

### Stage 4 — Recovery & landing detail (≈1 week)

**Tasks**

- recovery.py: drogue at apogee (optionally with a delay, to model motor ejection charges) and main at a set altitude. Parachute inflation modeled as Cd·A ramping up over a configurable fill time. Drag along −v_air, with wind drift.
- Report: descent rates under drogue and main, landing velocity, landing position, opening-shock peak acceleration.

**Tests:** terminal velocity under each chute matches analytic values; with zero wind, landing position equals the apogee's ground position; with constant wind, drift ≈ wind speed × descent time (within 5%).

### Stage 5 — Monte Carlo dispersion (≈1 week)

**Tasks**

- montecarlo.py: dispersions defined in YAML, each with a distribution (normal/uniform) and its parameters:
  - motor total impulse (scale the thrust curve), burn-time variation
  - Cd scale factor, dry mass, CG offset
  - rail elevation and heading, thrust misalignment
  - wind speed and direction, gust intensity
  - parachute Cd·A, deployment delay
- Run N flights (default 500) in parallel (`concurrent.futures.ProcessPoolExecutor`), each with its own seed derived from a master seed.
- Outputs: apogee histogram with mean/σ/percentiles, landing-point scatter with 1σ/2σ/3σ ellipses (from the covariance eigenvectors), rail-exit velocity and stability-margin distributions, plus a CSV of all runs.

**Tests:** same master seed → identical results; with all dispersions set to zero, every run equals the nominal run; covariance-ellipse code recovers a known covariance from synthetic Gaussian points.

**Acceptance:** 500 runs complete in < 5 min on a laptop; docs/NOTES_stage5.md.

### Stage 6 — Sensor output for the flight computer (≈1 week)

This is the interface to the flight-computer project. Freeze the CSV schema once it is agreed.

**Tasks**

- Resample the truth trajectory at fixed rates (IMU 500 Hz, barometer 50 Hz; configurable).
- Accelerometer: measures specific force in the body frame, f_b = Rᵀ·(a − g_vec). On the pad it reads ≈ +1 g along x_b; in free fall/coast it reads only aero deceleration. Add bias, white noise, scale-factor error, quantization (16-bit), and saturation (e.g. ±16 g).
- Gyroscope: ω_b + bias + bias random walk + white noise, saturation (e.g. ±2000 °/s).
- Barometer: ISA pressure at altitude + noise + quantization. Optional: a pressure disturbance near Mach 0.3+ (a known real-world issue).
- Output files:
  - sensors.csv: t, ax, ay, az, gx, gy, gz, pressure_pa, temperature_k (each sensor's columns filled at its own rate, empty otherwise, OR separate files per sensor — pick one and document it)
  - truth.csv: t, x, y, z, vx, vy, vz, qw, qx, qy, qz, wx, wy, wz, phase
  - events.json: true times of liftoff, rail exit, burnout, apogee, deployments, landing
- Noise parameters default to values typical of common hobby sensors (e.g. MPU-6050/ICM-20948 class IMU, BMP280/BMP390 class barometer). Cite datasheet values in the config comments.

**Tests:** with noise off, integrating the accelerometer (with gravity correction) reproduces truth velocity within 1%; the pad reading equals +1 g; noise statistics (σ) of generated samples match the configured values within 5%; bias shows up as a mean offset.

**Acceptance:** schema documented in docs/SENSOR_SCHEMA.md; docs/NOTES_stage6.md explains why an accelerometer can't "feel" gravity in free fall.

### Stage 7 — Validation report (≈1 week)

**Tasks**

- Model one real kit rocket (e.g. a common Estes kit, or an OpenRocket example file) in OpenRocket; export the simulation CSV into reference/openrocket/.
- Compare in layers:
  1. Dynamics only: use OpenRocket's Cd curve via table override → compare altitude, velocity, acceleration curves.
  2. Own aero: compare Cd(Mach) and CP directly with OpenRocket's values.
  3. Full: own aero + dynamics vs OpenRocket and vs RocketPy (optional `[validation]` extra).
- docs/VALIDATION.md: overlay plots, error table (apogee, max velocity, time to apogee, drift), and an explanation of every discrepancy > 2%.
- Stretch: compare against a real flight log (an altimeter or your own ESP32 data).

**Acceptance targets:** apogee within 5% of OpenRocket with table Cd; within 10% with own aero model; every larger gap explained.

## 5. Out of scope (for now)

Multi-stage rockets, hybrid/liquid motors, supersonic/transonic accuracy, Earth rotation/curvature, GUI, real-time 3D replay. Revisit after Stage 7.

## 6. How to drive this with Claude Code

1. Create a folder rocketsim/, put PLAN.md, CLAUDE.md and sim_1dof.py in it, git init.
2. Open Claude Code in that folder.
3. Use plan mode at the start of each stage (Shift+Tab to toggle) so Claude Code proposes its approach before writing code.

**Kickoff prompts:**

Stage 0

> Read PLAN.md and CLAUDE.md. Implement Stage 0 only. Move sim_1dof.py into reference/ unchanged. Show me the plan first, then implement, run the tests and stop.

Any later stage N

> Implement Stage N from PLAN.md. First list the tasks you'll do and any formula or design question you're unsure about. Write the tests listed for this stage before the implementation. Run the full test suite, then write docs/NOTES_stageN.md and stop for my review.

When something fails or looks wrong

> Don't patch the symptom. Explain the physical or numerical cause first, show the evidence (numbers, a plot), then propose a fix.

**Learning check after each stage** (ask yourself, or ask Claude to quiz you)

> Quiz me with 5 questions on the physics and design decisions in Stage N. Don't give answers until I reply.

Commit after every task that leaves tests green. Tag each finished stage (`git tag stage-3`).
