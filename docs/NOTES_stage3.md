# Stage 3 notes — full 6-DOF

## Physics in plain language

**State.** The state is position r and velocity v in the ENU frame, attitude q (a unit quaternion, body → ENU), and angular velocity ω in body axes: 13 numbers. The mass, CG and inertia [I_x, I_t, I_t] come from the burn time, not from the state.

**Quaternions vs Euler angles.** Three angles (yaw, pitch, roll) can't describe every attitude smoothly. At pitch = ±90°, yaw and roll turn about the same axis, and one degree of freedom disappears. This is **gimbal lock**, and their derivatives blow up there. A rocket starts at pitch 90°, exactly that worst case.

A quaternion uses four numbers with one constraint, |q| = 1. Its kinematics q̇ = ½·q ⊗ [0, ω] are linear in q and never singular. The price is that numerical error slowly breaks |q| = 1. So q is normalised before every use and at every phase boundary. The test checks that |q| drifts less than 1e-6 over a whole flight.

**Euler's rotation equations.** In body axes, I·ω̇ = M − ω × (I·ω). The ω × (I·ω) term is the gyroscopic coupling. It's why a spinning, tumbling object wobbles even with no torque.

The torque-free test integrates an *asymmetric* body for 10 s. It checks that the inertial angular momentum R·I·ω and the energy ½·ωᵀ·I·ω stay constant to 1e-8. If the quaternion product, R(q) or the gyroscopic term were wrong, that test would fail.

**Forces and moments (body axes).** The air-relative velocity v_a,b = Rᵀ·(v − w) sets α, Mach and q̄. Then:

- **Axial:** thrust − ½ρ·A·C_D0·|v_a|·v_bx.
- **Normal:** −½ρ·A·CN_α·|v_a|·[0, v_by, v_bz]. It points against the sideways airflow, with magnitude q̄·A·CN_α·sin α.
- **Moment:** [x_cg − X_cp, 0, 0] × F_n gives the restoring moment. Pitch/yaw damping −K·|ω_t|·ω_t and roll damping −C·ω_x are added.
- **Roll damping:** each fin strip at radius ξ sees an extra angle of attack ω·ξ/v (Niskanen 3.69/3.70). The tests check the 3.70 formula against a numerical strip integral.

**Launch attitude.** q = q_z(90° − ψ) ⊗ q_y(−θ). This puts the body x-axis at [cos θ·sin ψ, cos θ·cos ψ, sin θ], as PLAN §0 requires; the tests check three attitudes.

## Design decisions

- **One phase manager for both models.** 3-DOF and 6-DOF share `flight._fly(model, …)`. Each model module (`dof3`, `dof6`) provides the same small interface: state names, rail / free-flight / parachute derivatives, a rail-exit event and output info. There's one copy of the phase logic, events, warnings and summary.
- **Recovery and rail.** Recovery phases are point-mass with the attitude frozen (PLAN §3). On the rail, all motion is along the body axis and ω = 0.
- **Config: direction added.**
  - `launch.rail_heading` gives the rail's lean direction, in degrees clockwise from North.
  - `wind.direction` is the meteorological "blows FROM" direction, and `wind.speed` must now be ≥ 0.
  - 3-DOF uses only the wind component along the rail heading.
- **Outputs.** `trajectory.csv` holds the full truth state (`x…z, vx…vz, qw…qz, wx…wz`) plus accelerations, α, Mach and margin. The plots now include a ground track and a 3-D trajectory.

## Known simplifications

- **Rotation:**
  - variable-mass terms (İ·ω and jet damping) are ignored;
  - principal axes are assumed to lie along the body axes;
  - there's no fin cant, so no roll forcing — and with symmetric fins the rocket never starts to roll.
- **Aerodynamics:**
  - CN_α doesn't depend on roll orientation (Niskanen's choice for N ≥ 3 fins);
  - the CP doesn't move with α;
  - axial drag scales with cos α;
  - there's no body lift.
- **Wind:** no gusts or wind shear in direction yet; Stage 5 adds gusts.
- **Euler angles:** they're not implemented. The plots use α and position, not attitude angles.

## Results

The example rocket flies in the East–Up plane: the rail leans East, and the wind comes from the West. The 6-DOF results match Stage 2's 3-DOF to the printed digit: apogee 353.7 m, landing 435.3 m East. The North coordinate stays within about 1e-13 m of zero.

**Reduction tests:**
- **Vertical rail, no wind:** 6-DOF agrees with 3-DOF and 1-DOF to within 0.1%.
- **Wind in the plane:** 6-DOF agrees with 3-DOF to within 0.5%, and North motion stays below 1e-6 m.
- **Wind from four headings:** the rocket weathercocks upwind every time, then drifts downwind, with no motion across the wind.

## What to check next

- Give the rocket a crosswind relative to its rail heading. This is the first case that needs the 6-DOF model; the 3-DOF model can't represent it.
- Stage 4 (recovery detail) can reuse the point-mass parachute derivatives; only the deployment logic changes.
