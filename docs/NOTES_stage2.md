# Stage 2 notes — 3-DOF planar flight and real aerodynamics

## Physics in plain language

**CG and CP.**
- The **CG** (centre of gravity) is where the weight acts: Σm·x / Σm, recomputed every step as propellant burns.
- The **CP** (centre of pressure) is where the aerodynamic *normal* force acts. That force is the sideways push you get when the airflow meets the rocket at an angle of attack α.

If the CP is **aft** of the CG, that sideways push makes a moment that turns the nose back into the airflow, like a weathervane. If the CP is ahead of the CG, the moment grows α instead and the rocket tumbles. The **stability margin** (X_CP − X_CG)/d measures how far aft the CP is, in body diameters (calibers). 1–2 calibers is the usual target.

**Barrowman.**
- *Nose:* any smooth body section has CN_α = 2·[A(aft) − A(fore)]/A_ref, so a nose cone gets 2 whatever its shape, and a constant-diameter tube gets 0. The nose CP is X = (L·A − V)/A, where V is the nose volume. This gives 2L/3 for a cone, L/3 for an ellipsoid and L/2 for a Von Kármán. The tests check all three.
- *Fins:* Diederich's lift slope for a swept fin, with compressibility through β = √(1 − M²). It's multiplied by N/2 for N ≥ 3 fins, and by 1 + r/(s + r) because the body boosts the flow over the fins. The fin CP is at the quarter-chord of the mean aerodynamic chord.
- The total CP is the CN_α-weighted average of the component CPs (Niskanen eqs. 3.19, 3.28, 3.29, 3.34, 3.40, 3.54, 3.56).

**Drag (C_D0).** Four parts are added up (Niskanen §3.4):
- **Skin friction:** a turbulent C_f that depends on the Reynolds number, with a roughness limit and the (1 − 0.1M²) compressibility correction. It's scaled by the wetted area.
- **Nose pressure drag:** 0.8·sin²φ, which is zero for smooth nose shapes.
- **Fin leading and trailing edges:** these depend on the fin profile.
- **Base drag:** 0.12 + 0.13M² on the base area. It's smaller while the motor fires, because the exhaust fills part of the base.

The example rocket comes out at C_D0 ≈ 0.67 coasting at Mach 0.3, which is typical for a small finned model.

**Weathercocking.** On leaving the rail in a crosswind, the airflow comes partly from the side, so α ≠ 0. The normal force acts at the CP, behind the CG, and swings the nose *into* the wind. So the rocket climbs upwind. The parachute later drifts downwind, at the wind speed.

## Design decisions

- **Phases:** pad → rail → powered → coast → parachute → landed. Each is its own DOP853 integration that ends on an event.
- **Rail:** the rocket moves only along the rail and doesn't rotate. The rail takes up the normal force and the part of gravity across it. Liftoff is when thrust > m·g·sin θ_rail. The rail-exit speed and stability-margin warnings fire *at rail exit*, so they still appear if a bad design fails later in the flight.
- **Drag setting:** one of three, interchangeable in the config:
  - `buildup` (Niskanen);
  - a constant Cd, which the 3-DOF → 1-DOF reduction test uses;
  - a `mach,cd` CSV, for importing OpenRocket's Cd in Stage 7.
- **Mass properties:**
  - the nose and tube are thin shells of revolution, integrated numerically;
  - the fins are thin plates;
  - extra mass items are point masses or solid cylinders;
  - the motor is a solid cylinder whose mass shrinks as it burns.
- **CP check:** the CP and CN_α are checked against RocketPy's independent Barrowman code for four nose shapes. Ogive, cone, Von Kármán and ellipse all agree to about 1e-12.

## Known simplifications

- **Axial drag:** scales with cos α. This is exact at small α. Niskanen's empirical C_A(α) curve, which rises to 1.3× at 17°, isn't used.
- **CP and CN_α:**
  - Galejs body lift is left out. It's ∝ sin²α, so it doesn't affect CN_α at α = 0.
  - The fin CP doesn't move aft above Mach 0.5.
  - Launch lugs (parasitic drag) and fin–body interference drag are left out.
- **Motor:** the propellant's CG is fixed at the motor's centre. Variable-mass moment terms are ignored.
- **Parachute:** it opens instantly at apogee. Under it, attitude is frozen and only parachute drag acts.
- **Max α:** it's reported over powered + coast flight. Near apogee the airspeed is small, so α can be large (41° in the example). On the rail at near-zero speed it reaches about 90°. Neither matters for the trajectory.

## Deviation from PLAN.md, for you to decide

PLAN asks that the CP match **OpenRocket** within 2%. OpenRocket isn't scriptable from here, so the test uses **RocketPy**, the PLAN's own `[validation]` oracle, instead. Exporting an OpenRocket CP for the example rocket would add the real check (planned for Stage 7).

The 3-DOF → 1-DOF reduction test bounds lateral drift at 1e-6 m (PLAN's Stage 3 value). cos(π/2) is 6e-17 in floating point, which seeds about 7e-9 m of drift over the flight.

## Example result (`configs/example_rocket.yaml`: 85° rail, 3 m/s power-law wind)

| Quantity | Value |
|---|---|
| Apogee | 354 m, 11 m downrange |
| Max velocity | 115 m/s (Mach 0.34) |
| Rail-exit velocity | 17.8 m/s |
| Stability margin | 2.10 cal at rail exit, rising to 2.8 at burnout |
| Drift | 435 m |
| Landing velocity | 3.6 m/s |

## What to check next

- Build this rocket in OpenRocket and compare its CP, its Cd at Mach 0.3, and the apogee.
- Drift is dominated by the parachute. A smaller chute or a delay before it opens (Stage 4) cuts drift a lot.
