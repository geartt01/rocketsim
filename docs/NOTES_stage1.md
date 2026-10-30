# Stage 1 notes — 1-DOF vertical flight

## Physics in plain language

**Equation of motion.** The rocket moves only along z (up). Three forces act on it: thrust F(t) from the motor curve, drag D, and weight m(t)·g:

    m(t)·dv/dt = F(t) − ½·ρ(z)·Cd·A·v·|v| − m(t)·g

`F = m·a` still holds even though the mass changes. The reason is that the motor's thrust *is* the momentum the exhaust carries away each second. If you write momentum conservation for rocket + exhaust over a short time dt, the exhaust term turns into F, and what's left is m·a = ΣF. (Writing `d(mv)/dt = F` instead would be wrong. It counts the lost mass twice.)

**Mass.** m(t) = dry mass + motor casing + propellant left. The propellant burns in proportion to the impulse delivered so far. That is the same as assuming a constant exhaust velocity v_e = I_total / m_prop. A check of this: with constant thrust, the speed gained matches Tsiolkovsky's rocket equation, Δv = v_e·ln(m0/mf).

**Drag.** D = ½ρv²·Cd·A acts against the velocity, which is why the code writes `v·|v|`. ρ comes from the ISA model at launch-site elevation + z.

**Terminal velocity.** Under the parachute, the rocket stops accelerating when drag equals weight: ½ρv²·CdA = mg, so v_t = √(2mg / (ρ·CdA)). For the example rocket this gives 3.66 m/s.

**Why DOP853 and not Euler.** Euler's error shrinks only in proportion to the step size h, so it needs tiny steps to be accurate. RK4's error shrinks as h⁴, and DOP853's as h⁸. DOP853 also picks its own step size to meet `rtol = atol = 1e-8`. Events stop the integration exactly at apogee and at landing, instead of at the nearest step.

## Design decisions

- **Phases.** Each phase is a separate `solve_ivp` call:
  - pad → liftoff: the first time thrust > weight, found to machine precision with `brentq`;
  - powered: until the burnout time;
  - coast: until the apogee event (v crosses 0 going down);
  - parachute: until the landing event (z crosses 0 going down).

  No integration step ever crosses the jump at burnout or at chute deployment.
- **Outputs.** The trajectory is resampled every 0.01 s, plus at every point of the thrust curve, so the acceleration peak isn't missed. Acceleration is recomputed from the model, not estimated from differences between samples.
- **Config.** All keys are required, and unknown keys are an error, so a typo can't silently fall back to a default. The motor path is relative to the config file.
- **Oracle.** `reference/sim_1dof.py` is a separate, plain fixed-step RK4 script. It tracks mass as a state variable and has its own closed-form density formula. For the example rocket it agrees with the package to within 3·10⁻⁷ (relative).

## Known simplifications

- Cd is constant, the flight is perfectly vertical, and there is no launch rail or wind (Stage 2 adds these).
- The parachute opens instantly at apogee, and only its drag counts under canopy. The body's drag is ignored there (Stage 4 adds deployment delay and inflation time).
- If the rocket stops climbing before burnout, the simulation raises an error instead of modelling it.
- Constant g. The Mach check prints a warning above 0.8; the example reaches 0.35.

## Example result (`configs/example_rocket.yaml`, Estes C6, 35 g dry)

| Quantity | Value |
|---|---|
| Apogee | 401 m |
| Time to apogee | 8.3 s |
| Max velocity | 118 m/s |
| Max acceleration | 24 g |
| Landing velocity | 3.66 m/s |
| Flight time | 117 s |

## What to check next

- The example's dry mass and Cd are illustrative. Build the same rocket in OpenRocket and compare apogee. Expect differences mainly from Cd, which OpenRocket varies with speed.
- The 24 g peak is real: it happens at the C6's 14 N thrust spike, on a 58 g rocket.
