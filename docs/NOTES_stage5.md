# Stage 5 notes — Monte Carlo dispersion

## Idea in plain language

Every input is uncertain: the motor's impulse, the drag, the mass, how the rail is set up, and above all the wind. Instead of one "nominal" flight, we fly hundreds. Each one draws every uncertain input from a distribution, and we look at the spread of results.

That answers questions a single flight can't:
- How high will it *probably* go (the 5–95 % range)?
- Where might it land (the landing ellipse)?
- How close to the stability or rail-speed limits might a bad combination get?

**Inputs varied** (`dispersions:` in the config; each `normal: [mean, σ]` or `uniform: [low, high]`):

| Group | Parameters |
|---|---|
| Motor | impulse, burn time |
| Aerodynamics and mass | Cd scale, dry mass, dry CG |
| Launch | rail elevation and heading, thrust misalignment |
| Wind | speed, direction, gust intensity |
| Recovery | chute Cd·A, drogue delay |

Parameters named `*_scale` multiply the nominal; the others add to it.

**Gusts (turbulence).** These follow Niskanen §4.1.2:
- White noise goes through a 2-pole Kasdin IIR filter, giving pink noise with a 1/f^(5/3) spectrum — the high-frequency slope of real wind turbulence.
- It's divided by 2.252 so its standard deviation is 1; a test confirms the result really is 1.
- It's sampled at 20 Hz, giving U(t) = U + σ_u·x(t) along the mean wind direction.

**Thrust misalignment.** A nozzle tilted by δ pushes partly sideways. Because it acts at the nozzle, behind the CG, it also produces a moment that turns the rocket. The tilt's direction (azimuth) is random.

**Landing ellipses.** The covariance matrix of the landing points has eigenvectors along the ellipse axes, and the square roots of its eigenvalues give the 1-σ semi-axes. In 2-D, the 1/2/3-σ ellipses hold about 39 %, 86 % and 99 % of points. The test recovers a known covariance from synthetic points to 1 %.

## Reproducibility

One master seed (`--seed`) feeds `SeedSequence.spawn(n)`, giving one independent child seed per run. Each run draws its parameters, then its gust sequence, from its own child. So results depend only on (seed, run number): the test checks that 1 and 2 worker processes give identical rows. With every dispersion set to zero, every run reproduces the nominal flight bit for bit.

## Making 500 runs fit in 5 minutes

This laptop has 4 physical cores and 8 threads. At first, 500 runs took more than 10 minutes. Each fix was measured, not guessed:

1. **Gust interpolation.** Linear interpolation puts a kink every 0.05 s, and the adaptive solver cut its step at every kink, about 12× slower. A cubic spline through the same samples has no kinks. Niskanen only says "interpolated", so this is allowed. The turbulence spectrum is kept below 10 Hz.
2. **Hot-path overhead.** No physics change; results are identical to all printed digits.
   - the mass is constant after burnout, so it's cached;
   - `isa()` and the per-step aerodynamics now use plain floats instead of numpy single values;
   - the gust spline is evaluated piece by piece;
   - the body radius profile for pitch damping is cached.
3. **Solver for Monte Carlo only.** Monte Carlo uses RK45 at rtol 1e-6 / atol 1e-4; single flights keep DOP853 at 1e-8. Against the strict solver, over 8 dispersed flights, apogee moved by at most 0.15 m and the landing point by at most 0.44 m. That's under 1 % of the run-to-run spread. The full 500-run statistics match the strict-solver run to 0.03 m in mean apogee.
4. **Opening shock.** It's now measured on a 1 ms grid over each canopy's fill time plus 3 s, so it no longer depends on the output step. A side finding: near the ground, the power-law wind shear briefly loads the canopy to 1.08 g. That's not an opening shock and is no longer reported as one.

**Result:** 500 runs in 183 s, with 0 failures.

## Example result (`configs/example_rocket.yaml`, seed 1, illustrative spreads)

| Quantity | Value |
|---|---|
| Apogee | 353 m mean, σ 16.9 m, 5–95 % 326–381 m |
| Rail-exit speed | 17.8 m/s mean, minimum 16.8 m/s; never below the 15 m/s guideline |
| Margin at rail exit | 2.09 cal mean, minimum 1.73 cal |
| Landing ellipse | centre 412 m East; 2σ semi-axes 264 × 203 m, long axis along the wind |

## Known simplifications

- Dispersions are independent of each other; nothing is correlated.
- Gusts vary only the wind *speed* along the mean direction. There's no turbulence across the wind or vertically, and no change of wind direction with height.
- Dry-mass and CG dispersion shift the dry part as a whole, and its inertia scales with mass.
- Thrust misalignment acts in the 6-DOF model only.
- The spreads in the example config are illustrative, not measured.

## What to check next

- Replace the illustrative spreads with real ones, such as motor certification data or the scatter of your own weather forecasts.
- Use the 3-σ ellipse to choose a launch-field size or a safe rail heading for a given wind.
