# Stage 6 notes — sensor output for the flight computer

## Physics in plain language

**Why an accelerometer can't feel gravity in free fall.** Inside the chip, a tiny mass hangs on springs and the chip measures how far the springs stretch. Gravity pulls equally on the mass and on the chip's case, so on its own gravity stretches nothing. Only a *contact* force, one that pushes the case but not the mass, stretches the springs: thrust, drag, or the pad holding the rocket up.

So the sensor measures **specific force**, f = a − g_vec, not acceleration:
- **On the pad:** a = 0, so f = +g upward. It reads **+1 g**, though nothing is moving. That's the pad's push.
- **Coasting:** gravity accelerates everything together, so f is only drag / mass. Right after burnout that's strong (about −5 g on the example); near apogee it's almost zero, even though gravity is still acting.
- **Under thrust:** f = (thrust − drag) / mass.

In code: f_b = Rᵀ·(a − g_vec), rotated into body axes because the sensor is bolted to the rocket.

**Why this matters for your flight computer.** "Acceleration reads about zero" doesn't mean "stopped"; it means nothing is pushing. To get velocity you must rotate f into the inertial frame and *add gravity back*. The tests do exactly that: integrating the ideal accelerometer this way recovers the true velocity within 1 %.

**Sensor errors modelled:**

| Sensor | Model |
|---|---|
| Accelerometer | y = digitize(f·(1 + s) + b + n) |
| Gyroscope | y = digitize(ω + b + w(t) + n) |
| Barometer | p = quantize(p_ISA(altitude) + n) |

- **b:** constant bias. **s:** scale error. **n:** white noise.
- **w(t):** a random walk, so gyro bias slowly drifts.
- **digitize:** clips at ±range, then rounds to the ADC step.

White-noise σ per sample = noise density × √(bandwidth), with bandwidth = half the sample rate.

## Design decisions

- **One file.** `sensors.csv` has one row per IMU sample and empty barometer cells between barometer samples. That matches how a logger's stream looks, and needs only one file to read. Schema: `docs/SENSOR_SCHEMA.md`.
- **Exact truth, not resampled output.** Truth comes from each phase's dense solver output, so nothing is interpolated across a phase boundary.
- **Pad and post-landing data** (`t < 0`, then after landing) are in the log, so a flight computer can calibrate bias and detect landing.
- **Fixed-step RK4.** PLAN §3 keeps it as a cross-check. `integrators.py` has it; tests confirm 4th-order convergence and agreement with DOP853 on a 6-DOF powered flight. Fixed-rate sensor output comes from sampling the dense output, as the Stage 6 task describes, instead of a separate RK4 run.
- **Bug fixed on the way.** Under canopy the attitude was frozen but the stored spin rate kept its value from apogee, so a gyro would have reported rotation that wasn't happening. Rotation is now zeroed when the chute phase starts.

## Known simplifications

- **Barometer:** no pressure disturbance near Mach 0.3+ (PLAN marks this optional); no temperature drift.
- **IMU:** mounted at the CG. A real offset adds centripetal terms.
- **Under canopy:** attitude is frozen, so accelerometer axes there are at their apogee orientation, not hanging from the chute. Real logs would also show swinging and shock-cord snatch.
- **Values:** noise figures come from the datasheets, but the bias, scale-error and random-walk numbers are illustrative (inside the datasheet tolerances where the datasheet gives one).

## Result

Example flight: 54,387 IMU samples at 500 Hz, about 110 s long, written in about 30 s. Pad reading is +1 g along the nose on a vertical rail. The accelerometer saturates at 16 g during the C6's 24.7 g thrust spike.

## What to check next

- Feed `sensors.csv` to your apogee-detection code and score it against `events.json`.
- Note that saturation hides part of the thrust spike. A wider-range accelerometer (or an extra high-g one) avoids this, and integrating a clipped signal underestimates velocity.
