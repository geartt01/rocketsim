# Sensor output schema

This is the interface between RocketSim and the flight-computer project. Written by:

```bash
python -m rocketsim sensors configs/example_rocket.yaml --seed 1
```

Files go to `outputs/<config name>/`. **Time:** `t` is seconds since motor ignition, so samples before ignition have `t < 0`. The log starts `pad_time` before ignition and ends `landed_time` after landing. All units are SI.

## `sensors.csv` — what the flight computer sees

One file for both sensors, with one row per IMU sample. Barometer columns are filled only on barometer samples and are empty otherwise, so the barometer rate must divide the IMU rate.

| Column | Unit | Meaning |
|---|---|---|
| `t` | s | Sample time since ignition |
| `ax`, `ay`, `az` | m/s² | Accelerometer: specific force in body axes, with errors |
| `gx`, `gy`, `gz` | rad/s | Gyroscope: angular rate in body axes, with errors |
| `pressure_pa` | Pa | Barometer static pressure (every Nth row) |
| `temperature_k` | K | Barometer temperature (same rows) |

**Body axes:** x points toward the nose; y and z complete a right-handed frame. So on a vertical pad, `ax` ≈ +9.81 m/s², and in free fall it reads only drag (negative, pointing aft).

**Saturation:** readings clip at the configured range, as on real hardware. The example's 24.7 g thrust spike clips at ±16 g.

## `truth.csv` — the correct answer, for scoring your estimator

Same time grid as the IMU.

| Column | Unit | Meaning |
|---|---|---|
| `t` | s | Time since ignition |
| `x`, `y`, `z` | m | CG position: East, North, Up, from the rail base |
| `vx`, `vy`, `vz` | m/s | Velocity in East-North-Up |
| `qw`, `qx`, `qy`, `qz` | - | Unit quaternion rotating body → East-North-Up |
| `wx`, `wy`, `wz` | rad/s | Angular velocity in body axes |
| `phase` | text | `pad`, `rail`, `powered`, `coast`, `freefall`, `drogue`, `main`, `landed` |

`z` is the height above the launch site. Pressure uses `launch.site_altitude + z`.

Under a parachute the simulator freezes the attitude (point-mass recovery), so `q` is constant and `w` is 0 there. Don't use the attitude in these phases to test attitude estimators.

## `events.json` — true event times, in seconds since ignition

Keys: `liftoff`, `rail_exit`, `burnout`, `apogee`, `drogue`, `main` (only if the rocket has a main chute), `landing`.

Use these to score the flight computer: for example, its detected apogee time minus `apogee`.

## Reproducibility

The same config and `--seed` give identical files. Rates and noise are set in the config's `sensors:` section.
