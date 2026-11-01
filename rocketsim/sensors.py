"""IMU and barometer models: what a flight computer would log during a simulated flight.

Schema of the output files: docs/SENSOR_SCHEMA.md. All randomness comes from the
numpy Generator passed in. Output units are SI (m/s², rad/s, Pa, K).
"""

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from rocketsim import frames
from rocketsim.atmosphere import isa
from rocketsim.constants import G0
from rocketsim.dynamics import dof6
from rocketsim.io import read_yaml

STATE_COLUMNS = dof6.STATE
SENSOR_COLUMNS = ("t", "ax", "ay", "az", "gx", "gy", "gz", "pressure_pa", "temperature_k")


@dataclass
class Inertial:
    """One three-axis inertial sensor, in SI."""

    range: float  # full scale ±, m/s² or rad/s
    bits: int  # ADC resolution
    noise: float  # white noise σ per sample, m/s² or rad/s
    bias: tuple  # constant offset per axis, m/s² or rad/s
    scale_error: float = 0.0  # fractional, accelerometer only
    walk: float = 0.0  # bias random walk, rad/s per √s, gyroscope only


@dataclass
class Barometer:
    noise: float  # white noise σ per sample, Pa
    resolution: float  # output step, Pa


@dataclass
class SensorConfig:
    imu_rate: float  # Hz
    baro_rate: float  # Hz, must divide imu_rate
    pad_time: float  # s logged on the pad before ignition
    landed_time: float  # s logged after landing
    accel: Inertial
    gyro: Inertial
    baro: Barometer


def load_sensor_config(path):
    """Read the `sensors:` section of a rocket config; YAML uses g and deg/s, inside is SI.

    Raises ValueError naming the file and the offending key.
    """
    path = Path(path)

    def fail(msg):
        raise ValueError(f"{path}: {msg}")

    def section(parent, key, where, keys):
        sec = parent.get(key)
        if not isinstance(sec, dict):
            fail(f"'{where}' must be a mapping")
        missing, unknown = set(keys) - set(sec), set(sec) - set(keys)
        if missing or unknown:
            fail(f"'{where}': missing {sorted(missing)}, unknown {sorted(unknown)}")
        return sec

    def number(sec, key, where, positive=True):
        v = sec[key]
        if isinstance(v, bool) or not isinstance(v, (int, float)) or v != v:  # noqa: PLR0124
            fail(f"{where}.{key} must be a number, got {v!r}")
        if v <= 0 if positive else v < 0:
            fail(f"{where}.{key} must be > 0, got {v}" if positive
                 else f"{where}.{key} must be ≥ 0, got {v}")
        return float(v)

    def bits(sec, where):
        if not isinstance(sec["bits"], int) or isinstance(sec["bits"], bool) or sec["bits"] < 2:
            fail(f"{where}.bits must be an integer ≥ 2, got {sec['bits']!r}")
        return sec["bits"]

    def vector(sec, key, where, factor):
        v = sec[key]
        if not isinstance(v, list) or len(v) != 3:
            fail(f"{where}.{key} must be a list of 3 numbers")
        return tuple(float(c) * factor for c in v)

    cfg = read_yaml(path) or {}
    s = section(cfg, "sensors", "sensors",
                ("imu_rate", "baro_rate", "pad_time", "landed_time", "accel", "gyro", "baro"))
    a = section(s, "accel", "sensors.accel",
                ("range_g", "bits", "noise_g", "bias_g", "scale_error"))
    g = section(s, "gyro", "sensors.gyro",
                ("range_dps", "bits", "noise_dps", "bias_dps", "walk_dps_per_sqrt_s"))
    b = section(s, "baro", "sensors.baro", ("noise_pa", "resolution_pa"))
    deg = np.pi / 180
    imu_rate = number(s, "imu_rate", "sensors")
    baro_rate = number(s, "baro_rate", "sensors")
    if abs(imu_rate / baro_rate - round(imu_rate / baro_rate)) > 1e-9:
        fail("sensors.baro_rate must divide imu_rate exactly")
    return SensorConfig(
        imu_rate, baro_rate,
        number(s, "pad_time", "sensors", positive=False),
        number(s, "landed_time", "sensors", positive=False),
        Inertial(number(a, "range_g", "sensors.accel") * G0, bits(a, "sensors.accel"),
                 number(a, "noise_g", "sensors.accel", positive=False) * G0,
                 vector(a, "bias_g", "sensors.accel", G0),
                 scale_error=float(a["scale_error"])),
        Inertial(number(g, "range_dps", "sensors.gyro") * deg, bits(g, "sensors.gyro"),
                 number(g, "noise_dps", "sensors.gyro", positive=False) * deg,
                 vector(g, "bias_dps", "sensors.gyro", deg),
                 walk=number(g, "walk_dps_per_sqrt_s", "sensors.gyro", positive=False) * deg),
        Barometer(number(b, "noise_pa", "sensors.baro", positive=False),
                  number(b, "resolution_pa", "sensors.baro")),
    )


def truth_at_rate(flight, launch, rate, pad_time, landed_time):
    """Sample the true 6-DOF state and acceleration on a fixed grid, pad to after landing.

    t = 0 is ignition, so samples before it have t < 0. On the pad (until liftoff) and after
    landing, the rocket is held still: v = a = ω = 0. In flight, values come from each phase's
    dense solver output, so nothing is interpolated across a phase boundary.

    Args: flight (from flight.simulate), launch, rate (Hz), pad_time and landed_time (s).
    Returns dict of arrays: t, the 13 state columns, ax, ay, az (inertial, m/s²), phase.
    """
    t_end = flight.events["landing"] + landed_time
    n = int(np.ceil((t_end + pad_time) * rate)) + 1
    t = -pad_time + np.arange(n) / rate
    y = np.empty((n, 13))
    a = np.zeros((n, 3))
    phase = np.full(n, "landed", dtype="<U8")

    before = t < flight.events["liftoff"]
    y[before] = dof6.initial_state(launch)
    phase[before] = "pad"
    for sol, fun, args, label, _ in flight.segments:
        mask = (t >= sol.t[0]) & (t < sol.t[-1])
        for i in np.flatnonzero(mask):
            y[i] = sol.sol(t[i])
            a[i] = np.asarray(fun(t[i], y[i], *args))[3:6]
        phase[mask] = label
    after = t >= flight.events["landing"]
    y[after] = flight.final_state
    y[after, 3:6] = 0.0

    truth = {"t": t, **{name: y[:, i] for i, name in enumerate(STATE_COLUMNS)},
             "ax": a[:, 0], "ay": a[:, 1], "az": a[:, 2], "phase": phase}
    return truth


def specific_force(truth, g):
    """What an ideal accelerometer senses, in body axes: f_b = Rᵀ·(a − g_vec), m/s².

    g_vec = [0, 0, −g], so a body held still reads +g upward (the supporting force) and a body
    in free fall reads only its non-gravity forces (thrust, drag). Gravity pulls equally on the
    sensor's case and its proof mass, so it cannot be felt.
    Returns array [n, 3]. Source: standard strapdown inertial navigation, e.g. Titterton &
    Weston, Strapdown Inertial Navigation Technology, ch. 3.
    """
    q = np.array([truth[k] for k in ("qw", "qx", "qy", "qz")])
    q = q / np.sqrt((q**2).sum(axis=0))
    rot = frames.rotation_matrix(q)
    f = np.column_stack([truth["ax"], truth["ay"], truth["az"] + g])
    return np.einsum("jin,nj->ni", rot, f)


def _digitize(x, sensor):
    """Saturate at ±range and round to the ADC step 2·range / 2^bits."""
    lsb = 2 * sensor.range / 2**sensor.bits
    return np.round(np.clip(x, -sensor.range, sensor.range) / lsb) * lsb


def measure(truth, config, launch, rng, ideal=False):
    """Turn truth into sensor readings at the truth sample rate.

    accelerometer  y = digitize(f_b·(1 + s) + b + n),     n ~ N(0, σ²)
    gyroscope      y = digitize(ω + b + w(t) + n),        w random walk: Δw ~ N(0, σ_w²·Δt)
    barometer      p = quantize(p_ISA(altitude) + n),     temperature = T_ISA (0.01 K steps)

    ideal=True removes noise, bias, scale error, saturation and quantization (for tests).
    Assumptions: errors are independent per axis and constant-parameter; the barometer has no
    velocity or Mach pressure error; no temperature drift; no IMU offset from the CG.
    Returns dict: t, ax, ay, az (m/s²), gx, gy, gz (rad/s), pressure_pa, temperature_k.
    Sources for default values: MPU-6050 and BMP280 datasheets, see configs/example_rocket.yaml.
    """
    n = len(truth["t"])
    f = specific_force(truth, launch.g)
    w = np.column_stack([truth["wx"], truth["wy"], truth["wz"]])
    atm = isa(launch.site_altitude + truth["z"])
    out = {"t": truth["t"], "temperature_k": np.round(atm.T, 2)}
    if ideal:
        accel, gyro, pressure = f, w, atm.p
    else:
        dt = truth["t"][1] - truth["t"][0]
        a, g, b = config.accel, config.gyro, config.baro
        accel = _digitize(f * (1 + a.scale_error) + np.array(a.bias)
                          + rng.normal(0.0, a.noise, (n, 3)), a)
        walk = np.cumsum(rng.normal(0.0, g.walk * np.sqrt(dt), (n, 3)), axis=0)
        gyro = _digitize(w + np.array(g.bias) + walk + rng.normal(0.0, g.noise, (n, 3)), g)
        pressure = np.round((atm.p + rng.normal(0.0, b.noise, n)) / b.resolution) * b.resolution
    for i, axis in enumerate("xyz"):
        out[f"a{axis}"], out[f"g{axis}"] = accel[:, i], gyro[:, i]
    out["pressure_pa"] = pressure
    return out


def write_outputs(flight, truth, data, config, out):
    """Write sensors.csv, truth.csv and events.json (schema: docs/SENSOR_SCHEMA.md)."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    every = round(config.imu_rate / config.baro_rate)  # barometer cells empty between its samples
    with open(out / "sensors.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(SENSOR_COLUMNS)
        rows = zip(*(data[c] for c in SENSOR_COLUMNS))
        for i, row in enumerate(rows):
            writer.writerow([f"{row[0]:.6f}", *(f"{v:.6f}" for v in row[1:7]),
                             *((f"{row[7]:.2f}", f"{row[8]:.2f}") if i % every == 0
                               else ("", ""))])
    columns = ("t", *STATE_COLUMNS, "phase")
    with open(out / "truth.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(columns)
        writer.writerows(zip(*(truth[c] for c in columns)))
    events = {k: float(v) for k, v in flight.events.items()}
    (out / "events.json").write_text(json.dumps(events, indent=2), encoding="utf-8")
