"""Monte Carlo dispersion: many 6-DOF flights with randomly varied inputs, then statistics.

Reproducibility: one master seed → numpy SeedSequence.spawn(n) → one independent child seed
per run. Each run draws all its parameters, then its gust sequence, from its own child, so the
results depend only on (master seed, run index), not on worker count or scheduling.
"""

import dataclasses
import os
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from rocketsim.flight import simulate
from rocketsim.io import load_config, read_yaml
from rocketsim.recovery import Chute
from rocketsim.wind import GUST_DT, gust_signal, pink_noise

DT_OUT = 0.05  # s output step for Monte Carlo runs: only the summary is kept
# Solver settings for Monte Carlo runs. Single flights use DOP853 at 1e-8. Measured on 8
# dispersed example flights against that: RK45 at rtol 1e-6 / atol 1e-4 shifts apogee by
# ≤ 0.15 m and the landing point by ≤ 0.44 m (< 1 % of their run-to-run σ of ~17 m and
# ~135 m) and is 4.4x faster. Gusts put a spline knot every 0.05 s, where a high-order method
# gains nothing and still pays 12 evaluations per step.
SOLVER = {"method": "RK45", "rtol": 1e-6, "atol": 1e-4}
GUST_SPAN = 1800.0  # s of turbulence generated per run; held constant after that

# name → what the drawn value means. *_scale multiply the nominal; the rest add to it, except
# thrust_misalignment and gust_intensity, which are absolute (nominal 0).
PARAMETERS = {
    "impulse_scale": "× total impulse (thrust curve scaled)",
    "burn_time_scale": "× burn time (curve stretched in time, impulse kept)",
    "cd_scale": "× drag coefficient C_D0",
    "dry_mass": "+ kg to the dry mass (at the dry CG)",
    "cg_offset": "+ m to the dry CG position (aft positive)",
    "rail_elevation": "+ deg to the rail elevation (capped at 90)",
    "rail_heading": "+ deg to the rail heading",
    "thrust_misalignment": "|deg| thrust tilt; its azimuth is uniform on [0, 360)",
    "wind_speed": "+ m/s to the reference wind speed (floored at 0)",
    "wind_direction": "+ deg to the wind direction",
    "gust_intensity": "|m/s| turbulence standard deviation σ_u (Niskanen §4.1.2)",
    "chute_cda_scale": "× drogue and main Cd·A",
    "drogue_delay": "+ s to the drogue delay (floored at 0)",
}
_ONE = ("impulse_scale", "burn_time_scale", "cd_scale", "chute_cda_scale")
SUMMARY_COLUMNS = ("apogee", "landing_x", "landing_y", "drift", "rail_exit_velocity",
                   "margin_rail_exit", "min_margin", "max_velocity", "max_mach", "max_accel_g",
                   "opening_shock_g", "landing_velocity", "flight_time")


def load_dispersions(path):
    """Read the optional `dispersions:` section of a rocket config.

    Each entry is `name: {normal: [mean, sigma]}` or `name: {uniform: [low, high]}`; names are
    those in PARAMETERS. Missing names are not dispersed (scale 1, offset 0).
    Returns {name: (distribution, a, b)}. Raises ValueError naming the file and entry.
    """
    path = Path(path)
    cfg = read_yaml(path) or {}
    section = cfg.get("dispersions") or {}
    if not isinstance(section, dict):
        raise ValueError(f"{path}: dispersions must be a mapping")  # noqa: TRY004
    specs = {k: ("normal", 1.0 if k in _ONE else 0.0, 0.0) for k in PARAMETERS}
    for name, entry in section.items():
        where = f"{path}: dispersions.{name}"
        if name not in PARAMETERS:
            raise ValueError(f"{where}: unknown parameter; choose from {list(PARAMETERS)}")
        if not isinstance(entry, dict) or len(entry) != 1 or next(iter(entry)) not in (
                "normal", "uniform"):
            raise ValueError(f"{where}: give exactly one of normal or uniform")
        (dist, values), = entry.items()
        if (not isinstance(values, list) or len(values) != 2
                or not all(isinstance(v, (int, float)) and not isinstance(v, bool)
                           for v in values)):
            raise ValueError(f"{where}: needs two numbers, e.g. [{1 if name in _ONE else 0}, 0.1]")
        a, b = map(float, values)
        if dist == "normal" and b < 0:
            raise ValueError(f"{where}: normal needs σ ≥ 0, got {b}")
        if dist == "uniform" and a > b:
            raise ValueError(f"{where}: uniform needs low ≤ high, got [{a}, {b}]")
        specs[name] = (dist, a, b)
    return specs


def sample(specs, rng):
    """Draw one value per parameter, always in PARAMETERS order (keeps runs reproducible)."""
    out = {}
    for name in PARAMETERS:
        dist, a, b = specs[name]
        out[name] = rng.normal(a, b) if dist == "normal" else rng.uniform(a, b)
    return out


def apply(rocket, launch, p, rng):
    """Return (rocket, launch) with the sampled values p applied; rng supplies the
    misalignment azimuth and the gust sequence. Zero dispersions reproduce the nominal exactly.

    Dry mass/CG: the dry part's mass and CG are shifted and its inertia scaled by the mass
    ratio (shape unchanged). Impulse and burn time: thrust(t) → (k_I/k_t)·thrust(t/k_t).
    """
    mo = rocket.motor
    motor = dataclasses.replace(mo, time=mo.time * p["burn_time_scale"],
                                thrust_data=mo.thrust_data * p["impulse_scale"]
                                / p["burn_time_scale"])
    rec = rocket.recovery
    scale = p["chute_cda_scale"]
    recovery = dataclasses.replace(
        rec, drogue=Chute(rec.drogue.cda * scale, rec.drogue.fill_time),
        drogue_delay=max(rec.drogue_delay + p["drogue_delay"], 0.0),
        main=rec.main and Chute(rec.main.cda * scale, rec.main.fill_time))
    rocket = dataclasses.replace(rocket, motor=motor, recovery=recovery,
                                 cd_scale=rocket.cd_scale * p["cd_scale"],
                                 thrust_misalignment=np.radians(abs(p["thrust_misalignment"])),
                                 misalignment_azimuth=rng.uniform(0.0, 2 * np.pi))
    m, x, i_t, i_x = rocket._dry
    m_new = m + p["dry_mass"]
    if m_new <= 0:
        raise ValueError("dispersed dry mass is not positive")
    ratio = m_new / m  # exactly 1.0 for a zero dispersion, so the nominal is reproduced bit for bit
    rocket._dry = (m_new, x + p["cg_offset"], i_t * ratio, i_x * ratio)

    w = launch.wind
    sigma = abs(p["gust_intensity"])
    wind = dataclasses.replace(
        w, speed=max(w.speed + p["wind_speed"], 0.0),
        direction=w.direction + np.radians(p["wind_direction"]), gust_sigma=sigma,
        gust=gust_signal(pink_noise(rng, int(GUST_SPAN / GUST_DT))) if sigma else None)
    launch = dataclasses.replace(
        launch, wind=wind,
        rail_elevation=min(launch.rail_elevation + np.radians(p["rail_elevation"]), np.pi / 2),
        rail_heading=launch.rail_heading + np.radians(p["rail_heading"]))
    return rocket, launch


def run_one(config_path, specs, seed_seq):
    """One dispersed flight. Returns a row: run parameters, summary values, status."""
    rng = np.random.default_rng(seed_seq)
    params = sample(specs, rng)
    row = dict(params)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # per-run warnings would flood the console
            s = simulate(*apply(*load_config(config_path), params, rng), dt_out=DT_OUT,
                         **SOLVER).summary
        row |= {k: s[k] for k in SUMMARY_COLUMNS if k in s}
        row["landing_x"], row["landing_y"] = s["landing_position"]
        row["status"] = "ok"
    except (ValueError, RuntimeError) as exc:
        row |= dict.fromkeys(SUMMARY_COLUMNS, np.nan)
        row["status"] = f"failed: {exc}"
    return row


def run(config_path, specs, n, seed, workers=None):
    """Run n dispersed flights in parallel; rows come back in run order."""
    children = np.random.SeedSequence(seed).spawn(n)
    workers = workers or os.cpu_count()
    if workers == 1:
        return [run_one(config_path, specs, c) for c in children]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(run_one, [config_path] * n, [specs] * n, children))


def covariance_ellipse(points, n_sigma):
    """n-σ ellipse of 2-D points from their covariance matrix.

    The eigenvectors of the sample covariance are the ellipse axes and the square roots of its
    eigenvalues the 1-σ semi-axes. For a 2-D Gaussian, the 1/2/3-σ ellipses hold about 39 %,
    86 % and 99 % of points.
    Returns (center [2], semi-axes [major, minor], angle of the major axis from +x, rad).
    """
    points = np.asarray(points, dtype=float)
    values, vectors = np.linalg.eigh(np.cov(points.T))
    order = np.argsort(values)[::-1]
    values, vectors = values[order], vectors[:, order]
    angle = np.arctan2(vectors[1, 0], vectors[0, 0])
    return points.mean(axis=0), n_sigma * np.sqrt(values), angle


def statistics(rows):
    """Summary statistics of the successful runs."""
    ok = [r for r in rows if r["status"] == "ok"]
    stats = {"runs": len(rows), "failed": len(rows) - len(ok)}
    if not ok:
        return stats
    for key in ("apogee", "rail_exit_velocity", "margin_rail_exit", "drift"):
        v = np.array([r[key] for r in ok])
        stats[key] = {"mean": v.mean(), "std": v.std(ddof=1) if len(v) > 1 else 0.0,
                      "min": v.min(), "p5": np.percentile(v, 5), "p50": np.percentile(v, 50),
                      "p95": np.percentile(v, 95), "max": v.max()}
    if len(ok) > 2:
        landing = [[r["landing_x"], r["landing_y"]] for r in ok]
        stats["landing_ellipses"] = {k: covariance_ellipse(landing, k) for k in (1, 2, 3)}
    return stats
