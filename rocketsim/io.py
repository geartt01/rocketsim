"""YAML config loading and CSV export. See configs/example_rocket.yaml for the format."""

import csv
from pathlib import Path

import numpy as np
import yaml

from rocketsim.dynamics.dof3 import Launch
from rocketsim.geometry import FIN_PROFILES, NOSE_SHAPES, Body, Fins, MassItem, Nose, Rocket
from rocketsim.motor import load_eng
from rocketsim.wind import Wind


def load_config(path):
    """Load a rocket config into (geometry.Rocket, dof3.Launch).

    Every key is required unless noted, unknown keys are rejected, and numbers are range-checked,
    so a typo fails loudly instead of silently falling back to a default. Paths (motor, drag
    table) are relative to the config file. Angles are degrees in YAML, radians inside.
    Raises ValueError naming the file and the offending key.
    """
    path = Path(path)

    def fail(msg):
        raise ValueError(f"{path}: {msg}")

    def section(parent, key, where, required, optional=()):
        sec = parent.get(key) if isinstance(parent, dict) else None
        if not isinstance(sec, dict):
            fail(f"'{where}' must be a mapping")
        missing, unknown = set(required) - set(sec), set(sec) - set(required) - set(optional)
        if missing or unknown:
            fail(f"'{where}': missing {sorted(missing)}, unknown {sorted(unknown)}")
        return sec

    def number(sec, key, where, low=None, low_ok=False, high=None):
        value = sec[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or np.isnan(value):
            fail(f"{where}.{key} must be a number, got {value!r}")
        if low is not None and (value < low or (value == low and not low_ok)):
            fail(f"{where}.{key} must be {'≥' if low_ok else '>'} {low}, got {value}")
        if high is not None and value > high:
            fail(f"{where}.{key} must be ≤ {high}, got {value}")
        return float(value)

    def choice(sec, key, where, options):
        if sec[key] not in options:
            fail(f"{where}.{key} must be one of {list(options)}, got {sec[key]!r}")
        return sec[key]

    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    top = ("rocket", "motor", "recovery", "launch", "wind")
    if not isinstance(cfg, dict) or set(cfg) != set(top):
        fail(f"top-level keys must be exactly {list(top)}")

    r = section(cfg, "rocket", "rocket", ("nose", "body", "fins", "masses", "motor_position",
                                          "surface_roughness", "drag"))
    s = section(r, "nose", "rocket.nose", ("shape", "length", "mass"))
    nose = Nose(choice(s, "shape", "rocket.nose", NOSE_SHAPES),
                number(s, "length", "rocket.nose", 0), number(s, "mass", "rocket.nose", 0))
    s = section(r, "body", "rocket.body", ("length", "diameter", "mass"))
    body = Body(*(number(s, k, "rocket.body", 0) for k in ("length", "diameter", "mass")))
    if nose.shape == "ogive" and nose.length < body.diameter / 2:
        fail("rocket.nose: a tangent ogive needs length ≥ body radius")

    where = "rocket.fins"
    s = section(r, "fins", where, ("count", "root_chord", "tip_chord", "span", "sweep",
                                   "thickness", "position", "profile", "mass"))
    if not isinstance(s["count"], int) or isinstance(s["count"], bool) or s["count"] < 3:
        fail(f"{where}.count must be an integer ≥ 3 (evenly spaced fins)")
    fins = Fins(
        count=s["count"], root_chord=number(s, "root_chord", where, 0),
        tip_chord=number(s, "tip_chord", where, 0, low_ok=True), span=number(s, "span", where, 0),
        sweep=number(s, "sweep", where), thickness=number(s, "thickness", where, 0),
        position=number(s, "position", where, 0, low_ok=True),
        profile=choice(s, "profile", where, FIN_PROFILES), mass=number(s, "mass", where, 0))

    if not isinstance(r["masses"], list):
        fail("rocket.masses must be a list (may be empty: [])")
    masses = []
    for i, item in enumerate(r["masses"]):
        where = f"rocket.masses[{i}]"
        s = section({"m": item}, "m", where, ("name", "mass", "position"), ("length", "radius"))
        if ("length" in s) != ("radius" in s):
            fail(f"{where}: give both length and radius (cylinder) or neither (point mass)")
        masses.append(MassItem(str(s["name"]), number(s, "mass", where, 0),
                               number(s, "position", where, 0, low_ok=True),
                               *(number(s, k, where, 0) for k in ("length", "radius") if k in s)))

    drag = r["drag"]
    if drag == "buildup":
        pass
    elif isinstance(drag, str):
        drag = _load_cd_table(path.parent / drag, fail)
    else:
        drag = number(r, "drag", "rocket", 0)

    if not isinstance(cfg["motor"], str):
        fail("motor must be a path to a .eng file")
    s = section(cfg, "recovery", "recovery", ("chute_cda",))
    rocket = Rocket(nose, body, fins, masses, load_eng(path.parent / cfg["motor"]),
                    number(r, "motor_position", "rocket", 0),
                    number(r, "surface_roughness", "rocket", 0), drag,
                    number(s, "chute_cda", "recovery", 0))

    s = section(cfg, "wind", "wind", ("speed", "reference_height", "profile"))
    wind = Wind(number(s, "speed", "wind"), number(s, "reference_height", "wind", 0),
                choice(s, "profile", "wind", ("constant", "power_law")))
    s = section(cfg, "launch", "launch", ("site_altitude", "rail_length", "rail_elevation"))
    launch = Launch(number(s, "site_altitude", "launch", -2000, low_ok=True, high=20000),
                    number(s, "rail_length", "launch", 0),
                    np.radians(number(s, "rail_elevation", "launch", 0, high=90)), wind)
    return rocket, launch


def _load_cd_table(path, fail):
    """Read a two-column CSV 'mach,cd' (one header line) into (mach, cd) arrays."""
    try:
        data = np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2)
    except (OSError, ValueError) as exc:
        fail(f"cannot read drag table {path}: {exc}")
    if data.shape[1] != 2 or len(data) < 2 or np.any(np.diff(data[:, 0]) <= 0) \
            or np.any(data[:, 1] <= 0):
        fail(f"drag table {path} needs ≥ 2 rows of 'mach,cd' with increasing mach and cd > 0")
    return data[:, 0], data[:, 1]


def write_trajectory_csv(flight, path):
    """Write flight.trajectory as CSV, one column per key, one row per sample."""
    columns = list(flight.trajectory)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(columns)
        writer.writerows(zip(*(flight.trajectory[c] for c in columns)))
