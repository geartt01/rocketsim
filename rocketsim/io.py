"""YAML config loading and CSV export."""

import csv
from pathlib import Path

import yaml

from rocketsim.dynamics.dof1 import Rocket1D
from rocketsim.motor import load_eng

# section → required numeric keys; "motor" is a path string, relative to the config file
SCHEMA = {
    "rocket": ("dry_mass", "diameter", "cd"),
    "motor": None,
    "recovery": ("chute_cda",),
    "launch": ("elevation",),
}
POSITIVE = ("dry_mass", "diameter", "cd", "chute_cda")


def load_config(path):
    """Load a rocket config (see configs/example_rocket.yaml) into a Rocket1D.

    Every key is required and unknown keys are rejected, so a typo fails loudly instead of
    silently falling back to a default. Raises ValueError naming the file.
    """
    path = Path(path)

    def fail(msg):
        raise ValueError(f"{path}: {msg}")

    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(cfg, dict) or set(cfg) != set(SCHEMA):
        fail(f"top-level keys must be exactly {sorted(SCHEMA)}")
    values = {}
    for section, keys in SCHEMA.items():
        if keys is None:
            continue
        sec = cfg[section]
        if not isinstance(sec, dict) or set(sec) != set(keys):
            fail(f"'{section}' needs exactly the keys {list(keys)}")
        for key in keys:
            value = sec[key]
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                fail(f"{section}.{key} must be a number, got {value!r}")
            values[key] = float(value)
    for key in POSITIVE:
        if not values[key] > 0:
            fail(f"{key} must be positive, got {values[key]}")
    if not isinstance(cfg["motor"], str):
        fail("motor must be a path to a .eng file")

    return Rocket1D(motor=load_eng(path.parent / cfg["motor"]), **values)


def write_trajectory_csv(flight, path):
    """Write flight.trajectory as CSV, one column per key, one row per sample."""
    columns = list(flight.trajectory)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(columns)
        writer.writerows(zip(*(flight.trajectory[c] for c in columns)))
