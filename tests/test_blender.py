"""Stage 8: the Blender add-on, in real Blender when it is installed, and its data package."""

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from rocketsim.cli import find_blender
from rocketsim.export import write_flight_package

ROOT = Path(__file__).parents[1]
EXAMPLE = ROOT / "configs" / "example_rocket.yaml"


def test_flight_package_contents(tmp_path):
    out = tmp_path / "flight.npz"
    flight = write_flight_package(EXAMPLE, out)
    with np.load(out, allow_pickle=False) as z:
        meta = json.loads(str(z["meta"]))
        assert np.all(np.diff(z["t"]) > 0)
        assert z["z"].max() == pytest.approx(flight.summary["apogee"], rel=1e-3)
        assert z["profile_x"][0] == 0.0 and z["profile_x"][-1] == pytest.approx(meta["length"])
        assert z["profile_r"][0] == pytest.approx(0.0) and z["profile_r"][-1] == meta["radius"]
        assert z["drogue_open"][0] == 0.0 and z["drogue_open"][-1] == 1.0
        assert z["thrust"].max() <= meta["motor"]["peak_thrust"] + 1e-9
        assert set(z["phase"]) >= {"rail", "powered", "coast", "drogue", "landed"}
    assert meta["events"]["apogee"] == flight.events["apogee"]


def test_bad_rocket_gives_a_readable_error(tmp_path):
    config = tmp_path / "bad.yaml"
    config.write_text(EXAMPLE.read_text(encoding="utf-8").replace("cd: 0.0565", "cd: -1")
                      .replace("cda: 0.0565", "cda: -1"), encoding="utf-8")
    result = subprocess.run([sys.executable, "-m", "rocketsim", "flightdata", str(config), "-o",
                             str(tmp_path / "x.npz")], capture_output=True, text=True, cwd=ROOT,
                            encoding="utf-8", check=False)
    assert result.returncode == 1
    assert result.stderr.startswith("ERROR: ") and "cda" in result.stderr


@pytest.mark.skipif(find_blender() is None, reason="Blender is not installed")
def test_addon_inside_blender():
    result = subprocess.run(
        [find_blender(), "--background", "--factory-startup", "--python",
         str(ROOT / "tests" / "blender_selftest.py")],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        env=os.environ | {"ROCKETSIM_PYTHON": sys.executable}, timeout=540, check=False)
    assert "SELFTEST PASS" in result.stdout, result.stdout[-3000:] + result.stderr[-1000:]
