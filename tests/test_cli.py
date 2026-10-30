from pathlib import Path

import numpy as np
import pytest

from rocketsim.cli import main
from rocketsim.io import load_config

ROOT = Path(__file__).parents[1]
EXAMPLE = ROOT / "configs" / "example_rocket.yaml"
MOTOR = (Path(__file__).parent / "data" / "test_motor.eng").as_posix()
VALID = f"""
rocket:
  nose: {{shape: ogive, length: 0.08, mass: 0.005}}
  body: {{length: 0.3, diameter: 0.025, mass: 0.012}}
  fins: {{count: 3, root_chord: 0.06, tip_chord: 0.03, span: 0.045, sweep: 0.03,
          thickness: 0.002, position: 0.32, profile: rounded, mass: 0.005}}
  masses: [{{name: chute, mass: 0.008, position: 0.14}}]
  motor_position: 0.38
  surface_roughness: 60.0e-6
  drag: 0.5
motor: {MOTOR}
recovery: {{chute_cda: 0.05}}
launch: {{site_altitude: 300, rail_length: 1.0, rail_elevation: 85}}
wind: {{speed: 2.0, reference_height: 10.0, profile: constant}}
"""


def write(tmp_path, text):
    path = tmp_path / "rocket.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def test_example_config_loads():
    rocket, launch = load_config(EXAMPLE)
    assert rocket.motor.name == "C6"
    assert rocket.dry_mass == pytest.approx(0.034)
    assert launch.rail_elevation == pytest.approx(np.radians(85.0))


def test_valid_config(tmp_path):
    rocket, launch = load_config(write(tmp_path, VALID))
    assert launch.site_altitude == 300.0
    assert rocket.drag == 0.5
    assert rocket.motor.name == "TEST"


def test_drag_table(tmp_path):
    (tmp_path / "cd.csv").write_text("mach,cd\n0.0,0.45\n0.5,0.50\n")
    rocket, _ = load_config(write(tmp_path, VALID.replace("drag: 0.5", "drag: cd.csv")))
    assert np.allclose(rocket.drag[1], [0.45, 0.50])


@pytest.mark.parametrize(
    ("old", "new", "message"),
    [
        ("root_chord: 0.06", "rootchord: 0.06", "unknown"),
        ("chute_cda: 0.05", "chute_cda: -0.05", "must be >"),
        ("mass: 0.012", "mass: heavy", "must be a number"),
        ("count: 3", "count: 2", "≥ 3"),
        ("shape: ogive, length: 0.08", "shape: ogive, length: 0.01", "tangent ogive"),
        ("profile: rounded", "profile: wavy", "must be one of"),
        ("rail_elevation: 85", "rail_elevation: 95", "≤ 90"),
        ("drag: 0.5", "drag: missing.csv", "cannot read drag table"),
        ("wind:", "breeze:", "top-level keys"),
    ],
)
def test_bad_config_rejected(tmp_path, old, new, message):
    assert old in VALID
    with pytest.raises(ValueError, match=message):
        load_config(write(tmp_path, VALID.replace(old, new)))


def test_cli_run(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    main(["run", str(EXAMPLE)])
    printed = capsys.readouterr().out
    for label in ("Apogee", "Rail-exit velocity", "Stability margin", "Drift"):
        assert label in printed
    out = tmp_path / "outputs" / "example_rocket"
    header = (out / "trajectory.csv").read_text().splitlines()[0]
    assert header == "t,x,z,vx,vz,ax,az,theta,alpha,mach,margin,mass,phase"
    assert (out / "flight.png").stat().st_size > 0
