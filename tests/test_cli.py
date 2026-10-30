from pathlib import Path

import pytest

from rocketsim.cli import main
from rocketsim.io import load_config

ROOT = Path(__file__).parents[1]
EXAMPLE = ROOT / "configs" / "example_rocket.yaml"
MOTOR = (Path(__file__).parent / "data" / "test_motor.eng").as_posix()
VALID = f"""
rocket: {{dry_mass: 0.05, diameter: 0.025, cd: 0.5}}
motor: {MOTOR}
recovery: {{chute_cda: 0.05}}
launch: {{elevation: 300}}
"""


def write(tmp_path, text):
    path = tmp_path / "rocket.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def test_example_config_loads():
    rocket = load_config(EXAMPLE)
    assert rocket.motor.name == "C6"
    assert rocket.dry_mass == 0.035


def test_valid_config(tmp_path):
    rocket = load_config(write(tmp_path, VALID))
    assert rocket.elevation == 300.0
    assert rocket.motor.name == "TEST"


@pytest.mark.parametrize(
    ("old", "new", "message"),
    [
        ("cd: 0.5", "Cd: 0.5", "needs exactly"),
        ("chute_cda: 0.05", "chute_cda: -0.05", "must be positive"),
        ("dry_mass: 0.05", "dry_mass: heavy", "must be a number"),
        ("launch:", "site:", "top-level keys"),
    ],
)
def test_bad_config_rejected(tmp_path, old, new, message):
    with pytest.raises(ValueError, match=message):
        load_config(write(tmp_path, VALID.replace(old, new)))


def test_cli_run(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    main(["run", str(EXAMPLE)])
    assert "Apogee" in capsys.readouterr().out
    out = tmp_path / "outputs" / "example_rocket"
    header = (out / "trajectory.csv").read_text().splitlines()[0]
    assert header == "t,z,vz,az,mass,phase"
    assert (out / "flight.png").stat().st_size > 0
