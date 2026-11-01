"""Stage 6: IMU and barometer output, the interface to the flight-computer project."""

import dataclasses
import json
from pathlib import Path

import numpy as np
import pytest

from rocketsim import frames, sensors
from rocketsim.atmosphere import isa
from rocketsim.cli import main
from rocketsim.constants import G0
from rocketsim.flight import simulate
from rocketsim.io import load_config
from rocketsim.wind import Wind

EXAMPLE = Path(__file__).parents[1] / "configs" / "example_rocket.yaml"
CONFIG = sensors.load_sensor_config(EXAMPLE)


@pytest.fixture(scope="module")
def flown():
    """Example rocket on a vertical rail, no wind."""
    rocket, launch = load_config(EXAMPLE)
    launch = dataclasses.replace(launch, rail_elevation=np.pi / 2,
                                 wind=Wind(0.0, 0.0, 10.0, "constant"))
    flight = simulate(rocket, launch, dt_out=0.05)
    return flight, launch, sensors.truth_at_rate(flight, launch, 500.0, 2.0, 2.0)


def test_truth_grid_covers_pad_to_after_landing(flown):
    flight, _, truth = flown
    assert truth["t"][0] == pytest.approx(-2.0)
    assert truth["t"][-1] >= flight.events["landing"] + 2.0 - 1 / 500
    assert np.allclose(np.diff(truth["t"]), 1 / 500)
    assert list(dict.fromkeys(truth["phase"])) == ["pad", "rail", "powered", "coast", "drogue",
                                                   "landed"]


def test_accelerometer_reads_plus_1g_on_the_pad(flown):
    _, _, truth = flown
    f = sensors.specific_force(truth, G0)
    pad = truth["phase"] == "pad"
    assert f[pad, 0] == pytest.approx(G0)  # +1 g along the nose axis
    assert f[pad, 1:] == pytest.approx(0.0, abs=1e-12)


def test_accelerometer_reads_only_drag_while_coasting(flown):
    # In free fall gravity is not felt: the reading is drag / mass, pointing aft.
    _, _, truth = flown
    f = sensors.specific_force(truth, G0)
    coast = truth["phase"] == "coast"
    assert np.all(f[coast, 0] < 0)
    first_coast = np.flatnonzero(coast)[0]
    assert abs(f[first_coast, 0]) > 3 * G0  # strong drag right after burnout at ~115 m/s
    assert abs(f[np.flatnonzero(coast)[-1], 0]) < 0.05 * G0  # near apogee the air barely pushes


def test_integrating_ideal_accelerometer_recovers_velocity(flown):
    flight, launch, truth = flown
    data = sensors.measure(truth, CONFIG, launch, np.random.default_rng(0), ideal=True)
    ascent = np.isin(truth["phase"], ("pad", "rail", "powered", "coast"))
    rot = frames.rotation_matrix(np.array([truth[k] for k in ("qw", "qx", "qy", "qz")]))
    # inertial acceleration = R·f − g·ẑ, trapezoid-integrated at the sample rate
    a = np.einsum("ijn,nj->ni", rot, np.column_stack([data["ax"], data["ay"], data["az"]]))
    a[:, 2] -= G0
    dt = 1 / 500
    v = np.cumsum(0.5 * (a[1:] + a[:-1]) * dt, axis=0)
    i_burnout = np.searchsorted(truth["t"], flight.events["burnout"])
    assert v[i_burnout - 1, 2] == pytest.approx(truth["vz"][i_burnout], rel=0.01)
    last = np.flatnonzero(ascent)[-1]
    peak = truth["vz"][ascent].max()
    assert abs(v[last - 1, 2] - truth["vz"][last]) < 0.01 * peak


def test_noise_statistics_and_bias(flown):
    _, launch, truth = flown
    data = sensors.measure(truth, CONFIG, launch, np.random.default_rng(1))
    pad = truth["phase"] == "pad"
    ax_cfg, gy_cfg = CONFIG.accel, CONFIG.gyro
    # enough pad samples for σ within 5 %? 1000 samples give ±2.2 % (1σ)
    assert data["ay"][pad].std() == pytest.approx(ax_cfg.noise, rel=0.1)
    assert data["ay"][pad].mean() == pytest.approx(ax_cfg.bias[1], abs=4 * ax_cfg.noise / 30)
    assert data["gz"][pad].mean() == pytest.approx(gy_cfg.bias[2], abs=5 * gy_cfg.noise / 30)


def test_noise_statistics_long_sample():
    rng = np.random.default_rng(2)
    n = 200_000
    truth = {"t": np.arange(n) / 500.0, "z": np.zeros(n), "phase": np.full(n, "pad")}
    truth |= {k: np.zeros(n) for k in ("wx", "wy", "wz", "ax", "ay", "az", "qx", "qy", "qz")}
    truth["qw"] = np.ones(n)
    _, launch = load_config(EXAMPLE)
    config = dataclasses.replace(CONFIG, gyro=dataclasses.replace(CONFIG.gyro, walk=0.0))
    data = sensors.measure(truth, config, launch, rng)
    assert data["ay"].std() == pytest.approx(config.accel.noise, rel=0.05)
    assert data["gx"].std() == pytest.approx(config.gyro.noise, rel=0.05)
    assert data["ay"].mean() == pytest.approx(config.accel.bias[1], abs=0.01 * G0)
    assert data["gx"].mean() == pytest.approx(config.gyro.bias[0], rel=0.05)


def test_saturation_and_quantisation(flown):
    # The C6 thrust spike exceeds a ±16 g sensor: readings must clip, as on real hardware.
    _, launch, truth = flown
    data = sensors.measure(truth, CONFIG, launch, np.random.default_rng(3))
    assert data["ax"].max() <= CONFIG.accel.range
    assert np.isclose(data["ax"].max(), CONFIG.accel.range, rtol=1e-3)
    lsb = 2 * CONFIG.accel.range / 2**CONFIG.accel.bits
    steps = data["ax"] / lsb
    assert np.allclose(steps, np.round(steps), atol=1e-6)


def test_barometer(flown):
    _, launch, truth = flown
    ideal = sensors.measure(truth, CONFIG, launch, np.random.default_rng(4), ideal=True)
    expected = isa(launch.site_altitude + truth["z"]).p
    assert ideal["pressure_pa"] == pytest.approx(expected)
    noisy = sensors.measure(truth, CONFIG, launch, np.random.default_rng(4))
    pad = truth["phase"] == "pad"
    assert noisy["pressure_pa"][pad].std() == pytest.approx(CONFIG.baro.noise, rel=0.2)


def test_no_spin_reported_under_parachute(flown):
    _, _, truth = flown
    chute = truth["phase"] == "drogue"
    assert np.all(truth["wx"][chute] == 0) and np.all(truth["wy"][chute] == 0)


def test_same_seed_same_data(flown):
    _, launch, truth = flown
    a = sensors.measure(truth, CONFIG, launch, np.random.default_rng(9))
    b = sensors.measure(truth, CONFIG, launch, np.random.default_rng(9))
    assert np.array_equal(a["ax"], b["ax"]) and np.array_equal(a["gy"], b["gy"])


def test_cli_writes_schema_files(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    main(["sensors", str(EXAMPLE), "--seed", "3"])
    assert "sensors.csv" in capsys.readouterr().out
    out = tmp_path / "outputs" / "example_rocket"
    rows = (out / "sensors.csv").read_text().splitlines()
    assert rows[0] == "t,ax,ay,az,gx,gy,gz,pressure_pa,temperature_k"
    # 500 Hz IMU and 50 Hz barometer share one file; every 10th row has barometer values
    assert rows[1].split(",")[7] != "" and rows[2].split(",")[7] == ""
    assert (out / "truth.csv").read_text().splitlines()[0] == (
        "t,x,y,z,vx,vy,vz,qw,qx,qy,qz,wx,wy,wz,phase")
    events = json.loads((out / "events.json").read_text())
    assert events["liftoff"] < events["rail_exit"] < events["burnout"] < events["apogee"]


@pytest.mark.parametrize(("old", "new", "message"), [
    ("imu_rate: 500", "imu_rate: -5", "must be >"),
    ("bits: 16", "bits: 1.5", "must be an integer"),
    ("noise_pa:", "noise_kpa:", "unknown"),
])
def test_bad_sensor_config_rejected(tmp_path, old, new, message):
    text = EXAMPLE.read_text(encoding="utf-8")
    assert old in text
    text = text.replace(old, new, 1).replace("../motors/", EXAMPLE.parents[1].as_posix() + "/motors/")
    path = tmp_path / "bad.yaml"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        sensors.load_sensor_config(path)
