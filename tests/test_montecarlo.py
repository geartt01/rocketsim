"""Stage 5: Monte Carlo dispersion."""

import dataclasses
from pathlib import Path

import numpy as np
import pytest

from rocketsim import montecarlo
from rocketsim.flight import simulate
from rocketsim.io import load_config
from rocketsim.wind import pink_noise

EXAMPLE = Path(__file__).parents[1] / "configs" / "example_rocket.yaml"
SCALES = ("impulse_scale", "burn_time_scale", "cd_scale", "chute_cda_scale")
ZERO = {k: ("normal", 1.0 if k in SCALES else 0.0, 0.0) for k in montecarlo.PARAMETERS}


def test_same_seed_same_results():
    disp = montecarlo.load_dispersions(EXAMPLE)
    first = montecarlo.run(EXAMPLE, disp, n=4, seed=7, workers=2)
    second = montecarlo.run(EXAMPLE, disp, n=4, seed=7, workers=1)
    assert first == second  # also independent of the number of worker processes
    other = montecarlo.run(EXAMPLE, disp, n=4, seed=8, workers=2)
    assert [r["apogee"] for r in other] != [r["apogee"] for r in first]


def test_zero_dispersions_reproduce_nominal():
    nominal = simulate(*load_config(EXAMPLE), dt_out=montecarlo.DT_OUT, **montecarlo.SOLVER).summary
    for row in montecarlo.run(EXAMPLE, ZERO, n=3, seed=1, workers=1):
        assert row["status"] == "ok"
        for key in ("apogee", "rail_exit_velocity", "margin_rail_exit", "drift", "flight_time"):
            assert row[key] == nominal[key]
        assert (row["landing_x"], row["landing_y"]) == nominal["landing_position"]


def test_covariance_ellipse_recovers_known_covariance():
    rng = np.random.default_rng(3)
    angle = np.radians(30.0)
    rot = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    cov = rot @ np.diag([40.0**2, 10.0**2]) @ rot.T
    points = rng.multivariate_normal([100.0, -50.0], cov, size=200_000)
    center, semi_axes, angle_est = montecarlo.covariance_ellipse(points, n_sigma=2)
    assert center == pytest.approx([100.0, -50.0], abs=0.5)
    assert semi_axes == pytest.approx([80.0, 20.0], rel=0.01)
    assert np.cos(2 * (angle_est - angle)) == pytest.approx(1.0, abs=1e-3)  # axis, not sign


def test_pink_noise_is_normalised():
    x = pink_noise(np.random.default_rng(4), 400_000)
    assert x.mean() == pytest.approx(0.0, abs=0.05)
    assert x.std() == pytest.approx(1.0, rel=0.05)  # Niskanen's 2.252 normalisation


def test_motor_scaling():
    rocket, launch = load_config(EXAMPLE)
    params = dict.fromkeys(montecarlo.PARAMETERS, 0.0) | {k: 1.0 for k in SCALES}
    params |= {"impulse_scale": 1.1, "burn_time_scale": 0.9}
    scaled, _ = montecarlo.apply(rocket, launch, params, np.random.default_rng(0))
    assert scaled.motor.total_impulse == pytest.approx(1.1 * rocket.motor.total_impulse)
    assert scaled.motor.burn_time == pytest.approx(0.9 * rocket.motor.burn_time)


def test_thrust_misalignment_pushes_sideways():
    rocket, launch = load_config(EXAMPLE)
    launch = dataclasses.replace(launch, rail_elevation=np.pi / 2,
                                 wind=dataclasses.replace(launch.wind, speed=0.0))
    straight = simulate(rocket, launch).summary["apogee_position"]
    rocket.thrust_misalignment = np.radians(1.0)
    tilted = simulate(rocket, launch).summary["apogee_position"]
    assert np.linalg.norm(straight) < 1e-6
    assert np.linalg.norm(tilted) > 0.1


@pytest.mark.parametrize(("text", "message"), [
    ("dispersions: {cd_scal: {normal: [1, 0.1]}}", "unknown"),
    ("dispersions: {cd_scale: {gaussian: [1, 0.1]}}", "normal or uniform"),
    ("dispersions: {cd_scale: {normal: [1, -0.1]}}", "σ ≥ 0"),
    ("dispersions: {cd_scale: {uniform: [2, 1]}}", "low ≤ high"),
])
def test_bad_dispersions_rejected(tmp_path, text, message):
    config = EXAMPLE.read_text(encoding="utf-8").split("dispersions:")[0]
    config = config.replace("../motors/", (EXAMPLE.parent.parent / "motors").as_posix() + "/")
    path = tmp_path / "bad.yaml"
    path.write_text(config + text + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        montecarlo.load_dispersions(path)
