"""Stage 2: 3-DOF planar flight invariants."""

import contextlib
import dataclasses
import warnings
from pathlib import Path

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from rocketsim.dynamics import dof3
from rocketsim.dynamics.dof1 import Rocket1D
from rocketsim.flight import simulate_1dof, simulate_3dof
from rocketsim.io import load_config
from rocketsim.wind import Wind

EXAMPLE = Path(__file__).parents[1] / "configs" / "example_rocket.yaml"


def setup(wind_speed=0.0, elevation_deg=90.0, profile="constant", cd=None):
    rocket, launch = load_config(EXAMPLE)
    if cd is not None:
        rocket.drag = cd
    launch.rail_elevation = np.radians(elevation_deg)
    launch.rail_heading = np.radians(90.0)  # downrange x = East
    launch.wind = Wind(wind_speed, np.radians(270.0), 10.0, profile)  # blows toward +x
    return rocket, launch


def test_reduces_to_1dof_on_vertical_rail_without_wind():
    rocket, launch = setup(cd=0.5)
    flight = simulate_3dof(rocket, launch)
    r1 = Rocket1D(rocket.dry_mass, rocket.body.diameter, 0.5, rocket.motor,
                  rocket.recovery.drogue.cda,
                  launch.site_altitude)
    ref = simulate_1dof(r1)
    assert flight.summary["apogee"] == pytest.approx(ref.summary["apogee"], rel=1e-3)
    # cos(π/2) is 6e-17 in floating point, which seeds nm-level drift; PLAN's bound is 1e-6 m
    assert np.abs(flight.trajectory["x"]).max() < 1e-6
    assert np.allclose(flight.trajectory["theta"], np.pi / 2)


def test_stable_rocket_restores_angle_of_attack():
    """Coasting straight up with the body tilted 5°: it must turn back and the wobble decay."""
    rocket, launch = setup()
    launch = dataclasses.replace(launch, g=0.0, density=lambda z: 1.2)
    t0 = rocket.motor.burn_time + 1.0
    y0 = [0.0, 100.0, 0.0, 100.0, np.radians(95.0), 0.0]
    assert dof3.derivatives(t0, y0, rocket, launch)[5] < 0  # pitches back toward the velocity

    sol = solve_ivp(dof3.derivatives, (t0, t0 + 2.0), y0, method="DOP853", rtol=1e-8,
                    atol=1e-8, args=(rocket, launch), dense_output=True)
    t = np.linspace(t0, t0 + 2.0, 4001)
    y = sol.sol(t)
    alpha = y[4] - np.arctan2(y[3], y[2])  # signed angle between body axis and velocity
    first, last = np.abs(alpha[t < t0 + 0.5]).max(), np.abs(alpha[t > t0 + 1.5]).max()
    assert first == pytest.approx(np.radians(5.0), rel=1e-3)
    assert last < 0.5 * first


def test_weathercocks_into_wind():
    # Wind blows toward +x, i.e. comes from −x: the nose must turn toward −x (θ > 90°).
    flight = simulate_3dof(*setup(wind_speed=5.0))
    tr = flight.trajectory
    powered = tr["phase"] == "powered"
    assert tr["theta"][powered][-1] > np.pi / 2
    assert flight.summary["apogee_position"][0] < 0


def test_drifts_with_the_wind_under_parachute():
    flight = simulate_3dof(*setup(wind_speed=4.0))
    tr = flight.trajectory
    chute = tr["phase"] == "drogue"
    assert tr["vx"][chute][-1] == pytest.approx(4.0, rel=1e-3)
    assert flight.summary["landing_position"][0] > flight.summary["apogee_position"][0]


def test_rail_exit_at_rail_length():
    rocket, launch = setup(elevation_deg=80.0)
    flight = simulate_3dof(rocket, launch)
    rail = flight.trajectory["phase"] == "rail"
    assert np.allclose(flight.trajectory["theta"][rail], np.radians(80.0))
    t_exit = flight.events["rail_exit"]
    assert flight.events["liftoff"] < t_exit < flight.events["burnout"]
    assert flight.summary["rail_exit_velocity"] > 0


def test_warns_when_unstable():
    rocket, launch = setup()
    rocket.masses[0] = dataclasses.replace(rocket.masses[0], mass=0.2, position=0.37)
    rocket.__post_init__()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        # an unstable rocket may not fly a sensible trajectory; the warning is the point
        with contextlib.suppress(ValueError, RuntimeError):
            simulate_3dof(rocket, launch)
    assert any("stability margin" in str(w.message) for w in caught)
