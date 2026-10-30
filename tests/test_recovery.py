"""Stage 4: drogue/main recovery, inflation and landing."""

import dataclasses
from pathlib import Path

import numpy as np
import pytest

from rocketsim.constants import G0
from rocketsim.flight import simulate
from rocketsim.io import load_config
from rocketsim.recovery import Chute, Recovery, open_cda
from rocketsim.wind import Wind

EXAMPLE = Path(__file__).parents[1] / "configs" / "example_rocket.yaml"
RHO = 1.2  # constant density so terminal velocities have a closed form


def setup(recovery, speed=0.0, elevation_deg=90.0, constant_rho=False):
    rocket, launch = load_config(EXAMPLE)
    rocket.recovery = recovery
    launch = dataclasses.replace(launch, rail_elevation=np.radians(elevation_deg),
                                 rail_heading=np.radians(30.0),
                                 wind=Wind(speed, np.radians(250.0), 10.0, "constant"),
                                 density=(lambda z: RHO) if constant_rho else None)
    return rocket, launch


def terminal(rocket, t, cda):
    return np.sqrt(2 * rocket.mass_properties(t)[0] * G0 / (RHO * cda))


def test_open_cda_ramps_and_adds():
    drogue, main = Chute(0.01, 0.5), Chute(0.05, 0.0)
    deployments = [(drogue, 10.0), (main, 20.0)]
    assert open_cda(9.9, deployments) == 0.0
    assert open_cda(10.25, deployments) == pytest.approx(0.005)
    assert open_cda(15.0, deployments) == pytest.approx(0.01)
    assert open_cda(20.0, deployments) == pytest.approx(0.06)  # instant main + open drogue


def test_terminal_velocity_under_each_chute():
    rec = Recovery(Chute(0.01, 0.2), 0.0, Chute(0.06, 0.3), 120.0)
    rocket, launch = setup(rec, constant_rho=True)
    flight = simulate(rocket, launch)
    ev, s = flight.events, flight.summary
    assert s["drogue_descent_rate"] == pytest.approx(terminal(rocket, ev["main"], 0.01), rel=1e-4)
    assert s["main_descent_rate"] == pytest.approx(terminal(rocket, ev["landing"], 0.07), rel=1e-4)
    assert s["landing_velocity"] == pytest.approx(s["main_descent_rate"], rel=1e-6)


def test_main_deploys_at_its_altitude_and_drogue_after_delay():
    rec = Recovery(Chute(0.01, 0.2), 1.5, Chute(0.06, 0.3), 120.0)
    rocket, launch = setup(rec)
    flight = simulate(rocket, launch)
    ev, tr = flight.events, flight.trajectory
    assert ev["drogue"] == pytest.approx(ev["apogee"] + 1.5)
    i = np.searchsorted(tr["t"], ev["main"])
    assert tr["z"][i] == pytest.approx(120.0, abs=1e-6)
    assert list(dict.fromkeys(tr["phase"])) == ["rail", "powered", "coast", "freefall", "drogue",
                                                "main", "landed"]


def test_main_with_drogue_when_apogee_is_below_main_altitude():
    rec = Recovery(Chute(0.01, 0.0), 0.0, Chute(0.06, 0.0), 10_000.0)
    flight = simulate(*setup(rec))
    assert flight.events["main"] == flight.events["drogue"] == flight.events["apogee"]


def test_zero_wind_lands_under_apogee():
    rec = Recovery(Chute(0.0565, 0.0), 0.0)
    flight = simulate(*setup(rec, elevation_deg=80.0))
    apogee = np.array(flight.summary["apogee_position"])
    landing = np.array(flight.summary["landing_position"])
    assert np.linalg.norm(apogee) > 20  # tilted rail: apogee is well away from the pad
    # Only the horizontal speed left at apogee carries on; under the chute it dies in < 1 s.
    assert np.linalg.norm(landing - apogee) < 0.05 * np.linalg.norm(apogee)
    assert np.hypot(flight.trajectory["vx"][-1], flight.trajectory["vy"][-1]) < 1e-6


def test_constant_wind_drift_is_wind_times_descent_time():
    rec = Recovery(Chute(0.0565, 0.0), 0.0)
    rocket, launch = setup(rec, speed=5.0)
    flight = simulate(rocket, launch)
    s, ev = flight.summary, flight.events
    descent_drift = np.linalg.norm(np.subtract(s["landing_position"], s["apogee_position"]))
    assert descent_drift == pytest.approx(5.0 * (ev["landing"] - ev["apogee"]), rel=0.05)


def test_opening_shock():
    instant = Recovery(Chute(0.0565, 0.0), 2.0)  # falls 2 s first, then opens at once
    rocket, launch = setup(instant, constant_rho=True)
    flight = simulate(rocket, launch)
    tr, t_d = flight.trajectory, flight.events["drogue"]
    i = np.searchsorted(tr["t"], t_d)
    speed = np.sqrt(tr["vx"][i] ** 2 + tr["vy"][i] ** 2 + tr["vz"][i] ** 2)
    load = 0.5 * RHO * speed**2 * 0.0565 / (tr["mass"][i] * G0)  # drag / weight at opening
    assert flight.summary["opening_shock_g"] == pytest.approx(load, rel=1e-6)

    slow = Recovery(Chute(0.0565, 0.5), 2.0)
    assert simulate(*setup(slow, constant_rho=True)).summary["opening_shock_g"] < 0.8 * load
