"""Stage 1 physics invariants for the 1-DOF vertical flight."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from rocketsim.constants import G0
from rocketsim.dynamics.dof1 import Rocket1D
from rocketsim.flight import apogee_event, landing_event, liftoff_time, run_phase, simulate
from rocketsim.io import load_config
from rocketsim.motor import Motor, load_eng

ROOT = Path(__file__).parents[1]
TEST_ENG = Path(__file__).parent / "data" / "test_motor.eng"

_spec = importlib.util.spec_from_file_location("sim_1dof", ROOT / "reference" / "sim_1dof.py")
oracle = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(oracle)


def make_rocket(motor=None, **overrides):
    params = {"dry_mass": 0.05, "diameter": 0.025, "cd": 0.5, "chute_cda": 0.05, "elevation": 0.0}
    return Rocket1D(motor=motor or load_eng(TEST_ENG), **(params | overrides))


def constant_thrust_motor(thrust, burn_time, prop_mass, total_mass):
    times, thrusts = np.array([0.0, burn_time]), np.array([thrust, thrust])
    return Motor("CONST", 0.024, 0.07, "P", prop_mass, total_mass, "Test", times, thrusts)


def test_vacuum_coast_apogee():
    v0 = 50.0
    rocket = make_rocket(density=lambda z: 0.0)
    t0 = rocket.motor.burn_time + 1.0  # motor finished: no thrust, constant mass
    sol = run_phase(rocket, t0, [0.0, v0], t0 + 100.0, events=[apogee_event])
    assert sol.y_events[0][0][0] == pytest.approx(v0**2 / (2 * G0), rel=1e-6)


def test_rocket_equation():
    thrust, burn, m_prop = 20.0, 2.0, 0.02
    rocket = make_rocket(constant_thrust_motor(thrust, burn, m_prop, 0.03), g=0.0,
                         density=lambda z: 0.0)
    m0, mf = rocket.mass(0.0), rocket.mass(burn)
    assert m0 - mf == pytest.approx(m_prop)
    v_e = thrust * burn / m_prop  # constant thrust => constant mass flow F / v_e
    sol = run_phase(rocket, 0.0, [0.0, 0.0], burn)
    assert sol.y[1, -1] == pytest.approx(v_e * np.log(m0 / mf), rel=1e-6)


def test_terminal_velocity_under_chute():
    rho = 1.1
    rocket = make_rocket(density=lambda z: rho)
    t0 = rocket.motor.burn_time + 1.0
    sol = run_phase(rocket, t0, [500.0, 0.0], t0 + 1000.0, events=[landing_event], chute=True)
    v_terminal = np.sqrt(2 * rocket.mass(t0) * G0 / (rho * rocket.chute_cda))
    assert sol.y_events[0][0][1] == pytest.approx(-v_terminal, rel=1e-6)


@pytest.mark.parametrize(
    "rocket",
    [make_rocket(), load_config(ROOT / "configs" / "example_rocket.yaml")],
    ids=["test-motor", "example-config"],
)
def test_matches_oracle(rocket):
    m = rocket.motor
    ref = oracle.simulate(rocket.dry_mass, rocket.diameter, rocket.cd, m.time, m.thrust_data,
                          m.propellant_mass_initial, m.total_mass_initial)
    assert simulate(rocket).summary["apogee"] == pytest.approx(ref["apogee"], rel=5e-3)


def test_tolerance_convergence():
    rocket = make_rocket()
    coarse = simulate(rocket, rtol=1e-8, atol=1e-8).summary["apogee"]
    fine = simulate(rocket, rtol=1e-9, atol=1e-9).summary["apogee"]
    assert abs(fine - coarse) / fine < 1e-4


def test_stays_on_pad_until_thrust_exceeds_weight():
    rocket = make_rocket(dry_mass=0.5)  # weight ≈ 5 N; test motor ramps 0 → 10 N in 0.05 s
    t = liftoff_time(rocket)
    assert 0.0 < t < 0.05
    assert rocket.motor.thrust(t) == pytest.approx(rocket.mass(t) * G0, rel=1e-9)
    assert simulate(rocket).events["liftoff"] == t


def test_too_heavy_never_leaves_pad():
    with pytest.raises(ValueError, match="never exceeds weight"):
        liftoff_time(make_rocket(dry_mass=5.0))


def test_full_flight_sequence():
    rocket = make_rocket()
    flight = simulate(rocket)
    ev = flight.events
    assert ev["liftoff"] < ev["burnout"] < ev["apogee"] < ev["landing"]
    assert flight.trajectory["z"].min() > -1e-6
    assert flight.summary["apogee"] == pytest.approx(flight.trajectory["z"].max(), rel=1e-6)
    # From apogee the chute has long reached terminal velocity at ground level.
    v_terminal = np.sqrt(2 * rocket.mass(ev["landing"]) * G0 / (rocket.rho(0.0) * rocket.chute_cda))
    assert flight.summary["landing_velocity"] == pytest.approx(v_terminal, rel=1e-3)
