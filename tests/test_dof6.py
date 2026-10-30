"""Stage 3: full 6-DOF invariants."""

import dataclasses
from pathlib import Path

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from rocketsim import frames
from rocketsim.aero.damping import roll_damping_coefficient, trapezoid_roll_sum
from rocketsim.dynamics import dof6
from rocketsim.dynamics.dof1 import Rocket1D
from rocketsim.flight import simulate, simulate_1dof, simulate_3dof
from rocketsim.io import load_config
from rocketsim.wind import Wind

EXAMPLE = Path(__file__).parents[1] / "configs" / "example_rocket.yaml"


def setup(speed=0.0, direction_deg=0.0, elevation_deg=90.0, heading_deg=0.0, cd=None,
          profile="constant"):
    rocket, launch = load_config(EXAMPLE)
    if cd is not None:
        rocket.drag = cd
    launch = dataclasses.replace(launch, rail_elevation=np.radians(elevation_deg),
                                 rail_heading=np.radians(heading_deg),
                                 wind=Wind(speed, np.radians(direction_deg), 10.0, profile))
    return rocket, launch


# --- quaternion and frame math ---------------------------------------------------------------

def test_rotation_matrix_orthonormal():
    rng = np.random.default_rng(1)
    for q in rng.normal(size=(20, 4)):
        r = frames.rotation_matrix(frames.normalize(q))
        assert np.allclose(r @ r.T, np.eye(3), atol=1e-14)
        assert np.linalg.det(r) == pytest.approx(1.0, abs=1e-14)


def test_quaternion_product_composes_rotations():
    rng = np.random.default_rng(2)
    p, q = (frames.normalize(v) for v in rng.normal(size=(2, 4)))
    assert np.allclose(frames.rotation_matrix(frames.multiply(p, q)),
                       frames.rotation_matrix(p) @ frames.rotation_matrix(q), atol=1e-14)
    assert np.allclose(frames.multiply(q, frames.conjugate(q)), [1, 0, 0, 0], atol=1e-15)


@pytest.mark.parametrize(("elevation", "heading", "axis"), [
    (90, 0, [0, 0, 1]),
    (80, 90, [np.cos(np.radians(80)), 0, np.sin(np.radians(80))]),  # tilted toward East
    (60, 0, [0, 0.5, np.sin(np.radians(60))]),  # tilted toward North
])
def test_launch_attitude(elevation, heading, axis):
    q = frames.launch_quaternion(np.radians(elevation), np.radians(heading))
    assert frames.rotation_matrix(q)[:, 0] == pytest.approx(axis, abs=1e-15)


# --- rigid body ------------------------------------------------------------------------------

def test_torque_free_rigid_body_conserves_momentum_and_energy():
    inertia = np.array([0.5, 2.0, 3.0])  # asymmetric on purpose: the hardest case

    def rhs(t, y):
        q_dot, w_dot = dof6.rotational_derivatives(y[:4], y[4:], inertia, np.zeros(3))
        return np.concatenate([q_dot, w_dot])

    y0 = np.concatenate([[1.0, 0, 0, 0], [0.3, 4.0, -0.5]])
    sol = solve_ivp(rhs, (0, 10), y0, method="DOP853", rtol=1e-12, atol=1e-12, dense_output=True)
    for y in sol.sol(np.linspace(0, 10, 50)).T:
        q, w = frames.normalize(y[:4]), y[4:]
        momentum = frames.rotation_matrix(q) @ (inertia * w)
        energy = 0.5 * w @ (inertia * w)
        assert momentum == pytest.approx(inertia * y0[4:], rel=1e-8, abs=1e-10)
        assert energy == pytest.approx(0.5 * y0[4:] @ (inertia * y0[4:]), rel=1e-8)


def test_quaternion_norm_drift_over_flight():
    flight = simulate(*setup(speed=4.0, direction_deg=250.0, elevation_deg=85.0, heading_deg=30.0))
    tr = flight.trajectory
    norm = np.sqrt(tr["qw"] ** 2 + tr["qx"] ** 2 + tr["qy"] ** 2 + tr["qz"] ** 2)
    assert np.abs(norm - 1).max() < 1e-6


# --- reductions to the lower-order models -----------------------------------------------------

def test_reduces_to_3dof_and_1dof_on_vertical_rail_without_wind():
    rocket, launch = setup(cd=0.5)
    six = simulate(rocket, launch)
    three = simulate_3dof(rocket, launch)
    one = simulate_1dof(Rocket1D(rocket.dry_mass, rocket.body.diameter, 0.5, rocket.motor,
                                 rocket.chute_cda, launch.site_altitude))
    for ref in (three, one):
        assert six.summary["apogee"] == pytest.approx(ref.summary["apogee"], rel=1e-3)
    assert np.abs(six.trajectory["x"]).max() < 1e-6
    assert np.abs(six.trajectory["y"]).max() < 1e-6


def test_planar_reduction_with_wind():
    # Rail tilted toward East, wind from the West: everything happens in the East-Up plane.
    rocket, launch = setup(speed=4.0, direction_deg=270.0, elevation_deg=85.0, heading_deg=90.0,
                           profile="power_law")
    six, three = simulate(rocket, launch), simulate_3dof(rocket, launch)
    s6, s3 = six.summary, three.summary
    assert s6["apogee"] == pytest.approx(s3["apogee"], rel=5e-3)
    assert s6["apogee_position"][0] == pytest.approx(s3["apogee_position"][0], rel=5e-3)
    assert s6["landing_position"][0] == pytest.approx(s3["landing_position"][0], rel=5e-3)
    assert np.abs(six.trajectory["y"]).max() < 1e-6


# --- stability and weathercocking in 3-D ------------------------------------------------------

def test_stable_rocket_restores_angle_of_attack_3d():
    rocket, launch = setup()
    launch = dataclasses.replace(launch, g=0.0, density=lambda z: 1.2)
    t0 = rocket.motor.burn_time + 1.0
    axis, half = np.array([1.0, 2.0, 0.0]) / np.sqrt(5.0), np.radians(5.0) / 2
    tilt = np.array([np.cos(half), *(np.sin(half) * axis)])  # 5° about a horizontal axis
    q0 = frames.multiply(tilt, frames.launch_quaternion(np.pi / 2, 0.0))
    y0 = np.concatenate([[0, 0, 100.0], [0, 0, 100.0], q0, np.zeros(3)])
    sol = solve_ivp(dof6.derivatives, (t0, t0 + 2.0), y0, method="DOP853", rtol=1e-8,
                    atol=1e-8, args=(rocket, launch), dense_output=True)
    t = np.linspace(t0, t0 + 2.0, 2001)
    alpha = np.array([dof6.info(ti, yi, rocket, launch)[0] for ti, yi in zip(t, sol.sol(t).T)])
    first, last = alpha[t < t0 + 0.5].max(), alpha[t > t0 + 1.5].max()
    assert first == pytest.approx(np.radians(5.0), rel=1e-3)
    assert last < 0.5 * first


@pytest.mark.parametrize("direction_deg", [0.0, 90.0, 200.0, 315.0])
def test_weathercocks_into_wind_from_any_heading(direction_deg):
    flight = simulate(*setup(speed=5.0, direction_deg=direction_deg))
    d = np.radians(direction_deg)
    toward = -np.array([np.sin(d), np.cos(d)])  # horizontal direction the wind blows to
    apogee = np.array(flight.summary["apogee_position"])
    landing = np.array(flight.summary["landing_position"])
    assert apogee @ toward < 0  # climbed upwind
    assert (landing - apogee) @ toward > 0  # drifted downwind under the chute
    across = apogee - (apogee @ toward) * toward
    assert np.linalg.norm(across) < 1e-6  # no sideways motion across the wind


# --- roll damping -----------------------------------------------------------------------------

def test_trapezoid_roll_sum_matches_strip_integral():
    rocket, _ = setup()
    f, r = rocket.fins, rocket.radius
    y = np.linspace(0, f.span, 200001)
    c = f.root_chord + (f.tip_chord - f.root_chord) * y / f.span
    assert trapezoid_roll_sum(f, r) == pytest.approx(np.trapezoid(c * (r + y) ** 2, y), rel=1e-9)


def test_roll_damping_is_positive_and_grows_with_speed():
    rocket, _ = setup()
    slow, fast = (roll_damping_coefficient(rocket, 1.2, v, v / 340) for v in (30.0, 90.0))
    assert 0 < slow < fast
