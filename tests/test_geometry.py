"""Stage 2: mass properties and component geometry."""

from pathlib import Path

import numpy as np
import pytest

from rocketsim.aero.barrowman import nose_cn_alpha_cp
from rocketsim.geometry import (
    combine,
    nose_radius,
    shell_properties,
    solid_cylinder_inertia,
)
from rocketsim.io import load_config

EXAMPLE = Path(__file__).parents[1] / "configs" / "example_rocket.yaml"


def test_hollow_tube_inertia():
    m, r, length = 0.2, 0.03, 0.5
    x = np.linspace(0.0, length, 2001)
    x_cg, inertia, area = shell_properties(x, np.full_like(x, r), m)
    assert x_cg == pytest.approx(length / 2)
    assert inertia == pytest.approx(m * (r**2 / 2 + length**2 / 12), rel=1e-6)
    assert area == pytest.approx(2 * np.pi * r * length)


def test_solid_cylinder_inertia_matches_stack_of_disks():
    m, r, length = 0.3, 0.02, 0.1
    x = np.linspace(-length / 2, length / 2, 20001)
    xm = (x[1:] + x[:-1]) / 2
    dm = m / len(xm)
    # a thin disk about its diameter: dm·r²/4, shifted by xm
    numeric = np.sum(dm * (r**2 / 4 + xm**2))
    assert solid_cylinder_inertia(m, r, length) == pytest.approx(numeric, rel=1e-6)


def test_two_mass_cg_and_inertia():
    m, x_cg, inertia = combine([(1.0, 0.0, 0.0), (3.0, 4.0, 0.0)])
    assert (m, x_cg) == (4.0, 3.0)
    assert inertia == pytest.approx(1.0 * 3**2 + 3.0 * 1**2)


def test_conical_shell_cg_at_two_thirds():
    x = np.linspace(0.0, 0.1, 2001)
    x_cg, _, _ = shell_properties(x, nose_radius("cone", x, 0.1, 0.02), 1.0)
    assert x_cg == pytest.approx(2 / 3 * 0.1, rel=1e-5)


@pytest.mark.parametrize(("shape", "fraction"),
                         [("cone", 2 / 3), ("elliptical", 1 / 3), ("von_karman", 1 / 2)])
def test_nose_cp_analytic(shape, fraction):
    rocket, _ = load_config(EXAMPLE)
    rocket.nose.shape = shape
    rocket.__post_init__()  # recompute cached nose volume for the new shape
    cn, x_cp = nose_cn_alpha_cp(rocket)
    assert cn == 2.0
    assert x_cp == pytest.approx(fraction * rocket.nose.length, rel=1e-6)


@pytest.mark.parametrize("shape", ["cone", "ogive", "elliptical", "von_karman"])
def test_nose_profile_ends(shape):
    assert nose_radius(shape, 0.0, 0.1, 0.02) == pytest.approx(0.0, abs=1e-12)
    assert nose_radius(shape, 0.1, 0.1, 0.02) == pytest.approx(0.02)


def test_rectangular_fin_cg():
    rocket, _ = load_config(EXAMPLE)
    f = rocket.fins
    f.tip_chord, f.sweep = f.root_chord, 0.0
    x_cg, _ = f.properties(rocket.radius)
    assert x_cg == pytest.approx(f.position + f.root_chord / 2)


def test_motor_burn_moves_mass():
    rocket, _ = load_config(EXAMPLE)
    m0, cg0, _ = rocket.mass_properties(0.0)
    m1, cg1, _ = rocket.mass_properties(rocket.motor.burn_time)
    assert m0 == pytest.approx(rocket.dry_mass + rocket.motor.total_mass_initial)
    assert m0 - m1 == pytest.approx(rocket.motor.propellant_mass_initial)
    assert cg1 < cg0  # the motor is aft, so burning propellant moves the CG forward
