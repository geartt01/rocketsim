"""Stage 2: Barrowman normal force / CP, drag buildup, pitch damping."""

from pathlib import Path

import numpy as np
import pytest

from rocketsim.aero.barrowman import cn_alpha_cp, fins_cn_alpha_cp
from rocketsim.aero.damping import pitch_damping_moment
from rocketsim.aero.drag import buildup_cd, drag_coefficient, skin_friction_coefficient
from rocketsim.io import load_config

EXAMPLE = Path(__file__).parents[1] / "configs" / "example_rocket.yaml"


@pytest.fixture
def rocket():
    return load_config(EXAMPLE)[0]


def test_fin_formula_matches_plan_barrowman_form(rocket):
    # Niskanen's eq. 3.40 at M = 0 must equal the classic Barrowman form in PLAN.md.
    f, rad, d = rocket.fins, rocket.radius, rocket.body.diameter
    a, b, s, n = f.root_chord, f.tip_chord, f.span, f.count
    l_f = np.hypot(s, f.sweep + b / 2 - a / 2)  # mid-chord line length
    cn_plan = (1 + rad / (s + rad)) * 4 * n * (s / d) ** 2 / (1 + np.sqrt(1 + (2 * l_f / (a + b)) ** 2))
    x_plan = f.position + f.sweep * (a + 2 * b) / (3 * (a + b)) + ((a + b) - a * b / (a + b)) / 6
    cn, x_f = fins_cn_alpha_cp(rocket, 0.0)
    assert cn == pytest.approx(cn_plan, rel=1e-12)
    assert x_f == pytest.approx(x_plan, rel=1e-12)


def test_fin_cn_alpha_grows_with_mach(rocket):
    assert fins_cn_alpha_cp(rocket, 0.6)[0] > fins_cn_alpha_cp(rocket, 0.0)[0]


@pytest.mark.parametrize(("ours", "theirs"), [("ogive", "ogive"), ("cone", "conical"),
                                               ("von_karman", "von karman"),
                                               ("elliptical", "elliptical")])
def test_cp_matches_rocketpy(rocket, ours, theirs):
    """Independent Barrowman implementation (RocketPy, the PLAN's [validation] oracle).

    PLAN.md asks for OpenRocket within 2 %; OpenRocket is not scriptable here, so RocketPy
    stands in until the Stage 7 OpenRocket export.
    """
    rocketpy = pytest.importorskip("rocketpy")
    rocket.nose.shape = ours
    rocket.__post_init__()
    rp = rocketpy.Rocket(radius=rocket.radius, mass=1.0, inertia=(1e-3, 1e-3, 1e-5),
                         power_off_drag=0.5, power_on_drag=0.5,
                         center_of_mass_without_motor=0.2,
                         coordinate_system_orientation="nose_to_tail")
    rp.add_nose(length=rocket.nose.length, kind=theirs, position=0.0)
    f = rocket.fins
    rp.add_trapezoidal_fins(n=f.count, root_chord=f.root_chord, tip_chord=f.tip_chord,
                            span=f.span, position=f.position, sweep_length=f.sweep)
    cn, x_cp = cn_alpha_cp(rocket, 0.0)
    assert x_cp == pytest.approx(rp.cp_position(0.0), rel=0.02)
    assert cn == pytest.approx(rp.total_lift_coeff_der(0.0), rel=0.02)


def test_skin_friction_regimes():
    assert skin_friction_coefficient(5e3, 1e-4, 0.0) == 1.48e-2
    assert skin_friction_coefficient(1e6, 1e-6, 0.0) == pytest.approx(1 / (1.5 * np.log(1e6) - 5.6) ** 2)
    rough = 1e-3  # R_crit = 51·(1e-3)^−1.039 ≈ 6.3e4, so 1e6 is roughness-limited
    assert skin_friction_coefficient(1e6, rough, 0.0) == pytest.approx(0.032 * rough**0.2)
    assert skin_friction_coefficient(1e6, 1e-6, 0.5) == pytest.approx(
        skin_friction_coefficient(1e6, 1e-6, 0.0) * (1 - 0.1 * 0.25))


def test_buildup_cd_is_plausible(rocket):
    cd_coast = buildup_cd(rocket, 0.3, 2.6e6, thrusting=False)
    cd_burn = buildup_cd(rocket, 0.3, 2.6e6, thrusting=True)
    assert 0.3 < cd_coast < 0.9  # typical small model rockets: ~0.4–0.75
    assert cd_burn < cd_coast  # exhaust fills part of the base


def test_drag_table_and_constant(rocket):
    rocket.drag = (np.array([0.0, 0.5]), np.array([0.4, 0.6]))
    assert drag_coefficient(rocket, 0.25, 1e6, False) == pytest.approx(0.5)
    assert drag_coefficient(rocket, 0.9, 1e6, False) == pytest.approx(0.6)
    rocket.drag = 0.45
    assert drag_coefficient(rocket, 0.3, 1e6, True) == 0.45


def test_pitch_damping_opposes_rotation(rocket):
    x_f = fins_cn_alpha_cp(rocket, 0.0)[1]
    assert pitch_damping_moment(rocket, 1.2, 2.0, 0.27, x_f) < 0
    assert pitch_damping_moment(rocket, 1.2, -2.0, 0.27, x_f) > 0
    assert pitch_damping_moment(rocket, 1.2, 0.0, 0.27, x_f) == 0
