from pathlib import Path

import numpy as np
import pytest
from scipy.integrate import quad

from rocketsim.motor import load_eng

ENG = Path(__file__).parent / "data" / "test_motor.eng"
HEADER = "TEST 24 70 5-7-P 0.0125 0.0250 Test"


@pytest.fixture
def motor():
    return load_eng(ENG)


def test_header(motor):
    assert motor.name == "TEST"
    assert motor.diameter == pytest.approx(0.024)
    assert motor.length == pytest.approx(0.070)
    assert motor.propellant_mass_initial == 0.0125
    assert motor.total_mass_initial == 0.0250


def test_curve_starts_at_zero(motor):
    assert motor.time[0] == 0.0 and motor.thrust_data[0] == 0.0
    assert motor.burn_time == 1.2


def test_total_impulse_is_trapezoid(motor):
    assert motor.total_impulse == pytest.approx(np.trapezoid(motor.thrust_data, motor.time))


def test_impulse_matches_quadrature(motor):
    # impulse(t) is analytic; check it mid-segment against numerical integration of thrust(t).
    for t in (0.03, 0.3, 0.77, 1.2):
        kinks = motor.time[(motor.time > 0) & (motor.time < t)]
        expected, _ = quad(motor.thrust, 0.0, t, points=kinks if len(kinks) else None)
        assert motor.impulse(t) == pytest.approx(expected, rel=1e-9)


def test_thrust_zero_outside_burn(motor):
    assert motor.thrust(-1.0) == 0.0
    assert motor.thrust(motor.burn_time + 1.0) == 0.0
    assert motor.thrust(0.075) == pytest.approx(15.0)


def test_propellant_mass(motor):
    assert motor.propellant_mass(0.0) == pytest.approx(motor.propellant_mass_initial)
    assert motor.propellant_mass(motor.burn_time) == pytest.approx(0.0, abs=1e-15)
    assert motor.propellant_mass(motor.burn_time + 1.0) == pytest.approx(0.0, abs=1e-15)
    assert motor.propellant_mass(-1.0) == pytest.approx(motor.propellant_mass_initial)
    m = motor.propellant_mass(np.linspace(0, motor.burn_time, 50))
    assert np.all(np.diff(m) <= 0)


def test_utf8_comment_accepted(tmp_path):
    path = tmp_path / "utf8.eng"
    path.write_text("; Łukasz µ °\n" + HEADER + "\n0.1 5\n0.2 0\n", encoding="utf-8")
    assert load_eng(path).total_impulse > 0


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("; only comments\n", "no motor header"),
        ("TEST 24 70 P 0.01 0.02\n0.1 5\n", "7 fields"),
        ("TEST 24 70 P abc 0.02 Test\n0.1 5\n", "not a number"),
        (HEADER + "\n", "no thrust data"),
        (HEADER + "\n0.1 5\n0.1 6\n", "strictly increasing"),
        (HEADER + "\n0.1 5\n0.2 -1\n", "negative"),
        (HEADER + "\n0.1 5 7\n", "time thrust"),
        (HEADER + "\n0.1 5\n0.2 0\n" + HEADER + "\n0.1 5\n", "multiple motors"),
        ("TEST 24 70 P 0.03 0.02 Test\n0.1 5\n", "propellant mass"),
        (HEADER + "\n0.0 0\n", "positive time"),
        (HEADER + "\n0.1 0\n0.2 0\n", "no thrust"),
    ],
)
def test_malformed_file_rejected(tmp_path, text, message):
    path = tmp_path / "bad.eng"
    path.write_text(text)
    with pytest.raises(ValueError, match=message) as exc:
        load_eng(path)
    assert "bad.eng" in str(exc.value)
