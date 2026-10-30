import numpy as np
import pytest

from rocketsim.atmosphere import isa
from rocketsim.constants import H_TROPOPAUSE, R_EARTH


def geometric(h):
    """Geometric altitude for a geopotential altitude h (inverse of h = R·z/(R+z))."""
    return R_EARTH * h / (R_EARTH - h)


def test_sea_level():
    atm = isa(0.0)
    assert atm.rho == pytest.approx(1.225, abs=0.001)
    assert atm.a == pytest.approx(340.3, abs=0.05)
    assert atm.mu == pytest.approx(1.789e-5, rel=1e-3)


def test_tropopause():
    # ISA tables give 216.65 K and 22632 Pa at 11 km *geopotential*.
    atm = isa(geometric(H_TROPOPAUSE))
    assert atm.T == pytest.approx(216.65, abs=1e-9)
    assert atm.p == pytest.approx(22632.06, rel=1e-5)


def test_continuous_at_tropopause():
    z = geometric(H_TROPOPAUSE)
    below, above = isa(z - 1e-6), isa(z + 1e-6)
    assert above.T == pytest.approx(below.T, abs=1e-6)
    assert above.p == pytest.approx(below.p, rel=1e-9)


def test_isothermal_stratosphere():
    atm = isa(np.array([12e3, 15e3, 20e3]))
    assert np.allclose(atm.T, 216.65)
    assert np.all(np.diff(atm.p) < 0)


def test_out_of_range_raises():
    with pytest.raises(ValueError, match="valid range"):
        isa(25e3)
    with pytest.raises(ValueError, match="valid range"):
        isa(-3e3)
    with pytest.raises(ValueError, match="valid range"):
        isa(float("nan"))
