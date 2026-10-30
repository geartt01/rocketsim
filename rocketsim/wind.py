"""Horizontal wind: mean profile plus optional turbulence (gusts) along the mean direction."""

import math
from dataclasses import dataclass

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.signal import lfilter

GUST_DT = 0.05  # s, fixed turbulence sample interval (20 Hz, Niskanen §4.1.2)
_PINK_STD = 2.252  # std of the unscaled 2-pole pink-noise sequence (Niskanen §4.1.2)


def pink_noise(rng, n, alpha=5 / 3, poles=2):
    """n samples of unit-variance pink noise with spectrum ∝ 1/f^α (Kasdin IIR filter).

        x_n = w_n − a_1·x_{n−1} − … − a_p·x_{n−p},   a_0 = 1,  a_k = (k − 1 − α/2)·a_{k−1}/k
    w_n is Gaussian white noise from rng; the result is divided by 2.252, the standard deviation
    of a long unscaled 2-pole sequence. α = 5/3 matches the high-frequency slope of the Kaimal
    and von Kármán turbulence spectra. Source: Niskanen 2009 eqs. 4.5, 4.6 and §4.1.2;
    Kasdin 1995, Proc. IEEE 83(5).
    """
    a = [1.0]
    for k in range(1, poles + 1):
        a.append((k - 1 - alpha / 2) * a[-1] / k)
    return lfilter([1.0], a, rng.standard_normal(n)) / _PINK_STD  # the IIR recursion above


def gust_signal(samples):
    """x(t) through the 20 Hz samples by a cubic spline, held flat after the last sample.

    Niskanen interpolates between samples without fixing the method. A C² spline is used
    instead of straight lines: linear corners every 0.05 s force the adaptive integrator to
    cut its step at each corner (≈ 12× slower), while the spline keeps the sample values and
    so the turbulence spectrum below 10 Hz.
    """
    return CubicSpline(np.arange(len(samples)) * GUST_DT, samples)


@dataclass
class Wind:
    speed: float  # m/s at reference_height, ≥ 0
    direction: float  # rad clockwise from North that the wind blows FROM (meteorological)
    reference_height: float  # m above the launch site
    profile: str  # "constant" | "power_law"
    gust_sigma: float = 0.0  # m/s, turbulence standard deviation σ_u
    gust: CubicSpline | None = None  # unit turbulence x(t) from gust_signal(), t ≥ 0

    def __call__(self, z, t=0.0):
        """Wind speed along the mean direction at height z and time t, m/s.

        constant:   w = w_ref
        power_law:  w = w_ref·(z/z_ref)^(1/7) for z > 0, else 0
        gusts:      + σ_u·x(t), x through 20 Hz pink-noise samples (Niskanen §4.1.2, see
                    gust_signal); held at the last sample after the sequence ends.
        The 1/7 law is the classic empirical neutral-stability profile from PLAN.md (no
        primary reference checked yet).
        """
        w = self.speed
        if self.profile == "power_law":
            w = w * (max(z, 0.0) / self.reference_height) ** (1 / 7)
        if self.gust_sigma:
            # evaluate the spline piece directly: ~25× faster than CubicSpline.__call__
            x, c = self.gust.x, self.gust.c
            s = min(max(t, 0.0), x[-1])
            k = min(int(s / GUST_DT), len(x) - 2)
            d = s - x[k]
            w += self.gust_sigma * (((c[0, k] * d + c[1, k]) * d + c[2, k]) * d + c[3, k])
        return w

    def vector(self, z, t=0.0):
        """Wind velocity in ENU, m/s: −w(z, t)·[sin d, cos d, 0] (it blows toward d + 180°)."""
        w = self(z, t)
        return np.array([-w * math.sin(self.direction), -w * math.cos(self.direction), 0.0])
