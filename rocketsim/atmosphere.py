"""International Standard Atmosphere (ISA), troposphere and lower stratosphere."""

from typing import NamedTuple

import numpy as np

from rocketsim.constants import (
    G0,
    GAMMA_AIR,
    H_TROPOPAUSE,
    LAPSE_TROPO,
    P0,
    R_AIR,
    R_EARTH,
    SUTHERLAND_BETA,
    SUTHERLAND_S,
    T0,
)

Z_MIN, Z_MAX = -2000.0, 20000.0  # geometric altitude range covered by the two layers below, m

T_TROPOPAUSE = T0 + LAPSE_TROPO * H_TROPOPAUSE  # 216.65 K
P_TROPOPAUSE = P0 * (T_TROPOPAUSE / T0) ** (-G0 / (LAPSE_TROPO * R_AIR))  # ≈ 22632 Pa


class Atmosphere(NamedTuple):
    T: np.ndarray  # temperature, K
    p: np.ndarray  # pressure, Pa
    rho: np.ndarray  # density, kg/m^3
    a: np.ndarray  # speed of sound, m/s
    mu: np.ndarray  # dynamic viscosity, Pa·s


def isa(z):
    """ISA state at geometric altitude z above mean sea level.

    Equations:
        geopotential altitude   h = R_E·z / (R_E + z)
        troposphere (h ≤ 11 km): T = T0 + L·h,  p = p0·(T/T0)^(−g0/(L·R))
        stratosphere (h > 11 km): T = T11,      p = p11·exp(−g0·(h − h11)/(R·T11))
        ideal gas               ρ = p / (R·T)
        speed of sound          a = sqrt(γ·R·T)
        Sutherland's law        μ = β·T^1.5 / (T + S)

    Args:
        z: geometric altitude above mean sea level, m (scalar or array), −2 km ≤ z ≤ 20 km.

    Returns:
        Atmosphere(T [K], p [Pa], rho [kg/m^3], a [m/s], mu [Pa·s]), each shaped like z.

    Assumptions: dry air as an ideal gas in hydrostatic equilibrium; the pressure formulas come
    from integrating dp/dh = −ρ·g0 with the layer's temperature profile. Gravity's drop with
    altitude is absorbed by using geopotential h, which differs from z by ~0.05 % at 3 km.

    Source: U.S. Standard Atmosphere 1976 (NOAA-S/T 76-1562), eqs. 18, 33a, 33b, 42, 50, 51;
    identical to ISO 2533:1975 below 20 km.
    """
    z = np.asarray(z, dtype=float)
    if not np.all((z >= Z_MIN) & (z <= Z_MAX)):  # written this way so NaN is rejected too
        raise ValueError(f"altitude outside ISA valid range [{Z_MIN}, {Z_MAX}] m: {z}")
    h = R_EARTH * z / (R_EARTH + z)
    tropo = h <= H_TROPOPAUSE
    T = np.where(tropo, T0 + LAPSE_TROPO * h, T_TROPOPAUSE)
    p = np.where(
        tropo,
        P0 * (T / T0) ** (-G0 / (LAPSE_TROPO * R_AIR)),
        P_TROPOPAUSE * np.exp(-G0 * (h - H_TROPOPAUSE) / (R_AIR * T_TROPOPAUSE)),
    )
    rho = p / (R_AIR * T)
    a = np.sqrt(GAMMA_AIR * R_AIR * T)
    mu = SUTHERLAND_BETA * T**1.5 / (T + SUTHERLAND_S)
    return Atmosphere(T, p, rho, a, mu)
