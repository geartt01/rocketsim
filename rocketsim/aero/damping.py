"""Pitch damping moment (Niskanen 2009, §3.2.3)."""

import numpy as np

from rocketsim.geometry import nose_radius


def pitch_damping_moment(rocket, rho, omega, x_cg, x_fin_cp):
    """Moment opposing the pitch rate ω, N·m (sign opposite to ω).

    Body, each side of the CG treated as a cylinder spinning about the CG in cross-flow
    (C_D ≈ 1.1 on planform area):  M = 0.275·ρ·r̄·l⁴·ω²                     (3.57, 3.58)
    Fins:  M = 0.3·ρ·N·A_fin·ξ³·ω², ξ = |X_fin − x_cg|, N capped at 4        (3.60)
    (These are eqs. 3.59/3.60 multiplied by q·A_ref·d, which removes 1/v².)

    Args: rho (kg/m³), omega (rad/s), x_cg and x_fin_cp (m from nose tip).
    Assumptions: r̄ is the mean body radius of each side; a rough estimate is enough because the
    moment matters only when ω is large relative to v (e.g. near apogee).
    """
    r = rocket.radius
    x = np.linspace(0.0, rocket.length, 2001)
    radius = np.where(x < rocket.nose.length,
                      nose_radius(rocket.nose.shape, x, rocket.nose.length, r), r)
    fore, aft = x <= x_cg, x >= x_cg
    body = 0.0
    for side, length in ((fore, x_cg), (aft, rocket.length - x_cg)):
        if length > 0 and side.any():
            body += 0.275 * rho * radius[side].mean() * length**4
    f = rocket.fins
    fins = 0.3 * rho * min(f.count, 4) * f.area * abs(x_fin_cp - x_cg) ** 3
    return -(body + fins) * omega * abs(omega)
