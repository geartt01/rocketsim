"""Pitch/yaw and roll damping (Niskanen 2009, §3.2.3 and §3.3.2)."""

import math

import numpy as np

from rocketsim.geometry import nose_radius


def pitch_damping_coefficient(rocket, rho, x_cg, x_fin_cp):
    """K in the damping moment M = −K·ω·|ω| about a transverse axis, N·m·s².

    Body, each side of the CG treated as a cylinder spinning about the CG in cross-flow
    (C_D ≈ 1.1 on planform area):  K_body = 0.275·ρ·r̄·l⁴                     (3.57, 3.58)
    Fins:  K_fins = 0.3·ρ·N·A_fin·ξ³, ξ = |X_fin − x_cg|, N capped at 4         (3.60)
    (These are eqs. 3.59/3.60 multiplied by q·A_ref·d, which removes 1/v².)

    Args: rho (kg/m³), x_cg and x_fin_cp (m from nose tip).
    Assumptions: r̄ is the mean body radius of each side; a rough estimate is enough because the
    moment matters only when ω is large relative to v (e.g. near apogee).
    """
    x, cumsum = body_profile(rocket)
    n_fore = np.searchsorted(x, x_cg, side="right")  # samples with x ≤ x_cg
    n_before_aft = np.searchsorted(x, x_cg, side="left")  # samples with x < x_cg
    body = 0.0
    if x_cg > 0 and n_fore:
        body += 0.275 * rho * cumsum[n_fore - 1] / n_fore * x_cg**4
    if rocket.length > x_cg and n_before_aft < len(x):
        aft_sum = cumsum[-1] - (cumsum[n_before_aft - 1] if n_before_aft else 0.0)
        body += 0.275 * rho * aft_sum / (len(x) - n_before_aft) * (rocket.length - x_cg) ** 4
    f = rocket.fins
    return body + 0.3 * rho * min(f.count, 4) * f.area * abs(x_fin_cp - x_cg) ** 3


def body_profile(rocket):
    """Body radius r(x) on 2001 even samples and its running sum, cached on the rocket.

    The running sum gives the mean radius fore/aft of any CG in O(1); it is rebuilt if the
    nose or body geometry changes (tests edit them in place).
    """
    key = (rocket.nose.shape, rocket.nose.length, rocket.body.length, rocket.radius)
    cached = getattr(rocket, "_profile", None)
    if cached is None or cached[0] != key:
        x = np.linspace(0.0, rocket.length, 2001)
        radius = np.where(x < rocket.nose.length,
                          nose_radius(rocket.nose.shape, x, rocket.nose.length, rocket.radius),
                          rocket.radius)
        cached = rocket._profile = (key, x, np.cumsum(radius))
    return cached[1], cached[2]


def trapezoid_roll_sum(fins, body_radius):
    """Σ c·ξ²·Δξ over one trapezoidal fin, m⁴ (Niskanen 3.70):

        (C_r + C_t)/2·r²·s + (C_r + 2C_t)/3·r·s² + (C_r + 3C_t)/12·s³
    """
    cr, ct, s, r = fins.root_chord, fins.tip_chord, fins.span, body_radius
    return (cr + ct) / 2 * r**2 * s + (cr + 2 * ct) / 3 * r * s**2 + (cr + 3 * ct) / 12 * s**3


def roll_damping_coefficient(rocket, rho, speed, mach):
    """C in the roll damping moment L = −C·ω_x, N·m·s.

    Each fin strip at radius ξ sees an extra angle of attack ω·ξ/v; summing the strip lift
    (C_Nα0 = 2π/β, thin-airfoil slope) times ξ gives, for N fins,
        C_ld = N·C_Nα0·ω/(A_ref·d·v)·Σ c·ξ²·Δξ                                 (3.69)
    and in dimensional form (× q·A_ref·d):  C = ½·ρ·v·N·(2π/β)·Σ c·ξ²·Δξ.

    Args: rho (kg/m³), speed |v_air| (m/s), mach (-). Valid subsonic, |ω·ξ| ≪ v.
    """
    beta = math.sqrt(1 - min(mach, 0.99) ** 2)
    f = rocket.fins
    return 0.5 * rho * speed * f.count * 2 * math.pi / beta * trapezoid_roll_sum(f, rocket.radius)
