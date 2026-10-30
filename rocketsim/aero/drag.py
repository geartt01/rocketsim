"""Zero-angle-of-attack drag coefficient C_D0: component buildup, constant, or Cd(Mach) table.

Buildup source: Niskanen 2009, §3.4 (equation numbers below are Niskanen's). Subsonic only;
the simulator warns above Mach 0.8. Every coefficient is referred to A_ref = π·d²/4.
"""

import numpy as np


def skin_friction_coefficient(reynolds, roughness_ratio, mach):
    """Turbulent skin-friction coefficient C_f with subsonic compressibility correction.

        R < 1e4:          C_f = 1.48e-2                                 (3.81)
        1e4 ≤ R < R_crit: C_f = 1 / (1.50·ln R − 5.6)²                  (3.78)
        R ≥ R_crit:       C_f = 0.032·(R_s/L)^0.2  (roughness-limited)  (3.80)
        R_crit = 51·(R_s/L)^−1.039                                      (3.79)
        C_fc = C_f·(1 − 0.1·M²)                                         (3.82)

    Args: reynolds R = v·L/ν (-), roughness_ratio R_s/L (-), mach (-).
    Assumption: the boundary layer is fully turbulent (Niskanen §3.4.1).
    """
    if reynolds < 1e4:
        cf = 1.48e-2
    elif reynolds < 51 * roughness_ratio**-1.039:
        cf = 1 / (1.50 * np.log(reynolds) - 5.6) ** 2
    else:
        cf = 0.032 * roughness_ratio**0.2
    return cf * (1 - 0.1 * mach**2)


def base_drag_coefficient(mach):
    """(C_D)_base = 0.12 + 0.13·M² for M < 1, on the base area (Niskanen 3.94)."""
    return 0.12 + 0.13 * mach**2


def buildup_cd(rocket, mach, reynolds, thrusting):
    """C_D0 = friction + nose pressure + fin pressure + base drag  (Niskanen 3.97).

    friction:   C_fc·[(1 + 1/(2f_B))·A_wet,body + (1 + 2t/c̄)·A_wet,fins] / A_ref   (3.85)
                f_B = L/d fineness ratio, t fin thickness, c̄ mean aerodynamic chord
    nose:       0.8·sin²φ, φ = angle at the nose–body joint (0 for smooth shapes)  (3.86)
    fins:       leading edge (rounded/airfoil): [(1 − M²)^−0.417 − 1]·cos²Γ_L     (3.89, 3.91)
                leading edge (square): 0.85·(1 + M²/4 + M⁴/40)·cos²Γ_L            (3.90, B.1, B.2)
                trailing edge: base drag (square), half of it (rounded), 0 (airfoil)  (3.92)
                on the fin frontal area N·t·s                                       (3.93)
    base:       (0.12 + 0.13·M²) on A_base, minus the motor area while thrusting    (3.94)

    Args: mach (-), reynolds R = v·L/ν with L the rocket length (-), thrusting (bool).
    Not modelled: launch lugs/rail buttons (parasitic), fin–body interference, boattails.
    """
    f, a_ref, d = rocket.fins, rocket.area, rocket.body.diameter
    cf = skin_friction_coefficient(reynolds, rocket.roughness / rocket.length, mach)
    wet_body = rocket.nose_surface + np.pi * d * rocket.body.length
    wet_fins = 2 * f.count * f.area
    friction = cf * ((1 + d / (2 * rocket.length)) * wet_body
                     + (1 + 2 * f.thickness / f.mac) * wet_fins) / a_ref

    joint = np.arctan(rocket.radius / rocket.nose.length) if rocket.nose.shape == "cone" else 0.0
    nose = 0.8 * np.sin(joint) ** 2

    m = min(mach, 0.89)  # the subsonic leading-edge branch of eq. 3.89 holds for M < 0.9
    cos2_le = np.cos(np.arctan(f.sweep / f.span)) ** 2
    if f.profile == "square":
        leading = 0.85 * (1 + m**2 / 4 + m**4 / 40) * cos2_le
    else:
        leading = ((1 - m**2) ** -0.417 - 1) * cos2_le
    trailing = base_drag_coefficient(mach) * {"square": 1.0, "rounded": 0.5, "airfoil": 0.0}[f.profile]
    fin_pressure = (leading + trailing) * f.count * f.thickness * f.span / a_ref

    base_area = a_ref - (np.pi * rocket.motor.diameter**2 / 4 if thrusting else 0.0)
    base = base_drag_coefficient(mach) * base_area / a_ref
    return friction + nose + fin_pressure + base


def drag_coefficient(rocket, mach, reynolds, thrusting):
    """C_D0 from the rocket's drag setting: "buildup", a constant, or a (mach, cd) table.

    A table is interpolated linearly in Mach and held constant beyond its ends.
    """
    if isinstance(rocket.drag, str):
        return buildup_cd(rocket, mach, reynolds, thrusting)
    if isinstance(rocket.drag, tuple):
        return float(np.interp(mach, *rocket.drag))
    return rocket.drag
