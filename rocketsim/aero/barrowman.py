"""Normal-force slope CN_α and centre of pressure (CP) per component, subsonic Barrowman.

Sources: Barrowman 1967 (NASA TIR-33 / thesis), as extended in Niskanen 2009, "Development of
an Open Source Model Rocket Simulation Software", §3.2. Equation numbers below are Niskanen's.
CP positions are from the nose tip, positive aft. Valid for small α and Mach < ~0.8.
"""

import numpy as np

# Fin–fin interference for N fins in one set (Niskanen eq. 3.54)
_FIN_INTERFERENCE = {5: 0.948, 6: 0.913, 7: 0.854, 8: 0.810}


def nose_cn_alpha_cp(rocket):
    """Nose cone: CN_α = 2·[A(l) − A(0)]/A_ref = 2,  X_N = [l·A(l) − V] / [A(l) − A(0)].

    The nose base equals the body diameter, so A(l) = A_ref and A(0) = 0. Cone → 2l/3,
    ellipsoid → l/3, Von Kármán → l/2, tangent ogive in between (≈0.47·l at l/d ≈ 3).

    Returns: (CN_α [1/rad], X_N [m]). Source: Niskanen eqs. 3.19, 3.28 (sin α/α → 1).
    """
    a = rocket.area
    return 2.0, (rocket.nose.length * a - rocket.nose_volume) / a


def fins_cn_alpha_cp(rocket, mach):
    """Fin set with body interference.

    One fin (Diederich, compressibility via β = sqrt(1 − M²)):
        (CN_α)_1 = 2π·(s²/A_ref) / [1 + sqrt(1 + (β·s²/(A_fin·cos Γ_c))²)]          (3.40)
    N ≥ 3 evenly spaced fins: (CN_α)_N = (N/2)·(CN_α)_1 · interference(N)          (3.53, 3.54)
    Fin in presence of body: K_T(B) = 1 + r_t/(s + r_t)                             (3.56)
    CP at the quarter chord of the mean aerodynamic chord:
        X_F = X_LE + (X_t/3)·(C_r + 2C_t)/(C_r + C_t) + (1/6)·(C_r² + C_t² + C_r·C_t)/(C_r + C_t)  (3.34)

    with s = span, C_r/C_t = root/tip chord, X_t = sweep of the tip leading edge, Γ_c = mid-chord
    sweep angle, A_fin = one fin's area, r_t = body radius. At M = 0 this is algebraically the
    PLAN.md form [1 + R/(s+R)]·4N(s/d)² / [1 + sqrt(1 + (2·l_f/(a+b))²)].

    Assumption: CP stays at 25 % MAC (Niskanen moves it aft above M 0.5; ignored, M < 0.8 here).
    Args: mach, -. Returns: (CN_α [1/rad], X_F [m]).
    """
    f = rocket.fins
    cr, ct, s, xt = f.root_chord, f.tip_chord, f.span, f.sweep
    beta = np.sqrt(1 - min(mach, 0.99) ** 2)
    cos_gc = np.cos(np.arctan((xt + ct / 2 - cr / 2) / s))
    cn1 = 2 * np.pi * s**2 / rocket.area / (1 + np.sqrt(1 + (beta * s**2 / (f.area * cos_gc)) ** 2))
    interference = _FIN_INTERFERENCE.get(f.count, 1.0 if f.count < 5 else 0.75)
    k_tb = 1 + rocket.radius / (s + rocket.radius)
    x_f = f.position + xt / 3 * (cr + 2 * ct) / (cr + ct) + (cr**2 + ct**2 + cr * ct) / (6 * (cr + ct))
    return k_tb * f.count / 2 * cn1 * interference, x_f


def cn_alpha_cp(rocket, mach):
    """Whole rocket: CN_α = Σ CN_α,i,  X_CP = Σ(CN_α,i·X_i) / Σ CN_α,i  (Niskanen 3.29).

    The cylindrical body tube contributes nothing (A(l) = A(0)); Galejs' body-lift term is
    ∝ sin²α and adds no slope at α = 0, so it is left out.
    Returns: (CN_α [1/rad], X_CP [m from nose tip]).
    """
    (cn_n, x_n), (cn_f, x_f) = nose_cn_alpha_cp(rocket), fins_cn_alpha_cp(rocket, mach)
    cn = cn_n + cn_f
    return cn, (cn_n * x_n + cn_f * x_f) / cn
