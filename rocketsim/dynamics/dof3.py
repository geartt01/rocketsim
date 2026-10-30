"""3-DOF planar dynamics in the vertical plane of the rail: state [x, z, v_x, v_z, θ, θ̇].

x is horizontal distance downrange along the rail heading, z is up, θ is the pitch of the body
axis above the horizontal, θ̇ its rate (positive = nose rotating from +x toward +z). Only the
wind component along the rail heading is felt.
"""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from rocketsim.aero.barrowman import fins_cn_alpha_cp, nose_cn_alpha_cp
from rocketsim.aero.damping import pitch_damping_coefficient
from rocketsim.aero.drag import drag_coefficient
from rocketsim.atmosphere import isa
from rocketsim.constants import G0
from rocketsim.wind import Wind

STATE = ("x", "z", "vx", "vz", "theta", "omega")
Z, VZ, HORIZONTAL, VELOCITY = 1, 3, (0,), (2, 3)


@dataclass
class Launch:
    site_altitude: float  # m above mean sea level
    rail_length: float  # m the rocket travels along the rail before it is free
    rail_elevation: float  # rad above horizontal
    rail_heading: float  # rad clockwise from North
    wind: Wind
    g: float = G0  # m/s²; tests may change it
    density: Callable[[float], float] | None = None  # kg/m³ override for tests; None → ISA


def air(launch, z):
    """(ρ [kg/m³], speed of sound [m/s], kinematic viscosity ν = μ/ρ [m²/s]) at height z."""
    atm = isa(launch.site_altitude + z)
    rho = launch.density(z) if launch.density else atm.rho
    return rho, atm.a, atm.mu / atm.rho


def downrange_wind(launch, z):
    """Wind component along the rail heading, m/s (positive blows downrange)."""
    w = launch.wind.vector(z)
    return w[0] * np.sin(launch.rail_heading) + w[1] * np.cos(launch.rail_heading)


def initial_state(launch):
    return [0.0, 0.0, 0.0, 0.0, launch.rail_elevation, 0.0]


def normalize(y):
    return y


def aero_forces(t, y, rocket, launch):
    """Inertial non-gravity forces and pitching moment about the CG.

    With body axis e = [cos θ, sin θ] and air-relative velocity v_a = v − [w(z), 0]:
        v_ax = v_a·e,  v_lat = v_a − v_ax·e,  α = atan2(|v_lat|, v_ax),  q = ½ρ|v_a|²
        axial   F_a = (T(t) − ½ρ·A·C_D0·|v_a|·v_ax)·e          (= −q·A·C_D0·cos α + thrust)
        normal  F_n = −q·A·CN_α·sin α · v_lat/|v_lat| = −½ρ·A·CN_α·|v_a|·v_lat
        moment  M   = r_cp × F_n − K·θ̇·|θ̇|,  r_cp = −(X_cp − x_cg)·e  (CP aft of CG)

    Args: t (s since ignition), y state, rocket (geometry.Rocket), launch (Launch).
    Returns: (F_x [N], F_z [N], M [N·m], mass [kg], I_pitch [kg·m²],
              info = (α [rad], Mach [-], stability margin [calibers])).

    Assumptions: axial drag scales with cos α (exact at α = 0, PLAN's F_x = −q·S·C_A); CN_α
    and CP from Barrowman at the current Mach; variable-mass moment terms ignored.
    Sources: Niskanen 2009 §3.1–3.2 (forces), §3.2.3 (damping); Barrowman 1967.
    """
    _, z, vx, vz, theta, omega = y
    m, x_cg, inertia, _ = rocket.mass_properties(t)
    rho, sound, nu = air(launch, z)
    ex, ez = np.cos(theta), np.sin(theta)
    vax, vaz = vx - downrange_wind(launch, z), vz
    speed = np.hypot(vax, vaz)
    mach = speed / sound
    v_axial = vax * ex + vaz * ez
    lat_x, lat_z = vax - v_axial * ex, vaz - v_axial * ez

    thrusting = 0.0 <= t < rocket.motor.burn_time
    cd = drag_coefficient(rocket, mach, speed * rocket.length / nu, thrusting)
    (cn_n, x_n), (cn_f, x_f) = nose_cn_alpha_cp(rocket), fins_cn_alpha_cp(rocket, mach)
    cn = cn_n + cn_f
    x_cp = (cn_n * x_n + cn_f * x_f) / cn

    axial = rocket.motor.thrust(t) - 0.5 * rho * rocket.area * cd * speed * v_axial
    k = -0.5 * rho * rocket.area * cn * speed
    fnx, fnz = k * lat_x, k * lat_z
    moment = -(x_cp - x_cg) * (ex * fnz - ez * fnx)
    moment -= pitch_damping_coefficient(rocket, rho, x_cg, x_f) * omega * abs(omega)

    alpha = np.arctan2(np.hypot(lat_x, lat_z), v_axial)
    margin = (x_cp - x_cg) / rocket.body.diameter
    return axial * ex + fnx, axial * ez + fnz, moment, m, inertia, (alpha, mach, margin)


def info(t, y, rocket, launch):
    """(α [rad], Mach [-], stability margin [calibers], mass [kg]) for output."""
    *_, m, _, (alpha, mach, margin) = aero_forces(t, y, rocket, launch)
    return alpha, mach, margin, m


def derivatives(t, y, rocket, launch):
    """Free flight: [v_x, v_z, F_x/m, F_z/m − g, θ̇, M/I]  (Newton, Euler in the plane)."""
    fx, fz, moment, m, inertia, _ = aero_forces(t, y, rocket, launch)
    return [y[2], y[3], fx / m, fz / m - launch.g, y[5], moment / inertia]


def derivatives_rail(t, y, rocket, launch):
    """On the rail: motion only along the rail axis e, no rotation.

        a = (T + F_axial,drag − m·g·sin θ_rail) / m  along e
    The rail reacts the normal aerodynamic force and the gravity component across it. While
    the rocket has not started moving and the net force is not positive, it stays put.
    """
    fx, fz, _, m, _, _ = aero_forces(t, y, rocket, launch)
    ex, ez = np.cos(y[4]), np.sin(y[4])
    along = (fx * ex + fz * ez) / m - launch.g * ez
    if y[2] * ex + y[3] * ez <= 0 and along <= 0:
        along = 0.0
    return [y[2], y[3], along * ex, along * ez, 0.0, 0.0]


def derivatives_chute(t, y, rocket, launch):
    """Under parachute, point mass: a = −½ρ·CdA·|v_a|·v_a / m − g·ẑ; attitude frozen.

    Body drag is ignored under canopy (as in Stage 1); drag acts along −v_a, so the rocket
    drifts with the wind.
    """
    m = rocket.mass_properties(t)[0]
    rho = air(launch, y[1])[0]
    vax, vaz = y[2] - downrange_wind(launch, y[1]), y[3]
    k = -0.5 * rho * rocket.chute_cda * np.hypot(vax, vaz) / m
    return [y[2], y[3], k * vax, k * vaz - launch.g, 0.0, 0.0]


def rail_exit(t, y, rocket, launch):
    """Zero when the distance travelled along the rail reaches the rail length."""
    return y[0] * np.cos(y[4]) + y[1] * np.sin(y[4]) - launch.rail_length
