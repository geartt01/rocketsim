"""1-DOF vertical rocket dynamics: state [z, v_z]."""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from rocketsim.atmosphere import isa
from rocketsim.constants import G0
from rocketsim.motor import Motor


@dataclass
class Rocket1D:
    dry_mass: float  # kg, everything except the motor
    diameter: float  # m, reference diameter
    cd: float  # -, constant drag coefficient of the rocket body
    motor: Motor
    chute_cda: float  # m^2, parachute drag coefficient × area
    elevation: float = 0.0  # m, launch site above mean sea level
    g: float = G0  # m/s^2; tests set 0 to isolate the rocket equation
    density: Callable[[float], float] | None = None  # kg/m^3 at z; None → ISA at elevation + z

    @property
    def area(self):
        """Reference area A = π·d²/4, m²."""
        return np.pi * self.diameter**2 / 4

    def mass(self, t):
        """Total mass m(t) = m_dry + m_motor,0 − m_prop,0 + m_prop(t), kg (t in s)."""
        m = self.motor
        return self.dry_mass + m.total_mass_initial - m.propellant_mass_initial + m.propellant_mass(t)

    def rho(self, z):
        """Air density at height z above the launch site, kg/m³."""
        return self.density(z) if self.density else isa(self.elevation + z).rho


def derivatives(t, y, rocket, chute):
    """Right-hand side of the vertical equations of motion.

        dz/dt   = v
        dv/dt   = (F(t) + D) / m(t) − g,   D = −½·ρ(z)·CdA·v·|v|

    Args:
        t: time since ignition, s
        y: state [z (m, up from the launch site), v (m/s, up)]
        rocket: Rocket1D
        chute: True under parachute (CdA = chute Cd·A, body drag ignored), else CdA = Cd·A

    Returns:
        [dz/dt (m/s), dv/dt (m/s²)]

    Assumptions: purely vertical motion, constant g, drag always opposes velocity. Newton's
    second law in the form m·a = ΣF holds for a variable-mass rocket because the thrust F
    already contains the momentum carried away by the exhaust (Tsiolkovsky's derivation).

    Source: standard drag equation D = ½·ρ·v²·Cd·A (Anderson, Fundamentals of Aerodynamics,
    ch. 1); rocket equation of motion as in Niskanen 2009, §4.2.
    """
    z, v = y
    cda = rocket.chute_cda if chute else rocket.cd * rocket.area
    drag = -0.5 * rocket.rho(z) * cda * v * abs(v)
    return [v, (rocket.motor.thrust(t) + drag) / rocket.mass(t) - rocket.g]
