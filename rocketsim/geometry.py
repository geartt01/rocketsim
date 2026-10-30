"""Rocket components and mass properties.

Axial positions are measured from the nose tip, positive aft (PLAN.md §0). The body is one
nose cone followed by one cylindrical tube of constant diameter; fins are one trapezoidal set.
"""

from dataclasses import dataclass, field

import numpy as np
from scipy.integrate import quad

from rocketsim.motor import Motor
from rocketsim.recovery import Recovery

NOSE_SHAPES = ("cone", "ogive", "elliptical", "von_karman")
FIN_PROFILES = ("square", "rounded", "airfoil")
_N = 2001  # samples for numerical integration over a component


def nose_radius(shape, x, length, radius):
    """Nose-cone profile r(x), m, for 0 ≤ x ≤ length (x from the tip).

        cone        r = R·x/L                                            (Niskanen A.1)
        ogive       r = sqrt(ρ² − (L − x)²) − (ρ − R),  ρ = (R² + L²)/(2R)  (tangent ogive, A.2/A.5)
        elliptical  r = R·sqrt(1 − (1 − x/L)²)                           (A.6)
        von_karman  r = R/√π · sqrt(θ − sin(2θ)/2),  θ = acos(1 − 2x/L)   (Haack κ = 0, A.9/A.10)

    Source: Niskanen 2009, Appendix A. The tangent ogive needs L ≥ R.
    """
    x = np.clip(np.asarray(x, dtype=float), 0.0, length)
    u = x / length
    if shape == "cone":
        return radius * u
    if shape == "ogive":
        rho = (radius**2 + length**2) / (2 * radius)
        return np.sqrt(np.maximum(rho**2 - (length - x) ** 2, 0.0)) - (rho - radius)
    if shape == "elliptical":
        return radius * np.sqrt(np.maximum(1 - (1 - u) ** 2, 0.0))
    if shape == "von_karman":
        theta = np.arccos(1 - 2 * u)
        return radius / np.sqrt(np.pi) * np.sqrt(theta - np.sin(2 * theta) / 2)
    raise ValueError(f"unknown nose shape {shape!r}; choose from {NOSE_SHAPES}")


def shell_properties(x, r, mass):
    """Mass properties of a thin shell of revolution with uniform areal density.

    The profile (x, r) is cut into conical rings; ring i has surface dS = 2π·r_i·ds_i and mass
    dm = m·dS/S. A thin ring of radius r has I = dm·r² about the axis and dm·r²/2 about a
    diameter, so about the shell's CG:
        I_x = Σ dm·r²,   I_t = Σ dm·(r²/2 + (x − x_cg)²)

    Args: x, r: profile samples, m; mass, kg.
    Returns: (x_cg [m], I_transverse about x_cg [kg·m²], I_axial [kg·m²],
              wetted surface area S [m²]).
    """
    xm, rm = (x[1:] + x[:-1]) / 2, (r[1:] + r[:-1]) / 2
    ds = 2 * np.pi * rm * np.hypot(np.diff(x), np.diff(r))
    w = ds / ds.sum()
    x_cg = np.sum(w * xm)
    i_t = mass * np.sum(w * (rm**2 / 2 + (xm - x_cg) ** 2))
    return x_cg, i_t, mass * np.sum(w * rm**2), ds.sum()


def solid_cylinder_inertia(mass, radius, length):
    """Solid cylinder about its CG: (I_t = m·(3r² + L²)/12, I_x = m·r²/2), kg·m²."""
    return mass * (3 * radius**2 + length**2) / 12, mass * radius**2 / 2


@dataclass
class Nose:
    shape: str
    length: float  # m
    mass: float  # kg


@dataclass
class Body:
    length: float  # m, starts at the nose base
    diameter: float  # m, outer
    mass: float  # kg


@dataclass
class Fins:
    count: int  # ≥ 3, evenly spaced
    root_chord: float  # m
    tip_chord: float  # m
    span: float  # m, root to tip (semispan s)
    sweep: float  # m, tip leading edge aft of root leading edge
    thickness: float  # m
    position: float  # m, nose tip to root-chord leading edge
    profile: str  # square | rounded | airfoil
    mass: float  # kg, whole set

    @property
    def area(self):
        """One-sided area of one fin, m²."""
        return self.span * (self.root_chord + self.tip_chord) / 2

    @property
    def mac(self):
        """Mean aerodynamic chord c̄ = (1/A)∫c² dy = ⅔(Cr² + Cr·Ct + Ct²)/(Cr + Ct), m (Niskanen 3.30)."""
        cr, ct = self.root_chord, self.tip_chord
        return 2 / 3 * (cr**2 + cr * ct + ct**2) / (cr + ct)

    def properties(self, body_radius):
        """Mass properties of the fin set as thin flat plates.

        Each fin is cut into spanwise strips (chord c, radius r = R + y). For N ≥ 3 evenly
        spaced fins Σ sin²φ_k = N/2, so about the set's CG
            I_t = m·(E[(x − x_cg)²] + E[r²]/2),   I_x = m·E[r²],   averages weighted by area.
        Returns: (x_cg [m], I_transverse [kg·m²], I_axial [kg·m²]).
        """
        y = np.linspace(0.0, self.span, _N)
        ym, dy = (y[1:] + y[:-1]) / 2, np.diff(y)
        c = self.root_chord + (self.tip_chord - self.root_chord) * ym / self.span
        x_mid = self.position + self.sweep * ym / self.span + c / 2
        w = c * dy / np.sum(c * dy)
        x_cg = np.sum(w * x_mid)
        e_dx2 = np.sum(w * (c**2 / 12 + (x_mid - x_cg) ** 2))
        e_r2 = np.sum(w * (body_radius + ym) ** 2)
        return x_cg, self.mass * (e_dx2 + e_r2 / 2), self.mass * e_r2


@dataclass
class MassItem:
    name: str
    mass: float  # kg
    position: float  # m, nose tip to the item's CG
    length: float = 0.0  # m; with radius, a solid cylinder, else a point mass
    radius: float = 0.0  # m


@dataclass
class Rocket:
    nose: Nose
    body: Body
    fins: Fins
    masses: list[MassItem]
    motor: Motor
    motor_position: float  # m, nose tip to the motor's aft end
    roughness: float  # m, surface roughness height (Niskanen Table 3.2)
    drag: object  # "buildup" | float (constant Cd) | (mach array, cd array) table
    recovery: Recovery
    _dry: tuple = field(init=False, repr=False)  # (mass, x_cg, I about x_cg) without motor

    def __post_init__(self):
        r = self.radius
        x = np.linspace(0.0, self.nose.length, _N)
        *nose, self.nose_surface = shell_properties(
            x, nose_radius(self.nose.shape, x, self.nose.length, r), self.nose.mass)
        x = np.linspace(self.nose.length, self.length, _N)
        *tube, _ = shell_properties(x, np.full(_N, r), self.body.mass)
        parts = [(self.nose.mass, *nose), (self.body.mass, *tube),
                 (self.fins.mass, *self.fins.properties(r))]
        parts += [(m.mass, m.position, *solid_cylinder_inertia(m.mass, m.radius, m.length))
                  for m in self.masses]
        self._dry = combine(parts)
        n = self.nose  # V = ∫ π·r(x)² dx, cached because the CP needs it on every step
        self.nose_volume = quad(lambda s: np.pi * nose_radius(n.shape, s, n.length, r) ** 2,
                                0.0, n.length, limit=200)[0]

    @property
    def radius(self):
        return self.body.diameter / 2

    @property
    def area(self):
        """Reference area A_ref = π·d²/4 of the body tube, m²."""
        return np.pi * self.radius**2

    @property
    def length(self):
        """Nose tip to the aft end of the body tube, m."""
        return self.nose.length + self.body.length

    @property
    def dry_mass(self):
        return self._dry[0]

    def mass_properties(self, t):
        """Total mass, CG and inertia at time t since ignition.

        The motor is a solid cylinder whose mass m_casing + m_prop(t) sits at its geometric
        centre (propellant CG assumed fixed). Parallel-axis theorem (all parts on the axis):
            m = Σm_i,  x_cg = Σm_i·x_i / m,  I_t = Σ[I_t,i + m_i·(x_i − x_cg)²],  I_x = Σ I_x,i

        Returns: (m [kg], x_cg [m from nose tip], I_transverse (pitch = yaw) about the CG
                  [kg·m²], I_axial (roll) [kg·m²]).
        """
        mo = self.motor
        m_motor = mo.total_mass_initial - mo.propellant_mass_initial + mo.propellant_mass(t)
        motor_part = (m_motor, self.motor_position - mo.length / 2,
                      *solid_cylinder_inertia(m_motor, mo.diameter / 2, mo.length))
        return combine([self._dry, motor_part])


def combine(parts):
    """Combine (m, x_cg, I_t about own CG, I_x) parts with the parallel-axis theorem."""
    m = sum(p[0] for p in parts)
    x_cg = sum(p[0] * p[1] for p in parts) / m
    i_t = sum(p[2] + p[0] * (p[1] - x_cg) ** 2 for p in parts)
    return m, x_cg, i_t, sum(p[3] for p in parts)
