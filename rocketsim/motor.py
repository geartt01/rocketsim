"""Solid rocket motor: RASP .eng parser, thrust curve and propellant depletion."""

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass(eq=False)
class Motor:
    name: str
    diameter: float  # m
    length: float  # m
    delays: str  # ejection delays as written in the file, e.g. "5-7-P"; unused for now
    propellant_mass_initial: float  # kg
    total_mass_initial: float  # kg, casing + propellant
    manufacturer: str
    time: np.ndarray  # s, strictly increasing, starts at 0
    thrust_data: np.ndarray  # N, same length as time
    _cum_impulse: np.ndarray = field(init=False, repr=False)  # N·s at each data point

    def __post_init__(self):
        segment = 0.5 * (self.thrust_data[1:] + self.thrust_data[:-1]) * np.diff(self.time)
        self._cum_impulse = np.concatenate(([0.0], np.cumsum(segment)))

    @property
    def burn_time(self):
        """Time of the last thrust data point, s."""
        return float(self.time[-1])

    @property
    def total_impulse(self):
        """I_total = ∫F dt over the whole curve, N·s (exact for the piecewise-linear curve)."""
        return float(self._cum_impulse[-1])

    def thrust(self, t):
        """Thrust F(t), N, linearly interpolated between data points; 0 for t < 0 or t > t_burn.

        Args: t, s (scalar or array). Assumption: thrust is linear between RASP data points.
        """
        return np.interp(t, self.time, self.thrust_data, left=0.0, right=0.0)

    def impulse(self, t):
        """Impulse delivered so far, I(t) = ∫₀ᵗ F dt', N·s.

        On segment k, F is linear with slope s_k, so with τ = t − t_k:
            I(t) = I(t_k) + F_k·τ + ½·s_k·τ²
        which is exact (no numerical quadrature error). Args: t, s (scalar or array).
        """
        t = np.clip(np.asarray(t, dtype=float), self.time[0], self.time[-1])
        k = np.clip(np.searchsorted(self.time, t, side="right") - 1, 0, len(self.time) - 2)
        tau = t - self.time[k]
        slope = (self.thrust_data[k + 1] - self.thrust_data[k]) / (self.time[k + 1] - self.time[k])
        return self._cum_impulse[k] + self.thrust_data[k] * tau + 0.5 * slope * tau**2

    def propellant_mass(self, t):
        """Remaining propellant mass, kg:  m_p(t) = m_p0 · (1 − I(t)/I_total).

        Args: t, s (scalar or array).

        Assumption: constant effective exhaust velocity v_e = I_total/m_p0, so mass flow
        ṁ = F/v_e is proportional to thrust and propellant burns in proportion to impulse
        delivered (PLAN.md §4 Stage 0). Real motors deviate slightly at ignition and tail-off.
        """
        return self.propellant_mass_initial * (1.0 - self.impulse(t) / self.total_impulse)


def load_eng(path):
    """Parse a single-motor RASP .eng file.

    Format (https://www.thrustcurve.org/info/raspformat.html): ';' starts a comment; the first
    non-comment line is the header "name diameter_mm length_mm delays prop_mass_kg
    total_mass_kg manufacturer"; each following line is "time_s thrust_N".

    If the curve does not start at t = 0, a (0, 0) point is prepended so thrust ramps up from
    zero at ignition. Units are converted to SI (mm → m) on load.

    Raises:
        ValueError naming the file and line for anything malformed.
    """
    path = Path(path)

    def fail(msg, lineno=None):
        where = f"{path}:{lineno}" if lineno else str(path)
        raise ValueError(f"{where}: {msg}")

    def number(text, what, lineno):
        try:
            return float(text)
        except ValueError:
            fail(f"{what} is not a number: {text!r}", lineno)

    lines = []
    # errors="replace": odd bytes can only matter in comments, which are discarded
    text_lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    for lineno, raw in enumerate(text_lines, start=1):
        text = raw.split(";", 1)[0].strip()
        if text:
            lines.append((lineno, text.split()))
    if not lines:
        fail("no motor header found")

    lineno, header = lines[0]
    if len(header) != 7:
        fail(
            "header needs 7 fields (name diameter_mm length_mm delays prop_mass_kg "
            f"total_mass_kg manufacturer), got {len(header)}",
            lineno,
        )
    name, dia, length, delays, m_prop, m_total, manufacturer = header
    dia = number(dia, "diameter", lineno) / 1000.0
    length = number(length, "length", lineno) / 1000.0
    m_prop = number(m_prop, "propellant mass", lineno)
    m_total = number(m_total, "total mass", lineno)
    if dia <= 0 or length <= 0:
        fail("diameter and length must be positive", lineno)
    if not 0 < m_prop <= m_total:
        fail(f"need 0 < propellant mass ({m_prop}) <= total mass ({m_total})", lineno)

    times, thrusts = [], []
    for lineno, parts in lines[1:]:
        if len(parts) == 7:
            fail("multiple motors in one file are not supported", lineno)
        if len(parts) != 2:
            fail(f"expected 'time thrust', got {len(parts)} fields", lineno)
        t = number(parts[0], "time", lineno)
        f = number(parts[1], "thrust", lineno)
        if t < 0 or (times and t <= times[-1]):
            fail(f"times must be non-negative and strictly increasing (t={t})", lineno)
        if f < 0:
            fail(f"thrust is negative (F={f})", lineno)
        times.append(t)
        thrusts.append(f)
    if not times:
        fail("no thrust data after the header")
    if times[-1] <= 0:
        fail("thrust curve must end at a positive time")
    if max(thrusts) == 0:
        fail("thrust curve has no thrust (all values are 0)")
    if times[0] > 0:
        times.insert(0, 0.0)
        thrusts.insert(0, 0.0)

    return Motor(
        name=name,
        diameter=dia,
        length=length,
        delays=delays,
        propellant_mass_initial=m_prop,
        total_mass_initial=m_total,
        manufacturer=manufacturer,
        time=np.array(times),
        thrust_data=np.array(thrusts),
    )
