"""Phase manager for the 1-DOF flight: pad → powered → coast → parachute → landed."""

import warnings
from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

from rocketsim.atmosphere import isa
from rocketsim.constants import G0
from rocketsim.dynamics.dof1 import derivatives

T_PHASE_MAX = 3600.0  # s, safety cap on any one phase so a bad config cannot loop forever
MACH_LIMIT = 0.8  # validity limit of the aero model (PLAN.md §3)


def apogee_event(t, y, *args):
    """Zero when v_z = 0 while falling from positive to negative: apogee."""
    return y[1]


def landing_event(t, y, *args):
    """Zero when z = 0 while descending: touchdown at launch-site height."""
    return y[0]


for _event in (apogee_event, landing_event):
    _event.terminal, _event.direction = True, -1


def run_phase(rocket, t0, y0, t_end, events=(), chute=False, rtol=1e-8, atol=1e-8):
    """Integrate one flight phase with DOP853 until t_end or the first terminal event."""
    return solve_ivp(derivatives, (t0, t_end), y0, method="DOP853", rtol=rtol, atol=atol,
                     events=list(events), args=(rocket, chute), dense_output=True)


def liftoff_time(rocket):
    """First time thrust exceeds weight, F(t) = m(t)·g, in s since ignition.

    Assumption: the first crossing lies within a fine grid (data points + 2001 samples);
    brentq then finds it to machine precision.
    """
    motor = rocket.motor
    t = np.union1d(motor.time, np.linspace(0.0, motor.burn_time, 2001))
    excess = motor.thrust(t) - rocket.mass(t) * rocket.g
    above = np.flatnonzero(excess > 0)
    if len(above) == 0:
        raise ValueError("thrust never exceeds weight: the rocket does not leave the pad")
    i = above[0]
    if i == 0:
        return 0.0
    return brentq(lambda s: motor.thrust(s) - rocket.mass(s) * rocket.g, t[i - 1], t[i])


@dataclass
class Flight:
    trajectory: dict  # column name → array; see simulate()
    events: dict  # event name → time since ignition, s
    summary: dict  # see simulate()


def simulate(rocket, rtol=1e-8, atol=1e-8, dt_out=0.01):
    """Fly from the pad to landing, one solve_ivp call per phase.

    Each phase ends exactly on its event (burnout time, apogee, landing), so the integrator
    never steps across a discontinuity in thrust or drag.

    Returns:
        Flight with
          trajectory: t (s), z (m), vz (m/s), az (m/s²), mass (kg), phase, sampled every
                      dt_out seconds plus thrust-curve points, from liftoff to landing;
          events: liftoff, burnout, apogee, landing (s since ignition);
          summary: apogee (m), time_to_apogee (s from liftoff), max_velocity (m/s),
                   max_accel_g (-), landing_velocity (m/s, positive down), flight_time (s),
                   max_mach (-).
    """
    motor = rocket.motor
    events = {"liftoff": liftoff_time(rocket)}

    powered = run_phase(rocket, events["liftoff"], [0.0, 0.0], motor.burn_time,
                        [apogee_event, landing_event], rtol=rtol, atol=atol)
    if powered.status == 1:
        # shortcut: apogee or fall-back during the burn is not modelled; Stage 4 adds delays.
        raise ValueError("rocket stopped climbing before burnout: motor too weak for this rocket")
    events["burnout"] = motor.burn_time

    coast = run_phase(rocket, motor.burn_time, powered.y[:, -1], motor.burn_time + T_PHASE_MAX,
                      [apogee_event], rtol=rtol, atol=atol)
    if coast.status != 1:
        raise RuntimeError(f"no apogee within {T_PHASE_MAX} s of burnout")
    events["apogee"] = coast.t[-1]

    chute = run_phase(rocket, coast.t[-1], coast.y[:, -1], coast.t[-1] + T_PHASE_MAX,
                      [landing_event], chute=True, rtol=rtol, atol=atol)
    if chute.status != 1:
        raise RuntimeError(f"no landing within {T_PHASE_MAX} s of apogee")
    events["landing"] = chute.t[-1]

    parts = [
        _sample(powered, rocket, False, "powered", dt_out, motor.time),
        _sample(coast, rocket, False, "coast", dt_out),
        _sample(chute, rocket, True, "parachute", dt_out),
    ]
    landed = {"t": [chute.t[-1]], "z": [chute.y[0, -1]], "vz": [chute.y[1, -1]], "az": [0.0],
              "mass": [rocket.mass(chute.t[-1])], "phase": ["landed"]}
    trajectory = {k: np.concatenate([p[k] for p in parts] + [landed[k]]) for k in landed}

    z, vz = trajectory["z"], trajectory["vz"]
    max_mach = float(np.max(np.abs(vz) / isa(rocket.elevation + np.maximum(z, 0.0)).a))
    if max_mach > MACH_LIMIT:
        warnings.warn(f"max Mach {max_mach:.2f} exceeds the aero model's limit {MACH_LIMIT}",
                      stacklevel=2)
    summary = {
        "apogee": float(coast.y[0, -1]),
        "time_to_apogee": events["apogee"] - events["liftoff"],
        "max_velocity": float(vz.max()),
        "max_accel_g": float(trajectory["az"].max() / G0),
        "landing_velocity": float(-chute.y[1, -1]),
        "flight_time": events["landing"] - events["liftoff"],
        "max_mach": max_mach,
    }
    return Flight(trajectory, events, summary)


def _sample(sol, rocket, chute, phase, dt_out, extra_times=()):
    """Sample a phase on [t0, t1) from its dense output; az is re-evaluated from the model."""
    t0, t1 = sol.t[0], sol.t[-1]
    extra = [s for s in extra_times if t0 < s < t1]
    t = np.union1d(np.arange(t0, t1, dt_out), extra)
    z, vz = sol.sol(t)
    az = np.array([derivatives(ti, yi, rocket, chute)[1] for ti, yi in zip(t, zip(z, vz))])
    return {"t": t, "z": z, "vz": vz, "az": az, "mass": rocket.mass(t),
            "phase": np.full(len(t), phase)}
