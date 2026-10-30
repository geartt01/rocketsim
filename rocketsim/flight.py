"""Phase managers.

1-DOF (Stage 1):  pad → powered → coast → parachute → landed
3-DOF (Stage 2):  pad → rail → powered → coast → parachute → landed

Each phase is one solve_ivp (DOP853) call that ends exactly on its event, so the integrator
never steps across a jump in thrust, drag or constraints (PLAN.md §3).
"""

import warnings
from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

from rocketsim.atmosphere import isa
from rocketsim.constants import G0
from rocketsim.dynamics import dof1, dof3

T_PHASE_MAX = 3600.0  # s, safety cap on any one phase so a bad config cannot loop forever
MACH_LIMIT = 0.8  # validity limit of the aero model (PLAN.md §3)
RAIL_EXIT_MIN = 15.0  # m/s, common hobby guideline for a safe rail-exit speed
MARGIN_MIN = 1.0  # calibers, common minimum static stability margin


def _event(index, direction=-1):
    """Terminal event: state[index] crosses zero in the given direction."""
    def event(t, y, *args):
        return y[index]
    event.terminal, event.direction = True, direction
    return event


apogee_event, landing_event = _event(1), _event(0)  # 1-DOF state [z, v]
apogee_event_3, landing_event_3 = _event(3), _event(1)  # 3-DOF state [x, z, vx, vz, θ, ω]


def rail_exit_event(t, y, rocket, launch):
    """Zero when the distance travelled along the rail reaches the rail length."""
    return y[0] * np.cos(y[4]) + y[1] * np.sin(y[4]) - launch.rail_length


rail_exit_event.terminal, rail_exit_event.direction = True, 1


def run_phase(fun, t0, y0, t_end, args, events=(), rtol=1e-8, atol=1e-8):
    """Integrate one flight phase with DOP853 until t_end or the first terminal event."""
    return solve_ivp(fun, (t0, t_end), y0, method="DOP853", rtol=rtol, atol=atol,
                     events=list(events), args=args, dense_output=True)


def liftoff_time(motor, mass, g):
    """First time thrust exceeds the weight component along the rail: F(t) = m(t)·g.

    Args: motor; mass(t) → kg (vectorised); g, m/s² (already × sin θ_rail for a tilted rail).
    Assumption: the first crossing lies within a fine grid (data points + 2001 samples);
    brentq then finds it to machine precision.
    """
    t = np.union1d(motor.time, np.linspace(0.0, motor.burn_time, 2001))
    excess = motor.thrust(t) - mass(t) * g
    above = np.flatnonzero(excess > 0)
    if len(above) == 0:
        raise ValueError("thrust never exceeds weight: the rocket does not leave the pad")
    i = above[0]
    if i == 0:
        return 0.0
    return brentq(lambda s: motor.thrust(s) - mass(s) * g, t[i - 1], t[i])


@dataclass
class Flight:
    trajectory: dict  # column name → array; see simulate()
    events: dict  # event name → time since ignition, s
    summary: dict  # see simulate()


def _check(sol, phase, what):
    if sol.status != 1:
        raise RuntimeError(f"{phase}: {what}")


def _times(sol, dt_out, extra=()):
    """Output times on [t0, t1): a dt_out grid plus any extra times inside the phase."""
    t0, t1 = sol.t[0], sol.t[-1]
    return np.union1d(np.arange(t0, t1, dt_out), [s for s in extra if t0 < s < t1])


def _warn_mach(max_mach):
    if max_mach > MACH_LIMIT:
        warnings.warn(f"max Mach {max_mach:.2f} exceeds the aero model's limit {MACH_LIMIT}",
                      stacklevel=3)


def simulate_1dof(rocket, rtol=1e-8, atol=1e-8, dt_out=0.01):
    """Stage 1 vertical flight of a dof1.Rocket1D from the pad to landing.

    Returns Flight with trajectory columns t, z, vz, az, mass, phase; events liftoff, burnout,
    apogee, landing; summary apogee, time_to_apogee, max_velocity, max_accel_g,
    landing_velocity, flight_time, max_mach.
    """
    motor, tol = rocket.motor, {"rtol": rtol, "atol": atol}
    events = {"liftoff": liftoff_time(motor, rocket.mass, rocket.g)}
    powered = run_phase(dof1.derivatives, events["liftoff"], [0.0, 0.0], motor.burn_time,
                        (rocket, False), [apogee_event, landing_event], **tol)
    if powered.status == 1:
        # shortcut: apogee or fall-back during the burn is not modelled; Stage 4 adds delays.
        raise ValueError("rocket stopped climbing before burnout: motor too weak for this rocket")
    events["burnout"] = motor.burn_time
    coast = run_phase(dof1.derivatives, motor.burn_time, powered.y[:, -1],
                      motor.burn_time + T_PHASE_MAX, (rocket, False), [apogee_event], **tol)
    _check(coast, "coast", f"no apogee within {T_PHASE_MAX} s of burnout")
    events["apogee"] = coast.t[-1]
    chute = run_phase(dof1.derivatives, coast.t[-1], coast.y[:, -1], coast.t[-1] + T_PHASE_MAX,
                      (rocket, True), [landing_event], **tol)
    _check(chute, "parachute", f"no landing within {T_PHASE_MAX} s of apogee")
    events["landing"] = chute.t[-1]

    columns = {k: [] for k in ("t", "z", "vz", "az", "mass", "phase")}
    for sol, chute_open, phase, extra in ((powered, False, "powered", motor.time),
                                          (coast, False, "coast", ()),
                                          (chute, True, "parachute", ())):
        t = _times(sol, dt_out, extra)
        z, vz = sol.sol(t)
        az = [dof1.derivatives(ti, yi, rocket, chute_open)[1] for ti, yi in zip(t, zip(z, vz))]
        for k, v in zip(columns, (t, z, vz, az, rocket.mass(t), [phase] * len(t))):
            columns[k].append(v)
    t_land = chute.t[-1]
    for k, v in zip(columns, ([t_land], [chute.y[0, -1]], [chute.y[1, -1]], [0.0],
                              [rocket.mass(t_land)], ["landed"])):
        columns[k].append(v)
    trajectory = {k: np.concatenate(v) for k, v in columns.items()}

    z, vz = trajectory["z"], trajectory["vz"]
    max_mach = float(np.max(np.abs(vz) / isa(rocket.elevation + np.maximum(z, 0.0)).a))
    _warn_mach(max_mach)
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


def simulate(rocket, launch, rtol=1e-8, atol=1e-8, dt_out=0.01):
    """3-DOF planar flight of a geometry.Rocket from the pad to landing.

    Returns Flight with
      trajectory: t (s), x, z (m), vx, vz (m/s), ax, az (m/s²), theta (rad), alpha (rad),
                  mach (-), margin (calibers, NaN under parachute), mass (kg), phase;
      events: liftoff, rail_exit, burnout, apogee, landing (s since ignition);
      summary: apogee, apogee_x (m), time_to_apogee (s), max_velocity (m/s), max_accel_g,
               max_mach, max_alpha (rad), rail_exit_velocity (m/s), margin_rail_exit,
               min_margin (calibers, rail exit → apogee), landing_velocity (m/s, descent
               rate), drift (m, landing x), flight_time (s).
    Warns when Mach > 0.8, rail-exit speed < 15 m/s or margin at rail exit < 1 caliber.
    """
    motor, args, tol = rocket.motor, (rocket, launch), {"rtol": rtol, "atol": atol}
    th0 = launch.rail_elevation
    events = {"liftoff": liftoff_time(motor, lambda t: rocket.mass_properties(t)[0],
                                      launch.g * np.sin(th0))}
    y0 = [0.0, 0.0, 0.0, 0.0, th0, 0.0]
    rail = run_phase(dof3.derivatives_rail, events["liftoff"], y0, motor.burn_time, args,
                     [rail_exit_event], **tol)
    _check(rail, "rail", "still on the rail at burnout: rail too long or motor too weak")
    events["rail_exit"] = rail.t[-1]
    exit_speed = float(np.hypot(rail.y[2, -1], rail.y[3, -1]))
    margin_exit = float(dof3.aero_forces(rail.t[-1], rail.y[:, -1], *args)[5][2])
    # warn now, before free flight, so the warning survives if an unstable flight fails later
    if exit_speed < RAIL_EXIT_MIN:
        warnings.warn(f"rail-exit velocity {exit_speed:.1f} m/s is below {RAIL_EXIT_MIN} m/s",
                      stacklevel=2)
    if margin_exit < MARGIN_MIN:
        warnings.warn(f"stability margin {margin_exit:.2f} cal at rail exit is below "
                      f"{MARGIN_MIN}", stacklevel=2)

    powered = run_phase(dof3.derivatives, rail.t[-1], rail.y[:, -1], motor.burn_time, args,
                        [apogee_event_3, landing_event_3], **tol)
    if powered.status == 1:
        # shortcut: apogee or fall-back during the burn is not modelled; Stage 4 adds delays.
        raise ValueError("rocket stopped climbing before burnout: motor too weak for this rocket")
    events["burnout"] = motor.burn_time
    coast = run_phase(dof3.derivatives, motor.burn_time, powered.y[:, -1],
                      motor.burn_time + T_PHASE_MAX, args, [apogee_event_3], **tol)
    _check(coast, "coast", f"no apogee within {T_PHASE_MAX} s of burnout")
    events["apogee"] = coast.t[-1]
    chute = run_phase(dof3.derivatives_chute, coast.t[-1], coast.y[:, -1],
                      coast.t[-1] + T_PHASE_MAX, args, [landing_event_3], **tol)
    _check(chute, "parachute", f"no landing within {T_PHASE_MAX} s of apogee")
    events["landing"] = chute.t[-1]

    names = ("t", "x", "z", "vx", "vz", "ax", "az", "theta", "alpha", "mach", "margin", "mass",
             "phase")
    rows = []
    phases = ((rail, dof3.derivatives_rail, "rail", motor.time),
              (powered, dof3.derivatives, "powered", motor.time),
              (coast, dof3.derivatives, "coast", ()),
              (chute, dof3.derivatives_chute, "parachute", ()))
    for sol, fun, phase, extra in phases:
        for t in _times(sol, dt_out, extra):
            rows.append(_row(t, sol.sol(t), fun, rocket, launch, phase))
    rows.append(_row(chute.t[-1], chute.y[:, -1], dof3.derivatives_chute, rocket, launch,
                     "landed"))
    trajectory = {n: np.array([r[i] for r in rows]) for i, n in enumerate(names)}

    tr, ascent = trajectory, np.isin(trajectory["phase"], ("powered", "coast"))
    speed = np.hypot(tr["vx"], tr["vz"])
    summary = {
        "apogee": float(coast.y[1, -1]),
        "apogee_x": float(coast.y[0, -1]),
        "time_to_apogee": events["apogee"] - events["liftoff"],
        "max_velocity": float(speed.max()),
        "max_accel_g": float(np.hypot(tr["ax"], tr["az"]).max() / G0),
        "max_mach": float(np.nanmax(tr["mach"])),
        "max_alpha": float(np.abs(tr["alpha"][ascent]).max()),
        "rail_exit_velocity": exit_speed,
        "margin_rail_exit": margin_exit,
        "min_margin": float(tr["margin"][ascent].min()),
        "landing_velocity": float(-chute.y[3, -1]),
        "drift": float(chute.y[0, -1]),
        "flight_time": events["landing"] - events["liftoff"],
    }
    _warn_mach(summary["max_mach"])
    return Flight(trajectory, events, summary)


def _row(t, y, fun, rocket, launch, phase):
    """One trajectory sample; accelerations come from the model, not finite differences."""
    d = fun(t, y, rocket, launch)
    if phase in ("parachute", "landed"):
        alpha = mach = margin = np.nan
        mass = rocket.mass_properties(t)[0]
    else:
        *_, mass, _, (alpha, mach, margin) = dof3.aero_forces(t, y, rocket, launch)
    if phase == "landed":
        d = [0.0] * 6
    return (t, y[0], y[1], y[2], y[3], d[2], d[3], y[4], alpha, mach, margin, mass, phase)
