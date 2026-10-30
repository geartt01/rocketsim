"""Phase managers.

1-DOF (Stage 1):        pad → powered → coast → parachute → landed
3-DOF / 6-DOF (2, 3):   pad → rail → powered → coast → parachute → landed

Each phase is one solve_ivp (DOP853) call that ends exactly on its event, so the integrator
never steps across a jump in thrust, drag or constraints (PLAN.md §3).
"""

import warnings
from dataclasses import dataclass
from functools import partial

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

from rocketsim.atmosphere import isa
from rocketsim.constants import G0
from rocketsim.dynamics import dof1, dof3, dof6

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


def _event_fn(fun, direction=1):
    """Terminal event from a function of (t, y, *args) crossing zero in the given direction."""
    def event(t, y, *args):
        return fun(t, y, *args)
    event.terminal, event.direction = True, direction
    return event


apogee_event, landing_event = _event(1), _event(0)  # 1-DOF state [z, v]


def run_phase(fun, t0, y0, t_end, args, events=(), rtol=1e-8, atol=1e-8, method="DOP853"):
    """Integrate one flight phase until t_end or the first terminal event (DOP853 by default)."""
    return solve_ivp(fun, (t0, t_end), y0, method=method, rtol=rtol, atol=atol,
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
    trajectory: dict  # column name → array
    events: dict  # event name → time since ignition, s
    summary: dict


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
        # shortcut: apogee during the burn is not modelled, add it if underpowered designs matter
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


def _fly(model, rocket, launch, rtol=1e-8, atol=1e-8, dt_out=0.01, method="DOP853"):
    """Rail-launched flight with a 3-DOF (dof3) or 6-DOF (dof6) model, pad to landing.

    After apogee: free fall for the drogue delay ("freefall"), then the drogue ("drogue"),
    then the main once descending through its altitude ("main"); both stay attached.

    Returns Flight with
      trajectory: t, the model's state columns (model.STATE), accelerations a<axis> (m/s²),
                  alpha (rad), mach (-), margin (calibers; NaN under parachute), mass (kg),
                  phase;
      events: liftoff, rail_exit, burnout, apogee, drogue, main (if any), landing
              (s since ignition);
      summary: apogee (m), apogee_position and landing_position (horizontal coordinates, m),
               drift (m, horizontal distance pad → landing), time_to_apogee (s),
               max_velocity (m/s), max_accel_g, max_mach, max_alpha (rad, powered + coast),
               rail_exit_velocity (m/s), margin_rail_exit and min_margin (calibers),
               drogue_descent_rate and main_descent_rate (m/s at the end of each phase, None
               if that phase does not happen), opening_shock_g (peak parachute drag / weight),
               landing_velocity (m/s, descent rate), flight_time (s).
    Warns when Mach > 0.8, rail-exit speed < 15 m/s or margin at rail exit < 1 caliber.
    method/rtol/atol go to solve_ivp for every phase (Monte Carlo trades accuracy for speed).
    """
    motor, args = rocket.motor, (rocket, launch)
    tol = {"rtol": rtol, "atol": atol, "method": method}
    rail_exit = _event_fn(model.rail_exit)
    apogee, landing = _event(model.VZ), _event(model.Z)

    events = {"liftoff": liftoff_time(motor, lambda t: rocket.mass_properties(t)[0],
                                      launch.g * np.sin(launch.rail_elevation))}
    rail = run_phase(model.derivatives_rail, events["liftoff"], model.initial_state(launch),
                     motor.burn_time, args, [rail_exit], **tol)
    _check(rail, "rail", "still on the rail at burnout: rail too long or motor too weak")
    events["rail_exit"] = rail.t[-1]
    exit_speed = float(np.linalg.norm(rail.y[list(model.VELOCITY), -1]))
    margin_exit = float(model.info(rail.t[-1], rail.y[:, -1], *args)[2])
    # warn now, before free flight, so the warning survives if an unstable flight fails later
    if exit_speed < RAIL_EXIT_MIN:
        warnings.warn(f"rail-exit velocity {exit_speed:.1f} m/s is below {RAIL_EXIT_MIN} m/s",
                      stacklevel=2)
    if margin_exit < MARGIN_MIN:
        warnings.warn(f"stability margin {margin_exit:.2f} cal at rail exit is below "
                      f"{MARGIN_MIN}", stacklevel=2)

    powered = run_phase(model.derivatives, rail.t[-1], model.normalize(rail.y[:, -1]),
                        motor.burn_time, args, [apogee, landing], **tol)
    if powered.status == 1:
        # shortcut: apogee during the burn is not modelled, add it if underpowered designs matter
        raise ValueError("rocket stopped climbing before burnout: motor too weak for this rocket")
    events["burnout"] = motor.burn_time
    coast = run_phase(model.derivatives, motor.burn_time, model.normalize(powered.y[:, -1]),
                      motor.burn_time + T_PHASE_MAX, args, [apogee], **tol)
    _check(coast, "coast", f"no apogee within {T_PHASE_MAX} s of burnout")
    events["apogee"] = coast.t[-1]
    segments = [(rail, model.derivatives_rail, args, "rail", motor.time),
                (powered, model.derivatives, args, "powered", motor.time),
                (coast, model.derivatives, args, "coast", ())]

    rec, t_now, y = rocket.recovery, coast.t[-1], model.normalize(coast.y[:, -1])
    if rec.drogue_delay > 0:
        fall = run_phase(model.derivatives, t_now, y, t_now + rec.drogue_delay, args, [landing],
                         **tol)
        if fall.status == 1:
            raise RuntimeError("landed before the drogue deployed: drogue delay too long")
        segments.append((fall, model.derivatives, args, "freefall", ()))
        t_now, y = fall.t[-1], model.normalize(fall.y[:, -1])
    events["drogue"] = t_now
    deployments = [(rec.drogue, t_now)]
    descent = {"drogue_descent_rate": None, "main_descent_rate": None}

    if rec.main is not None and y[model.Z] <= rec.main_altitude:
        deployments.append((rec.main, t_now))  # already below the main altitude: open both
        events["main"] = t_now
    events_drogue = [landing]
    if rec.main is not None and "main" not in events:
        events_drogue.append(_event_fn(lambda t, s, *a: s[model.Z] - rec.main_altitude, -1))
    phase = "main" if "main" in events else "drogue"
    while True:
        chute_args = (*args, tuple(deployments))
        sol = run_phase(model.derivatives_chute, t_now, y, t_now + T_PHASE_MAX, chute_args,
                        events_drogue if phase == "drogue" else [landing], **tol)
        _check(sol, phase, f"no landing within {T_PHASE_MAX} s")
        fills = [t_d + c.fill_time for c, t_d in deployments]
        segments.append((sol, model.derivatives_chute, chute_args, phase, fills))
        descent[f"{phase}_descent_rate"] = float(-sol.y[model.VZ, -1])
        t_now, y = sol.t[-1], model.normalize(sol.y[:, -1])
        if phase == "main" or sol.t_events[0].size:  # landing event fired
            break
        phase = "main"
        events["main"] = t_now
        deployments.append((rec.main, t_now))
    events["landing"] = t_now

    accel_names = tuple("a" + model.STATE[i][1:] for i in model.VELOCITY)
    names = ("t", *model.STATE, *accel_names, "alpha", "mach", "margin", "mass", "phase")
    rows = []
    for sol, fun, fun_args, label, extra in segments:
        for t in _times(sol, dt_out, extra):
            rows.append(_row(model, t, sol.sol(t), fun, fun_args, label))
    rows.append(_row(model, t_now, y, None, args, "landed"))
    tr = {n: np.array([r[i] for r in rows]) for i, n in enumerate(names)}

    ascent = np.isin(tr["phase"], ("powered", "coast"))
    horizontal = list(model.HORIZONTAL)
    summary = {
        "apogee": float(coast.y[model.Z, -1]),
        "apogee_position": tuple(float(c) for c in coast.y[horizontal, -1]),
        "landing_position": tuple(float(c) for c in y[horizontal]),
        "drift": float(np.linalg.norm(y[horizontal])),
        "time_to_apogee": events["apogee"] - events["liftoff"],
        "max_velocity": float(np.linalg.norm([tr[model.STATE[i]] for i in model.VELOCITY],
                                             axis=0).max()),
        "max_accel_g": float(np.linalg.norm([tr[a] for a in accel_names], axis=0).max() / G0),
        "max_mach": float(np.nanmax(tr["mach"])),
        "max_alpha": float(np.abs(tr["alpha"][ascent]).max()),
        "rail_exit_velocity": exit_speed,
        "margin_rail_exit": margin_exit,
        "min_margin": float(tr["margin"][ascent].min()),
        **descent,
        "opening_shock_g": max(_opening_shock(model, s, launch.g)
                               for s in segments if s[3] in ("drogue", "main")),
        "landing_velocity": float(-y[model.VZ]),
        "flight_time": events["landing"] - events["liftoff"],
    }
    _warn_mach(summary["max_mach"])
    return Flight(tr, events, summary)


def _opening_shock(model, segment, g, window=3.0, step=1e-3):
    """Peak parachute load in g, |a + g·ẑ|/g, on a 1 ms grid over the canopy fill + 3 s.

    Scanned separately from the trajectory output so the peak does not depend on dt_out.
    """
    sol, fun, args, _, fills = segment
    t0 = sol.t[0]
    t1 = min(sol.t[-1], max(fills) + window)
    peak = 0.0
    for t in np.linspace(t0, t1, max(2, int((t1 - t0) / step) + 1)):
        d = fun(t, sol.sol(t), *args)
        a = [d[i] for i in model.VELOCITY]
        a[-1] += g
        peak = max(peak, float(np.sqrt(np.dot(a, a))))
    return peak / G0


def _row(model, t, y, fun, args, phase):
    """One trajectory sample; accelerations come from the model, not finite differences."""
    if fun is None:  # landed
        accel = [0.0] * len(model.VELOCITY)
    else:
        d = fun(t, y, *args)
        accel = [d[i] for i in model.VELOCITY]
    if phase in ("drogue", "main", "landed"):
        extra = (np.nan, np.nan, np.nan, args[0].mass_properties(t)[0])
    else:
        extra = model.info(t, y, *args)
    return (t, *y, *accel, *extra, phase)


simulate_3dof = partial(_fly, dof3)
simulate = partial(_fly, dof6)
