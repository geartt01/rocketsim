"""Stage 7 validation: RocketSim against OpenRocket (and RocketPy when installed).

Layers, from PLAN.md: (1) dynamics only, using OpenRocket's Cd(Mach); (2) aerodynamics directly:
Cd(Mach), CP, mass properties; (3) full: own aerodynamics and dynamics.

    python validation/compare.py     prints the numbers, writes docs/validation/*.png and .json
"""

import csv
import dataclasses
import json
import warnings
from pathlib import Path

import numpy as np
from matplotlib.figure import Figure

from rocketsim.aero.barrowman import cn_alpha_cp
from rocketsim.aero.drag import buildup_cd
from rocketsim.flight import simulate
from rocketsim.io import load_config
from rocketsim.wind import Wind

ROOT = Path(__file__).parents[1]
CONFIG = ROOT / "configs" / "simple_rocket.yaml"
REF = ROOT / "reference" / "openrocket"
OUT = ROOT / "docs" / "validation"
T_CHARGE = 8.86  # s, OpenRocket's ejection charge for the C6-7: burnout + 7 s


def load_reference(case):
    with open(REF / f"{case}.csv", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    data = np.array([[float(v) for v in r] for r in rows[1:]])
    table = {name: data[:, i] for i, name in enumerate(rows[0])}
    table["events"] = dict(json.loads((REF / f"{case}_events.json").read_text(encoding="utf-8")))
    return table


def reference_summary(ref):
    apogee = int(np.argmax(ref["z"]))
    return {"apogee": float(ref["z"][apogee]), "time_from_ignition": float(ref["t"][apogee]),
            "max_velocity": float(ref["v"][: apogee + 1].max()),
            "max_accel_g": float(ref["az"].max() / 9.80665),
            # OpenRocket's descent rate wobbles ±3 % step to step, so average the last 10 s
            "landing_velocity": float(-ref["vz"][ref["t"] > ref["t"][-1] - 10.0].mean()),
            "drift": float(np.hypot(ref["x"][-1], ref["y"][-1]))}


def cd_table(ref):
    """OpenRocket's coasting Cd as a (mach, cd) table, averaged into 0.01 Mach bins."""
    coast = (ref["t"] > ref["events"]["Motor burnout"]) & (ref["t"] < ref["events"]["Apogee"])
    bins = np.round(ref["mach"][coast], 2)
    mach = np.unique(bins)
    return mach, np.array([ref["cd"][coast][bins == m].mean() for m in mach])


def fly(drag=None, wind_speed=0.0):
    """Fly the example with the chute opening at OpenRocket's charge time, 8.86 s."""
    rocket, launch = load_config(CONFIG)
    if drag is not None:
        rocket.drag = drag
    launch = dataclasses.replace(launch, wind=Wind(wind_speed, np.radians(270.0), 10.0, "constant"))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        first = simulate(rocket, launch, dt_out=0.05, method="RK45", rtol=1e-6, atol=1e-4)
        rocket.recovery.drogue_delay = T_CHARGE - first.events["apogee"]
        flight = simulate(rocket, launch)
    flight.summary["time_from_ignition"] = flight.events["apogee"]
    return rocket, flight


def errors(ours, reference):
    return {k: {"ours": ours[k], "openrocket": reference[k],
                "percent": 100 * (ours[k] / reference[k] - 1)}
            for k in reference if k in ours and abs(reference[k]) > 1.0}  # 0.3 m calm drift: noise


def aero_comparison(rocket, ref):
    """Cd and CP against OpenRocket at the same Mach and Reynolds number, in coast."""
    coast = (ref["t"] > ref["events"]["Motor burnout"]) & (ref["t"] < ref["events"]["Apogee"])
    rows = []
    for mach in (0.05, 0.10, 0.15, 0.20, 0.25):
        i = np.flatnonzero(coast)[np.argmin(np.abs(ref["mach"][coast] - mach))]
        reynolds = ref["v"][i] * rocket.length / 1.5e-5
        cd = buildup_cd(rocket, ref["mach"][i], reynolds, thrusting=False)
        x_cp = cn_alpha_cp(rocket, ref["mach"][i])[1]
        rows.append({"mach": float(ref["mach"][i]), "cd_ours": float(cd),
                     "cd_openrocket": float(ref["cd"][i]), "cp_ours": float(x_cp),
                     "cp_openrocket": float(ref["cp"][i])})
    return rows


def mass_comparison(rocket, ref):
    m, x_cg, i_t, _ = rocket.mass_properties(0.0)
    m_b, x_b, i_b, _ = rocket.mass_properties(rocket.motor.burn_time)
    burnout = np.argmin(np.abs(ref["t"] - rocket.motor.burn_time))
    return {"launch_mass": [m, float(ref["mass"][0])], "launch_cg": [x_cg, float(ref["cg"][0])],
            "launch_inertia": [i_t, float(ref["i_long"][0])],
            "burnout_mass": [m_b, float(ref["mass"][burnout])],
            "burnout_cg": [x_b, float(ref["cg"][burnout])],
            "burnout_inertia": [i_b, float(ref["i_long"][burnout])]}


def rocketpy_flight(rocket, cd, launch_rail=1.0):
    """The same rocket and Cd table in RocketPy; returns None when it is not installed."""
    try:
        import rocketpy
    except ImportError:
        return None
    m_dry, x_dry, i_t, i_x = rocket._dry
    env = rocketpy.Environment(latitude=0, longitude=0, elevation=0)
    env.set_atmospheric_model(type="standard_atmosphere")
    motor = rocketpy.GenericMotor.load_from_eng_file(
        str(REF / "simple_rocket_C6.eng"),
        coordinate_system_orientation="nozzle_to_combustion_chamber")
    table = [list(p) for p in zip(*(c.tolist() for c in cd))]
    rp = rocketpy.Rocket(radius=rocket.radius, mass=m_dry, inertia=(i_t, i_t, i_x),
                         power_off_drag=table, power_on_drag=table,
                         center_of_mass_without_motor=x_dry,
                         coordinate_system_orientation="nose_to_tail")
    rp.add_motor(motor, position=rocket.motor_position)
    rp.add_nose(length=rocket.nose.length, kind="ogive", position=0.0)
    f = rocket.fins
    rp.add_trapezoidal_fins(n=f.count, root_chord=f.root_chord, tip_chord=f.tip_chord, span=f.span,
                            position=f.position, sweep_length=f.sweep)
    flight = rocketpy.Flight(rocket=rp, environment=env, rail_length=launch_rail, inclination=90,
                             heading=0, terminate_on_apogee=True)
    return {"apogee": float(flight.apogee), "time_from_ignition": float(flight.apogee_time),
            "max_velocity": float(flight.max_speed)}


def plot_overlay(ref, table_flight, own_flight):
    fig = Figure(figsize=(11, 9), layout="constrained")
    axes = fig.subplots(3, 1, sharex=True)
    ascent = ref["t"] < ref["events"]["Apogee"] + 1
    for ax, key, label, factor in ((axes[0], "z", "altitude [m]", 1.0),
                                   (axes[1], "vz", "vertical velocity [m/s]", 1.0),
                                   (axes[2], "az", "vertical acceleration [g]", 1 / 9.80665)):
        ax.plot(ref["t"][ascent], ref[key][ascent] * factor, "k", lw=2.5, alpha=0.45,
                label="OpenRocket")
        for flight, style, name in ((table_flight, "C0-", "RocketSim, OpenRocket's Cd"),
                                    (own_flight, "C3--", "RocketSim, own Cd")):
            tr = flight.trajectory
            keep = tr["t"] < ref["events"]["Apogee"] + 1
            ax.plot(tr["t"][keep], tr[key][keep] * factor, style, lw=1.2, label=name)
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.3)
    axes[0].legend()
    axes[-1].set_xlabel("time since ignition [s]")
    fig.savefig(OUT / "overlay_ascent.png", dpi=110)


def plot_aero(rocket, ref, cd):
    fig = Figure(figsize=(10, 4), layout="constrained")
    ax_cd, ax_cp = fig.subplots(1, 2)
    ax_cd.plot(*cd, "k", lw=2.5, alpha=0.45, label="OpenRocket")
    mach = cd[0][cd[0] > 0.02]
    ours = [buildup_cd(rocket, m, m * 340 * rocket.length / 1.5e-5, False) for m in mach]
    ax_cd.plot(mach, ours, "C3--", label="RocketSim buildup")
    ax_cd.set_xlabel("Mach")
    ax_cd.set_ylabel("Cd (coasting)")
    ax_cd.legend()
    ax_cd.grid(True, alpha=0.3)
    coast = (ref["t"] > ref["events"]["Motor burnout"]) & (ref["t"] < ref["events"]["Apogee"] - 1)
    ax_cp.plot(ref["mach"][coast], ref["cp"][coast], "k", lw=2.5, alpha=0.45, label="OpenRocket")
    grid = np.linspace(ref["mach"][coast].min(), ref["mach"][coast].max(), 40)
    ax_cp.plot(grid, [cn_alpha_cp(rocket, m)[1] for m in grid], "C3--", label="RocketSim")
    ax_cp.set_xlabel("Mach")
    ax_cp.set_ylabel("CP from nose tip [m]")
    ax_cp.legend()
    ax_cp.grid(True, alpha=0.3)
    fig.savefig(OUT / "aero.png", dpi=110)


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    calm, wind = load_reference("calm"), load_reference("wind2")
    cd = cd_table(calm)
    rocket, table_flight = fly(cd)
    _, own_flight = fly()
    _, table_wind = fly(cd, wind_speed=2.0)
    _, own_wind = fly(wind_speed=2.0)
    result = {
        "layer1_table_cd": errors(table_flight.summary, reference_summary(calm)),
        "layer1_table_cd_wind": errors(table_wind.summary, reference_summary(wind)),
        "layer2_aero": aero_comparison(rocket, calm),
        "layer2_mass": mass_comparison(rocket, calm),
        "layer3_own_aero": errors(own_flight.summary, reference_summary(calm)),
        "layer3_own_aero_wind": errors(own_wind.summary, reference_summary(wind)),
        "rocketpy_table_cd": rocketpy_flight(rocket, cd),
    }
    plot_overlay(calm, table_flight, own_flight)
    plot_aero(rocket, calm, cd)
    (OUT / "results.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    return result


if __name__ == "__main__":
    summary = run()
    for key in ("layer1_table_cd", "layer1_table_cd_wind", "layer3_own_aero",
                "layer3_own_aero_wind"):
        print(key)
        for metric, v in summary[key].items():
            print(f"  {metric:20} ours {v['ours']:9.3f}  OR {v['openrocket']:9.3f}  "
                  f"{v['percent']:+6.2f} %")
    for row in summary["layer2_aero"]:
        print(f"M {row['mach']:.3f}  Cd {row['cd_ours']:.3f} vs {row['cd_openrocket']:.3f}  "
              f"CP {row['cp_ours']:.4f} vs {row['cp_openrocket']:.4f}")
    for key, (a, b) in summary["layer2_mass"].items():
        print(f"{key:18} {a:.5f} vs {b:.5f} ({100 * (a / b - 1):+.2f} %)")
    print("rocketpy", summary["rocketpy_table_cd"])
