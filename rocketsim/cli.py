"""Command line: python -m rocketsim run|montecarlo|sensors|flightdata|blender."""

import argparse
import csv
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from rocketsim import montecarlo, sensors
from rocketsim.export import write_flight_package
from rocketsim.flight import simulate
from rocketsim.io import load_config, write_trajectory_csv
from rocketsim.plots import plot_3d, plot_flight, plot_montecarlo


def main(argv=None):
    parser = argparse.ArgumentParser(prog="rocketsim")
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="simulate one flight and save results to outputs/")
    run.add_argument("config", help="rocket config YAML")
    mc = commands.add_parser("montecarlo", help="dispersed flights using the config's "
                             "dispersions: section; saves statistics to outputs/")
    mc.add_argument("config", help="rocket config YAML")
    mc.add_argument("-n", type=int, default=500, help="number of flights (default 500)")
    mc.add_argument("--seed", type=int, default=1, help="master random seed (default 1)")
    mc.add_argument("--workers", type=int, default=None, help="processes (default: all cores)")
    sens = commands.add_parser("sensors", help="fly once and write simulated IMU and barometer "
                               "logs using the config's sensors: section")
    sens.add_argument("config", help="rocket config YAML")
    sens.add_argument("--seed", type=int, default=1, help="random seed (default 1)")
    data = commands.add_parser("flightdata", help="fly once and package it for a 3-D viewer")
    data.add_argument("config", help="rocket config YAML")
    data.add_argument("-o", "--output", required=True, help="output .npz file")
    view = commands.add_parser("blender", help="open the rocket in Blender (blender/ add-on)")
    view.add_argument("config", nargs="?", default=None, help="rocket config YAML (optional)")
    args = parser.parse_args(argv)
    if args.command == "flightdata":
        _flightdata(args)
    elif args.command == "blender":
        _blender(args)
    else:
        out = Path("outputs") / Path(args.config).stem
        {"run": _run, "montecarlo": _montecarlo, "sensors": _sensors}[args.command](args, out)


def _run(args, out):
    flight = simulate(*load_config(args.config))
    out.mkdir(parents=True, exist_ok=True)
    write_trajectory_csv(flight, out / "trajectory.csv")
    plot_flight(flight, out / "flight.png")
    plot_3d(flight, out / "trajectory_3d.png")

    s = flight.summary
    (ax, ay), (lx, ly) = s["apogee_position"], s["landing_position"]
    print(f"Apogee              {s['apogee']:8.1f} m   (E {ax:+.1f} m, N {ay:+.1f} m)")
    print(f"Time to apogee      {s['time_to_apogee']:8.2f} s")
    print(f"Max velocity        {s['max_velocity']:8.1f} m/s  (Mach {s['max_mach']:.2f})")
    print(f"Max acceleration    {s['max_accel_g']:8.1f} g")
    print(f"Rail-exit velocity  {s['rail_exit_velocity']:8.1f} m/s")
    print(f"Stability margin    {s['margin_rail_exit']:8.2f} cal at rail exit "
          f"(min {s['min_margin']:.2f})")
    print(f"Max angle of attack {np.degrees(s['max_alpha']):8.1f} deg")
    for chute in ("drogue", "main"):
        if s[f"{chute}_descent_rate"] is not None:
            print(f"Descent under {chute:6} {s[f'{chute}_descent_rate']:6.2f} m/s")
    print(f"Opening shock       {s['opening_shock_g']:8.1f} g (peak chute drag / weight)")
    print(f"Landing velocity    {s['landing_velocity']:8.2f} m/s")
    print(f"Landing point       {s['drift']:8.1f} m from the rail (E {lx:+.1f} m, N {ly:+.1f} m)")
    print(f"Flight time         {s['flight_time']:8.1f} s")
    print(f"Saved {out / 'trajectory.csv'}, {out / 'flight.png'} and {out / 'trajectory_3d.png'}")


def _flightdata(args):
    try:
        write_flight_package(args.config, args.output)
    except (ValueError, RuntimeError, OSError) as exc:  # shown as is in the viewer
        sys.exit(f"ERROR: {exc}")


def find_blender():
    """Blender's executable: ROCKETSIM_BLENDER, then PATH, then the usual install folders."""
    candidates = [os.environ.get("ROCKETSIM_BLENDER"), shutil.which("blender")]
    local = Path(os.environ.get("LOCALAPPDATA", "")) / "Programs"
    program_files = Path(os.environ.get("ProgramFiles", r"C:\Program Files"))
    candidates += sorted(local.glob("blender-*/blender.exe"), reverse=True)
    candidates += sorted(program_files.glob("Blender Foundation/*/blender.exe"), reverse=True)
    return next((str(c) for c in candidates if c and Path(c).is_file()), None)


def prepare_blender(blender):
    """Blender settings private to this project, so its welcome screen stays away.

    BLENDER_USER_CONFIG keeps them apart from any other use of Blender on this machine, whose
    settings are left untouched. Returns (start file to open, environment to run with).
    """
    folder = Path(__file__).resolve().parents[1] / "outputs" / "blender"
    start = folder / "start.blend"
    env = os.environ | {"ROCKETSIM_PYTHON": sys.executable,
                        "BLENDER_USER_CONFIG": str(folder / "config")}
    if not start.exists():
        folder.mkdir(parents=True, exist_ok=True)
        setup = ("import bpy; bpy.context.preferences.view.show_splash = False; "
                 "bpy.ops.wm.save_userpref(); "
                 f"bpy.ops.wm.save_as_mainfile(filepath={str(start)!r})")
        subprocess.run([blender, "--background", "--factory-startup", "--python-expr", setup],
                       check=True, capture_output=True, env=env)
    return start, env


def _blender(args):
    blender = find_blender()
    if blender is None:
        sys.exit("Blender not found. Install it from blender.org, or set ROCKETSIM_BLENDER to "
                 "blender.exe.")
    addon = Path(__file__).resolve().parents[1] / "blender" / "rocketsim_addon.py"
    start, env = prepare_blender(blender)
    command = [blender, "--window-maximized", str(start), "--python", str(addon)]
    if args.config:
        command += ["--", str(Path(args.config).resolve())]
    subprocess.Popen(command, env=env)
    print(f"Started Blender: {blender}")


def _sensors(args, out):
    config = sensors.load_sensor_config(args.config)
    rocket, launch = load_config(args.config)
    flight = simulate(rocket, launch)
    truth = sensors.truth_at_rate(flight, launch, config.imu_rate, config.pad_time,
                                  config.landed_time)
    data = sensors.measure(truth, config, launch, np.random.default_rng(args.seed))
    sensors.write_outputs(flight, truth, data, config, out)
    print(f"{len(truth['t'])} IMU samples at {config.imu_rate:.0f} Hz, barometer at "
          f"{config.baro_rate:.0f} Hz, t = {truth['t'][0]:.1f} to {truth['t'][-1]:.1f} s (seed "
          f"{args.seed})")
    print(f"Saved {out / 'sensors.csv'}, {out / 'truth.csv'} and {out / 'events.json'}")


def _montecarlo(args, out):
    specs = montecarlo.load_dispersions(args.config)
    start = time.perf_counter()
    rows = montecarlo.run(Path(args.config), specs, args.n, args.seed, args.workers)
    elapsed = time.perf_counter() - start
    stats = montecarlo.statistics(rows)
    out.mkdir(parents=True, exist_ok=True)
    columns = ["run", "status", *montecarlo.PARAMETERS, *montecarlo.SUMMARY_COLUMNS]
    with open(out / "montecarlo.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, columns)
        writer.writeheader()
        writer.writerows({"run": i, **r} for i, r in enumerate(rows))

    print(f"{stats['runs']} runs in {elapsed:.0f} s (seed {args.seed}), {stats['failed']} failed")
    for msg in sorted({r["status"] for r in rows if r["status"] != "ok"}):
        print(f"  {msg}")
    if "apogee" in stats:
        plot_montecarlo(rows, stats, out / "montecarlo.png")
        for key, label, unit in (("apogee", "Apogee", "m"),
                                 ("rail_exit_velocity", "Rail-exit velocity", "m/s"),
                                 ("margin_rail_exit", "Margin at rail exit", "cal"),
                                 ("drift", "Landing distance", "m")):
            s = stats[key]
            print(f"{label:20} mean {s['mean']:7.2f} {unit:3}  σ {s['std']:6.2f}  "
                  f"5–95 % {s['p5']:7.2f}–{s['p95']:.2f}  (min {s['min']:.2f})")
        for k, (center, axes, angle) in stats.get("landing_ellipses", {}).items():
            print(f"Landing {k}σ ellipse  centre E {center[0]:+.0f} N {center[1]:+.0f} m, "
                  f"semi-axes {axes[0]:.0f} × {axes[1]:.0f} m, "
                  f"major axis {np.degrees(angle):.0f}° from East")
        print(f"Saved {out / 'montecarlo.csv'} and {out / 'montecarlo.png'}")
