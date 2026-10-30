"""Command line: python -m rocketsim run <config.yaml>."""

import argparse
from pathlib import Path

import numpy as np

from rocketsim.flight import simulate
from rocketsim.io import load_config, write_trajectory_csv
from rocketsim.plots import plot_3d, plot_flight


def main(argv=None):
    parser = argparse.ArgumentParser(prog="rocketsim")
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="simulate one flight and save results to outputs/")
    run.add_argument("config", help="rocket config YAML")
    args = parser.parse_args(argv)

    flight = simulate(*load_config(args.config))
    out = Path("outputs") / Path(args.config).stem
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
