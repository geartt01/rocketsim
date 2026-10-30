"""Command line: python -m rocketsim run <config.yaml>."""

import argparse
from pathlib import Path

from rocketsim.flight import simulate
from rocketsim.io import load_config, write_trajectory_csv
from rocketsim.plots import plot_flight


def main(argv=None):
    parser = argparse.ArgumentParser(prog="rocketsim")
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="simulate one flight and save results to outputs/")
    run.add_argument("config", help="rocket config YAML")
    args = parser.parse_args(argv)

    flight = simulate(load_config(args.config))
    out = Path("outputs") / Path(args.config).stem
    out.mkdir(parents=True, exist_ok=True)
    write_trajectory_csv(flight, out / "trajectory.csv")
    plot_flight(flight, out / "flight.png")

    s = flight.summary
    print(f"Apogee            {s['apogee']:8.1f} m")
    print(f"Time to apogee    {s['time_to_apogee']:8.2f} s")
    print(f"Max velocity      {s['max_velocity']:8.1f} m/s  (Mach {s['max_mach']:.2f})")
    print(f"Max acceleration  {s['max_accel_g']:8.1f} g")
    print(f"Landing velocity  {s['landing_velocity']:8.2f} m/s")
    print(f"Flight time       {s['flight_time']:8.1f} s")
    print(f"Saved {out / 'trajectory.csv'} and {out / 'flight.png'}")
