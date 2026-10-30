"""Flight plots, rendered straight to file (no GUI backend needed)."""

import numpy as np
from matplotlib.figure import Figure

from rocketsim.constants import G0


def plot_flight(flight, path):
    """3-DOF flight: altitude, speed, acceleration, angle of attack, margin vs time; x-z path."""
    tr = flight.trajectory
    fig = Figure(figsize=(11, 10), layout="constrained")
    axes = fig.subplots(3, 2)
    series = [
        (tr["z"], "altitude [m]"),
        (np.hypot(tr["vx"], tr["vz"]), "speed [m/s]"),
        (np.hypot(tr["ax"], tr["az"]) / G0, "acceleration [g]"),
        (np.degrees(tr["alpha"]), "angle of attack [deg]"),
        (tr["margin"], "stability margin [cal]"),
    ]
    for ax, (y, label) in zip(axes.flat, series):
        ax.plot(tr["t"], y)
        ax.set_ylabel(label)
        ax.set_xlabel("time since ignition [s]")
        ax.grid(True, alpha=0.3)
        for t in flight.events.values():
            ax.axvline(t, color="grey", linestyle=":", linewidth=1)
    for name, t in flight.events.items():
        axes[0, 0].annotate(name, (t, 1), xycoords=("data", "axes fraction"), rotation=90,
                            va="top", ha="right", fontsize=8, color="grey")
    path_ax = axes[2, 1]
    path_ax.plot(tr["x"], tr["z"])
    path_ax.set_xlabel("downrange x [m]")
    path_ax.set_ylabel("altitude z [m]")
    path_ax.set_aspect("equal", adjustable="datalim")
    path_ax.grid(True, alpha=0.3)
    fig.savefig(path, dpi=110)
