"""Flight plots for the 6-DOF trajectory, rendered straight to file (no GUI backend needed)."""

import numpy as np
from matplotlib.figure import Figure

from rocketsim.constants import G0


def plot_flight(flight, path):
    """Altitude, speed, acceleration, angle of attack, margin vs time, and the ground track."""
    tr = flight.trajectory
    fig = Figure(figsize=(11, 10), layout="constrained")
    axes = fig.subplots(3, 2)
    series = [
        (tr["z"], "altitude [m]"),
        (np.sqrt(tr["vx"] ** 2 + tr["vy"] ** 2 + tr["vz"] ** 2), "speed [m/s]"),
        (np.sqrt(tr["ax"] ** 2 + tr["ay"] ** 2 + tr["az"] ** 2) / G0, "acceleration [g]"),
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
    track = axes[2, 1]
    track.plot(tr["x"], tr["y"])
    track.plot(0, 0, "k^", label="rail")
    track.plot(*flight.summary["landing_position"], "rv", label="landing")
    track.set_xlabel("East [m]")
    track.set_ylabel("North [m]")
    track.set_title("ground track")
    track.set_aspect("equal", adjustable="datalim")
    track.legend()
    track.grid(True, alpha=0.3)
    fig.savefig(path, dpi=110)


def plot_3d(flight, path):
    """3-D trajectory in ENU."""
    tr = flight.trajectory
    fig = Figure(figsize=(8, 7), layout="constrained")
    ax = fig.add_subplot(projection="3d")
    ax.plot(tr["x"], tr["y"], tr["z"])
    # same span on both horizontal axes, so a near-planar flight isn't stretched across 1e-14 m
    half = max(np.ptp(tr["x"]), np.ptp(tr["y"]), 1.0) / 2
    for axis, c in ((ax.set_xlim, tr["x"]), (ax.set_ylim, tr["y"])):
        mid = (c.max() + c.min()) / 2
        axis(mid - half, mid + half)
    ax.set_xlabel("East [m]")
    ax.set_ylabel("North [m]")
    ax.set_zlabel("Up [m]")
    fig.savefig(path, dpi=110)
