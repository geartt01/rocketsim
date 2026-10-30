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


def plot_montecarlo(rows, stats, path):
    """Apogee histogram, landing scatter with 1/2/3-σ ellipses, rail-exit speed and margin."""
    from matplotlib.patches import Ellipse

    ok = [r for r in rows if r["status"] == "ok"]
    fig = Figure(figsize=(11, 9), layout="constrained")
    (ax_apo, ax_land), (ax_rail, ax_margin) = fig.subplots(2, 2)
    for ax, key, label, limit in ((ax_apo, "apogee", "apogee [m]", None),
                                  (ax_rail, "rail_exit_velocity", "rail-exit speed [m/s]", 15.0),
                                  (ax_margin, "margin_rail_exit", "margin at rail exit [cal]",
                                   1.0)):
        values = [r[key] for r in ok]
        ax.hist(values, bins=30)
        ax.axvline(np.mean(values), color="k", label="mean")
        if limit is not None:
            ax.axvline(limit, color="r", linestyle="--", label="guideline")
        ax.set_xlabel(label)
        ax.set_ylabel("runs")
        ax.legend()
    s = stats["apogee"]
    ax_apo.set_title(f"mean {s['mean']:.1f} m, σ {s['std']:.1f} m, "
                     f"5–95 % {s['p5']:.0f}–{s['p95']:.0f} m")

    ax_land.scatter([r["landing_x"] for r in ok], [r["landing_y"] for r in ok], s=6)
    ax_land.plot(0, 0, "k^", label="rail")
    for k, (center, axes, angle) in stats.get("landing_ellipses", {}).items():
        ax_land.add_patch(Ellipse(center, 2 * axes[0], 2 * axes[1], angle=np.degrees(angle),
                                  fill=False, color="r", alpha=1.2 - 0.3 * k, label=f"{k}σ"))
    ax_land.set_xlabel("East [m]")
    ax_land.set_ylabel("North [m]")
    ax_land.set_title(f"landing points ({len(ok)} of {len(rows)} runs)")
    ax_land.set_aspect("equal", adjustable="datalim")
    ax_land.legend()
    ax_land.grid(True, alpha=0.3)
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
