"""Flight plots, rendered straight to file (no GUI backend needed)."""

from matplotlib.figure import Figure

from rocketsim.constants import G0


def plot_flight(flight, path):
    """Altitude, velocity and acceleration vs time, with phase events marked."""
    tr = flight.trajectory
    fig = Figure(figsize=(8, 9), layout="constrained")
    axes = fig.subplots(3, 1, sharex=True)
    series = [(tr["z"], "altitude [m]"), (tr["vz"], "velocity [m/s]"),
              (tr["az"] / G0, "acceleration [g]")]
    for ax, (y, label) in zip(axes, series):
        ax.plot(tr["t"], y)
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.3)
        for t in flight.events.values():
            ax.axvline(t, color="grey", linestyle=":", linewidth=1)
    for name, t in flight.events.items():
        axes[0].annotate(name, (t, 1), xycoords=("data", "axes fraction"), rotation=90,
                         va="top", ha="right", fontsize=8, color="grey")
    axes[-1].set_xlabel("time since ignition [s]")
    fig.savefig(path, dpi=120)
