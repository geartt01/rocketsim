"""Fixed-step RK4, the cross-check on the adaptive solve_ivp used for flight phases."""

import numpy as np


def rk4(fun, t0, y0, t1, dt, args=()):
    """Integrate dy/dt = fun(t, y, *args) from t0 to t1 with the classic 4th-order Runge-Kutta.

        k1 = f(t, y),  k2 = f(t + h/2, y + h·k1/2),  k3 = f(t + h/2, y + h·k2/2),
        k4 = f(t + h, y + h·k3),  y⁺ = y + h·(k1 + 2k2 + 2k3 + k4)/6

    The last step is shortened to land exactly on t1. Global error ∝ h⁴ for a smooth f.
    Returns (t [n], y [n, len(y0)]). Source: any numerical methods text, e.g. Press et al.,
    Numerical Recipes, §17.1.
    """
    steps = int(np.ceil((t1 - t0) / dt - 1e-12))
    times = np.linspace(t0, t1, steps + 1)
    out = np.empty((steps + 1, len(y0)))
    out[0] = y = np.asarray(y0, dtype=float)
    for i in range(steps):
        t, h = times[i], times[i + 1] - times[i]
        k1 = np.asarray(fun(t, y, *args))
        k2 = np.asarray(fun(t + h / 2, y + h / 2 * k1, *args))
        k3 = np.asarray(fun(t + h / 2, y + h / 2 * k2, *args))
        k4 = np.asarray(fun(t + h, y + h * k3, *args))
        out[i + 1] = y = y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return times, out
