"""Fixed-step RK4 as a cross-check on the adaptive integrator (PLAN.md §3)."""

from pathlib import Path

import numpy as np
import pytest

from rocketsim.dynamics import dof6
from rocketsim.flight import run_phase
from rocketsim.integrators import rk4
from rocketsim.io import load_config

EXAMPLE = Path(__file__).parents[1] / "configs" / "example_rocket.yaml"


def test_rk4_is_fourth_order():
    # dy/dt = −y: halving the step must cut the error by about 2⁴ = 16.
    def fun(t, y):
        return -y

    err = [abs(rk4(fun, 0.0, np.array([1.0]), 1.0, dt)[1][-1, 0] - np.exp(-1.0))
           for dt in (0.1, 0.05)]
    assert err[0] / err[1] == pytest.approx(16.0, rel=0.05)


def test_rk4_agrees_with_dop853_on_6dof_flight():
    # Powered free flight in wind after the rail: both integrators must land on the same state.
    rocket, launch = load_config(EXAMPLE)
    t0, t1 = 0.2, rocket.motor.burn_time
    y0 = dof6.initial_state(launch)
    y0[3:6] = 20.0 * np.array([np.sin(launch.rail_heading), np.cos(launch.rail_heading), 8.0])
    adaptive = run_phase(dof6.derivatives, t0, y0, t1, (rocket, launch))
    _, fixed = rk4(dof6.derivatives, t0, y0, t1, 1e-3, (rocket, launch))
    assert fixed[-1][:6] == pytest.approx(adaptive.y[:6, -1], rel=1e-4, abs=1e-3)
