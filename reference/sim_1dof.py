"""Standalone 1-DOF vertical rocket simulation: the test oracle for RocketSim.

Deliberately independent of the rocketsim package (imports only numpy) and deliberately
simple: fixed-step RK4 from ignition to apogee, mass carried as a state variable, density
from the closed-form ISA troposphere formula. Do not edit (CLAUDE.md rule 8).

Model:
    dz/dt = v
    dv/dt = (F(t) − ½·ρ(z)·Cd·A·v·|v|) / m − g
    dm/dt = −F(t) / v_e,   v_e = I_total / m_prop   (mass flow proportional to thrust)
    ρ(z)  = ρ0·(1 + L·z/T0)^(−g/(R·L) − 1)          (ISA troposphere, z treated as geopotential)
On the pad (z ≤ 0, v ≤ 0) the rocket is held still while thrust ≤ weight.
"""

import numpy as np

G = 9.80665  # m/s^2
R = 287.05287  # J/(kg·K)
T0 = 288.15  # K
P0 = 101325.0  # Pa
L = -0.0065  # K/m


def density(z):
    """ISA troposphere density, kg/m^3, at altitude z (m)."""
    rho0 = P0 / (R * T0)
    return rho0 * (1 + L * z / T0) ** (-G / (R * L) - 1)


def simulate(dry_mass, diameter, cd, motor_time, motor_thrust, prop_mass, motor_mass, dt=1e-3):
    """Fly straight up from ignition to apogee.

    Args:
        dry_mass: rocket mass without motor, kg
        diameter: reference diameter, m
        cd: constant drag coefficient, -
        motor_time, motor_thrust: thrust curve points, s and N (linear between points, 0 outside)
        prop_mass: initial propellant mass, kg
        motor_mass: initial total motor mass (casing + propellant), kg
        dt: time step, s

    Returns:
        dict with apogee (m), t_apogee (s since ignition), v_max (m/s).
    """
    motor_time = np.asarray(motor_time, dtype=float)
    motor_thrust = np.asarray(motor_thrust, dtype=float)
    area = np.pi * diameter**2 / 4
    v_e = np.trapezoid(motor_thrust, motor_time) / prop_mass

    def thrust(t):
        return np.interp(t, motor_time, motor_thrust, left=0.0, right=0.0)

    def deriv(t, s):
        z, v, m = s
        f = thrust(t)
        if z <= 0 and v <= 0 and f <= m * G:
            return np.array([0.0, 0.0, -f / v_e])
        drag = 0.5 * density(z) * cd * area * v * abs(v)
        return np.array([v, (f - drag) / m - G, -f / v_e])

    t = 0.0
    s = np.array([0.0, 0.0, dry_mass + motor_mass])
    apogee, t_apogee, v_max = 0.0, 0.0, 0.0
    lifted = False
    while True:
        k1 = deriv(t, s)
        k2 = deriv(t + dt / 2, s + dt / 2 * k1)
        k3 = deriv(t + dt / 2, s + dt / 2 * k2)
        k4 = deriv(t + dt, s + dt * k3)
        s = s + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        t += dt
        z, v, _ = s
        lifted = lifted or z > 0
        v_max = max(v_max, v)
        if z > apogee:
            apogee, t_apogee = z, t
        if lifted and v < 0:
            return {"apogee": apogee, "t_apogee": t_apogee, "v_max": v_max}
        if t > 600:
            raise RuntimeError("no apogee within 600 s")


if __name__ == "__main__":
    # Illustrative run with a made-up constant-thrust motor.
    print(simulate(0.05, 0.025, 0.5, [0, 0.01, 1.0, 1.01], [0, 10, 10, 0], 0.01, 0.02))
