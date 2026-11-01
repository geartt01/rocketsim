"""Full 6-DOF rigid-body dynamics: state [r (3), v (3), q (4), ω (3)].

r, v are inertial ENU (origin at the rail base); q = [w, x, y, z] rotates body → ENU; ω is the
body-frame angular velocity [roll, pitch, yaw rates about x_b, y_b, z_b] (PLAN.md §0).
"""

import math

import numpy as np

from rocketsim import frames
from rocketsim.aero.barrowman import fins_cn_alpha_cp, nose_cn_alpha_cp
from rocketsim.aero.damping import pitch_damping_coefficient, roll_damping_coefficient
from rocketsim.aero.drag import drag_coefficient
from rocketsim.dynamics.dof3 import air
from rocketsim.recovery import open_cda

STATE = ("x", "y", "z", "vx", "vy", "vz", "qw", "qx", "qy", "qz", "wx", "wy", "wz")
Z, VZ, HORIZONTAL, VELOCITY = 2, 5, (0, 1), (3, 4, 5)
_STILL = np.zeros(7)  # dq/dt and dω/dt when the attitude is held


def initial_state(launch):
    q = frames.launch_quaternion(launch.rail_elevation, launch.rail_heading)
    return np.concatenate([np.zeros(6), q, np.zeros(3)])


def normalize(y):
    """Renormalise the quaternion part of a state (done at every phase boundary)."""
    y = np.array(y, dtype=float)
    y[6:10] = frames.normalize(y[6:10])
    return y


def stop_rotation(y):
    """State with zero angular velocity, for the point-mass recovery phases."""
    y = np.array(y, dtype=float)
    y[10:13] = 0.0
    return y


def rotational_derivatives(q, omega, inertia, moment):
    """Attitude kinematics and Euler's rotation equations, body frame.

        q̇ = ½·q ⊗ [0, ω]
        I·ω̇ = M − ω × (I·ω),   I = diag(I_x, I_y, I_z)

    Args: q unit quaternion, omega (rad/s, body), inertia principal moments (kg·m²),
          moment about the CG (N·m, body).
    Assumption: principal axes along the body axes; variable-mass terms (İ·ω, jet damping)
    ignored. Source: Euler's equations, e.g. Goldstein, Classical Mechanics, ch. 5.
    """
    q_dot = 0.5 * frames.multiply(q, np.array([0.0, *omega]))
    (wx, wy, wz), (hx, hy, hz) = omega, inertia * omega
    gyro = np.array([wy * hz - wz * hy, wz * hx - wx * hz, wx * hy - wy * hx])  # ω × (I·ω)
    return q_dot, (moment - gyro) / inertia


def aero_forces(t, y, rocket, launch):
    """Non-gravity forces (inertial) and moment about the CG (body).

    v_a,b = Rᵀ·(v − w(z))  air-relative velocity in body axes,  q̄ = ½ρ|v_a|²
    axial   F_x = −½ρ·A·C_D0·|v_a|·v_bx                  (= −q̄·A·C_D0·cos α)
    normal  F_n = −½ρ·A·CN_α·|v_a|·[0, v_by, v_bz]       (= q̄·A·CN_α·sin α, opposing v_lat)
    thrust  F_T = T(t)·[cos δ, sin δ·cos φ, sin δ·sin φ], δ misalignment, φ its azimuth,
            acting at the nozzle x_noz (the motor's aft end)
    moment  M = [x_cg − X_cp, 0, 0] × F_n + [x_cg − x_noz, 0, 0] × F_T
                − K·|ω_t|·ω_t − C_roll·ω_x·x̂
            ω_t = [0, ω_y, ω_z]; K pitch/yaw damping, C_roll roll damping (Niskanen 3.58–3.69)

    Returns: (F [N, ENU], M [N·m, body], mass [kg], inertia [I_x, I_t, I_t] [kg·m²],
              info = (α [rad], Mach [-], margin [calibers])).
    Assumptions as in dof3.aero_forces; no fin cant, so no roll forcing.
    """
    v, q, omega = y[3:6], frames.normalize(y[6:10]), y[10:13]
    z = y[2]
    m, x_cg, i_t, i_x = rocket.mass_properties(t)
    rho, sound, nu = air(launch, z)
    rot = frames.rotation_matrix(q)
    vb = rot.T @ (v - launch.wind.vector(z, t))
    speed = math.sqrt(vb @ vb)
    mach = speed / sound

    thrusting = 0.0 <= t < rocket.motor.burn_time
    cd = drag_coefficient(rocket, mach, speed * rocket.length / nu, thrusting)
    (cn_n, x_n), (cn_f, x_f) = nose_cn_alpha_cp(rocket), fins_cn_alpha_cp(rocket, mach)
    cn = cn_n + cn_f
    x_cp = (cn_n * x_n + cn_f * x_f) / cn

    k = -0.5 * rho * rocket.area * cn * speed
    force_b = np.array([-0.5 * rho * rocket.area * cd * speed * vb[0], k * vb[1], k * vb[2]])
    arm = x_cg - x_cp
    moment = np.array([0.0, -arm * force_b[2], arm * force_b[1]])  # [arm, 0, 0] × F_n
    # thrust tilted by δ toward azimuth φ, acting at the nozzle (motor aft end)
    delta, phi = rocket.thrust_misalignment, rocket.misalignment_azimuth
    thrust = rocket.motor.thrust(t) * np.array([math.cos(delta), math.sin(delta) * math.cos(phi),
                                                math.sin(delta) * math.sin(phi)])
    nozzle_arm = x_cg - rocket.motor_position
    moment += np.array([0.0, -nozzle_arm * thrust[2], nozzle_arm * thrust[1]])
    force_b += thrust
    w_t = np.array([0.0, omega[1], omega[2]])
    moment -= pitch_damping_coefficient(rocket, rho, x_cg, x_f) * math.hypot(*omega[1:]) * w_t
    moment[0] -= roll_damping_coefficient(rocket, rho, speed, mach) * omega[0]

    alpha = math.atan2(math.hypot(vb[1], vb[2]), vb[0])
    margin = (x_cp - x_cg) / rocket.body.diameter
    return rot @ force_b, moment, m, np.array([i_x, i_t, i_t]), (alpha, mach, margin)


def info(t, y, rocket, launch):
    """(α [rad], Mach [-], stability margin [calibers], mass [kg]) for output."""
    *_, m, _, (alpha, mach, margin) = aero_forces(t, y, rocket, launch)
    return alpha, mach, margin, m


def derivatives(t, y, rocket, launch):
    """Free flight: ṙ = v,  m·v̇ = F + m·g,  q̇, ω̇ from rotational_derivatives."""
    force, moment, m, inertia, _ = aero_forces(t, y, rocket, launch)
    q_dot, w_dot = rotational_derivatives(frames.normalize(y[6:10]), y[10:13], inertia, moment)
    accel = force / m - np.array([0.0, 0.0, launch.g])
    return np.concatenate([y[3:6], accel, q_dot, w_dot])


def derivatives_rail(t, y, rocket, launch):
    """On the rail: motion only along the body axis e, no rotation.

        a = (F·e) / m − g·e_z  along e
    The rail reacts everything across it. While the rocket has not started moving and the
    net force is not positive, it stays put.
    """
    force, _, m, _, _ = aero_forces(t, y, rocket, launch)
    e = frames.rotation_matrix(frames.normalize(y[6:10]))[:, 0]
    along = force @ e / m - launch.g * e[2]
    if y[3:6] @ e <= 0 and along <= 0:
        along = 0.0
    return np.concatenate([y[3:6], along * e, _STILL])


def derivatives_chute(t, y, rocket, launch, deployments):
    """Under parachute, point mass: a = −½ρ·CdA(t)·|v_a|·v_a / m − g·ẑ; attitude frozen.

    CdA(t) from recovery.open_cda; body drag ignored under canopy. Drag acts along −v_a, so
    the rocket drifts with the wind (PLAN.md §3: point-mass recovery).
    """
    m = rocket.mass_properties(t)[0]
    rho = air(launch, y[2])[0]
    w = launch.wind.vector(y[2], t)
    vax, vay, vaz = y[3] - w[0], y[4] - w[1], y[5] - w[2]
    k = -0.5 * rho * open_cda(t, deployments) * math.sqrt(vax**2 + vay**2 + vaz**2) / m
    return np.array([y[3], y[4], y[5], k * vax, k * vay, k * vaz - launch.g, *_STILL])


def rail_exit(t, y, rocket, launch):
    """Zero when the distance travelled along the rail reaches the rail length."""
    e = frames.rotation_matrix(frames.normalize(y[6:10]))[:, 0]
    return y[0:3] @ e - launch.rail_length
