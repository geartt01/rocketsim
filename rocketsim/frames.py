"""Quaternion math. q = [w, x, y, z], scalar first, rotating body → inertial (ENU), PLAN.md §0.

Source: standard Hamilton quaternion algebra, e.g. Diebel 2006, "Representing Attitude: Euler
Angles, Unit Quaternions, and Rotation Vectors" (Stanford). The tests check R(p⊗q) = R(p)·R(q)
and orthonormality rather than relying on the citation.
"""

import numpy as np


def multiply(p, q):
    """Hamilton product p ⊗ q, so that R(p ⊗ q) = R(p)·R(q)."""
    pw, px, py, pz = p
    qw, qx, qy, qz = q
    return np.array([
        pw * qw - px * qx - py * qy - pz * qz,
        pw * qx + px * qw + py * qz - pz * qy,
        pw * qy - px * qz + py * qw + pz * qx,
        pw * qz + px * qy - py * qx + pz * qw,
    ])


def conjugate(q):
    """q* = [w, −x, −y, −z]; for a unit quaternion this is the inverse rotation."""
    return np.array([q[0], -q[1], -q[2], -q[3]])


def normalize(v):
    """v / |v| (works for quaternions and plain vectors)."""
    v = np.asarray(v, dtype=float)
    return v / np.linalg.norm(v)


def rotation_matrix(q):
    """R(q) for a unit quaternion: v_inertial = R·v_body.

        R = [[1−2(y²+z²), 2(xy−wz),   2(xz+wy)],
             [2(xy+wz),   1−2(x²+z²), 2(yz−wx)],
             [2(xz−wy),   2(yz+wx),   1−2(x²+y²)]]
    """
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def launch_quaternion(elevation, heading):
    """Attitude on the rail: body x-axis = [cos θ·sin ψ, cos θ·cos ψ, sin θ] in ENU.

    q = q_z(π/2 − ψ) ⊗ q_y(−θ): first pitch the nose up from East by θ (rotation about y by −θ),
    then turn it about Up from East to the heading ψ (clockwise from North). Roll is arbitrary
    for an axisymmetric rocket with ≥ 3 fins; this choice keeps body y horizontal.
    Args: elevation θ, heading ψ, rad.
    """
    a, b = (np.pi / 2 - heading) / 2, -elevation / 2
    q_z = np.array([np.cos(a), 0.0, 0.0, np.sin(a)])
    q_y = np.array([np.cos(b), 0.0, np.sin(b), 0.0])
    return multiply(q_z, q_y)
