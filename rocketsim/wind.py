"""Horizontal wind models."""

from dataclasses import dataclass

import numpy as np


@dataclass
class Wind:
    speed: float  # m/s at reference_height, ≥ 0
    direction: float  # rad clockwise from North that the wind blows FROM (meteorological)
    reference_height: float  # m above the launch site
    profile: str  # "constant" | "power_law"

    def __call__(self, z):
        """Wind speed at height z above the launch site, m/s.

        constant:   w(z) = w_ref
        power_law:  w(z) = w_ref·(z/z_ref)^(1/7) for z > 0, else 0
        Source: the classic empirical 1/7 power law for a neutrally stable boundary layer over
        open ground, as specified in PLAN.md (no primary reference checked yet).
        Assumption: no gusts (Stage 5 adds them).
        """
        if self.profile == "constant":
            return self.speed
        return self.speed * (max(z, 0.0) / self.reference_height) ** (1 / 7)

    def vector(self, z):
        """Wind velocity in ENU, m/s: −w(z)·[sin d, cos d, 0] (it blows toward d + 180°)."""
        w = self(z)
        return np.array([-w * np.sin(self.direction), -w * np.cos(self.direction), 0.0])
