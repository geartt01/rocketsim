"""Horizontal wind models (3-DOF: wind blows along +x)."""

from dataclasses import dataclass


@dataclass
class Wind:
    speed: float  # m/s at reference_height; positive blows toward +x (downrange)
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
