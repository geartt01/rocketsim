"""Drogue and main parachutes: deployment rules and inflation."""

from dataclasses import dataclass


@dataclass
class Chute:
    cda: float  # m², drag coefficient × reference area when fully open
    fill_time: float  # s from deployment to fully open (0 = instant)


@dataclass
class Recovery:
    drogue: Chute  # deploys `drogue_delay` s after apogee (a single-chute rocket uses only this)
    drogue_delay: float  # s; models a motor ejection charge firing after apogee
    main: Chute | None = None  # deploys when descending through `main_altitude`
    main_altitude: float = 0.0  # m above the launch site


def open_cda(t, deployments):
    """Total parachute Cd·A at time t, m², for [(chute, t_deploy), ...].

        CdA(t) = Σ CdA_i · clip((t − t_i) / t_fill,i, 0, 1)

    Assumption: each canopy's Cd·A grows linearly from 0 to full over its fill time (a common
    first-order inflation model; real canopies grow roughly with the square of the filled
    fraction, so this overestimates early drag a little). Chutes stay attached once deployed,
    so under main the drogue still pulls. Args: t (s), deployments.
    """
    total = 0.0
    for chute, t_deploy in deployments:
        if t >= t_deploy:
            fill = 1.0 if chute.fill_time == 0 else min(1.0, (t - t_deploy) / chute.fill_time)
            total += chute.cda * fill
    return total
