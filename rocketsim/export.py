"""Package one simulated flight for viewers such as the Blender add-on (blender/).

Blender's bundled Python has no scipy, so it cannot import the simulator. It runs this module
through the project's own Python instead and reads the result, a plain .npz file.
"""

import json
import warnings

import numpy as np

from rocketsim.flight import simulate
from rocketsim.geometry import nose_radius
from rocketsim.io import load_config
from rocketsim.recovery import open_cda

DT_OUT = 1 / 60  # s, smooth enough for 60 fps playback
PROFILE_POINTS = 49  # samples along the nose outline


def write_flight_package(config_path, out_path):
    """Fly the rocket in config_path and write out_path (.npz), for display only.

    Arrays: the trajectory columns; x_cg (m from nose tip), thrust (N), drogue_open and main_open
    (0..1 fraction of full Cd·A); profile_x, profile_r (body outline from nose tip to tail, m).
    `meta` is a JSON string: events, summary, warnings, fins, rail, motor and chute sizes.
    Raises ValueError or RuntimeError with a readable message when the flight cannot be flown.
    """
    rocket, launch = load_config(config_path)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        flight = simulate(rocket, launch, dt_out=DT_OUT)
    tr, events, rec = flight.trajectory, flight.events, rocket.recovery
    t = tr["t"]

    def opening(chute, event):
        if chute is None or event not in events:
            return np.zeros(len(t))
        return np.array([open_cda(s, [(chute, events[event])]) / chute.cda for s in t])

    u = np.linspace(0.0, 1.0, PROFILE_POINTS) ** 2  # denser near the tip, where the nose curves
    nose_x = u * rocket.nose.length
    nose_r = nose_radius(rocket.nose.shape, nose_x, rocket.nose.length, rocket.radius)
    f, mo = rocket.fins, rocket.motor
    meta = {
        "events": {k: float(v) for k, v in events.items()},
        "summary": {k: (list(v) if isinstance(v, tuple) else v)
                    for k, v in flight.summary.items()},
        "warnings": [str(w.message) for w in caught],
        "length": rocket.length,
        "radius": rocket.radius,
        "nose_length": rocket.nose.length,
        "fins": {"count": f.count, "root_chord": f.root_chord, "tip_chord": f.tip_chord,
                 "span": f.span, "sweep": f.sweep, "thickness": f.thickness,
                 "position": f.position},
        "motor": {"name": mo.name, "diameter": mo.diameter, "length": mo.length,
                  "aft_end": rocket.motor_position, "burn_time": mo.burn_time,
                  "peak_thrust": float(mo.thrust_data.max())},
        "rail": {"length": launch.rail_length, "elevation": launch.rail_elevation,
                 "heading": launch.rail_heading},
        "wind": {"speed": launch.wind.speed, "direction": launch.wind.direction},
        "chutes": {"drogue_cda": rec.drogue.cda,
                   "main_cda": rec.main.cda if rec.main else 0.0},
    }
    np.savez(
        out_path, **{k: (v.astype(str) if k == "phase" else v) for k, v in tr.items()},
        x_cg=np.array([rocket.mass_properties(s)[1] for s in t]),
        thrust=mo.thrust(t),
        drogue_open=opening(rec.drogue, "drogue"),
        main_open=opening(rec.main, "main"),
        profile_x=np.append(nose_x, rocket.length),
        profile_r=np.append(nose_r, rocket.radius),
        meta=np.array(json.dumps(meta)),
    )
    return flight
