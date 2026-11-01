"""Runs inside Blender (see tests/test_blender.py), checks the add-on, prints PASS or the failure.

Usage: blender --background --factory-startup --python tests/blender_selftest.py
"""

import importlib.util
import json
import subprocess
import sys
import traceback
from pathlib import Path

import bpy
import numpy as np
from mathutils import Quaternion, Vector

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("rocketsim_addon",
                                              ROOT / "blender" / "rocketsim_addon.py")
addon = importlib.util.module_from_spec(spec)
sys.modules["rocketsim_addon"] = addon
spec.loader.exec_module(addon)
CONFIG = ROOT / "configs" / "example_rocket.yaml"


def simulate_blocking(scene):
    addon.start_simulation(scene, autoplay=False)
    addon.STATE.proc.wait()
    return addon.finish_simulation(scene)


def frame_at(flight, t):
    return int(np.argmin(np.abs(flight.frames["t"] - t))) + 1


def check_rocket_follows_the_simulation(scene, flight):
    a, ev = flight.arrays, flight.meta["events"]
    rocket = bpy.data.objects["RS Rocket"]
    for t in (0.5, 1.5, 3.0, ev["apogee"] - 0.5):
        frame = frame_at(flight, t)
        scene.frame_set(frame)
        t_exact = flight.frames["t"][frame - 1]
        cg = Vector([np.interp(t_exact, a["t"], a[k]) for k in ("x", "y", "z")])
        x_cg = np.interp(t_exact, a["t"], a["x_cg"])
        q = Quaternion([np.interp(t_exact, a["t"], a[k]) for k in ("qw", "qx", "qy", "qz")])
        expected = cg + q.normalized() @ Vector((x_cg, 0.0, 0.0))
        assert (rocket.location - expected).length < 1e-3, (t, rocket.location, expected)
        # the mesh's nose points along the simulated body x-axis
        nose_axis = rocket.rotation_quaternion @ Vector((1.0, 0.0, 0.0))
        assert (nose_axis - q.normalized() @ Vector((1.0, 0.0, 0.0))).length < 1e-5


def check_trail_ends_at_the_rocket(scene, flight):
    curve = bpy.data.curves["RS Trail"]
    scene.frame_set(frame_at(flight, -1.0))
    assert curve.bevel_factor_end == 0.0, "no trail on the pad"
    f = flight.frames
    frame = frame_at(flight, 3.0)
    scene.frame_set(frame)
    steps = np.linalg.norm(np.diff(np.c_[f["x"], f["y"], f["z"]], axis=0), axis=1)
    flown = steps[: frame - 1].sum() / steps.sum()
    assert abs(curve.bevel_factor_end - flown) < 1e-6
    scene.frame_set(scene.frame_end)
    assert abs(curve.bevel_factor_end - 1.0) < 1e-6, "the whole path by the end"


def check_events_show_on_screen(scene, flight):
    ev = flight.meta["events"]
    flame, chute = bpy.data.objects["RS Flame"], bpy.data.objects["RS Drogue"]
    scene.frame_set(frame_at(flight, -1.0))
    assert flame.hide_viewport and chute.hide_viewport, "nothing lit on the pad"
    assert flight.frames["phase"][scene.frame_current - 1] == "pad"
    scene.frame_set(frame_at(flight, 1.0))
    assert not flame.hide_viewport, "flame while the motor burns"
    scene.frame_set(frame_at(flight, ev["burnout"] + 1.0))
    assert flame.hide_viewport, "no flame after burnout"
    assert chute.hide_viewport, "no chute before deployment"
    scene.frame_set(frame_at(flight, ev["drogue"] + 2.0))
    assert not chute.hide_viewport, "chute open after deployment"
    scene.frame_set(scene.frame_end)
    assert flight.frames["phase"][scene.frame_current - 1] == "landed"
    assert flight.frames["t"][-1] >= ev["landing"] + addon.TAIL_TIME - 0.1


def check_rocket_mesh_matches_inputs(flight):
    meta = flight.meta
    xs = [v.co.x for v in bpy.data.objects["RS Rocket"].data.vertices]
    assert abs(max(xs)) < 1e-6, "nose tip at the origin"
    assert min(xs) <= -meta["length"], "reaches the tail"
    ys = [v.co.y for v in bpy.data.objects["RS Rocket"].data.vertices]
    fin_tip = meta["radius"] + meta["fins"]["span"]
    assert fin_tip - 1e-4 <= max(ys) + meta["fins"]["thickness"] and max(ys) <= fin_tip + 1e-3


def check_panel_reproduces_the_config_file(scene, flight):
    # The same numbers run straight from the YAML file must give the same flight.
    out = addon.WORK / "direct.npz"
    subprocess.run([addon.python_exe(), "-m", "rocketsim", "flightdata", str(CONFIG), "-o",
                    str(out)], check=True, cwd=ROOT)
    direct = json.loads(str(np.load(out)["meta"]))["summary"]
    for key in ("apogee", "drift", "flight_time", "rail_exit_velocity"):
        assert abs(direct[key] - flight.meta["summary"][key]) < 1e-9, key


def check_bad_input_is_reported(scene):
    p = scene.rocketsim
    saved = p.nose_length
    p.nose_length = 0.0
    assert not simulate_blocking(scene), "a zero-length nose must fail"
    assert "nose" in addon.STATE.error and "length" in addon.STATE.error, addon.STATE.error
    p.nose_length = saved
    assert simulate_blocking(scene), addon.STATE.error
    assert addon.STATE.error == ""


def check_main_chute_and_masses(scene):
    p = scene.rocketsim
    p.use_main, p.main_cda, p.main_altitude = True, 0.3, 120.0
    item = p.masses.add()
    item.name, item.mass, item.position = "camera", 0.005, 0.2
    assert simulate_blocking(scene), addon.STATE.error
    flight = addon.STATE.flight
    ev = flight.meta["events"]
    assert "main" in ev
    scene.frame_set(frame_at(flight, ev["main"] - 1.0))
    assert bpy.data.objects["RS Main"].hide_viewport
    scene.frame_set(frame_at(flight, ev["main"] + 1.0))
    assert not bpy.data.objects["RS Main"].hide_viewport
    assert abs(flight.frames["z"][scene.frame_current - 1]) < 120.0


def check_playback_is_real_time_by_default(flight):
    dt = np.diff(flight.frames["t"])
    assert np.allclose(dt, 1 / addon.FPS)


def check_fast_descent_keeps_the_ascent_real_time(scene, flight):
    scene.rocketsim.descent_speed = "10"
    ev = flight.meta["events"]
    f = flight.frames
    assert len(f["t"]) < (ev["landing"] + addon.TAIL_TIME + addon.LEAD_TIME) * addon.FPS / 2
    ascent = f["t"] < ev["apogee"]
    assert np.allclose(np.diff(f["t"][ascent]), 1 / addon.FPS)
    scene.rocketsim.descent_speed = "1"


def main():
    addon.register()
    scene = bpy.context.scene
    addon.props_from_config(scene.rocketsim, addon.read_config(CONFIG), CONFIG.parent)
    assert simulate_blocking(scene), addon.STATE.error
    flight = addon.STATE.flight
    check_playback_is_real_time_by_default(flight)
    check_rocket_follows_the_simulation(scene, flight)
    check_trail_ends_at_the_rocket(scene, flight)
    check_events_show_on_screen(scene, flight)
    check_rocket_mesh_matches_inputs(flight)
    check_panel_reproduces_the_config_file(scene, flight)
    check_fast_descent_keeps_the_ascent_real_time(scene, flight)
    check_bad_input_is_reported(scene)
    check_main_chute_and_masses(scene)
    addon.unregister()


try:
    main()
    print("SELFTEST PASS")
except Exception:  # noqa: BLE001 (any failure must reach pytest as an exit code)
    traceback.print_exc()
    print("SELFTEST FAIL")
    sys.stdout.flush()
    sys.exit(1)
