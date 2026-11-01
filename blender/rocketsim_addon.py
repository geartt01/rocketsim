# SPDX-License-Identifier: GPL-3.0-or-later
"""RocketSim in Blender: type in a rocket, press Simulate & Fly, watch it fly with live data.

Start it with `python -m rocketsim blender`. The physics stays in the rocketsim package: this
add-on writes a config file, runs the simulator with the project's own Python (Blender's has no
scipy), reads the flight back and plays it. Nothing here changes the simulated numbers.

Display-only choices, which are not simulated: the rocket hangs nose-up under its chute (the
simulator treats it as a point there) and lies down on landing; canopy size assumes Cd 0.8;
the flame's size follows thrust; the rail is drawn beside the rocket.
"""

import json
import math
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import blf
import bpy
import gpu
import numpy as np
from bpy.props import (
    BoolProperty,
    CollectionProperty,
    EnumProperty,
    FloatProperty,
    IntProperty,
    PointerProperty,
    StringProperty,
)
from bpy_extras.io_utils import ExportHelper, ImportHelper
from gpu_extras.batch import batch_for_shader
from mathutils import Euler, Quaternion, Vector

bl_info = {
    "name": "RocketSim",
    "author": "RocketSim project",
    "version": (1, 0, 0),
    "blender": (4, 2, 0),
    "location": "3D Viewport > Sidebar (N) > RocketSim",
    "description": "Simulate a model rocket with RocketSim and watch it fly with live data",
    "category": "Simulation",
}

PROJECT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT / "configs" / "example_rocket.yaml"
WORK = PROJECT / "outputs" / "blender"
FPS = 30
LEAD_TIME = 2.0  # s shown on the pad before ignition
TAIL_TIME = 4.0  # s shown after landing
G0 = 9.80665
VISUAL_CD = 0.8  # only sizes the drawn canopy
GROUND_SIZE = 40000.0  # m, big enough that its edge stays out of sight from altitude

STATE = SimpleNamespace(proc=None, out=None, autoplay=False, flight=None, status="Starting...",
                        error="", extras={}, draw=None)


def python_exe():
    """The project's Python, which has the simulator's dependencies."""
    if os.environ.get("ROCKETSIM_PYTHON"):
        return os.environ["ROCKETSIM_PYTHON"]
    return str(PROJECT / ".venv" / "Scripts" / "python.exe")


def run_hidden(args, **kwargs):
    flags = {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {}
    return subprocess.Popen(args, cwd=PROJECT, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            text=True, encoding="utf-8", **flags, **kwargs)


# --- rocket inputs ------------------------------------------------------------------------------

def length(name, **kw):
    return FloatProperty(name=name, unit="LENGTH", min=kw.pop("min", 0.0), precision=4,
                         step=0.1, **kw)


def mass(name, **kw):
    return FloatProperty(name=name, unit="MASS", min=0.0, precision=4, step=0.01, **kw)


def number(name, **kw):
    return FloatProperty(name=name, precision=kw.pop("precision", 3), **kw)


class RSMassItem(bpy.types.PropertyGroup):
    name: StringProperty(name="Name", default="payload")
    mass: mass("Mass")
    position: length("Position", description="Nose tip to this item's centre of mass")
    length: length("Length", description="With a radius: a solid cylinder. 0: a point mass")
    radius: length("Radius", description="With a length: a solid cylinder. 0: a point mass")


def camera_changed(self, context):
    update_scene(context.scene)


def speed_changed(self, context):
    if STATE.flight:
        prepare_playback(STATE.flight, context.scene)
        update_scene(context.scene)


class RSProps(bpy.types.PropertyGroup):
    nose_shape: EnumProperty(name="Shape", items=[
        ("cone", "Cone", ""), ("ogive", "Tangent ogive", ""),
        ("elliptical", "Elliptical", ""), ("von_karman", "Von Karman", "")])
    nose_length: length("Length")
    nose_mass: mass("Mass")
    body_length: length("Length")
    body_diameter: length("Diameter")
    body_mass: mass("Mass")
    fin_count: IntProperty(name="Count", min=3, max=8, default=3)
    fin_root: length("Root chord")
    fin_tip: length("Tip chord")
    fin_span: length("Span")
    fin_sweep: FloatProperty(name="Sweep", unit="LENGTH", precision=4, step=0.1)
    fin_thickness: length("Thickness")
    fin_position: length("Position", description="Nose tip to the root chord's leading edge")
    fin_profile: EnumProperty(name="Profile", items=[
        ("square", "Square", ""), ("rounded", "Rounded", ""), ("airfoil", "Airfoil", "")])
    fin_mass: mass("Mass")
    masses: CollectionProperty(type=RSMassItem)
    masses_index: IntProperty()
    motor_file: StringProperty(name="Motor file", subtype="FILE_PATH",
                               description="RASP .eng thrust curve (thrustcurve.org)")
    motor_position: length("Aft end", description="Nose tip to the motor's aft end")
    roughness: FloatProperty(name="Roughness (mm)", min=0.0, precision=3, step=0.1,
                             description="Regular paint is about 0.06 mm")
    drag_mode: EnumProperty(name="Drag", items=[
        ("buildup", "Calculated (Niskanen)", "Component buildup from the geometry"),
        ("constant", "Constant Cd", "One Cd for all speeds"),
        ("table", "Cd(Mach) table", "CSV with header mach,cd")])
    cd_value: number("Cd", min=0.001, default=0.5)
    cd_table: StringProperty(name="Table file", subtype="FILE_PATH")
    drogue_cda: number("Cd x A (m²)", min=0.0001, precision=4,
                       description="Drag coefficient times canopy area")
    drogue_fill: number("Fill time (s)", min=0.0)
    drogue_delay: number("Delay (s)", min=0.0, description="Seconds after apogee")
    use_main: BoolProperty(name="Main parachute", default=False)
    main_cda: number("Cd x A (m²)", min=0.0001, precision=4, default=0.3,
                     description="Drag coefficient times canopy area")
    main_fill: number("Fill time (s)", min=0.0, default=0.5)
    main_altitude: length("Opens at", default=100.0,
                          description="Altitude above the launch site")
    site_altitude: FloatProperty(name="Site altitude", unit="LENGTH", min=-2000.0, max=20000.0)
    rail_length: length("Rail length")
    rail_elevation: number("Elevation (deg)", min=1.0, max=90.0, precision=1,
                           description="90 is straight up")
    rail_heading: number("Heading (deg)", min=0.0, max=360.0, precision=1,
                         description="Clockwise from North, the way the rail leans")
    wind_speed: number("Speed (m/s)", min=0.0, precision=2)
    wind_direction: number("Blows from (deg)", min=0.0, max=360.0, precision=1,
                           description="Clockwise from North, the direction it comes FROM")
    wind_height: length("Ref. height", default=10.0, min=0.1,
                        description="The height at which the wind speed applies")
    wind_profile: EnumProperty(name="Profile", items=[
        ("constant", "Constant", "Same wind at all heights"),
        ("power_law", "Power law", "Slower near the ground (1/7 law)")])
    camera_mode: EnumProperty(name="Camera", default="GROUND", update=camera_changed, items=[
        ("GROUND", "From the ground", "Stands by the pad and zooms to follow"),
        ("FOLLOW", "Follow rocket", "Flies alongside the rocket"),
        ("OVERVIEW", "Whole flight", "Far enough to see the full path")])
    descent_speed: EnumProperty(name="Descent", default="1", update=speed_changed, items=[
        ("1", "Real time", "The whole flight at 1x"),
        ("4", "4x", "Fast-forward the slow parachute descent"),
        ("10", "10x", "Fast-forward the slow parachute descent")])


def _exact(value):
    """Undo Blender's 32-bit float storage (0.0248 reads back 0.024800000712...)."""
    if isinstance(value, float):
        return float(f"{value:.7g}")
    if isinstance(value, dict):
        return {k: _exact(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_exact(v) for v in value]
    return value


def config_from_props(p):
    """The panel's values as a RocketSim config dictionary (angles in degrees, SI otherwise)."""
    if p.drag_mode == "buildup":
        drag = "buildup"
    elif p.drag_mode == "constant":
        drag = p.cd_value
    else:
        drag = bpy.path.abspath(p.cd_table)
    masses = []
    for m in p.masses:
        item = {"name": m.name, "mass": m.mass, "position": m.position}
        if m.length > 0 and m.radius > 0:
            item |= {"length": m.length, "radius": m.radius}
        masses.append(item)
    recovery = {"drogue": {"cda": p.drogue_cda, "fill_time": p.drogue_fill,
                           "delay": p.drogue_delay}}
    if p.use_main:
        recovery["main"] = {"cda": p.main_cda, "fill_time": p.main_fill,
                            "altitude": p.main_altitude}
    return _exact({
        "rocket": {
            "nose": {"shape": p.nose_shape, "length": p.nose_length, "mass": p.nose_mass},
            "body": {"length": p.body_length, "diameter": p.body_diameter, "mass": p.body_mass},
            "fins": {"count": p.fin_count, "root_chord": p.fin_root, "tip_chord": p.fin_tip,
                     "span": p.fin_span, "sweep": p.fin_sweep, "thickness": p.fin_thickness,
                     "position": p.fin_position, "profile": p.fin_profile, "mass": p.fin_mass},
            "masses": masses,
            "motor_position": p.motor_position,
            "surface_roughness": p.roughness / 1000.0,
            "drag": drag,
        },
        "motor": bpy.path.abspath(p.motor_file),
        "recovery": recovery,
        "launch": {"site_altitude": p.site_altitude, "rail_length": p.rail_length,
                   "rail_elevation": p.rail_elevation, "rail_heading": p.rail_heading},
        "wind": {"speed": p.wind_speed, "direction": p.wind_direction,
                 "reference_height": p.wind_height, "profile": p.wind_profile},
    })


def props_from_config(p, cfg, folder):
    """Fill the panel from a RocketSim config dictionary. Paths resolve against folder."""
    r = cfg["rocket"]
    n, b, f = r["nose"], r["body"], r["fins"]
    p.nose_shape, p.nose_length, p.nose_mass = n["shape"], n["length"], n["mass"]
    p.body_length, p.body_diameter, p.body_mass = b["length"], b["diameter"], b["mass"]
    p.fin_count, p.fin_root, p.fin_tip = f["count"], f["root_chord"], f["tip_chord"]
    p.fin_span, p.fin_sweep, p.fin_thickness = f["span"], f["sweep"], f["thickness"]
    p.fin_position, p.fin_profile, p.fin_mass = f["position"], f["profile"], f["mass"]
    p.masses.clear()
    for m in r["masses"]:
        item = p.masses.add()
        item.name, item.mass, item.position = str(m["name"]), m["mass"], m["position"]
        item.length, item.radius = m.get("length", 0.0), m.get("radius", 0.0)
    p.motor_file = str((Path(folder) / cfg["motor"]).resolve())
    p.motor_position = r["motor_position"]
    p.roughness = r["surface_roughness"] * 1000.0
    if r["drag"] == "buildup":
        p.drag_mode = "buildup"
    elif isinstance(r["drag"], str):
        p.drag_mode, p.cd_table = "table", str((Path(folder) / r["drag"]).resolve())
    else:
        p.drag_mode, p.cd_value = "constant", r["drag"]
    d = cfg["recovery"]["drogue"]
    p.drogue_cda, p.drogue_fill, p.drogue_delay = d["cda"], d["fill_time"], d["delay"]
    main = cfg["recovery"].get("main")
    p.use_main = main is not None
    if main:
        p.main_cda, p.main_fill, p.main_altitude = main["cda"], main["fill_time"], main["altitude"]
    s, w = cfg["launch"], cfg["wind"]
    p.site_altitude, p.rail_length = s["site_altitude"], s["rail_length"]
    p.rail_elevation, p.rail_heading = s["rail_elevation"], s["rail_heading"]
    p.wind_speed, p.wind_direction = w["speed"], w["direction"]
    p.wind_height, p.wind_profile = w["reference_height"], w["profile"]
    STATE.extras = {k: cfg[k] for k in ("dispersions", "sensors") if k in cfg}


def read_config(path):
    """Parse a YAML config with the project's Python (Blender has no YAML library)."""
    code = ("import json,sys;from rocketsim.io import read_yaml;"
            "print(json.dumps(read_yaml(sys.argv[1])))")
    proc = run_hidden([python_exe(), "-c", code, str(path)])
    out, err = proc.communicate()
    if proc.returncode:
        raise RuntimeError(err.strip().splitlines()[-1] if err.strip() else "cannot read config")
    return json.loads(out)


# --- running the simulator ----------------------------------------------------------------------

def start_simulation(scene, autoplay):
    """Write the panel's rocket to a config file and run the simulator in the background."""
    if STATE.proc:
        return
    WORK.mkdir(parents=True, exist_ok=True)
    config = WORK / "current_rocket.yaml"
    # JSON is valid YAML, so the simulator's own loader validates exactly what the panel holds.
    config.write_text(json.dumps(config_from_props(scene.rocketsim), indent=2), encoding="utf-8")
    STATE.out = WORK / "flight.npz"
    STATE.autoplay = autoplay
    STATE.error, STATE.status = "", "Simulating..."
    STATE.proc = run_hidden([python_exe(), "-m", "rocketsim", "flightdata", str(config),
                             "-o", str(STATE.out)])
    if not bpy.app.background:
        bpy.app.timers.register(poll_simulation, first_interval=0.1)


def finish_simulation(scene):
    """Collect the finished simulator run: load the flight, or report why it failed."""
    proc, STATE.proc = STATE.proc, None
    _, err = proc.communicate()
    if proc.returncode:
        lines = [ln for ln in err.strip().splitlines() if ln.strip()]
        message = lines[-1].removeprefix("ERROR: ") if lines else "The simulator failed"
        STATE.error = message.split(": ", 1)[-1]  # drop the temporary config file's path
        STATE.status = "Simulation failed"
        return False
    STATE.flight = load_flight(STATE.out)
    STATE.status = ""
    build_scene(scene, STATE.flight)
    prepare_playback(STATE.flight, scene)
    scene.frame_set(1)
    update_scene(scene)
    return True


def poll_simulation():
    if STATE.proc is None:
        return None
    if STATE.proc.poll() is None:
        redraw_all()
        return 0.1
    scene = bpy.context.scene
    if finish_simulation(scene) and STATE.autoplay:
        play_animation()
    redraw_all()
    return None


def load_flight(path):
    with np.load(path, allow_pickle=False) as z:
        arrays = {k: z[k] for k in z.files}
    meta = json.loads(str(arrays.pop("meta")))
    arrays["phase"] = arrays["phase"].astype(str)
    return SimpleNamespace(arrays=arrays, meta=meta, frames=None)


def prepare_playback(flight, scene):
    """Build the per-frame table, so playback and scrubbing are exact table lookups."""
    a, ev = flight.arrays, flight.meta["events"]
    k = float(scene.rocketsim.descent_speed)
    fast_from = ev.get("main", ev["drogue"]) + 4.0
    fast_to = ev["landing"] - 6.0
    t_end = ev["landing"] + TAIL_TIME
    times = []
    t = -LEAD_TIME
    while t < t_end:
        times.append(t)
        t += (k if fast_from < t < fast_to else 1.0) / FPS
    tf = np.array(times)
    n = len(tf)
    row = np.clip(np.searchsorted(a["t"], tf, side="right") - 1, 0, len(a["t"]) - 1)
    f = {"t": tf, "rate": np.where((tf > fast_from) & (tf < fast_to), k, 1.0)}
    for key in ("x", "y", "z", "vx", "vy", "vz", "ax", "ay", "az", "qw", "qx", "qy", "qz",
                "mass", "x_cg", "drogue_open", "main_open"):
        f[key] = np.interp(tf, a["t"], a[key])
    for key in ("mach", "alpha", "margin"):
        f[key] = a[key][row]
    f["thrust"] = np.interp(tf, np.r_[0.0, a["t"]], np.r_[0.0, a["thrust"]], left=0.0)
    f["phase"] = a["phase"][row]
    still = (tf < ev["liftoff"]) | (tf >= ev["landing"])
    for key in ("vx", "vy", "vz", "ax", "ay", "az"):
        f[key][still] = 0.0
    f["phase"][tf < ev["liftoff"]] = "pad"
    f["phase"][tf >= ev["landing"]] = "landed"
    for key in ("mach", "alpha", "margin"):
        f[key][tf >= ev["landing"]] = np.nan
    f["mach"][tf < ev["liftoff"]], f["alpha"][tf < ev["liftoff"]] = 0.0, 0.0
    f["speed"] = np.sqrt(f["vx"] ** 2 + f["vy"] ** 2 + f["vz"] ** 2)
    f["accel_g"] = np.sqrt(f["ax"] ** 2 + f["ay"] ** 2 + f["az"] ** 2) / G0
    f["distance"] = np.hypot(f["x"], f["y"])

    # Display attitude and position, kept apart from the simulated values above.
    meta = flight.meta
    ground = -(meta["length"] - f["x_cg"][0]) - 0.002
    hanging = Vector((1, 0, 0)).rotation_difference(Vector((0, 0, 1)))
    heading = meta["rail"]["heading"]
    lying = Euler((0, 0, math.pi / 2 - heading)).to_quaternion()
    loc, rot = np.empty((n, 3)), np.empty((n, 4))
    for i in range(n):
        q = Quaternion((f["qw"][i], f["qx"][i], f["qy"][i], f["qz"][i])).normalized()
        cg = Vector((f["x"][i], f["y"][i], f["z"][i]))
        tt = tf[i]
        if tt > ev["drogue"]:
            q = q.slerp(hanging, _smooth((tt - ev["drogue"]) / 2.0))
        if tt > ev["landing"] - 0.1:
            q = q.slerp(lying, _smooth((tt - ev["landing"] + 0.1) / 0.5))
            rest = ground + meta["radius"] + 0.002
            cg.z = max(rest, cg.z) if tt < ev["landing"] else rest
        rot[i] = tuple(q)
        loc[i] = cg + q @ Vector((f["x_cg"][i], 0.0, 0.0))  # nose tip, the mesh origin
    f["loc"], f["rot"], f["ground"] = loc, rot, ground
    f["collapse"] = np.clip((tf - ev["landing"]) / 1.0, 0.0, 1.0)
    flight.frames = f
    scene.frame_start, scene.frame_end = 1, n
    scene.render.fps = FPS
    scene.sync_mode = "FRAME_DROP"  # keep real time even if the viewport can't reach 30 fps
    set_trail_points(f)


def _smooth(u):
    u = min(max(u, 0.0), 1.0)
    return u * u * (3 - 2 * u)


# --- the 3-D scene ------------------------------------------------------------------------------

def material(name, color, emission=0.0, roughness=0.5, metallic=0.0):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    node = mat.node_tree.nodes["Principled BSDF"]
    node.inputs["Base Color"].default_value = (*color, 1.0)
    node.inputs["Roughness"].default_value = roughness
    node.inputs["Metallic"].default_value = metallic
    if emission:
        node.inputs["Emission Color"].default_value = (*color, 1.0)
        node.inputs["Emission Strength"].default_value = emission
    return mat


def new_mesh_object(name, verts, faces, materials=(), face_materials=None, smooth=False):
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(verts, [], faces)
    for m in materials:
        mesh.materials.append(m)
    if face_materials is not None:
        mesh.polygons.foreach_set("material_index", face_materials)
    if smooth:
        mesh.shade_smooth()
    mesh.validate()
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)
    return obj


def revolve(profile, segments=32):
    """Surface of revolution about the rocket's axis. profile: (axial x, radius), nose to tail."""
    verts, faces = [], []
    for x, r in profile:
        for k in range(segments):
            a = 2 * math.pi * k / segments
            verts.append((-float(x), float(r) * math.cos(a), float(r) * math.sin(a)))
    for i in range(len(profile) - 1):
        for k in range(segments):
            k2 = (k + 1) % segments
            a, b = i * segments + k, i * segments + k2
            faces.append((a, b, b + segments, a + segments))
    return verts, faces


def build_rocket(meta, arrays):
    """One mesh from the simulator's outline and fin sizes. Origin at the nose tip; +x to the nose."""
    px, pr = arrays["profile_x"], arrays["profile_r"]
    nose_end = meta["nose_length"]
    verts, faces = revolve(list(zip(px, pr)))
    segments = 32
    face_mat = [0 if px[i] < nose_end - 1e-9 else 1 for i in range(len(px) - 1)
                for _ in range(segments)]
    # tail cap with the motor nozzle sticking out of it
    motor = meta["motor"]
    r_nozzle = motor["diameter"] / 2
    start = len(verts)
    tail = meta["length"]
    aft = max(motor["aft_end"], tail)
    cap, cap_faces = revolve([(tail, meta["radius"]), (tail, r_nozzle), (aft + 0.004, r_nozzle),
                              (aft + 0.004, r_nozzle * 0.5)])
    verts += cap
    faces += [tuple(v + start for v in f) for f in cap_faces]
    face_mat += [3] * len(cap_faces)
    f = meta["fins"]
    half = f["thickness"] / 2
    x0, cr, ct, span, sweep = f["position"], f["root_chord"], f["tip_chord"], f["span"], f["sweep"]
    r = meta["radius"]
    outline = [(x0, r), (x0 + cr, r), (x0 + sweep + ct, r + span), (x0 + sweep, r + span)]
    for n in range(f["count"]):
        angle = 2 * math.pi * n / f["count"]
        c, s = math.cos(angle), math.sin(angle)
        start = len(verts)
        for side in (-half, half):
            for ax, rad in outline:
                verts.append((-ax, rad * c - side * s, rad * s + side * c))
        faces += [tuple(start + v for v in q) for q in (
            (0, 1, 2, 3), (7, 6, 5, 4), (0, 4, 5, 1), (1, 5, 6, 2), (2, 6, 7, 3), (3, 7, 4, 0))]
        face_mat += [2] * 6
    mats = (material("RS Nose", (0.85, 0.12, 0.05), roughness=0.3),
            material("RS Body", (0.93, 0.93, 0.9), roughness=0.35),
            material("RS Fins", (0.1, 0.1, 0.12), roughness=0.4),
            material("RS Nozzle", (0.2, 0.2, 0.2), roughness=0.5, metallic=0.8))
    obj = new_mesh_object("RS Rocket", verts, faces, mats, face_mat)
    for poly in obj.data.polygons:
        poly.use_smooth = poly.material_index in (0, 1)
    obj.rotation_mode = "QUATERNION"
    return obj


def build_chute(name, color):
    """Canopy of radius 1 with lines meeting at the origin, which attaches to the rocket."""
    rings, segments, line = 7, 24, 2.2
    verts, faces = [], []
    for i in range(rings):
        phi = math.pi / 2 * (i + 1) / rings
        for k in range(segments):
            a = 2 * math.pi * k / segments
            verts.append((math.sin(phi) * math.cos(a), math.sin(phi) * math.sin(a),
                           line + 0.7 * math.cos(phi)))
    apex = len(verts)
    verts.append((0.0, 0.0, line + 0.7))
    for k in range(segments):
        faces.append((apex, k, (k + 1) % segments))
    for i in range(rings - 1):
        for k in range(segments):
            k2 = (k + 1) % segments
            a, b = i * segments + k, i * segments + k2
            faces.append((a, a + segments, b + segments, b))
    rim = (rings - 1) * segments
    for k in range(0, segments, 3):  # suspension lines as thin triangles
        i = len(verts)
        p = Vector(verts[rim + k])
        side = Vector((-p.y, p.x, 0.0)).normalized() * 0.008
        verts += [tuple(p - side), tuple(p + side), (0.0, 0.0, 0.0)]
        faces.append((i, i + 1, i + 2))
    obj = new_mesh_object(name, verts, faces,
                          (material(name, color, emission=0.45, roughness=0.6),), smooth=True)
    obj.rotation_mode = "XYZ"
    return obj


def build_flame(meta):
    """Unit-length cone along −x, scaled by thrust; emissive."""
    r = meta["motor"]["diameter"] * 0.45
    profile = [(0.0, r), (0.3, r * 1.1), (1.0, 0.0)]
    verts, faces = revolve(profile, 16)
    obj = new_mesh_object("RS Flame", verts, faces,
                          (material("RS Flame", (1.0, 0.42, 0.06), emission=3.0),), smooth=True)
    return obj


def build_scene(scene, flight):
    """Rebuild the world for this rocket. Everything made here is named 'RS ...'."""
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj, do_unlink=True)
    for coll in (bpy.data.meshes, bpy.data.materials, bpy.data.curves, bpy.data.cameras,
                 bpy.data.lights):
        for item in list(coll):
            if item.users == 0:
                coll.remove(item)
    meta, arrays = flight.meta, flight.arrays
    build_rocket(meta, arrays)
    flame = build_flame(meta)
    flame.parent = bpy.data.objects["RS Rocket"]
    flame.location = (-max(meta["motor"]["aft_end"], meta["length"]) - 0.004, 0.0, 0.0)
    build_chute("RS Drogue", (0.95, 0.3, 0.1))
    build_chute("RS Main", (0.1, 0.45, 0.95))

    ground_z = -(meta["length"] - arrays["x_cg"][0]) - 0.002
    build_ground(ground_z)
    build_rail(meta, ground_z)

    marker = new_mesh_object("RS Marker", *uv_sphere(), (material(
        "RS Marker", (1.0, 0.2, 0.05), emission=1.2),), smooth=True)
    marker.hide_render = False
    focus = bpy.data.objects.new("RS Focus", None)
    scene.collection.objects.link(focus)

    curve = bpy.data.curves.new("RS Trail", "CURVE")
    curve.dimensions = "3D"
    curve.bevel_depth = 0.004
    curve.bevel_resolution = 1
    curve.use_fill_caps = True
    curve.materials.append(material("RS Trail", (1.0, 0.72, 0.3), emission=1.0))
    trail = bpy.data.objects.new("RS Trail", curve)
    scene.collection.objects.link(trail)

    cam_data = bpy.data.cameras.new("RS Camera")
    cam_data.clip_start, cam_data.clip_end = 0.05, 100000.0
    cam_data.show_passepartout = False
    cam_data.show_limits = False
    scene.render.resolution_x, scene.render.resolution_y = 1920, 1080
    cam = bpy.data.objects.new("RS Camera", cam_data)
    scene.collection.objects.link(cam)
    track = cam.constraints.new("TRACK_TO")
    track.target, track.track_axis, track.up_axis = focus, "TRACK_NEGATIVE_Z", "UP_Y"
    scene.camera = cam

    sun_data = bpy.data.lights.new("RS Sun", "SUN")
    sun_data.energy = 2.2
    sun = bpy.data.objects.new("RS Sun", sun_data)
    sun.rotation_euler = (math.radians(50), 0.0, math.radians(35))
    scene.collection.objects.link(sun)
    build_world(scene)
    setup_viewports(scene)


def uv_sphere(rings=8, segments=16):
    verts = [(0.0, 0.0, 1.0)]
    for i in range(1, rings):
        phi = math.pi * i / rings
        for k in range(segments):
            a = 2 * math.pi * k / segments
            verts.append((math.sin(phi) * math.cos(a), math.sin(phi) * math.sin(a), math.cos(phi)))
    bottom = len(verts)
    verts.append((0.0, 0.0, -1.0))
    faces = []
    for k in range(segments):
        faces.append((0, 1 + k, 1 + (k + 1) % segments))
        faces.append((bottom, 1 + (rings - 2) * segments + (k + 1) % segments,
                      1 + (rings - 2) * segments + k))
    for i in range(rings - 2):
        for k in range(segments):
            a, b = 1 + i * segments + k, 1 + i * segments + (k + 1) % segments
            faces.append((a, a + segments, b + segments, b))
    return verts, faces


def build_ground(z):
    s = GROUND_SIZE
    obj = new_mesh_object("RS Ground", [(-s, -s, z), (s, -s, z), (s, s, z), (-s, s, z)],
                          [(0, 1, 2, 3)])
    mat = bpy.data.materials.new("RS Ground")
    mat.use_nodes = True
    nodes, links = mat.node_tree.nodes, mat.node_tree.links
    coord = nodes.new("ShaderNodeTexCoord")
    fine = nodes.new("ShaderNodeTexChecker")  # 10 m squares give a sense of scale
    fine.inputs["Scale"].default_value = 0.1
    fine.inputs["Color1"].default_value = (0.045, 0.11, 0.025, 1)
    fine.inputs["Color2"].default_value = (0.035, 0.09, 0.02, 1)
    links.new(coord.outputs["Object"], fine.inputs["Vector"])
    bsdf = nodes["Principled BSDF"]
    bsdf.inputs["Roughness"].default_value = 0.95
    links.new(fine.outputs["Color"], bsdf.inputs["Base Color"])
    obj.data.materials.append(mat)
    pad = new_mesh_object("RS Pad", *disc(0.35, z + 0.003), (material(
        "RS Pad", (0.3, 0.3, 0.32), roughness=0.8),))
    pad.hide_select = True


def disc(radius, z, segments=32):
    verts = [(radius * math.cos(2 * math.pi * k / segments),
              radius * math.sin(2 * math.pi * k / segments), z) for k in range(segments)]
    return verts, [tuple(range(segments))]


def build_rail(meta, ground_z):
    rail = meta["rail"]
    e = Vector((math.cos(rail["elevation"]) * math.sin(rail["heading"]),
                math.cos(rail["elevation"]) * math.cos(rail["heading"]),
                math.sin(rail["elevation"])))
    # the rod runs from the pad, up beside the rocket, for the simulated rail travel
    length_shown = rail["length"] + meta["length"] * 0.6
    verts, faces = revolve([(0.0, 0.004), (length_shown, 0.004)], 8)
    obj = new_mesh_object("RS Rail", verts, faces, (material(
        "RS Rail", (0.75, 0.75, 0.78), roughness=0.3, metallic=0.9),))
    obj.rotation_mode = "QUATERNION"
    obj.rotation_quaternion = Vector((-1, 0, 0)).rotation_difference(e)
    side = e.cross(Vector((0.0, 0.0, 1.0)))
    side = side.normalized() if side.length > 1e-6 else Vector((1.0, 0.0, 0.0))
    obj.location = Vector((0.0, 0.0, ground_z)) + side * (meta["radius"] + 0.006)


def build_world(scene):
    world = bpy.data.worlds.get("RS World") or bpy.data.worlds.new("RS World")
    world.use_nodes = True
    nodes, links = world.node_tree.nodes, world.node_tree.links
    nodes.clear()
    sky = nodes.new("ShaderNodeTexSky")
    sky.sky_type = "NISHITA"
    sky.sun_elevation = math.radians(62.0)
    sky.dust_density = 0.4
    sky.sun_disc = False
    background = nodes.new("ShaderNodeBackground")
    background.inputs["Strength"].default_value = 0.13
    output = nodes.new("ShaderNodeOutputWorld")
    links.new(sky.outputs["Color"], background.inputs["Color"])
    links.new(background.outputs["Background"], output.inputs["Surface"])
    scene.world = world


def set_trail_points(frames):
    curve = bpy.data.curves.get("RS Trail")
    if curve is None:
        return
    curve.splines.clear()
    # the simulated path, not the display pose, so lying down on landing leaves no kink
    path = np.c_[frames["x"], frames["y"], frames["z"]]
    points = path[:: max(1, len(path) // 1500)]
    spline = curve.splines.new("POLY")
    spline.points.add(len(points) - 1)
    spline.points.foreach_set("co", np.c_[points, np.ones(len(points))].ravel())
    curve.bevel_factor_mapping_end = "SPLINE"  # the factor is a fraction of the arc length
    curve.resolution_u = 1
    steps = np.linalg.norm(np.diff(path, axis=0), axis=1)
    travelled = np.r_[0.0, np.cumsum(steps)]
    frames["trail"] = travelled / max(travelled[-1], 1e-9)  # fraction of the path flown, per frame


def tidy_blender_ui():
    """Give the flight the whole window. Only this session; nothing is saved to Blender."""
    for cls in bpy.types.Panel.__subclasses__():
        if (getattr(cls, "bl_space_type", "") == "VIEW_3D"
                and getattr(cls, "bl_region_type", "") == "UI"
                and not issubclass(cls, SidebarPanel)):  # one tab left, so it is the open one
            try:
                bpy.utils.unregister_class(cls)
            except RuntimeError:  # child panels go with their parent
                pass
    for window in bpy.context.window_manager.windows:
        for kind in ("OUTLINER", "PROPERTIES"):
            for area in list(window.screen.areas):
                if area.type == kind:
                    with bpy.context.temp_override(window=window, area=area):
                        bpy.ops.screen.area_close()


def setup_viewports(scene):
    for window in bpy.context.window_manager.windows:
        for area in window.screen.areas:
            if area.type != "VIEW_3D":
                continue
            space = area.spaces.active
            space.camera = scene.camera
            space.region_3d.view_perspective = "CAMERA"
            space.region_3d.view_camera_zoom = 29.0
            space.region_3d.view_camera_offset = (0.0, 0.0)
            space.clip_end = 100000.0
            space.shading.type = "MATERIAL"
            space.shading.use_scene_lights = True
            space.shading.use_scene_world = True
            space.show_region_ui = True
            space.show_region_toolbar = False
            space.show_region_tool_header = False
            overlay = space.overlay
            overlay.show_floor = overlay.show_axis_x = overlay.show_axis_y = False
            overlay.show_cursor = overlay.show_object_origins = False
            overlay.show_outline_selected = overlay.show_extras = False
            overlay.show_text = overlay.show_relationship_lines = False
            overlay.show_annotation = False
    scene.unit_settings.system = "METRIC"
    scene.view_settings.view_transform = "Standard"  # keeps the rocket's colours saturated
    scene.view_settings.look = "None"


def update_scene(scene):
    """Put everything where this frame says. Runs on every frame change."""
    flight = STATE.flight
    objects = bpy.data.objects
    if flight is None or flight.frames is None or "RS Rocket" not in objects:
        return
    f = flight.frames
    i = min(max(scene.frame_current - 1, 0), len(f["t"]) - 1)
    meta = flight.meta
    rocket = objects["RS Rocket"]
    tip = Vector(f["loc"][i])
    q = Quaternion(f["rot"][i])
    rocket.location, rocket.rotation_quaternion = tip, q

    peak = max(meta["motor"]["peak_thrust"], 1e-9)
    flame_length = f["thrust"][i] / peak * meta["length"] * 1.3
    flame = objects["RS Flame"]
    flame.hide_viewport = flame.hide_render = flame_length < 1e-4
    width = 0.6 + 0.4 * f["thrust"][i] / peak
    flame.scale = (max(flame_length, 1e-4), width, width)

    chutes = meta["chutes"]
    shown = [(n, c, o) for n, c, o in (("RS Drogue", chutes["drogue_cda"], f["drogue_open"][i]),
                                       ("RS Main", chutes["main_cda"], f["main_open"][i]))]
    both = sum(o > 0 for _, _, o in shown) == 2
    for sign, (name, cda, opened) in zip((1, -1), shown):
        obj = objects[name]
        visible = opened > 0 and f["collapse"][i] < 1.0
        obj.hide_viewport = obj.hide_render = not visible
        if visible:
            radius = math.sqrt(cda / (VISUAL_CD * math.pi)) if cda else 0.0
            s = radius * math.sqrt(opened)
            flat = 1.0 - f["collapse"][i]
            obj.location = tip
            obj.scale = (s * (1 + 0.3 * f["collapse"][i]), s * (1 + 0.3 * f["collapse"][i]),
                         s * max(flat, 0.02))
            obj.rotation_euler = (sign * 0.35 if both else 0.0, 0.0, 0.0)

    bpy.data.curves["RS Trail"].bevel_factor_end = f["trail"][i]
    objects["RS Trail"].hide_viewport = f["trail"][i] <= 0.0

    center = tip + q @ Vector((-meta["length"] / 2, 0.0, 0.0))
    objects["RS Focus"].location = center
    # with a canopy out, the follow camera backs off and looks up to fit it in
    biggest = max(meta["chutes"]["drogue_cda"], meta["chutes"]["main_cda"])
    opened = max(f["drogue_open"][i], f["main_open"][i]) * (1.0 - f["collapse"][i])
    canopy = 3.0 * math.sqrt(biggest / (VISUAL_CD * math.pi)) * opened
    if scene.rocketsim.camera_mode == "FOLLOW":
        objects["RS Focus"].location = center + Vector((0.0, 0.0, canopy * 0.5))
    place_camera(scene, objects["RS Camera"], center, f, i, meta, objects["RS Focus"], canopy)


def place_camera(scene, cam, center, f, i, meta, focus, canopy):
    mode = scene.rocketsim.camera_mode
    z0 = f["ground"]
    length_ = meta["length"]
    if mode == "FOLLOW":
        reach = length_ + canopy
        offset = Vector((reach * 3.0, -reach * 4.0, reach * 0.8))
        pos = center + offset
        pos.z = max(pos.z, z0 + 0.25)
        lens = 35.0
    elif mode == "OVERVIEW":
        loc = f["loc"]
        low, high = loc.min(axis=0), loc.max(axis=0)
        middle = Vector(((low + high) / 2).tolist())
        size = max(float((high - low).max()), 30.0)
        pos = middle + Vector((0.2, -1.0, 0.18)) * size * 1.9
        pos.z = max(pos.z, z0 + 2.0)
        focus.location = middle
        lens = 35.0
    else:
        pos = Vector((6.0, -14.0, z0 + 1.7))
        distance = (center - pos).length
        lens = min(max(0.12 * distance * 36.0 / (length_ + canopy), 28.0), 1200.0)
        if canopy:
            focus.location = center + Vector((0.0, 0.0, canopy * 0.5))
    cam.location = pos
    cam.data.lens = lens
    marker = bpy.data.objects["RS Marker"]
    distance = (center - pos).length
    # a trail of constant apparent width, whatever the camera distance and zoom
    bpy.data.curves["RS Trail"].bevel_depth = max(0.002, distance * 0.0012 * 35.0 / lens)
    size = distance * 0.0045 if mode == "OVERVIEW" else 0.0  # too small to see from that far
    marker.hide_viewport = marker.hide_render = size == 0.0
    marker.location = center
    marker.scale = (size, size, size)


def frame_handler(scene, depsgraph=None):
    update_scene(scene)
    flight = STATE.flight
    if (flight is not None and scene.frame_current >= scene.frame_end
            and bpy.context.screen is not None and bpy.context.screen.is_animation_playing):
        bpy.app.timers.register(stop_animation, first_interval=0.0)


def stop_animation():
    with_view3d(lambda: bpy.ops.screen.animation_cancel(restore_frame=False))


def play_animation():
    def start():
        if not bpy.context.screen.is_animation_playing:
            bpy.ops.screen.animation_play()
    with_view3d(start)


def with_view3d(action):
    for window in bpy.context.window_manager.windows:
        for area in window.screen.areas:
            if area.type == "VIEW_3D":
                with bpy.context.temp_override(window=window, area=area):
                    action()
                return


def redraw_all():
    for window in bpy.context.window_manager.windows:
        for area in window.screen.areas:
            area.tag_redraw()


# --- live data on screen ------------------------------------------------------------------------

PHASE_LABEL = {"pad": "ON THE PAD", "rail": "ON THE RAIL", "powered": "POWERED FLIGHT",
               "coast": "COASTING", "freefall": "FREE FALL", "drogue": "UNDER DROGUE",
               "main": "UNDER MAIN", "landed": "LANDED"}
PHASE_COLOR = {"pad": (0.8, 0.8, 0.8), "rail": (1.0, 0.8, 0.3), "powered": (1.0, 0.5, 0.15),
               "coast": (0.4, 0.8, 1.0), "freefall": (1.0, 0.4, 0.4), "drogue": (1.0, 0.6, 0.3),
               "main": (0.4, 0.6, 1.0), "landed": (0.5, 1.0, 0.5)}


def _rect(shader, x, y, w, h, color):
    batch = batch_for_shader(shader, "TRI_FAN", {"pos": [(x, y), (x + w, y), (x + w, y + h),
                                                         (x, y + h)]})
    shader.uniform_float("color", color)
    batch.draw(shader)


def _line(shader, points, color):
    batch = batch_for_shader(shader, "LINE_STRIP", {"pos": points})
    shader.uniform_float("color", color)
    batch.draw(shader)


def _text(font, x, y, text, size, color=(1, 1, 1, 1)):
    blf.size(font, size)
    blf.color(font, *color)
    blf.position(font, x, y, 0)
    blf.draw(font, text)


def _fmt(value, pattern):
    if value is None or math.isnan(value):
        return "--"
    text = pattern % value
    return pattern % 0.0 if float(text) == 0.0 else text  # never "-0.0"


def draw_hud():
    """Telemetry panel and strip charts over the 3-D view, updated every frame."""
    context = bpy.context
    if context.area is None or context.space_data.type != "VIEW_3D":
        return
    scale = context.preferences.system.ui_scale
    region = context.region
    shader = gpu.shader.from_builtin("UNIFORM_COLOR")
    gpu.state.blend_set("ALPHA")
    font = 0
    pad = 14 * scale
    x0, top = pad, region.height - pad
    flight = STATE.flight

    if STATE.proc is not None or STATE.error or flight is None or flight.frames is None:
        message = STATE.error or STATE.status or ""
        if message:
            w, h = min(region.width - 2 * pad, 760 * scale), 46 * scale
            _rect(shader, x0, top - h, w, h, (0.02, 0.03, 0.05, 0.9))
            color = (1.0, 0.45, 0.4, 1) if STATE.error else (1, 1, 1, 1)
            _text(font, x0 + pad, top - h + 15 * scale, message[:110], 15 * scale, color)
        if flight is None or flight.frames is None:
            gpu.state.blend_set("NONE")
            return
        top -= 54 * scale

    f = flight.frames
    i = min(max(context.scene.frame_current - 1, 0), len(f["t"]) - 1)
    phase = str(f["phase"][i])
    t = f["t"][i]
    rate = f["rate"][i]
    rows = [
        ("Altitude", _fmt(f["z"][i], "%.1f"), "m"),
        ("Vertical speed", _fmt(f["vz"][i], "%+.1f"), "m/s"),
        ("Speed", _fmt(f["speed"][i], "%.1f"), "m/s"),
        ("Mach", _fmt(f["mach"][i], "%.3f"), ""),
        ("Acceleration", _fmt(f["accel_g"][i], "%.1f"), "g"),
        ("Thrust", _fmt(f["thrust"][i], "%.1f"), "N"),
        ("Mass", _fmt(f["mass"][i] * 1000.0, "%.1f"), "g"),
        ("Angle of attack", _fmt(math.degrees(f["alpha"][i]), "%.1f"), "deg"),
        ("Stability margin", _fmt(f["margin"][i], "%.2f"), "cal"),
        ("Distance from pad", _fmt(f["distance"][i], "%.1f"), "m"),
    ]
    row_h, width = 23 * scale, 300 * scale
    height = 84 * scale + row_h * len(rows)
    _rect(shader, x0, top - height, width, height, (0.02, 0.03, 0.05, 0.82))
    color = PHASE_COLOR.get(phase, (1, 1, 1))
    _rect(shader, x0, top - 5 * scale, width, 5 * scale, (*color, 1.0))
    _text(font, x0 + pad, top - 36 * scale, PHASE_LABEL.get(phase, phase.upper()), 19 * scale,
          (*color, 1.0))
    clock = f"T{'+' if t >= 0 else '-'}{abs(t):06.2f} s"
    if rate > 1:
        clock += f"   ({rate:.0f}x)"
    _text(font, x0 + pad, top - 64 * scale, clock, 17 * scale)
    y = top - 94 * scale
    for label, value, unit in rows:
        _text(font, x0 + pad, y, label, 14 * scale, (0.72, 0.72, 0.72, 1))
        blf.size(font, 16 * scale)
        tw = blf.dimensions(font, value)[0]
        _text(font, x0 + width - 52 * scale - tw, y, value, 16 * scale)
        _text(font, x0 + width - 46 * scale, y, unit, 13 * scale, (0.72, 0.72, 0.72, 1))
        y -= row_h

    chart_w = min(420 * scale, region.width * 0.36)
    chart_h = 104 * scale
    arrays = [("Altitude [m]", f["z"], (0.45, 0.85, 1.0, 1)),
              ("Vertical speed [m/s]", f["vz"], (1.0, 0.65, 0.25, 1))]
    # the charts cover the flight down to shortly after the chute opens, then the full flight
    n = len(f["t"])
    y0 = pad + 8 * scale
    for title, series, color in arrays:
        _rect(shader, x0, y0, chart_w, chart_h, (0.02, 0.03, 0.05, 0.82))
        step = max(1, n // 400)
        low, high = float(series.min()), float(series.max())
        span = (high - low) or 1.0
        margin = 8 * scale
        fx = (chart_w - 2 * margin) / max(n - 1, 1)
        fy = (chart_h - 30 * scale) / span
        pts = [(x0 + margin + k * fx, y0 + margin + (float(series[k]) - low) * fy)
               for k in range(0, min(i + 1, n), step)]
        pts.append((x0 + margin + i * fx, y0 + margin + (float(series[i]) - low) * fy))
        if low < 0 < high:
            zero = y0 + margin - low * fy
            _line(shader, [(x0 + margin, zero), (x0 + chart_w - margin, zero)],
                  (0.5, 0.5, 0.5, 0.7))
        if len(pts) > 1:
            _line(shader, pts, color)
        marker_x, marker_y = pts[-1]
        _rect(shader, marker_x - 2.5 * scale, marker_y - 2.5 * scale, 5 * scale, 5 * scale, color)
        _text(font, x0 + margin, y0 + chart_h - 19 * scale, title, 13 * scale,
              (0.8, 0.8, 0.8, 1))
        label = _fmt(float(series[i]), "%.1f")
        blf.size(font, 15 * scale)
        _text(font, x0 + chart_w - margin - blf.dimensions(font, label)[0],
              y0 + chart_h - 20 * scale, label, 15 * scale, color)
        y0 += chart_h + 8 * scale
    gpu.state.blend_set("NONE")


# --- buttons and panels -------------------------------------------------------------------------

class RS_OT_fly(bpy.types.Operator):
    bl_idname = "rocketsim.fly"
    bl_label = "Simulate & Fly"
    bl_description = "Run the simulator on these values, then fly the result"

    def execute(self, context):
        for attr in ("animation_cancel",):
            if context.screen.is_animation_playing:
                getattr(bpy.ops.screen, attr)(restore_frame=False)
        context.scene.frame_set(1)
        start_simulation(context.scene, autoplay=True)
        return {"FINISHED"}


class RS_OT_replay(bpy.types.Operator):
    bl_idname = "rocketsim.replay"
    bl_label = "Replay"
    bl_description = "Watch the last simulated flight again from the start"

    @classmethod
    def poll(cls, context):
        return STATE.flight is not None

    def execute(self, context):
        if context.screen.is_animation_playing:
            bpy.ops.screen.animation_cancel(restore_frame=False)
        context.scene.frame_set(1)
        bpy.ops.screen.animation_play()
        return {"FINISHED"}


class RS_OT_pause(bpy.types.Operator):
    bl_idname = "rocketsim.pause"
    bl_label = "Play / Pause"

    @classmethod
    def poll(cls, context):
        return STATE.flight is not None

    def execute(self, context):
        bpy.ops.screen.animation_play()
        return {"FINISHED"}


class RS_OT_mass_add(bpy.types.Operator):
    bl_idname = "rocketsim.mass_add"
    bl_label = "Add mass"

    def execute(self, context):
        p = context.scene.rocketsim
        item = p.masses.add()
        item.name, item.mass, item.position = "payload", 0.001, p.body_length / 2 + p.nose_length
        p.masses_index = len(p.masses) - 1
        return {"FINISHED"}


class RS_OT_mass_remove(bpy.types.Operator):
    bl_idname = "rocketsim.mass_remove"
    bl_label = "Remove mass"

    @classmethod
    def poll(cls, context):
        return len(context.scene.rocketsim.masses) > 0

    def execute(self, context):
        p = context.scene.rocketsim
        p.masses.remove(p.masses_index)
        p.masses_index = min(p.masses_index, max(len(p.masses) - 1, 0))
        return {"FINISHED"}


class RS_OT_load(bpy.types.Operator, ImportHelper):
    bl_idname = "rocketsim.load"
    bl_label = "Load Rocket"
    bl_description = "Fill the panel from a RocketSim config file"
    filename_ext = ".yaml"
    filter_glob: StringProperty(default="*.yaml;*.yml", options={"HIDDEN"})

    def execute(self, context):
        try:
            props_from_config(context.scene.rocketsim, read_config(self.filepath),
                              Path(self.filepath).parent)
        except (RuntimeError, KeyError, TypeError, ValueError) as exc:
            self.report({"ERROR"}, f"Could not load this config: {exc}")
            return {"CANCELLED"}
        return {"FINISHED"}


class RS_OT_save(bpy.types.Operator, ExportHelper):
    bl_idname = "rocketsim.save"
    bl_label = "Save Rocket"
    bl_description = "Save the panel's values as a RocketSim config file"
    filename_ext = ".yaml"
    filter_glob: StringProperty(default="*.yaml;*.yml", options={"HIDDEN"})

    def execute(self, context):
        cfg = config_from_props(context.scene.rocketsim) | STATE.extras
        Path(self.filepath).write_text(json.dumps(cfg, indent=2), encoding="utf-8")
        self.report({"INFO"}, f"Saved {self.filepath}")
        return {"FINISHED"}


class RS_UL_masses(bpy.types.UIList):
    def draw_item(self, context, layout, data, item, icon, active_data, active_propname):
        layout.prop(item, "name", text="", emboss=False)
        layout.label(text=f"{item.mass * 1000:.1f} g")


class SidebarPanel:
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "RocketSim"


class RS_PT_main(SidebarPanel, bpy.types.Panel):
    bl_label = "RocketSim"

    def draw(self, context):
        layout, p = self.layout, context.scene.rocketsim
        col = layout.column(align=True)
        col.scale_y = 1.8
        busy = STATE.proc is not None
        col.enabled = not busy
        col.operator("rocketsim.fly", text="Simulating..." if busy else "Simulate & Fly",
                     icon="PLAY")
        row = layout.row(align=True)
        row.operator("rocketsim.replay", icon="LOOP_BACK")
        row.operator("rocketsim.pause", text="", icon="PAUSE")
        layout.prop(p, "camera_mode")
        layout.prop(p, "descent_speed")
        row = layout.row(align=True)
        row.operator("rocketsim.load", icon="FILE_FOLDER", text="Load")
        row.operator("rocketsim.save", icon="FILE_TICK", text="Save")
        if STATE.error:
            box = layout.box()
            box.alert = True
            box.label(text="Simulation failed", icon="ERROR")
            for start in range(0, min(len(STATE.error), 210), 42):
                box.label(text=STATE.error[start:start + 42])
        if STATE.flight:
            for message in STATE.flight.meta["warnings"]:
                box = layout.box()
                box.label(text=message[:42], icon="ERROR")


class RS_PT_results(SidebarPanel, bpy.types.Panel):
    bl_label = "Flight results"
    bl_parent_id = "RS_PT_main"
    bl_options = {"DEFAULT_CLOSED"}  # noqa: RUF012 (Blender's API uses a class attribute)

    @classmethod
    def poll(cls, context):
        return STATE.flight is not None

    def draw(self, context):
        s = STATE.flight.meta["summary"]
        ev = STATE.flight.meta["events"]
        col = self.layout.column(align=True)
        rows = [("Apogee", f"{s['apogee']:.1f} m"),
                ("Time to apogee", f"{s['time_to_apogee']:.2f} s"),
                ("Max speed", f"{s['max_velocity']:.1f} m/s"),
                ("Max Mach", f"{s['max_mach']:.3f}"),
                ("Max acceleration", f"{s['max_accel_g']:.1f} g"),
                ("Rail-exit speed", f"{s['rail_exit_velocity']:.1f} m/s"),
                ("Stability at rail exit", f"{s['margin_rail_exit']:.2f} cal"),
                ("Opening shock", f"{s['opening_shock_g']:.1f} g"),
                ("Landing speed", f"{s['landing_velocity']:.2f} m/s"),
                ("Landing distance", f"{s['drift']:.1f} m"),
                ("Flight time", f"{s['flight_time']:.1f} s"),
                ("Burnout", f"T+{ev['burnout']:.2f} s")]
        for label, value in rows:
            split = col.split(factor=0.62)
            split.label(text=label)
            split.label(text=value)


def _section(label, draw_fn, closed=True):
    """Make one collapsible input section under the main panel."""
    name = "RS_PT_" + "".join(c for c in label if c.isalnum())

    def draw(self, context):
        self.layout.use_property_split = True
        self.layout.use_property_decorate = False
        draw_fn(self.layout, context.scene.rocketsim)

    return type(name, (SidebarPanel, bpy.types.Panel),
                {"bl_label": label, "bl_parent_id": "RS_PT_main",
                 "bl_options": {"DEFAULT_CLOSED"} if closed else set(), "draw": draw})


def _props(layout, p, names):
    for name in names:
        layout.prop(p, name)


def _draw_masses(layout, p):
    layout.use_property_split = False
    row = layout.row()
    row.template_list("RS_UL_masses", "", p, "masses", p, "masses_index", rows=3)
    col = row.column(align=True)
    col.operator("rocketsim.mass_add", text="", icon="ADD")
    col.operator("rocketsim.mass_remove", text="", icon="REMOVE")
    if p.masses and p.masses_index < len(p.masses):
        layout.use_property_split = True
        item = p.masses[p.masses_index]
        _props(layout, item, ("name", "mass", "position", "length", "radius"))


def _draw_drag(layout, p):
    layout.prop(p, "drag_mode")
    if p.drag_mode == "constant":
        layout.prop(p, "cd_value")
    elif p.drag_mode == "table":
        layout.prop(p, "cd_table")
    layout.prop(p, "roughness")


def _draw_recovery(layout, p):
    layout.label(text="Drogue (or your only parachute)")
    _props(layout, p, ("drogue_cda", "drogue_fill", "drogue_delay"))
    layout.prop(p, "use_main")
    if p.use_main:
        _props(layout, p, ("main_cda", "main_fill", "main_altitude"))


SECTIONS = [
    _section("Nose cone", lambda lay, p: _props(lay, p, ("nose_shape", "nose_length", "nose_mass")),
             closed=False),
    _section("Body tube", lambda lay, p: _props(lay, p, ("body_length", "body_diameter",
                                                         "body_mass")), closed=False),
    _section("Fins", lambda lay, p: _props(lay, p, (
        "fin_count", "fin_root", "fin_tip", "fin_span", "fin_sweep", "fin_thickness",
        "fin_position", "fin_profile", "fin_mass"))),
    _section("Extra masses", _draw_masses),
    _section("Motor", lambda lay, p: _props(lay, p, ("motor_file", "motor_position")),
             closed=False),
    _section("Drag", _draw_drag),
    _section("Recovery", _draw_recovery),
    _section("Launch", lambda lay, p: _props(lay, p, ("site_altitude", "rail_length",
                                                      "rail_elevation", "rail_heading")),
             closed=False),
    _section("Wind", lambda lay, p: _props(lay, p, ("wind_speed", "wind_direction", "wind_height",
                                                    "wind_profile")), closed=False),
]
CLASSES = (RSMassItem, RSProps, RS_OT_fly, RS_OT_replay, RS_OT_pause, RS_OT_mass_add,
           RS_OT_mass_remove, RS_OT_load, RS_OT_save, RS_UL_masses, RS_PT_main, *SECTIONS,
           RS_PT_results)


def register():
    for cls in CLASSES:
        bpy.utils.register_class(cls)
    bpy.types.Scene.rocketsim = PointerProperty(type=RSProps)
    if frame_handler not in bpy.app.handlers.frame_change_post:
        bpy.app.handlers.frame_change_post.append(frame_handler)
    STATE.draw = bpy.types.SpaceView3D.draw_handler_add(draw_hud, (), "WINDOW", "POST_PIXEL")


def unregister():
    bpy.types.SpaceView3D.draw_handler_remove(STATE.draw, "WINDOW")
    if frame_handler in bpy.app.handlers.frame_change_post:
        bpy.app.handlers.frame_change_post.remove(frame_handler)
    del bpy.types.Scene.rocketsim
    for cls in reversed(CLASSES):
        bpy.utils.unregister_class(cls)


def startup(config_path=DEFAULT_CONFIG, autoplay=False):
    """Fill the panel from a config and simulate it, so the rocket is on the pad right away."""
    scene = bpy.context.scene
    props_from_config(scene.rocketsim, read_config(config_path), Path(config_path).parent)
    scene.render.engine = "BLENDER_EEVEE_NEXT"
    if not bpy.app.background:
        tidy_blender_ui()
    start_simulation(scene, autoplay)


def _command_line_config():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    return Path(argv[0]) if argv else DEFAULT_CONFIG


if __name__ == "__main__":
    register()
    if not bpy.app.background:
        bpy.app.timers.register(lambda: startup(_command_line_config()), first_interval=0.3)
