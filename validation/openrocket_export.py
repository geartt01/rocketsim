r"""Run OpenRocket's own engine on its bundled "A simple model rocket" and export the results.

Not part of the simulator and not imported by it. Needs OpenRocket 23.09 installed and JPype in
a separate environment (it is not a project dependency):

    python -m venv orvenv && orvenv\Scripts\pip install jpype1
    orvenv\Scripts\python validation\openrocket_export.py

Writes reference/openrocket/: <case>.csv flight data, simple_rocket_C6.eng (OpenRocket's own
motor curve), and components.json (masses, positions, motor mounting).
"""

import io
import json
import re
import sys
import tempfile
import time
import zipfile
from pathlib import Path

import jpype

JAR = r"C:\Program Files\OpenRocket\OpenRocket.jar"
JVM = r"C:\Program Files\OpenRocket\jre\bin\server\jvm.dll"
OUT = Path(__file__).parents[1] / "reference" / "openrocket"
SIM_INDEX = 4  # "Simulation 5" in the example: Estes C6-7, charge fires well after apogee
COLUMNS = {
    "t": "TYPE_TIME", "z": "TYPE_ALTITUDE", "vz": "TYPE_VELOCITY_Z",
    "az": "TYPE_ACCELERATION_Z", "v": "TYPE_VELOCITY_TOTAL", "mach": "TYPE_MACH_NUMBER",
    "cd": "TYPE_DRAG_COEFF", "cp": "TYPE_CP_LOCATION", "cg": "TYPE_CG_LOCATION",
    "mass": "TYPE_MASS", "thrust": "TYPE_THRUST_FORCE", "x": "TYPE_POSITION_X",
    "y": "TYPE_POSITION_Y", "margin": "TYPE_STABILITY", "aoa": "TYPE_AOA",
    "i_long": "TYPE_LONGITUDINAL_INERTIA",
}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    data = zipfile.ZipFile(JAR).read("datafiles/examples/A simple model rocket.ork")
    ork = Path(tempfile.gettempdir()) / "a_simple_model_rocket.ork"
    ork.write_bytes(data)
    jpype.startJVM(JVM, "-Djava.awt.headless=true", classpath=[JAR], convertStrings=True)
    pkg = jpype.JPackage("net").sf.openrocket
    gui = pkg.startup.GuiModule()
    injector = jpype.JPackage("com").google.inject.Guice.createInjector(
        gui, pkg.plugin.PluginModule())
    pkg.startup.Application.setInjector(injector)
    gui.startLoader()
    time.sleep(8)  # headless: loading a rocket before the motor database is ready crashes

    doc = pkg.file.GeneralRocketLoader(jpype.java.io.File(str(ork))).load()
    sim = doc.getSimulation(SIM_INDEX)
    types = pkg.simulation.FlightDataType
    export_components(doc, sim, data)
    export_motor(sim)
    for case, wind in (("calm", 0.0), ("wind2", 2.0)):
        options = sim.getOptions()
        options.setWindSpeedAverage(wind)
        options.setWindTurbulenceIntensity(0.0)
        options.setTimeStep(0.005)
        sim.simulate(jpype.JArray(pkg.simulation.listeners.SimulationListener)(0))
        branch = sim.getSimulatedData().getBranch(0)
        arrays = {k: list(branch.get(getattr(types, v))) for k, v in COLUMNS.items()}
        with open(OUT / f"{case}.csv", "w", encoding="utf-8") as f:
            f.write(",".join(arrays) + "\n")
            f.writelines(",".join(f"{float(v):.8g}" for v in row) + "\n" for row in zip(*arrays.values()))
        events = [(str(e.getType()), float(e.getTime())) for e in branch.getEvents()]
        (OUT / f"{case}_events.json").write_text(json.dumps(events, indent=1), encoding="utf-8")
        print(case, "max altitude", float(sim.getSimulatedData().getMaxAltitude()))
    jpype.shutdownJVM()


def export_components(doc, sim, data):
    coordinate = jpype.JPackage("net").sf.openrocket.util.Coordinate
    parts = []
    it = doc.getRocket().iterator(True)
    while it.hasNext():
        c = it.next()
        top = float(c.toAbsolute(coordinate.ZERO)[0].x)
        parts.append({"name": str(c.getName()), "kind": str(c.getClass().getSimpleName()),
                      "mass": float(c.getMass()), "cg": top + float(c.getComponentCG().x),
                      "top": top, "length": float(c.getLength())})
    xml = zipfile.ZipFile(io.BytesIO(data)).read("rocket.ork").decode("utf-8")
    mount = re.search(r"<motormount>.*?</motormount>", xml, re.DOTALL).group(0)
    overhang = float(re.search(r"<overhang>(.*?)</overhang>", mount).group(1))
    inner = next(p for p in parts if p["kind"] == "InnerTube")
    motor = next(iter(sim.getActiveConfiguration().getActiveMotors())).getMotor()
    result = {"parts": parts, "motor_aft_end": inner["top"] + inner["length"] + overhang,
              "motor_length": float(motor.getLength()), "motor_diameter": float(motor.getDiameter())}
    (OUT / "components.json").write_text(json.dumps(result, indent=1), encoding="utf-8")


def export_motor(sim):
    motor = next(iter(sim.getActiveConfiguration().getActiveMotors())).getMotor()
    times = list(motor.getTimePoints())
    thrust = list(motor.getThrustPoints())
    launch, burnout = float(motor.getLaunchMass()), float(motor.getBurnoutMass())
    lines = [f"; {motor.getDesignation()} thrust curve as used by OpenRocket 23.09",
             (f"C6 {motor.getDiameter() * 1000:.1f} {motor.getLength() * 1000:.1f} 7 "
             f"{launch - burnout:.5f} {launch:.5f} Estes")]
    lines += [f"{float(t):.4f} {float(f):.4f}" for t, f in zip(times, thrust) if float(t) > 0]
    (OUT / "simple_rocket_C6.eng").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
