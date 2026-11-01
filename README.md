# RocketSim

6-DOF model-rocket flight simulator in Python, built stage by stage (see `PLAN.md`). It simulates a rocket on its launch rail and in powered and coasting flight, then its descent under a drogue and an optional main parachute, in wind. It prints a summary and saves the trajectory and plots.

## Setup (once)

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows PowerShell; Linux/macOS: source .venv/bin/activate
pip install -e ".[dev]"         # add ,validation for the RocketPy cross-check tests
```

In VS Code: open the folder, then choose *Python: Select Interpreter* → `.venv`.

## Run a flight

```bash
python -m rocketsim run configs/example_rocket.yaml
```

This prints apogee, velocities, rail-exit speed, stability margin, descent rates, opening shock and landing point. It also writes `outputs/<config name>/`:

| File | Contents |
|---|---|
| `trajectory.csv` | Truth state every 0.01 s: position, velocity, quaternion, angular rate, acceleration, α, Mach, margin, mass, phase |
| `flight.png` | Altitude, speed, acceleration, angle of attack, stability margin, ground track |
| `trajectory_3d.png` | 3-D path in East-North-Up |

## Monte Carlo (many flights with uncertain inputs)

```bash
python -m rocketsim montecarlo configs/example_rocket.yaml -n 500
```

This varies the inputs listed in the config's `dispersions:` section and flies all runs in parallel. It prints the apogee spread, rail-exit speed and stability-margin ranges, and the 1/2/3-σ landing ellipses. It saves `montecarlo.csv` (every run's inputs and results) and `montecarlo.png`.

`--seed` makes results repeatable; the same seed always gives the same runs. `--workers` limits how many processes run at once. 500 runs take about 3 minutes on a 4-core laptop.

## Fly it in Blender

Needs [Blender](https://www.blender.org/) 4.5. Double-click `Start RocketSim in Blender.bat`, or:

```bash
python -m rocketsim blender
```

Type in the rocket's details, press **Simulate & Fly**, and watch it fly and land with live data on screen. See [docs/BLENDER.md](docs/BLENDER.md).

## Sensor data for a flight computer

```bash
python -m rocketsim sensors configs/example_rocket.yaml
```

Writes simulated IMU and barometer readings in the format in [docs/SENSOR_SCHEMA.md](docs/SENSOR_SCHEMA.md).

## Validation

Results against OpenRocket and RocketPy are in [docs/VALIDATION.md](docs/VALIDATION.md).

## Make your own rocket

Copy `configs/example_rocket.yaml` and edit it. Every key is required unless marked optional, and typos are rejected with a message naming the key. Units are SI, angles are in degrees, and axial positions are measured from the nose tip, positive aft.

- **Motor:** download a RASP `.eng` file from thrustcurve.org into `motors/` and point `motor:` at it.
- **Drag:** `buildup` uses the built-in model (Niskanen 2009). A number gives a constant Cd. A path to a `mach,cd` CSV uses that table, for example one exported from OpenRocket.
- **Recovery:** `drogue` is required (a single-chute rocket uses only it); `main` is optional.

## Test and lint

```bash
pytest -q
ruff check .
```

What each stage built and why is in `docs/NOTES_stage*.md`.
