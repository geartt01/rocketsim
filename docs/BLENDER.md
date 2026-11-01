# RocketSim in Blender

Type in a rocket, press **Simulate & Fly**, and watch the modelled rocket fly and land with live data on screen.

## Start it

Double-click **`Start RocketSim in Blender.bat`** in the project folder, or run:

```bash
python -m rocketsim blender
```

Blender opens with the example rocket on the pad. To open another rocket: `python -m rocketsim blender configs/simple_rocket.yaml`.

## Use it

The **RocketSim** sidebar is on the right (press `N` if it's hidden).

1. Edit any value: nose, body, fins, extra masses, motor file, drag, parachutes, launch rail, wind.
2. Press **Simulate & Fly**. It takes a few seconds, then the flight plays.
3. Watch the data panel and charts on the left. **Flight results** at the bottom of the sidebar has the summary.

| Control | What it does |
|---|---|
| Camera | From the ground (zooms to follow), Follow rocket, or Whole flight |
| Descent | Real time, or 4× / 10× to fast-forward the parachute descent; the ascent stays real time |
| Replay / Pause | Replays the last flight; the timeline at the bottom also scrubs through it |
| Load / Save | Reads or writes a RocketSim config file |

If a value can't fly, the reason appears in red and the previous flight stays on screen.

## How it works

Blender's Python has no scipy, so the add-on doesn't contain any physics. It writes your values to a config file, runs the same validated simulator as the command line (`python -m rocketsim flightdata`) with the project's own Python, reads the result, and plays it back. The rocket's mesh is built from the outline and fin sizes the simulator returns.

**"Real time" means playback.** The whole flight is computed first, then played at true speed with its data in sync. Changing a value mid-flight has no effect until you press the button again.

## Shown for looks, not simulated

- Under a parachute the simulator treats the rocket as a point. On screen it hangs nose-up and lies down on landing.
- Canopy size assumes Cd 0.8. The flame's size follows thrust.
- The rail is drawn beside the rocket, and the ground is drawn at the rocket's tail.

The numbers on screen come from the simulator directly, not from these display choices.

## Tests

`tests/test_blender.py` runs `tests/blender_selftest.py` inside real Blender. It checks that the rocket's position and attitude match the simulated flight, that flame, chutes and trail appear at the right times, that the panel's values give the same flight as the config file, that a bad value is reported, and that main chutes and extra masses work.

Settings stay in `outputs/blender/config`, so any other use of Blender on the machine isn't affected.
