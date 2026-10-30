# RocketSim

6-DOF model-rocket flight simulator in Python, built stage by stage (see `PLAN.md`).

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows; on Linux/macOS: source .venv/bin/activate
pip install -e ".[dev]"
```

## Test and lint

```bash
pytest -q
ruff check .
```

## Run (Stage 1+)

```bash
python -m rocketsim run configs/example_rocket.yaml
```

Motor files (`.eng`, RASP format) go in `motors/`; download them from thrustcurve.org.
