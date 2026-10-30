# RocketSim — instructions for Claude Code

6-DOF model-rocket flight simulator in Python. The full spec and stage plan are in `PLAN.md`; read it at the start of every session.

## Who I am / what I want

* I'm building this to learn rocket flight physics and as a portfolio project for student rocketry teams and NewSpace internships. I must be able to explain every equation.
* Treat me as an experienced programmer.

## Rules

1. Work on one stage at a time (PLAN.md §4). Stop at the end of a stage for my review. Never start the next stage on your own.
2. Tests first for physics: write the invariant tests listed for the stage before (or alongside) the implementation. A stage is done only when `pytest` is green and `ruff check .` is clean.
3. Conventions in PLAN.md §0 are fixed (SI units, ENU inertial frame, body x toward nose, axial positions from nose tip positive aft, scalar-first quaternion body→inertial). Ask before changing any of them.
4. Every physics function gets a docstring with: the equation, units of every input/output, assumptions, and the source (e.g. "Niskanen 2009, §3.2.2" or "Barrowman 1967"). If you're not sure a formula is right, say so and ask. Don't guess silently.
5. No new dependencies beyond PLAN.md §1 without asking.
6. No rocket-specific numbers in code; they belong in `configs/*.yaml`.
7. All randomness through a seeded `numpy.random.Generator` passed in explicitly.
8. `reference/sim_1dof.py` is a test oracle. Never edit it.
9. When a result looks wrong: explain the cause with evidence before changing code. No tolerance-loosening to make tests pass without telling me why.
10. Commit after each task that leaves tests green, with a clear message. Don't push unless I ask.
11. At the end of each stage, write `docs/NOTES_stageN.md` (max ~1 page): the physics in plain language, the design decisions, known simplifications, and what to check next.

## Commands

* Install: `pip install -e ".[dev]"`
* Test: `pytest -q`
* Lint: `ruff check .`
* Run: `python -m rocketsim run configs/example_rocket.yaml`
* Monte Carlo (Stage 5+): `python -m rocketsim montecarlo configs/example_rocket.yaml -n 500`
* Sensors (Stage 6+): `python -m rocketsim sensors configs/example_rocket.yaml`
