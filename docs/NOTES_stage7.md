# Stage 7 notes — validation

The numbers and plots are in [VALIDATION.md](VALIDATION.md). These notes cover the reasoning.

## Physics in plain language

**Why compare in layers.** A simulator's apogee can be right because two errors cancel. So the comparison separates them:

1. **Dynamics only:** give RocketSim OpenRocket's Cd table. Any gap left comes from the equations of motion, not the aerodynamics. Apogee differs by −0.01 %.
2. **Aerodynamics directly:** compare Cd, CP and mass properties with no flight involved. CP matches to 4 digits. Cd is about 5 % low.
3. **Everything together:** own aerodynamics and dynamics. Apogee is +4.03 %.

Layer 1 shows the +4.03 % isn't a dynamics error, and layer 2 finds its cause.

**Why low drag raises apogee by less than 5 %.** Drag is only part of what slows the rocket on the way up; gravity does the rest. So 5 % less Cd gives about 4 % more apogee here.

**The cause of the gap.** OpenRocket's drag breakdown at Mach 0.2 has 0.039 of Cd from the launch lug, which RocketSim doesn't model. The other parts agree closely.

## Design decisions

- **Reference data comes from OpenRocket's own engine,** driven through JPype in a separate environment. JPype isn't a project dependency; the results are stored in `reference/openrocket/`, so the tests don't need OpenRocket.
- **C6-7, not C6-5.** The bundled example's C6-5 ejects before apogee, so it wouldn't test apogee at all.
- **Descent rate is averaged over the last 10 s,** because OpenRocket's final samples wobble.
- **Lug drag was not added.** Niskanen's eq. 3.95 gives about 0.067 by hand and OpenRocket's code reports 0.039. Adding it means choosing between them without evidence, so the gap stays visible and explained.
- **RocketPy is optional** (`[validation]` extra). Without it, 4 tests skip.

## Known simplifications

- One subsonic rocket (Mach 0.28), small and near vertical. Other designs may differ more.
- Pitch inertia is about 6 % high, because small parts are point masses and the nose is a thin shell.
- Agreeing with OpenRocket shows consistency with an accepted tool, not that either matches reality.
- Monte Carlo statistics and attitude under strong crosswind weren't compared.

## What to check next

- PLAN's stretch goal: compare with a real flight log from an altimeter or your ESP32. This is the only check against reality, and it needs a flight from you.
- If your rocket has a lug or rail buttons, expect real apogee below the own-aero prediction.
