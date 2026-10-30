# Stage 4 notes — recovery and landing

## Physics in plain language

**Sequence after apogee.**
1. **Free fall:** the rocket keeps flying ballistically (full 6-DOF) for the drogue delay. This models a motor ejection charge that fires late.
2. **Drogue:** the drogue opens, and the rocket becomes a point mass under the canopy.
3. **Main:** when it descends through the main altitude, the main opens too. The drogue stays attached, so the total Cd·A is drogue + main.

If the apogee is already below the main altitude, both chutes open together. A single-chute model rocket just configures a drogue with no main.

**Parachute force.** D = ½ρ·Cd·A·|v_a|·v_a, opposing the *air-relative* velocity v_a = v − w. So the canopy is carried along with the wind. Once horizontal, the rocket's ground speed equals the wind speed, and the drift is about wind speed × descent time.

**Terminal velocity.** The rocket stops speeding up when drag equals weight: ½ρv²·CdA = mg, giving v_t = √(2mg/(ρ·CdA)). Air density rises as it descends, so the descent rate slowly drops all the way down.

**Inflation and opening shock.** A canopy doesn't open instantly. Here Cd·A grows linearly from 0 to its full value over `fill_time`. The opening shock is the peak parachute drag divided by weight, in g; 1 g means steady descent.
- An instant opening at speed v gives exactly ½ρv²·CdA/(m·g), and the test checks this to 1e-6.
- A 0.5 s fill cuts that peak, because the rocket slows down while the canopy is still growing.
- Opening right at apogee, where the speed is near zero, is the gentlest case: 1.1 g for the example.

## Design decisions

- **Config.** The `recovery` section has `drogue` {cda, fill_time, delay} and an optional `main` {cda, fill_time, altitude}. Delay and altitude are measured from apogee and from the launch site.
- **Phase manager.** Each recovery phase is its own solve_ivp call, and events end each phase exactly: main altitude, landing. The phase manager steps through freefall → drogue → main in a short loop. The deployment list goes to the parachute derivative, so Cd·A(t) is known at every point.
- **Sampling.** Deployment and fill-end times are added to the output samples, so the shock peak isn't missed between samples.
- **New summary fields:** `drogue_descent_rate` (at main deployment or landing), `main_descent_rate`, `opening_shock_g`, and events `drogue` and `main`.

## Known simplifications

- **Inflation:** linear Cd·A growth. Measured canopies grow closer to quadratically, so this slightly overestimates early drag. There's no canopy oscillation or snatch force from the shock cord.
- **Under canopy:** point mass with attitude frozen and body drag ignored (PLAN §3).
- **Ejection timing:** the charge is timed from apogee, not from burnout. A real C6-5 fires 5 s after burnout, which on the example is about 0.8 s *before* apogee. That case can't be configured yet.

## Results

**Example rocket** (one 30 cm chute at apogee, 0.3 s fill):

| Quantity | Value |
|---|---|
| Descent rate | 3.61 m/s |
| Opening shock | 1.1 g |
| Landing point | 435 m East |

**Tests:**
- **Terminal velocity:** under drogue and under drogue + main, it matches √(2mg/(ρ·CdA)) to 1e-4.
- **Main deployment:** happens exactly at its altitude; the drogue opens exactly at apogee + delay.
- **Zero wind:** the rocket lands 2.8 m from the point under apogee, compared with a 103 m apogee offset. That 2.8 m is the horizontal speed left at apogee dying out under the canopy.
- **Constant wind:** descent drift is 99.4% of wind speed × descent time; the requirement was within 5%.

## What to check next

- Let the ejection be timed from burnout, so a real "C6-5" delay can be modelled. It's one more option in `drogue`.
- Stage 5 (Monte Carlo) can now vary chute Cd·A and deployment delay directly.
