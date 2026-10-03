# HTD-45H deployment qualification

This workflow ports the WildRobot 2.650 kg BAM fixture campaign into WR2 and
adds dynamic-torque preflight, staged execution, multi-servo qualification,
and generation of a machine-readable deployment specification.

See the [servo specification and evidence index](../design/servo_model.md) for
the value consumed by training or deployment, its canonical definition,
collection status, required fixture, and implemented script structure.

The replacement weight is 2.650 kg. With the arm and mounting hardware, the
modeled moving mass is 2.722761 kg and the load inertia is approximately
0.0430185 kg m^2. The physical weight's center of mass must remain at the
modeled 120.4 mm location. Update the fixture MJCF if its shape, mounting, or
center of mass differs.

## Safety requirements

Use a rigid fixture with a counter-bearing for the output shaft, verified
clearance through +/-72 degrees, a catcher capable of receiving the complete
moving mass after automatic unload, and a reachable independent power cutoff.
Only the fixture servo may be connected to the selected bus.

Use a current-logging supply and preferably a load cell. The HTD protocol
reports position, voltage, temperature, and torque-enable state, but does not
report current or output torque. Known-load torque is otherwise inferred from
the fixture model.

The software torque, temperature, position, and 5.0 V hard-floor limits are
safety aborts, not servo ratings. Servo-reported voltage below the qualified
9.6 V operating minimum produces a yellow warning and a recorded voltage event,
but does not stop motion unless it crosses the hard floor. The dynamic preflight
estimates gravity plus commanded-trajectory inertial torque. Step transients,
fixture compliance, impacts, and controller overshoot remain unmodeled. The
5-degree tracking limit must persist for 0.15 seconds before aborting, so an
intentional step edge is not mistaken for a stalled servo.

## Preflight

Install the required packages:

```bash
uv sync --extra sysid --extra dev
```

Run the complete no-I/O preflight:

```bash
uv run python -m wr2.tools.servo_sysid run \
  --plan legacy_deployment \
  --servo-id 100 \
  --board-port /dev/serial/by-id/YOUR_ADAPTER
```

The preflight checks all seven conditions without opening the serial port.
Immediately before torque is enabled, hardware mode also reads the servo's
EEPROM angle, voltage, temperature, and motor-mode settings. It refuses motion
when the requested travel exceeds the configured angle limits, the software
temperature ceiling exceeds the EEPROM ceiling, position mode is disabled, or
the live supply is outside the EEPROM voltage range.

## Low-load repeatability commissioning

### Automated bounded BAM plan

The `bam_position` plan automates the currently permitted low-load sequence: five
+10-degree commissioning repeats, five -10-degree commissioning repeats, and
the gravity-neutral E3 inertial-bandwidth capture. It does not include E1, E2,
or E4--E7. It runs every mathematical preflight without opening the serial port
unless hardware execution is explicitly enabled.

```bash
uv run python -m wr2.tools.servo_sysid run \
  --plan bam_position \
  --servo-id 100 \
  --servo-label htd45h-unit-a \
  --board-port /dev/serial/by-id/YOUR_ADAPTER
```

For hardware execution, start the external voltage/current logger first. The
runner records a distinct synchronization label for every condition, enforces a
55 C initial test ceiling, independently cools and unloads around the child
captures, continues through qualified-range voltage warnings, stops on the
first other failed safety or repeatability gate, and writes a
top-level manifest with the plan hash and exact clean Git revision. `--run-all`
is an explicit request to continue between all three low-load conditions
without a human review pause:

```bash
uv run python -m wr2.tools.servo_sysid run \
  --plan bam_position \
  --servo-id 100 \
  --servo-label htd45h-unit-a \
  --board-port /dev/serial/by-id/YOUR_ADAPTER \
  --run-dir results/servo_sysid/bam-low-load-unit-a-run01 \
  --measured-weight-kg 2.650 \
  --measured-com-radius-m 0.1204 \
  --external-log-label-prefix supply-unit-a-run01 \
  --run-all \
  --execute --confirm-fixture-safe
```

For staged review, omit `--run-all` and select one bounded range with
`--start-at` and `--stop-after`. Reusing `--run-dir` skips conditions already
recorded as complete. An interrupted condition is never silently adopted;
inspect its artifacts and start a new run directory. The software cannot start
or verify an arbitrary external logger, so its raw clock-aligned file still
needs to be archived alongside the run.

### Current unit-A checkpoint

The unit-A set now contains four accepted +10-degree repeats, five accepted
-10-degree repeats, gravity-neutral E3 fit/validation runs07/08, signed loaded
fit runs10/11, and signed loaded validation runs12/13. Run14 combines the three
fit traces and three held-out traces. It selected no extra delay, kept every
continuous parameter away from an optimizer bound, and achieved 0.865 degree
fit and 0.855 degree held-out mean replay RMSE. Exact observations and
limitations are retained in
[`htd45h_unit_a.json`](evidence/htd45h_unit_a.json).

The fitter identifies one effective velocity-damping term. WR2 training keeps
`kv_sim=0.5`, so the mapped joint damping is the fitted 0.673622 total minus
0.5, or 0.173622 N m s/rad. Never assign the full fitted total to the joint
while retaining `kv_sim`; that would double-count velocity feedback.

Equivalent unit-A low-load dynamics captures are complete. Further hardware
work should add a clock-aligned current/voltage logger, repeat the safe
conditions on independently labeled servos, or use the upgraded fixtures
required for backlash, torque-speed, and braking. Do not run the unsafe legacy
E4/E5 conditions with the current fixture.

Directories created by the former `bam_suite.py` implementation contain a
different manifest schema and are evidence archives only; do not resume them
with the plan runner. The deprecated `bam_suite` and `commission` module names
translate old command-line arguments into new campaigns, but new procedures
should use the unified command above.

Before the 3 N m deployment campaign, run five independent +10-degree trials.
Each repetition performs its own cooldown and safety checks, returns to zero,
unloads the servo, and writes a separate JSON/NPZ pair. The runner then reports
zero-position and loaded-position repeatability across all five trials.

```bash
uv run python -m wr2.tools.servo_sysid run \
  --plan bam_repeatability \
  --servo-id 100 \
  --servo-label htd45h-unit-a \
  --board-port /dev/serial/by-id/YOUR_ADAPTER \
  --center-deg 10 \
  --repeats 5 \
  --measured-weight-kg 2.650 \
  --measured-com-radius-m 0.1204 \
  --external-log-label supply-unit-a-commission-plus10 \
  --run-all \
  --execute --confirm-fixture-safe
```

The default repeatability limit is 0.5 degrees standard deviation. The entire
timestamped result directory can be copied as one logical result with
`./scripts/scp_from_remote.sh --wrdev --latest servo_sysid`.

## Staged hardware execution

The original high-load conditions below are retained for future deployment
qualification, but are not prerequisites for starting policy training. Do not
run E1, E2, or E4--E7 with the current unit-A evidence: WR2 has validated only
approximately 1 N m actual fixture torque for short holds, and approximately
0.317 N m reached the 80 C software cutoff before ten minutes. Run the
gravity-neutral E3 dynamic condition first and redesign later conditions from
the trained policy's measured torque, speed, and duty-cycle distribution.

Run and inspect one condition at a time. Hardware motion requires both safety
flags and an explicit stopping condition:

```bash
uv run python -m wr2.tools.servo_sysid run \
  --plan legacy_deployment \
  --servo-id 100 \
  --servo-label htd45h-unit-a \
  --board-port /dev/serial/by-id/YOUR_ADAPTER \
  --campaign-dir results/servo_sysid/htd45h-unit-a \
  --measured-weight-kg 2.650 \
  --measured-com-radius-m 0.1204 \
  --start-at E1_static_plus72 \
  --stop-after E1_static_plus72 \
  --external-log-label supply-unit-a-run-1 \
  --execute --confirm-fixture-safe
```

Inspect the JSON/NPZ pair before continuing with E2, then E3, and so forth.
Reuse the same `--campaign-dir` for every condition belonging to that servo.
The runner rejects changes to fixture geometry, measured mass/COM, direction,
or safety limits within a partially completed campaign.
The conditions are:

1. `E1_static_plus72`: +3.03 N m slow static ramp.
2. `E2_static_minus72`: -3.04 N m slow static ramp.
3. `E3_inertial_bandwidth`: gravity-neutral 0.1-4 Hz motion.
4. `E4_loaded_plus60`: +2.76 N m loaded response.
5. `E5_loaded_minus60`: -2.77 N m loaded response.
6. `E6_deployment_deadband_plus60`: loaded response with runtime deadband.
7. `E7_thermal_hold_plus60`: ten-minute 2.76 N m hold.

The loaded dynamic profiles use 2- and 5-degree amplitudes. The original WR1
8-degree profile reached approximately 3.18 N m after including fixture
inertia, leaving almost no uncertainty margin below the 3.2 N m safety
ceiling. The reduced profile still reaches approximately 3.00 N m in the
positive direction and 3.03 N m in the negative direction.

Repeat the complete campaign for at least three independently labeled servos,
at the actual deployment supply and wiring configuration. Additional campaigns
at the minimum battery voltage and after warm-up are strongly recommended.

After each campaign, copy
`wr2/tools/servo_sysid/external_measurements.example.json` into the campaign
directory as `external_measurements.json` and replace every example value with
clock-aligned supply/logger measurements. The analyzer requires this evidence
by default. Use `--allow-missing-external-measurements` only for a fixture-only
engineering report that must not be treated as deployment-qualified.

## Generate the deployment specification

Pass all completed campaign directories to the analyzer:

```bash
uv run python -m wr2.tools.servo_sysid report \
  results/servo_sysid/htd45h-unit-a \
  results/servo_sysid/htd45h-unit-b \
  results/servo_sysid/htd45h-unit-c \
  --output results/servo_sysid/htd45h_deployment_spec.json
```

The default qualification requires three distinct servo labels, complete
signed static and loaded-dynamic coverage, voltage >=9.6 V, temperature below
the inclusive 80 C shutdown, tracking p95 <=5 degrees, and a final thermal
slope <=0.5 C/min. The report
applies a 1.2 safety factor before recommending peak and continuous torque
limits. Thresholds are explicit command-line parameters and should ultimately
be tied to the WR2 policy's measured error tolerance and required operating
duration.

A ten-minute hold is not a continuous-duty result when temperature is still
rising. Extend the hold or derate the continuous limit until the final
temperature slope demonstrates equilibrium.

Fit the identifiable position-loop dynamics from zero, positive-load, and
negative-load captures. Keep separate repeated captures for validation:

```bash
uv run python -m wr2.tools.servo_sysid fit \
  CAMPAIGN_A/03_E3_inertial_bandwidth.npz \
  CAMPAIGN_A/04_E4_loaded_plus60.npz \
  CAMPAIGN_A/05_E5_loaded_minus60.npz \
  --validation-capture CAMPAIGN_B/04_E4_loaded_plus60.npz \
  --validation-capture CAMPAIGN_B/05_E5_loaded_minus60.npz \
  --controller-kv 0.5 \
  --output results/servo_sysid/htd45h_dynamics_fit.json
```

The included fitter identifies position gain, effective total velocity
damping, friction, armature, and whole-response delay. It emits both the
effective parameters and an explicit training mapping in which
`effective_velocity_damping = kv_sim + damping`. It deliberately does not
auto-edit the robot model; review its held-out error and uncertainty first.
A voltage/temperature-dependent torque-speed model still requires the external
current/load-cell data and multiple load, speed, voltage, and temperature
conditions.
