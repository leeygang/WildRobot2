# WildRobot2 TODO plan

This file tracks work that must not be inferred from simulation results or
vendor headline specifications. Training may proceed with the current
randomized provisional model, but the hardware-calibration items below must be
completed before treating a policy as deployment-qualified.

## Walking training

- [ ] Run v0.21.0 heading-aware 20M refinement from the retained v0.20.1 best
  checkpoint after explicit input migration. Check speed MAE, heading/path drift,
  phase-binned speed fluctuation and scripted standing/walking survival against
  the parent; retain Gaussian reward, servo model and command range unchanged.
- [ ] Confirm the accepted policy at 0.05/0.075/0.10 m/s on held-out randomized
  seeds and with `evaluate --transitions` before expanding command speeds.

- [x] Replace the WR1/static actuator midpoint with the held-out-validated
  unit-A low-load fit. Training now uses `kp_sim=24.1574`, `kv_sim=0.5`, joint
  `damping=0.173622`, `frictionloss=0.465471`, and `armature=0.020951`. The
  active velocity term and joint damping sum to the fitted effective damping
  of 0.673622 N m s/rad; this remains a provisional single-servo training
  nominal, not a deployment rating.
- [x] Run the cold v0.11.0 residual-action canary. It produced stable bilateral
  phase stepping and safe torque but remained near zero forward speed because
  the strict velocity reward supplied negligible gradient at 0.08--0.18 m/s.
- [x] Run the cold v0.11.1 acquisition canary at 0.05--0.10 m/s with 20%
  standing commands. It improved stochastic survival from 20% to 6% falls but
  still stepped mostly in place and reached about 10% target clipping.
- [x] Run the cold v0.14.0 moderate-randomization canary. It acquired stable
  bilateral stepping with low action saturation, but optimized a backward gait
  because the mixed 0.05--0.10 m/s commands supplied little early velocity
  gradient relative to the foot-phase return.
- [ ] Warm-start v0.15.0 from the v0.14.0 12,042,240-step checkpoint and run a
  20M fixed-0.05-m/s acquisition canary. Require positive forward velocity
  before expanding the active command curriculum back toward 0.10 m/s.
- [x] Implement the ToddlerBot RSL-RL path with adaptive-KL scheduling, a
  current RTX-5070-compatible PyTorch build, full-state checkpoint resume, and
  the same deterministic MJX evaluator/metrics used by Brax.
- [x] Verify that a 1000-step horizon is reported as a Brax truncation and is
  forwarded to RSL-RL as `time_outs=1` for value bootstrapping.
- [x] Verify that a true unhealthy-state fall remains a terminal transition
  with `truncation=0`.
- [x] Verify that WR2's command, action, target, IMU, torque-exposure, and step
  state reset after both timeout and fall without changing truncation semantics.
- [ ] Measure RSL-RL throughput and peak GPU memory against the Brax baseline on
  the RTX 5070 before committing the full one-billion-step acquisition run.
- [x] Align acquisition and evaluation with ToddlerBot's active strict
  `exp(-1000 * velocity_error^2)` kernel (`sigma=0.0316 m/s`). Keep it fixed in
  the v0.11.x canaries so command sampling is the only new mechanism.
- [ ] Pass the P0 walking gates at 0.10, 0.15, and 0.20 m/s under independent
  randomized confirmation.
- [ ] Export the accepted policy and implement the physical runtime adapter for
  `wr2_proprio_v3` and the limit-aware ten-leg-action contract.

## Hardware calibration required before deployment

The [HTD-45H servo specification and evidence
index](design/servo_model.md) maps every training and deployment parameter to
its definition, collection script, required setup, evidence status, and current
value. Use it as the traceability record; this section tracks the remaining
work.

### Tests possible with the current 2.650 kg BAM fixture

The current fixture has a 2.650 kg removable weight, approximately 2.722761 kg
total moving mass, and a measured 0.1204 m weight-center radius. It is useful
for known-gravity-load and low-speed system-identification work. It is not a
dynamometer: without an inline torque sensor, output torque is inferred from
fixture geometry, and inertia, bearing friction, compliance, and gravity cannot
be separated reliably during fast motion.

Before every powered campaign, verify the measured mass and center radius,
rigid mounting, output-shaft counter-bearing, full commanded-path clearance,
catcher, independent power cutoff, EEPROM limits, supply voltage, and the
selected cooldown limit. The bounded low-load plan may then run all stages in
one command; it independently cools, returns to neutral, and unloads each stage
and stops on the first blocking failure.

The following work can be completed with the current fixture:

Use `python -m wr2.tools.servo_sysid run --plan bam_position` to automate the
bounded `+10 degree -> -10 degree -> E3` sequence. It is preflight-only by
default, requires explicit fixture confirmation for hardware execution, records
the plan and fixture hashes, clean Git revision, and per-condition artifacts,
and stops on the first non-voltage failed gate. Qualified-range voltage
excursions are warnings recorded in capture metadata; the external supply
logger must still be started and synchronized separately.

Use `--plan bam_low_load_qualification` for each additional servo. Its seven
stages combine signed repeatability, gravity-neutral dynamics, signed loaded
dynamics, and signed coarse hysteresis under the same current-fixture limits.
Run it twice per servo and reserve the second complete campaign for held-out
validation; it deliberately excludes every test requiring new instrumentation.

- [x] Collect preliminary unit-A low-load commissioning at +10 and -10 degrees.
  The accepted data comprise four +10 repeats and five -10 repeats, with zero
  and loaded position repeatability, direction-dependent error, internal
  voltage and temperature, and geometry-inferred fixture torque preserved in
  `docs/hardware/evidence/htd45h_unit_a.json`.
- [ ] Extend low-load commissioning to several angles and repeat all accepted
  conditions on additional labeled servos. Unit B now has complete fit and
  held-out campaigns preserved in
  `docs/hardware/evidence/htd45h_unit_b.json`; unit C remains.
- [ ] Run slow forward/reverse sweeps at cold and warmed conditions to obtain
  preliminary *loaded* backlash and hysteresis evidence. Approach every target
  from both directions and retain the complete command/position trajectory;
  endpoint-only measurements are insufficient. The automated
  first `bam_hysteresis` signed pair is complete: H1 measured 0.443 degree mean
  center loop width and H2 measured 1.588 degrees, a preliminary 3.58x
  direction asymmetry. Unit B independently reproduced the signed effect in
  two sessions: positive center means were 0.474/0.468 degrees and negative
  means were 1.575/1.625 degrees. Controlled cold/warm comparisons remain. The
  servo's 0.24-degree telemetry resolution limits the smallest deadband this
  setup can resolve.
- [x] Collect the first gravity-neutral, low-amplitude E3 chirp on unit A. The
  run completed without voltage warnings and is retained as preliminary
  position-loop, whole-response, and low-load speed evidence.
- [x] Repeat E3 in a separate session for held-out dynamic validation. Run08
  reproduced the run07 response closely and is reserved from fitting.
- [x] Collect and validate bounded signed loaded dynamics near +/-10 degrees.
  Runs10/11 form the fit set and runs12/13 are held out. Run14 selected no
  extra 20 ms delay, kept every continuous parameter away from its optimizer
  bounds, and achieved 0.865 degree fit and 0.855 degree held-out mean replay
  RMSE. Its total damping is explicitly split into `kv_sim` plus joint damping
  before use by training.
- [x] Collect and validate the complete unit-B low-load matrix. Run01 and the
  held-out run02 each contain all 15 captures. The effective fit selected zero
  extra delay, kept every continuous parameter inside its bounds, and achieved
  0.7565 degree fit and 0.7558 degree held-out mean replay RMSE. This is
  accepted as unit-B variability evidence, not a population specification.
- [ ] Run bounded low-load thermal/current-duration tests only after adding a
  clock-aligned external supply logger. Record supply voltage and current in
  addition to command, position, derived velocity, servo voltage, and
  temperature. A rising final temperature slope is not a continuous rating.
- [ ] Repeat safe conditions on at least three independently labeled servos and
  at the intended deployment wiring and supply configuration before treating
  the observations as a population range. The bounded
  `bam_low_load_qualification` fit and held-out campaigns are complete for unit
  B; two complete campaigns on unit C remain to be collected.

For the present unit A, do not run the documented E1, E2, or E4--E7 high-load
conditions. Existing evidence covers only about 1 N m for short holds, and an
approximately 0.317 N m sustained condition reached the inclusive 80 C cutoff
after about 402 seconds. The 2.7--3.0 N m fixture profiles are retained only as
future conditions to be redesigned from policy demand and new instrumentation.

Every current-fixture capture must retain monotonic and wall-clock timestamps,
scheduled command position and derived command velocity, measured position and
derived velocity, direction of approach, servo voltage, temperature and
torque-enable state, measured mass and center radius, fixture-model hash,
external logger identity/path when used, test conditions, abort reason, and the
exact robot/config Git revision. Clearly label gravity-model torque as
`inferred_fixture_torque`, never as measured servo output torque.

### Tests requiring a different or upgraded setup

#### Instrumented BAM fixture for low-speed torque and hysteresis

Add a bidirectional inline torque transducer, or a calibrated bidirectional load
cell with independently measured moment arm, plus a synchronized current- and
voltage-logging supply. Use a rigid, counter-bearing-supported torque path and
removable or interchangeable masses/radii so load can be increased from a safe
low-energy condition. Calibrate sensor zero, scale, sign, cross-axis sensitivity,
and fixture friction before and after each session. Use a shared hardware
trigger when available; otherwise measure and report clock offset and drift.

- [ ] Measure quasi-static torque, current, position error, backlash, and
  hysteresis in both directions at several joint angles, loads, supply voltages,
  and cold/warm temperatures.
- [ ] Repeat each condition after approaching from both directions and include a
  truly unloaded or counterbalanced sweep. The fixed 2.650 kg configuration
  alone is not an unloaded test.
- [ ] Use an external encoder when sub-0.24-degree backlash or compliance must be
  resolved; do not infer it from the quantized servo telemetry alone.

This upgraded fixture can qualify low-speed static behavior. Its swinging mass
still makes it unsuitable for identifying the full high-speed torque envelope
or controlled back-driven braking.

#### Guarded motor dynamometer for torque-speed and braking

Use a bidirectional drive/load motor, inline torque transducer, independent
shaft encoder, programmable current-logging supply, rigid guarded coupling,
counter-bearing as required, reachable independent power cutoff, and hardware
over-torque, over-speed, over-current, and inclusive 80 C trips. Size the load
motor and sensor for the vendor peak bound but begin with substantially lower
software and supply limits. Validate sensor signs and automatic unload with the
servo disabled before enabling closed-loop motion.

- [ ] Sweep steady motoring points in both shaft directions over multiple loads
  and speeds at minimum, nominal, and maximum deployment voltage and at cold and
  warmed temperatures. Capture commanded/actual position and velocity, measured
  shaft torque, current, voltage, temperature, timestamps, and all limit events.
- [ ] Measure controlled deceleration and externally back-driven motion in all
  four torque/velocity quadrants. Include repeated speed ramps and steady points
  so active braking, passive/back-driven response, and direction asymmetry can
  be separated.
- [ ] Fit peak-torque plateau, taper onset, torque at maximum speed, maximum
  speed, braking limit, passive/active ratio, voltage/temperature dependence,
  direction dependence, and uncertainty intervals. Reserve complete sessions
  from different servos as held-out validation data.

Do not update the simulator or PPO randomization ranges from a best-fit curve
alone. The fitted result must include raw-capture references, calibration
records, coverage by operating quadrant, held-out replay error, uncertainty,
and an explicit conservative margin.

#### Handoff contract for the three actuator qualification campaigns

The next agent should treat torque-speed, braking, and backlash/hysteresis as
three separate campaigns. The existing `capture.py` and YAML plans support only
the servo bus and the gravity-loaded BAM. They must not be relabeled as
dynamometer or isolated-backlash tests. `campaign.py` intentionally rejects
plans that declare unsupported instruments, so new powered plans must wait
until each required instrument has a tested adapter, calibration record,
timestamp mapping, and preflight-only dry run.

Common requirements for all three campaigns:

- Start with a written fixture diagram, signal/clock diagram, instrument ranges,
  calibration procedure, automatic abort path, and conservative initial limits.
  Keep the reachable independent power cutoff and inclusive 80 C shutdown.
- Record raw, synchronized command, servo telemetry, external encoder, measured
  torque, supply voltage/current, temperature, limit state, and monotonic time.
  Preserve unmodified raw samples; derived values belong in a separate report.
- Validate signs, units, zero, scale, clock offset/drift, automatic unload, and
  abort behavior with the servo disabled before any powered identification run.
- Use labeled servos and complete-session splits: fitting and held-out replay
  must come from different sessions, and deployment qualification requires at
  least three servos. Report uncertainty and direction/temperature dependence.
- An analyzer may emit a proposed model/config patch, but it must never edit
  `robot.yml`, `default.yml`, or PPO ranges automatically. Adoption requires a
  reviewed evidence record and a conservative deployment margin.

**Campaign T1 -- motoring torque versus speed**

- Required setup: guarded bidirectional dynamometer, inline torque transducer,
  independent output encoder, programmable current-logging supply, rigid
  coupling/counter-bearing, and hardware over-current/torque/speed protection.
- Cover both shaft directions, multiple steady speeds and loads, the 9.6 V,
  11.1 V, and 12.6 V operating points, and cold/warmed conditions. Begin at a
  low-energy point; do not approach the vendor stall bound until lower points,
  unloading, and every abort path have passed.
- Definition of done: held-out replay bounds the measured torque-speed surface;
  the report proposes plateau torque, taper onset, torque at maximum speed,
  maximum speed, voltage/temperature terms, direction asymmetry, uncertainty,
  and the corresponding simulator fields. No endpoint may be called continuous
  torque without a separate thermal-equilibrium result.

**Campaign T2 -- active and passive/reverse braking**

- Use the same guarded dynamometer and measure all four torque/velocity
  quadrants. Include commanded deceleration, speed reversals, steady externally
  back-driven points, and repeated ramps in both directions; do not infer
  braking torque from position error alone.
- Separate actively commanded opposing torque from passive/back-driven drag and
  record regeneration or supply-current behavior where the hardware permits.
- Definition of done: held-out sessions support conservative
  `brake_torque_limit_nm` and `passive_active_ratio` proposals, including speed,
  direction, voltage, and temperature dependence. Automatic unload and abort
  tests must pass before the result can affect training randomization.

**Campaign T3 -- bidirectional backlash and loaded hysteresis**

- Required setup: external output encoder with resolution materially better
  than the servo's 0.24-degree telemetry step, synchronized torque/load sensing,
  and a truly unloaded or counterbalanced condition in addition to loaded BAM
  conditions. Repeat at representative angles and cold/warmed temperatures.
- Approach every measurement point from both directions and retain complete
  slow forward/reverse trajectories. Report deadband, center loop width, zero
  shift, repeatability, load dependence, and direction asymmetry separately.
- Definition of done: held-out sessions distinguish isolated mechanical
  backlash from controller/friction/compliance hysteresis. Before changing
  `backlash_rad`, decide whether each observed effect belongs in physics,
  actuator target bias, or encoder observation noise, then demonstrate replay
  improvement without double-counting the fitted friction/controller model.

#### Installed-robot BNO085 reference setup

The servo fixture does not qualify the installed IMU. Rigidly secure the
assembled robot in several known orientations, then use a rate table, accurately
timed manual fixture, motion-capture system, or another independently validated
orientation/rate reference. Timestamp raw sensor events and host receipt with a
common clock or a measured synchronization offset.

- [ ] Record long stationary captures in several orientations, synchronized
  slow and fast rotations, repeated start-up/reinitialization, and intentional
  bus/host load. Preserve raw quaternions, angular velocity, sensor timestamps,
  host timestamps, sequence/drop indicators, reference motion, and temperature.
- [ ] Estimate mounting transform/error, gyro white and correlated noise, bias
  drift, projected-gravity error, sample-age distribution, stale/drop rate, and
  end-to-end latency. Validate the fitted observation/delay model on held-out
  motions before changing training ranges.
- [ ] For v4 heading input, validate the BNO085 six-axis/game rotation vector in
  a zero-initial-heading frame, measure yaw drift during stationary and walking
  motion over deployment-duration windows, and verify commanded heading
  integration/reset matches simulation. Do not claim magnetometer-free means
  drift-free; fit the heading noise only from installed measurements.

### Required fitted outputs

- [ ] **HTD-45H torque versus speed:** capture commanded/actual position,
  velocity, current, voltage, temperature, and measured output torque at
  multiple loads, speeds, directions, supply voltages, and temperatures. Fit
  the peak-torque plateau, taper, maximum speed, and uncertainty ranges used by
  `wr2/locomotion/walking_env.py` and `ppo_walking.yaml`.
- [ ] **HTD-45H reverse/braking torque:** measure controlled deceleration and
  back-driven motion in both directions. Fit the braking limit and
  passive/active ratio instead of relying on the current vendor-bounded
  provisional values.
- [ ] **Bidirectional backlash and hysteresis:** run slow unloaded and loaded
  forward/reverse sweeps at representative joint angles and temperatures.
  Measure direction-dependent deadband, repeatability, and zero offset, then
  replace the provisional backlash and target-bias distributions.
- [ ] **Installed BNO085 behavior:** log stationary data in several
  orientations and synchronized slow/fast motions on the assembled robot.
  Estimate gyro white noise, correlated noise, bias drift, projected-gravity
  error, mounting error, sample age, dropped/stale samples, and end-to-end
  latency; update the observation-noise and delay model from those results.

Each calibration campaign must retain raw captures, a machine-readable fitted
result, held-out replay error, test conditions, and the exact robot/config Git
revision. Do not replace the training ranges from one run without held-out
validation or an explicit conservative margin.

## Hardware rollout

- [ ] Verify the accepted policy first with the robot suspended and torque
  limited.
- [ ] Run restrained standing and low-amplitude stepping while recording servo
  temperature, voltage, current, command tracking, IMU data, and safety stops.
- [ ] Permit untethered walking only after the fitted hardware model and
  restrained tests satisfy the deployment limits.
