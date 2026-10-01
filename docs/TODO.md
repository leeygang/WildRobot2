# WildRobot2 TODO plan

This file tracks work that must not be inferred from simulation results or
vendor headline specifications. Training may proceed with the current
randomized provisional model, but the hardware-calibration items below must be
completed before treating a policy as deployment-qualified.

## Walking training

- [ ] Start a new cold PPO gait-acquisition campaign with training contract
  v0.9. Existing checkpoints are incompatible with the new 10-action actor
  and asymmetric 825/1440 actor/critic observations.
- [ ] After gait acquisition, tighten `velocity_reward_sigma` from 0.15 m/s
  toward the unchanged 0.0316 m/s final tracking tolerance and confirm that
  the gait survives fine-tuning.
- [ ] Pass the P0 walking gates at 0.10, 0.15, and 0.20 m/s under independent
  randomized confirmation.
- [ ] Export the accepted policy and implement the physical runtime adapter for
  `wr2_proprio_v3` and the limit-aware ten-leg-action contract.

## Hardware calibration required before deployment

### Tests possible with the current 2.650 kg BAM fixture

The current fixture has a 2.650 kg removable weight, approximately 2.722761 kg
total moving mass, and a measured 0.1204 m weight-center radius. It is useful
for known-gravity-load and low-speed system-identification work. It is not a
dynamometer: without an inline torque sensor, output torque is inferred from
fixture geometry, and inertia, bearing friction, compliance, and gravity cannot
be separated reliably during fast motion.

Before every powered test, verify the measured mass and center radius, rigid
mounting, output-shaft counter-bearing, full commanded-path clearance, catcher,
independent power cutoff, EEPROM limits, supply voltage, and a starting
temperature no greater than the selected cooldown limit. Run only one bounded
condition at a time and inspect its JSON/NPZ artifacts before continuing.

The following work can be completed with the current fixture:

- [ ] Repeat low-load commissioning in both directions and at several angles.
  Record zero and loaded position repeatability, direction-dependent position
  error, servo-reported voltage and temperature, and the gravity torque inferred
  from measured position, mass, radius, and the versioned fixture model.
- [ ] Run slow forward/reverse sweeps at cold and warmed conditions to obtain
  preliminary *loaded* backlash and hysteresis evidence. Approach every target
  from both directions and retain the complete command/position trajectory;
  endpoint-only measurements are insufficient. The servo's 0.24-degree
  telemetry resolution limits the smallest deadband this setup can resolve.
- [ ] Run the gravity-neutral, low-amplitude E3 chirp first to estimate the
  effective position-loop response, whole-response delay, repeatability, and
  low-load speed tracking. Keep separate captures for fitting and held-out
  replay validation.
- [ ] Run bounded low-load thermal/current-duration tests only after adding a
  clock-aligned external supply logger. Record supply voltage and current in
  addition to command, position, derived velocity, servo voltage, and
  temperature. A rising final temperature slope is not a continuous rating.
- [ ] Repeat safe conditions on at least three independently labeled servos and
  at the intended deployment wiring and supply configuration before treating
  the observations as a population range.

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
