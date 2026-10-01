# WildRobot2 TODO plan

This file tracks work that must not be inferred from simulation results or
vendor headline specifications. Training may proceed with the current
randomized provisional model, but the hardware-calibration items below must be
completed before treating a policy as deployment-qualified.

## Walking training

- [ ] Start a new cold PPO gait-acquisition campaign with training contract
  v0.8. Existing checkpoints are incompatible with the new 10-action actor
  and asymmetric 825/1440 actor/critic observations.
- [ ] After gait acquisition, tighten `velocity_reward_sigma` from 0.15 m/s
  toward the unchanged 0.0316 m/s final tracking tolerance and confirm that
  the gait survives fine-tuning.
- [ ] Pass the P0 walking gates at 0.10, 0.15, and 0.20 m/s under independent
  randomized confirmation.
- [ ] Export the accepted policy and implement the physical runtime adapter for
  `wr2_proprio_v3` and the limit-aware ten-leg-action contract.

## Hardware calibration required before deployment

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
