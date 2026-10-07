# Changelog

## 2026-10-07

- Reviewed v0.20.1 `wr2_ppo_20261007_084251_seed0`: no evaluated update beat
  the restored policy. Worst-endpoint nominal score changed 0.492 -> 0.437;
  0.10 m/s speed remained 0.0905 m/s but MAE increased to 0.0270 m/s. Independent
  randomized final-policy confirmation covered 3 speeds x 3 seeds x 128
  environments: all 1152 episodes survived without target clipping, while
  speed MAE worsened versus the parent at every speed. The value-loss spike
  persisted after B1; it cannot be attributed solely to missing bootstrapping.
- Prepared v0.21.0: ToddlerBot full orientation-to-path reward, observable
  six-axis-IMU-relative heading sin/cos, explicit action-preserving v3-to-v4
  checkpoint/Adam migration, path and phase diagnostics, and separate scripted
  standing/walking evaluation. Gaussian velocity reward, actuator envelope,
  dynamics randomization and PPO hyperparameters remain unchanged. Actor/critic
  input shapes are 855/1470; installed heading drift remains unqualified.
- Validation: 121 tests pass, including quaternion parity, hardware heading
  input, historical v3 loading, timeout/fall resets, migration/Adam update, and
  transition metric isolation. MJX smoke, an 80-step restored CPU PPO run, and
  a two-environment 1000-step scripted transition evaluation also passed.

- Completed the 20M-step v0.20.0 RSL-RL velocity-refinement continuation
  (`wr2_ppo_20261006_205755_seed0`). The restored step-0 policy remained best:
  at a 0.10 m/s command it achieved 0.0902 m/s forward velocity, 0.0247 m/s
  mean absolute velocity error, 1000-step episodes, zero falls, 86.1% contact
  match, and zero action saturation. Continued optimization regressed the
  worst-endpoint walking score from 0.491 to 0.390, so the final checkpoint was
  not promoted over the restored policy.
- Confirmed and fixed the RSL-RL timeout-bootstrap bug. WR2 now returns only
  unhealthy/non-finite states as environment terminals while still resetting
  recurrent episode state at the 1000-step horizon; Brax consequently marks the
  horizon as a truncation and RSL-RL receives `time_outs=1` for bootstrapping.
- Added regression coverage for timeout truncation, true-fall termination,
  recurrent-state reset, and the Brax-to-RSL timeout bridge. The complete suite
  passes 115 tests and 12 subtests. Training configuration v0.20.1 isolates this
  correction without changing rewards, actuator behavior, or randomization.

## 2026-09-28

- Created the initial ToddlerBot-aligned WildRobot2 project structure.
- Defined source-of-truth boundaries for descriptions, calibration, simulation,
  hardware, policies, and training.
- Added the single-assembly WR2 Onshape-to-MuJoCo export configuration.
