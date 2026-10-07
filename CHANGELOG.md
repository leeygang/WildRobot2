# Changelog

## 2026-10-07

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
