# Changelog

## 2026-10-10

- Confirmed the 200M continuation review (`wr2_fresh_20261009_205016_seed0`):
  exact source/Adam/LR/RNG restore, 200,007,680 additional transitions, all 11
  checkpoints finite, and the selected best still at restored step zero.
  Paired fully randomized held-out fixed-0.10 speed declined 0.0798 to
  0.0770 m/s while scripted start/stop errors improved; scripted clipping
  remained about 8.8%. Random-command falls changed from 0/384 to 2/384;
  this is a limited robustness warning, not demonstrated broad collapse.
  Retain the original 920M source. No clipping, solver or PPO root cause was
  established for the reproduced 180-degree restart failure.
- Prepared a dedicated 200M/11-evaluation continuation from that same 920M
  checkpoint, disabling only actor mirror loss to match ToddlerBot's default.
  Historical configs, rewards, actions, physics, servo envelope, randomization,
  native PPO and checkpoint-selection behavior remain unchanged. Full-state
  restoration is retained; no exploration or optimizer reset.
- Added `evaluate --random-commands` for training-distribution held-out
  confirmation, with explicit variation/schedule metadata and sample-count
  conditional tracking. Scripted/random reports no longer assign a misleading
  fixed walking score. See [the experiment, confirmed review and GPU command](docs/design/mirror_off_continuation.md).
- Validation: 197 full-suite tests and six targeted evaluation regressions
  pass; Ruff and diff checks pass. An 80-transition restored Mac MJX/Torch
  smoke verifies exact source model/Adam/LR/counter/CPU-RNG restore, a finite
  native PPO update and saved state with mirror loss disabled. A two-environment
  1000-step random-command CLI smoke verifies the full-variation schedule and
  conditional reports. These validate implementation, not gait qualification.
  No GPU training or hardware motion was launched.

## 2026-10-09

- Prepared a dedicated 1B fresh-start walking config with 51 evaluations and
  generated `wr2_fresh_*` run IDs. It preserves the verified mirror treatment's
  rewards, actions, physics, randomization, servo limits and PPO settings;
  the original continuation config remains unchanged. Added regression checks
  for config parity, empty native optimizer/zero counters, fresh exploration
  and checkpoint cadence. See [the GPU command and setup](docs/design/fresh_walking_training.md).
  Validation: 169 tests pass; an 80-transition fresh MJX/Torch PPO smoke verifies
  zero initial counters, empty Adam state, exploration std 0.5, and a finite
  optimizer update and saved final checkpoint. No GPU training was launched.

## 2026-10-08

- Extended the matched-state diagnostic with configurable bidirectional
  pre-restart actor handovers and explicit pre-restart fall/non-finite
  accounting. Existing first/second-start defaults and full state/history
  retention are unchanged. See the
  [reproduction and diagnostic report](docs/design/walking_restart_recovery_diagnostic.md).
  Validation: 166 tests and 30 subtests pass. No training/model/servo config
  changes; the running actor mirror-loss experiment is unaffected.

- Prepared a controlled 20M actor mirror-loss treatment using ToddlerBot's
  optional native RSL-RL loss (coefficient 1.0, no transition/critic augmentation).
  The dedicated `ppo_walking_mirror.yaml` retains the saved Newton-10 control's
  rewards, servo limits, randomization and PPO settings. Historical/default
  configs remain mirror-disabled. Shared model-derived reflection covers every
  actor history frame; checkpoints retain action/Adam/LR/RNG compatibility and
  logs include raw/weighted mirror MSE in the total loss. See
  [the experiment and GPU command](docs/design/actor_mirror_loss_experiment.md).
  Validation: 165 tests and 24 subtests pass; an 80-transition restored
  MJX/Torch smoke completed the native loss update, with exact source
  parameter/Adam/LR/counter/CPU-RNG restore and a finite saved model.
  GPU treatment results remain pending.

- Added matched pre-start controller handovers for old/new walking checkpoints.
  The diagnostic changes actor weights only, retains complete physics, sensor,
  history and command-delay state, checks untouched donor prefixes, and can
  verify diagonal controls against the independent original replay. Foot offsets
  now have an explicit signed left-minus-right robot-forward convention.
  Training configuration, rewards, physics and checkpoint contents are unchanged.
  Validation: 156 tests and 21 subtests pass; 2,304 paired diagnostic episodes
  completed on Mac across three held-out seeds with verified donor prefixes.

- Added paired held-out replay with pre-autoreset physical traces, original
  evaluation RNG streams, noise/dynamics ablations, restart timing probes and
  diagnostic actor reflection. Local and remote walking-agent confirmation now
  includes randomized standing/walking transitions, with walking-only tracking
  denominators and existing stage/hard-safety thresholds. PPO, rewards and the
  default solver budget are unchanged. Training-result conclusions remain
  subject to review; see [the investigation](docs/design/walking_solver_followup_20261008.md).
  Validation: 151 tests and 21 subtests pass.

- Added `scp_from_remote.sh --results <folder>` to recursively download any
  custom folder or subtree under remote `results/` to the matching local path.
  Existing local-only files are preserved; remote deletions are not mirrored.
  Validation covers nested runs, rsync/scp fallback, dry-run, missing folders,
  path traversal rejection and the existing transfer commands.

## 2026-10-07

- Added matched Newton solver-budget overrides to training, held-out evaluation
  and heading/contact diagnostics, plus a diagnostic all-history heading-input
  ablation. Historical/default physics remains one iteration; no symmetry loss,
  reward or servo change is enabled. Runs now record the effective physics
  budget/timing/model hash and restore-checkpoint hash for controlled comparisons.
  Regression coverage checks legacy defaults, override isolation, TB orientation
  noise composition, unchanged old input features and physics, and reward-only
  orientation changes. See [the audit and A/B handoff](docs/design/walking_regression_ab_20261007.md).
  Validation: 139 tests, 200 twenty-second Mac diagnostic episodes without
  falls/non-finite states, and an 80-transition restored RSL-RL smoke pass.
  GPU causal comparison remains pending; development checks do not qualify
  the source policy for deployment or prove the solver caused its asymmetry.

- Added pre-training model/reference symmetry, all-history actor reflection,
  half-cycle stride and full-physics-substep contact-impulse verification.
  Diagnostic-only solver convergence and mirrored-action projection probes
  run on Mac/GPU without changing training rewards, model files or checkpoints.
  Validation: 134 tests pass, with model/actor reflection, physical target
  replay, momentum reference points and integrated-impulse regression coverage.
  See [the verification report](docs/design/walking_symmetry_verification_20261007.md).

- Reviewed v0.21.0 `wr2_ppo_20261007_152924_seed0`: the starting checkpoint
  remained best. Final 0.10 m/s MAE worsened 0.0247 -> 0.0411 m/s despite
  heading improving 4.16 -> 3.41 degrees; scripted-transition saturation
  increased 1.19% -> 3.93%. Do not promote the final policy.
- Added and ran paired heading/stance-force/contact-slip/joint-target
  diagnostics on Mac: 48 twenty-second episodes, no falls or non-finite states.
  The final policy corrects added heading errors but retains uneven propulsion;
  established-stance slip is not a gross foot-support failure. Active clipping
  is isolated to left-knee extension, predominantly during standing, with
  requests as high as +8.91 degrees against the correct 0-degree boundary.
  See [the diagnostic report](docs/design/walking_diagnostics_20261007.md).
  Training rewards, PPO, joint bounds, and servo settings are unchanged.
- Diagnostic validation: 126 tests and 12 subtests pass, including contact
  force sign, point-velocity slip, delayed target traces, conditional metrics,
  and heading injection with preserved reference/observation history.

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
