# Changelog

## 2026-10-12

### XY 200M review and TB-aligned continuation

- Confirmed `wr2_xy_20261012_085006_seed0`: 200,007,680 transitions / 4,883
  RSL-RL updates and eleven retained checkpoints. All ten final inline command
  endpoints reach 1,000 steps without falls, but all drift backward. At forward
  +0.10 m/s, actual X is -0.00908 m/s and XY MAE is 0.12147 m/s; return is
  309.33, including 221.27 foot-phase reward versus only 0.27 velocity reward.
  Scripted standing XY speed remains 0.05111 m/s. Inline checks omit actor
  noise and vector model variation; continuous hardware safety is not proven.
- Fixed primary XY headline scoring to use the same vector-error measurement
  as endpoint selection. XY progress and conditional start/stop logs now show
  XY MAE; forward-only behavior is preserved. Missing XY measurements remain
  unscored, without silently substituting X error. Regression first reproduced
  an incorrect 0.879 headline instead of 0.177 for 0.04 m/s lateral error.
- Prepared an unchanged-reward 800M continuation from the final `params.pt`,
  not `best_params.pt` (which is the untrained step-zero policy). Preserve
  native policy/Adam/LR/counters/Torch RNG state, 2,048 environments, Newton-10
  physics and servo/randomization settings. Forty-one evaluations retain
  checkpoints every roughly 20M added transitions, reaching 1,000,038,400
  cumulative transitions. See [review and GPU command](docs/design/xy_walking_training.md#confirmed-200m-review-and-continuation).
- Verification: all 231 tests pass, including end-to-end headline/endpoint/
  transition-axis regression, missing-XY handling, forward-only preservation
  and the 800M update/evaluation cadence. The actual final-200M checkpoint
  restores policy/Adam/LR/counters/CPU RNG/model contract exactly on Mac,
  then completes 80 added transitions / two finite native RSL-RL updates.
  The short smoke does not establish gait or CUDA-resume qualification.
- Independent Mac readiness evaluation: one held-out seed (1101), four
  environments, four cardinal directions plus their scripted start/stop
  schedules; all 32 episodes reach 1,000 steps without falls/non-finite state.
  Actor noise, vector model variation and reset/servo variation are enabled.
  Tracking remains poor, and scripted standing XY speed is 0.0533-0.0565 m/s.
  This reproduces the learning gap, not a reporting artifact; the small sample
  is not final walking/robustness or hardware qualification.

- Fixed the omitted TB GPU startup setting: training now defaults
  `XLA_PYTHON_CLIENT_PREALLOCATE=false` before importing GPU backends and logs
  its effective value. This prevents JAX's default 75% reservation from starving
  RSL-RL's Torch rollout storage. Explicit user overrides are preserved, as in
  WR1's runtime; PPO, model, randomization, 2,048 environments and training
  budgets remain unchanged. GPU memory fit still requires the GPU retry.
- Verification: all 229 tests passed, including fresh-process startup ordering
  and user-override regressions. A CPU XY smoke completed 80 transitions and
  two native RSL-RL updates with finite policy, Adam state and metrics; it
  skipped the ten-direction evaluator and does not establish CUDA memory fit.

## 2026-10-11

### Six-DOF hip-yaw model revision

- Imported the updated Onshape model with independent left/right hip yaw,
  neck yaw and head pitch. Re-imported the user's mass-property correction:
  neck/head bodies are 5.087/13.467 g, with total CAD mass 2.058328 kg.
  Preserve exported physical limits and inertias; neck/head, arms and waist
  stay at home while all twelve leg joints participate in walking PPO.
- Updated canonical/MJX models, actuator order, robot configuration, observation
  builders, action contracts, model-derived reflection and current training
  templates for 21 actuators / 12 leg actions. `wr2_proprio_v5` has 1005 actor
  and 1740 critic inputs. TB's Normal residual action and 0.25-rad scale remain
  unchanged; native checkpoints now guard model hash and actuator orders.
  Old ten-action policies require fresh training, not heading migration.
- Derive foot collision boxes, sole sites and home heights from the new CAD.
  Home/walk-home root heights are 0.289479/0.288261 m with 1 mm initial ground
  clearance. Six-DOF reference IK controls full foot orientation. Separate
  measured crouched CoM height (0.205281 m) from root-reference height
  (0.283261 m), and correct the viewer's CoM overlay accordingly.
- The old extra 10 mm reference crouch fails full orientation IK at WR2's
  90-degree knee boundary; 5 mm passes without loosening joint, ground, support
  or IK bounds. All 53 signed-XY geometry checks pass with at least 8.96 mm
  support margin. Preserve TB's timing/speed prior: measured leg-height ratio
  is 0.9782, corresponding to about 0.9891 speed/time scaling. Pure turning
  remains a separately tracked implementation, despite the new hip-yaw hardware.
- Reference diagnostics now reflect lateral commands as well as phase, and
  distinguish CAD home-site mismatch from generated-motion asymmetry. The
  updated model has 0.2515 mm home-sole mirror mismatch; generated XY foot
  displacement mirrors within 0.0086 mm. Do not force unequal CAD legs to
  share identical IK coordinates. See [the import and verification guide](docs/design/hip_yaw_model_import.md).
- Validation: 227 full-suite tests and ten focused import/symmetry tests pass;
  Ruff and diff checks pass. Canonical/MJX compile, two-second home/walk-home
  settling and deterministic re-import pass. The corrected model's fresh Mac
  native smoke completes 80 transitions / two finite RSL-RL updates and saves
  matching model-contract metadata plus finite policy/Adam state. It skips
  held-out rollout evaluation; no GPU training or physical hardware was started.

### Earlier signed-XY baseline on the five-DOF model

- Added a dedicated `ppo_walking_xy.yaml` baseline for signed forward/backward,
  left/right and diagonal walking, with 20% standing and 80% TB-style ellipse
  sampling. Preserve the historical forward-only configs and ten leg actions;
  waist yaw remains inactive and pure turning is explicitly excluded because
  WR2 lacks TB's independent left/right hip-yaw axes. Mirror loss is disabled
  as in TB's default trainer; PPO/rewards/servo limits/physics are unchanged.
- Extended the privileged reference to XY, with TB's heading-frame correction
  and lookup coverage for rotated commands. The existing WR2 periodic LIPM
  proxy now accounts for lateral stance-center advance; numerical five-DOF IK
  remains an explicit difference from TB's planner. All 49 lookup points pass
  unchanged geometry/support bounds with at least 8 mm of support margin.
- Added signed XY CLI evaluation/viewer support, conditional XY/X/Y errors,
  lateral start/stop tests and ten-direction checkpoint ranking based on XY
  error. Standing/mixed schedules remain unscored; lateral walking is measured
  against commanded Y rather than treated as drift. Existing autonomous-agent
  acceptance criteria remain forward-only; the new baseline uses manual runs.
  Ranked diagonals lie within TB's sampled ellipse, not its enclosing box;
  preflight checks their exact references in addition to the 49 lookup points.
  See [scope, parity rationale and GPU commands](docs/design/xy_walking_training.md).
- Validation: the final full-suite run passes 217 tests, including 14 signed-XY
  regressions for sampler parity, references, transitions and lateral scoring.
  Sampler golden values match current TB with turning excluded; independent
  LIPM finite differences, heading correction and physical joint limits pass.
  An 80-transition Mac native RSL-RL smoke completes two finite PPO updates
  and saves finite model/Adam state. It skips held-out evaluation; the full
  ten-condition CPU CLI smoke was stopped during first evaluation/compilation
  before updates, so complete CLI validation remains pending on GPU.
  No physical hardware or GPU training was started.

## 2026-10-10

- Confirmed the seed1 mirror-off review (`wr2_mirror_off_20261010_125338_seed1`):
  exact full-state restore, all 11 checkpoints finite, 200,007,680 added steps.
  Paired full-variation evaluation covered 4,608 episodes without falls or
  non-finite states. Selected best120 improves fixed-0.10 speed 0.0819 to
  0.0829 m/s, MAE 0.0265 to 0.0259 m/s, and heading 3.86 to 2.99 degrees.
  Scripted stop MAE improves 0.0487 to 0.0373 m/s, but start MAE worsens
  0.0939 to 0.1017 m/s; this is a tradeoff, not completed walking. Final200
  is slower at 0.0796 m/s. Mixed-mode clipping and hardware qualification
  remain unresolved; no new restore/metric/solver bug was identified.
- Prepared another 200M/11-evaluation continuation from seed1 best120 using
  environment/evaluation seed 2. Only seed and version labels change;
  mirror-off TB-aligned PPO, rewards, actions, servo envelope, randomization
  ranges and physics remain unchanged. Preserve full native model/Adam/LR/
  counter/Torch-RNG state, retain best80 as fallback, and check launch behavior
  alongside tracking/stopping before promotion. See
  [the confirmed review and next GPU command](docs/design/mirror_off_seed2_continuation.md).
- Validation: 203 full-suite tests and five focused continuation regressions
  pass; Ruff and diff checks pass. An 80-transition Mac native MJX/Torch smoke
  restores actual best120 model/Adam/LR/counters/CPU RNG exactly, completes a
  finite PPO update and saves seed 2 with mirror loss off. Short smoke episodes
  verify startup, not gait qualification; no GPU training or hardware started.

- Confirmed the mirror-off 200M review (`wr2_mirror_off_20261010_083343_seed0`):
  exact full-state restoration and finite state across all 11 checkpoints.
  Paired full-variation native evaluation covered 6,144 episodes. Best80
  improved fixed-0.10 speed 0.0798 to 0.0819 m/s and MAE 0.0289 to 0.0265 m/s,
  with no falls in its 1,536 candidate episodes. Scripted stop MAE worsened
  0.0405 to 0.0487 m/s and heading increased; this is not final walking success
  or hardware qualification. Reject final200: 27/384 scripted and 5/384 random
  falls, with degraded tracking. No new PPO/solver root cause was established.
- Prepared another 200M/11-evaluation continuation from best80, changing only
  the environment/evaluation seed to 1 and version labels. Mirror loss stays
  off, matching ToddlerBot's default; rewards, actions, randomization ranges,
  servo envelope and physics remain unchanged. Full model/Adam/LR/counter/
  Torch-RNG restoration is retained, without resetting exploration. See
  [the confirmed review and next GPU command](docs/design/mirror_off_seed1_continuation.md).
- Validation: 201 full-suite tests and three focused continuation regressions
  pass; Ruff and diff checks pass. An 80-transition Mac native MJX/Torch smoke
  restores the actual best80 model/Adam/LR/counters/CPU RNG exactly, completes
  a finite PPO update and saves seed 1 with mirror loss off. The short smoke
  verifies startup, not gait quality; no GPU training or hardware was started.

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
