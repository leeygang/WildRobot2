# TB-aligned signed XY walking — 2026-10-11

## Scope and hardware

Use `wr2/locomotion/configs/ppo_walking_xy.yaml` for the new task: standing,
forward/backward, left/right and diagonal translation. Turning is disabled.
The updated model has 12 leg actions. Old ten-action checkpoints are historical
and cannot be restored on this model; the autonomous
walking-agent campaign still has forward-only acceptance criteria. Use the
manual training/evaluation commands below for this XY baseline.

ToddlerBot has **two independent hip-yaw axes**, one in each leg. On 2xc each
has a drive/driven gear pair (`left_hip_yaw_drive` / `left_hip_yaw_driven`,
and the corresponding right pair). It also has waist yaw. WR2's new export
likewise has six actuated joints per leg, including independent hip yaw.
Keep only the twelve leg actions active; waist, arms, neck and head stay at
home. The mass/inertia of those inactive parts still affects walking.
Turning remains a separate reference/command implementation, not something
enabled automatically by adding the hip-yaw hardware. See
[the model import and mass-property audit](hip_yaw_model_import.md).

## Command and PPO parity

Local sources of truth:

- TB `toddlerbot/locomotion/walk_env.py:235`: signed ellipse-style XY sampler.
- TB `walk.gin:2,46`: 1B walking budget, X [-0.10,0.10] m/s,
  Y [-0.05,0.05] m/s, XY radial deadzones 0.05 m/s.
- TB `mjx_config.py:194`: 3-second resampling, 20% standing and optional 20%
  pure turning; WR2 excludes turning and allocates that probability to XY.
- TB `walk_zmp_ref.py:141`: heading-corrected XY lookup in the critic.
- TB `train_mjx.py:981`: mirror loss excluded unless explicitly requested.
- WR1 `training/algos/ppo/ppo_core.py:533`: mirror loss is likewise optional.
- Public reference: [ToddlerBot paper](https://arxiv.org/abs/2502.00893).
  Latest active local implementation takes precedence over older paper details.

The new sampler matches TB with `zero_chance=.2, turn_chance=0`: 20% standing
and 80% signed XY walking. It uses the same theta/radial random draws, including
the shared draw for X/Y. It is not a uniform Cartesian command box. Portable
regression tests include golden outputs extracted from the current TB method.

The new config retains Normal/log-std RSL-RL PPO, adaptive KL, existing network,
rewards, action scale/delay, domain randomization,
servo model and Newton-10 physics. Mirror coefficient is zero. This is an
expanded-task TB-default baseline, not a controlled mirror-only experiment.
The new model requires 1005 actor / 1740 critic inputs and twelve leg actions;
all six-DOF pose weights and reflection transforms follow the current geometry.

Measured canonical walk-home hip-to-foot height: WR2 0.23820 m, compared with
TB2xc home 0.24350 m. Length ratio 0.9782 implies dynamically similar
speed/time scale sqrt(ratio) about 0.9891. Retain TB's initial speed bounds,
0.72 s cycle, 0.04 m swing and
20 ms control, documenting this small scaling approximation. WR2's lower mass
and different actuators remain represented in its model and hardware limits.

## Reference implementation and checks

The critic lookup now conditions on both X and Y and applies TB's
`reference_heading - torso_heading` XY correction. Translation lookup bounds
also cover rotated commands. The deployable actor receives no reference input;
actor receives the model-sized observation and uses twelve leg residuals.

WR2 retains its existing periodic LIPM reference and model-derived numerical
six-DOF IK, rather than copying TB's analytic IK or its full preview-controller
trajectory. This is an explicit implementation difference, not exact planner
parity. Lateral stepping moves the stance centers twice per cycle: an analytic
second harmonic follows that advance, with each harmonic satisfying
`ZMP = CoM - h/g * CoM_acceleration`. At zero lateral speed the historical
forward trajectory equations are unchanged. LIPM CoM height is now measured
from the actual crouched model, separately from the reference root height.
The reference's additional 5 mm crouch respects WR2's 90-degree knee limits
with full foot-orientation IK, without relaxing geometric gates.

The initial uncompensated XY extension failed the extra heading-envelope
support checks by about 4 mm. The lateral-advance correction passes every
point of the 49-command lookup, including standing and envelope corners,
plus four exact ranked diagonals: minimum support margin 8.96 mm. Existing
ground/support/IK bounds are not relaxed. Tests independently check the LIPM
acceleration by finite differences, physical joint ranges, reference continuity,
stationary stance-foot XY motion and the heading correction.

A full-variation Mac native smoke completes 80 transitions and two RSL-RL PPO
updates with finite model, Adam state and metrics. It skips held-out evaluation.
The full ten-condition CLI smoke was stopped during expensive first CPU
evaluation/cold compilation, before any update; it is not counted as passed.
The subsequent `wr2_xy_20261012_085006_seed0` GPU run completed 200,007,680
transitions / 4,883 native updates and all eleven evaluation checkpoints.
This confirms the full GPU path, not walking or hardware qualification.

These are reference geometry/implementation checks, not proof of learned
walking, dynamic balance, continuous servo thermal safety or deployment readiness.

## Training and evaluation

The updated CAD neck/head masses have been re-imported and geometry rechecked.
They still need hardware mass/servo qualification before deployment.
After pushing the preparation commit, on GPU:

```bash
git pull --ff-only
uv run --extra training python -m wr2.locomotion.train \
  --config wr2/locomotion/configs/ppo_walking_xy.yaml
```

The trainer defaults `XLA_PYTHON_CLIENT_PREALLOCATE=false` before GPU backend
imports, matching TB's `train_mjx.py:12` and WR1's training runtime. JAX grows
its pool on demand instead of reserving 75% of VRAM before Torch allocates PPO
storage. Startup logs print the effective setting; explicit environment
overrides are preserved. Keep this setting false when sharing the RTX 5070
between MJX and RSL-RL. See [JAX GPU memory allocation](https://docs.jax.dev/en/latest/gpu_memory_allocation.html).

Launch without `--restore-checkpoint` for the fresh 1B baseline. Run ID is
generated as `wr2_xy_<timestamp>_seed0`. There are 51 evaluations including
step zero, roughly every 20M added transitions. Review intermediate results
at 200M, not only the final model. Old forward checkpoints remain historical
evidence; they cannot be replayed on the new dynamics for a paired comparison.
For a bounded initial 200M run, append `--num-timesteps 200000000 --num-evals 11`.

Each training evaluation covers ten fixed directions: X +/-0.05 and +/-0.10,
Y +/-0.05, and four diagonals (X +/-0.07071, Y +/-0.03536), inside TB's sampled
ellipse rather than the unsampled enclosing-box corners. Preflight also checks
the four exact diagonal references, for 53 reference conditions in total.
Checkpoint selection uses the worst
direction's walking score based on XY-vector tracking error, not X error alone.
Primary progress metrics still show fixed forward 0.10 for historical comparison.
The existing training transition evaluator remains forward-only; independent
scripted XY transitions below are required before promotion.

## Confirmed 200M review and continuation

The first six-DOF XY run learned stable alternating stepping, not commanded
translation. All ten final inline endpoints reached 1,000 steps without falls,
but all moved backward: at X +0.10, actual X was -0.00908 m/s with 0.12147 m/s
mean instantaneous XY error. Lateral +/-0.05 commands produced actual Y of
+0.00508/+0.00355 m/s. Forward return rose 201.55 to 309.33, with 221.27 from
foot-height tracking and only 0.27 from XY velocity tracking. The standing
part of the scripted forward schedule still had 0.05111 m/s XY speed.
Inline evaluation disables actor noise and omits vector model variation,
while retaining local reset/servo variation; these are not held-out guarantees.

The reporting fix makes the primary headline score use the same XY error as
endpoint ranking, and logs sample-count-normalized XY start/stop errors.
Historical forward-only scoring remains unchanged. Missing XY measurements
remain unscored rather than falling back to X error. Reward computation,
PPO, reference, physics, servo model and randomization are unchanged.

`best_params.pt` for this run is the untrained step-zero policy, because its
worst-direction score exceeds all trained checkpoints. Continue from the
**final `params.pt`**, not that selected-best file. Its SHA256 is
`f2033077df91a2db70371087083b66ff72f8b4d89b23e14af4df03a7bedc0e19`.
The model hash matches the current twelve-action geometry. After pushing the
preparation commit from Mac, run on GPU:

```bash
git pull --ff-only
uv run --extra training python -m wr2.locomotion.train \
  --config wr2/locomotion/configs/ppo_walking_xy.yaml \
  --num-timesteps 800000000 --num-evals 41 \
  --restore-checkpoint results/wr2_walking/wr2_xy_20261012_085006_seed0/params.pt
```

This adds 19,532 updates / 800,030,720 actual transitions, reaching cumulative
24,415 updates / 1,000,038,400 transitions, the TB-aligned 1B budget.
The 41 evaluations include the restored step-zero policy and retain checkpoints
roughly every 20M **added** transitions. Run IDs are generated automatically;
checkpoint filenames and progress steps are run-relative, while native
`total_steps` and `iteration` counters remain cumulative. Full policy/Adam/LR/
Torch RNG/model-contract restoration is retained. Expected restored LR is
1.5e-5 and mean residual std 0.06903 (about 0.99 degrees of target noise),
not the fresh-policy std of 0.5. Simulation state is reset on resume.

Keep the original run and every intermediate checkpoint. Review at each 200M
milestone; extending the budget is not a guarantee that the learned stepping
pattern will acquire command tracking. Do not add acquisition-only rewards or
reset exploration as part of this continuation. The existing TB velocity and
foot-phase kernels are preserved; see the local-code references above.

Preparation verification: all 231 tests pass, including an end-to-end progress
regression (headline, endpoint ranking, XY switch log, missing-XY handling and
forward-only preservation) and the 800M cadence calculation. A native Mac
smoke restores the actual final-200M policy/Adam/LR/counters/CPU RNG/model
contract exactly, completes 80 added transitions / two finite PPO updates
with full configured variation, and saves matching cumulative counters.
Its five-step episodes and omitted evaluator test startup, not walking quality
or CUDA restoration. No hardware motion is included.

A separate full-variation CPU readiness probe evaluates seed 1101 with four
environments, four cardinal directions and all four scripted start/stop
schedules: 32 completed episodes, all 1,000 steps with no falls/non-finite
states. It reproduces poor command tracking and 0.0533-0.0565 m/s standing
XY speed. Reports are `results/wr2_xy_smoke/held_out_cardinal_200m_ready.json`
and `held_out_transition_200m_ready.json`. This is a small-sample implementation
check, not final robustness qualification or evidence that longer training
will necessarily succeed.

For numeric reference checks (Mac or GPU, no hardware):

```bash
uv run python -m wr2.tools.view_zmp_reference \
  --config wr2/locomotion/configs/ppo_walking_xy.yaml \
  --xy-commands '0.1,0;-0.1,0;0,0.05;0,-0.05;0.1,0.05;-0.1,-0.05;0,0' \
  --validate-only
```

For visual reference playback on Mac, use the same module through `mjpython`,
remove `--validate-only` and choose a single `--xy-commands '0,0.05'`. The
camera and planned CoM/ZMP overlays follow motion in both axes.

After copying the run locally, set `XY_RUN` to its actual result directory:

```bash
XY_RUN=results/wr2_walking/wr2_xy_REPLACE_WITH_RUN_ID_seed0
uv run python -m wr2.locomotion.evaluate \
  --config "$XY_RUN/training_config.yaml" --checkpoint "$XY_RUN/best_params.pt" \
  --allow-cpu --seeds 707,808,909 --num-envs 128 \
  --xy-commands '0.05,0;0.1,0;-0.05,0;-0.1,0;0,0.05;0,-0.05;0.070710678,0.035355339;0.070710678,-0.035355339;-0.070710678,0.035355339;-0.070710678,-0.035355339' \
  --output "$XY_RUN/held_out_xy.json"
```

Repeat with `--transitions` and a separate output to test standing/start/stop
for every direction. Also use `--random-commands` without `--xy-commands` for
the unaltered training-distribution transitions. Compare retained intermediate
checkpoints, not only `params.pt`. Include new validation seeds before final
qualification; the listed seeds have already been used in earlier reviews.

Schema v3 reports both commanded axes, full distribution metadata and
sample-count-normalized XY/X/Y tracking. Lateral error is measured against
commanded Y, so intentional side-stepping is not called drift. Standing and
mixed-command schedules have no fixed walking score. Inspect fall/length,
per-axis and XY tracking, contact/phase, target clipping and servo metrics;
do not infer success from a signed velocity mean that cancels across directions.
The old forward-only walking-agent gates are not XY success gates.
