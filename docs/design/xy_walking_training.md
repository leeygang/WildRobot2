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
The complete training/evaluation CLI still needs confirmation on GPU.

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
