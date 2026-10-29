# Hip action-center PPO comparison — 2026-10-28

## Purpose

Compare additional learning with the current action window against learning
with the tested model-home hip window. The [GPU validation](wr2_hip_action_gpu_review_20261028.md)
found usable initialization and better posterior stance reference coverage,
while the frozen candidate remained slower and asymmetric. These runs test
whether PPO can use that added posterior range to improve right-foot placement
and forward progress.

Both runs continue the qualified goal-XY checkpoint:
`results/wr2_walking/wr2_paper_goal_xy_20261026_193545_seed3/checkpoints/000200294400`.
Expected checkpoint SHA256:
`689ebaf1c0c8b10508bc674207954e5b433385a62eb071aa918de23b6f153253`.

## Configurations

| | Control | Candidate |
|---|---|---|
| Config | `ppo_walking_hip_control.yaml` | `ppo_walking_hip_model_home.yaml` |
| Hip-pitch target center | Grounded home, about 16.63° forward | Model home, about 6.88° forward |
| Output prefix | `wr2_paper_hip_control` | `wr2_paper_hip_model_home` |
| Additional budget | 200M | 200M |

Version 0.32.0 adds one default-false configuration flag,
`environment.model_home_hip_pitch_action_center`. Only the named left/right
hip-pitch action centers change when enabled. Native delay, target bias,
mechanical bounds, quantization and slew remain in the production target path.

Both complete configurations preserve the qualified goal-XY recipe: seed 3,
1024 environments, .25-rad residual scale, fixed 1e-4 learning rate, original
actor/critic/normalizer, and the same fresh optimizer restart used by native
Brax parameter-only continuation. Both keep heading feedback, reference-frame
XY tracking, physics/randomization, commands, gait timing, all reward weights,
and air-time credit after sustained landing. No learning rate, entropy or
initial distribution adjustment is introduced.

The grounded actual reset, knee/ankle centers, position-observation baseline,
standing-reward home and initial zero action history remain identical. Thus
initial observations and actor outputs match. The candidate's physical target
interpretation changes immediately; the frozen-policy speed/backward-lean
tradeoff measured in the GPU validation is its starting condition. This is
not a behavior-preserving checkpoint conversion. Holding offsets validated
by the diagnostic are not applied to training.

Local TB **`paper-walking-baseline`** `mjx_env.py` uses a fixed model-home
center + .25-rad delayed residual, and moving leg-reference imitation.
The candidate follows its hip-center choice while retaining WR2's supported
knee/ankle pose. Wholesale model-home replacement is outside this experiment:
WR2's grounded knee would exceed that model-home residual window.
See [TB paper v2](https://arxiv.org/html/2502.00893v2) and the linked GPU audit
for the measured rationale.

## GPU commands

After syncing the local implementation commit to the GPU checkout, run these
sequentially from that checkout. Both use the same original checkpoint.
The regular training entry point creates timestamped result folders.

Control:

```bash
uv run --locked --python 3.10 --extra training python -m wr2.locomotion.train \
  --config wr2/locomotion/configs/ppo_walking_hip_control.yaml \
  --restore-checkpoint results/wr2_walking/wr2_paper_goal_xy_20261026_193545_seed3/checkpoints/000200294400
```

Candidate:

```bash
uv run --locked --python 3.10 --extra training python -m wr2.locomotion.train \
  --config wr2/locomotion/configs/ppo_walking_hip_model_home.yaml \
  --restore-checkpoint results/wr2_walking/wr2_paper_goal_xy_20261026_193545_seed3/checkpoints/000200294400
```

The startup banner identifies the hip center. `run_config.json` records both
`policy_home_motor_position_rad` (the preserved observation/reset home) and
`policy_action_center_motor_position_rad` (the actual target center).
The effective saved YAML preserves the flag for evaluation and live viewing.
PPO and walking diagnostics now use that actual center when reconstructing
reference actions, coverage and learning direction; historical runs retain
the original mapping.

## Local verification

58 tests pass across the hip validation, walking diagnostics, Paper recipe
and native PPO restore suites. Checks cover unchanged historical mapping,
opposite hip signs, candidate/reference-action conversion, identical reset
observations, native holding-target equivalence, delayed target reconstruction,
and parameter-only PPO continuation.

Both complete configurations match the original run's saved recipe after
accounting for run labels and the candidate flag. Both saved-config round trips
preserve the flag. The original qualified checkpoint loads without any heading
input conversion; both mappings produce identical initial deterministic actions.
Standing and .20 TB-command startup checks complete 20 steps per origin/case
(80 native transitions total), all finite, without falls and with zero target
reconstruction error. The record is at
`results/wr2_walking/wr2_paper_goal_xy_20261026_193545_seed3/hip_action_training_setup_mac/review.json`.
These are nominal CPU startup checks, not a trained-gait qualification. The
earlier GPU validation supplies the larger physical-startup comparison.

## Comparison and selection

Both retain checkpoints at every evaluation, about 13.35M steps apart;
the rounded final budget is about 200.2944M additional transitions. Keep the
standard selector unchanged and independently inspect intermediate/final
policies for the forward-walking priority.

Compare both runs against their step-zero policy and the original qualified
checkpoint at .10/.15/.20 TB commands, with paired reset draws and the same
randomized no-push profile. Include standing and stop/restart behavior. The
primary measurements are requested-direction progress, right landing lead,
left/right same-foot strides, falls, and foot self-contact. Check right-hip
boundary occupancy and delivered-target motion to understand the change.
Assess torso rocking and height variation at matched achieved speed.

Select the candidate only if it recovers forward progress and improves actual
placement without a material standing/stability regression. Lower saturation,
better reference coverage or smaller rocking at slower speed is insufficient.
This comparison can establish whether the changed action window helps PPO;
it does not assume that every excluded reference angle is necessary for gait.
