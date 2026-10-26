# Heading-error feedback experiment — 0.30.0

## Question and scope

Does giving actor and critic the desired-heading error let PPO use the existing
orientation reward to reduce persistent turning?

This is a **feedback-only experiment**. The XY training reward remains in the
body frame with coefficient15. All reward formulas and weights, including yaw5,
orientation1, moving leg-reference imitation and qualified landing, retain the
completed 0.29.0 setup. Action origin/scale, physics, ZMP reference, command/noise
distributions and PPO settings also retain that setup. The reference-frame XY
reward proposed in the [GPU review](wr2_heading_gpu_review_20261025.md) is a
separate follow-up if this input change is insufficient.

TB `paper-walking-baseline` observes actual Euler yaw and gyro plus commanded
yaw rate, while its orientation reward uses a reference quaternion omitted from
actor/critic inputs. WR2 previously had the same limitation. Explicit heading
error is an intentional extension of this observation contract. Sources:
[ToddlerBot Table VII](https://arxiv.org/html/2502.00893v2), local
`paper-walking-baseline:toddlerbot/locomotion/mjx_env.py`, and
[Legged Gym's heading feedback](https://github.com/leggedrobotics/legged_gym/blob/8fa29acc6fd1910c3d9659eef6310bdd301cde0a/legged_gym/envs/base/legged_robot.py#L327).

## Implementation

For each history frame, append:

```
sin(actual_yaw - reference_heading)
cos(actual_yaw - reference_heading)
```

The actor derives these from its existing noisy yaw measurement, preserving
the original noise draw and its correlation with Euler yaw. The critic uses
true simulator yaw. The reference starts at the episode's initial heading and
integrates commanded yaw rate. The step's features use the same updated
reference heading as the existing orientation reward. Angles are dimensionless;
the existing TB velocity/time/size normalization remains unchanged.

Actor input grows **975→1005** (65→67 per frame), critic **1485→1515**
(99→101), with the existing 15-frame history. Original features retain their
ordering; each frame's two new features are appended. The Paper path previously
declared these additional dimensions when `heading_observation` was enabled,
but did not emit the features. That opt-in path is now completed; prior Paper
runs had the flag disabled.

The standard Brax trainer automatically expands the restored input kernels,
inserting **zero weights** at the two new columns in every frame. Existing
weights/biases, actor distribution and critic are retained. Normalizer history
is mapped identically, with mean/variance0 and standard deviation1 for the new
fields. Logits, deterministic actions and critic outputs are checked before
training at explicit high matrix precision; this verification setting does not
change the training precision. The conversion is saved in `restore_report.json` and run provenance.
Native Brax restores parameters with new Adam state and a new step counter.

## Evaluation and checkpoints

Independent evaluation now logs reference-frame XY velocities/error and mean
absolute wrapped heading error alongside existing body-frame metrics. The
experiment selects retained candidates by **reference-frame XY error**, within
the existing finite/≤1% fall gate, across the standard 15 Paper conditions.
This changes checkpoint reporting/selection, not PPO's reward or updates.
All intermediate checkpoints remain saved.

The .15/.20 high-command comparison remains a separate 20-TB-second heading
evaluation; these are not added to the standard Paper condition list. Retained
checkpoints should be reviewed at those commands, rather than treating the
automatically selected checkpoint as sufficient evidence of improvement.

First milestone at .20: approach the earlier policy's **.14 TB m/s** progress
in the requested direction and roughly **45°** or less absolute net turn,
while preserving stability. Check .10 tracking and .15/.20 posture and lateral
drift. If feedback alone fails, the result tests the observation hypothesis;
it does not invalidate the independently supported reference-frame XY reward.

## Run on the GPU host

From the WR2 checkout, first verify restored inference and physics:

```bash
uv run --locked --python 3.10 --extra training python -m wr2.locomotion.train \
  --config wr2/locomotion/configs/ppo_walking_heading_feedback.yaml \
  --restore-checkpoint results/wr2_walking/wr2_paper_progress_20261025_085931_seed3/checkpoints/000300134400 \
  --smoke --smoke-steps 20
```

Then start the **200M-step continuation** with the standard trainer:

```bash
uv run --locked --python 3.10 --extra training python -m wr2.locomotion.train \
  --config wr2/locomotion/configs/ppo_walking_heading_feedback.yaml \
  --restore-checkpoint results/wr2_walking/wr2_paper_progress_20261025_085931_seed3/checkpoints/000300134400
```

The restored-policy input expansion is automatic; no migration wrapper script
or TB checkout is required. Keep the restore argument: omitting it initializes
a fresh network. New runs use `results/wr2_walking/wr2_paper_heading_feedback_*`.

## Verification artifacts

Local preparation checks are under the base run's `heading_feedback_preparation`.
The checkpoint check initially found maximum differences of **7.2e-7 in policy
logits, 1.5e-7 in actions and 2.9e-6 in critic values**. Additional tests cover
real Paper frame dimensions, noisy heading feedback, reference update timing,
unchanged physical transitions/rewards, history-column mapping, native restore
and a native PPO update. The config delta consists of heading observations,
the 200M continuation budget, evaluation selection and version/output identity.
Across 2,472 recorded actor observations at .10/.15/.20, expanded checkpoint
logits agree within **1.2e-6** and initial actions within **3.9e-7**.
The standard Mac training entry point passed its full **227-command** reference
preflight and **20 restored-policy physics steps**, with actor1005/critic1515,
finite state/reward and no fall. A native Brax PPO update also passed with the
expanded restored parameters. Legacy evaluation regression checks include the
existing discount parameter in their test fixture.
