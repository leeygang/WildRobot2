# Goal-frame XY continuation — 0.31.0

The run completed. See the [result and next action-space audit](wr2_goal_xy_result_20261027.md).

## Purpose and evidence

Continue the qualified+120M heading-feedback policy for200M additional PPO
steps. The single training change is XY velocity tracking in the desired
heading frame, with coefficient15 and the existing TB-normalized velocity
units. The goal is better commanded forward progression with less uncommanded
turning. Record foot contact and landing geometry to assess gait quality.

The [full confirmation](wr2_heading_feedback_120_confirmation_20261026.md)
qualified+120M on both seeds, with1/3,840 falls and no nonfinite states.
The [GPU comparison](wr2_heading_feedback_gpu_review_20261026.md) found
13.5° net turn for+120M versus28.1° for+173M at .20 TB m/s. Their native
returns were almost equal; the frozen-policy goal-frame XY calculation favored
the straighter policy. Its reset-discounted comparison did not clearly
separate them, so the proposed PPO benefit remains an experimental question.

TB `paper-walking-baseline` uses actual torso-frame XY velocity. This is an
intentional departure supported by the measured direction loss and the
objective comparison. Sources: local Paper `mjx_env.py:_reward_lin_vel_xy`,
[ToddlerBot Table VII](https://arxiv.org/html/2502.00893v2), and the reviews above.

## Implementation

With velocity scale `s=.8240547565`, the weighted per-step XY reward becomes:

```
heading_goal = previous_heading_goal + commanded_yaw_rate * dt
v_goal = Rz(-heading_goal) * cached_world_torso_velocity
r_xy = 15 * exp(-100 * ||v_goal.xy / s - command.xy / s||²) * dt
```

The heading starts at reset heading and integrates commanded turning. It uses
the same updated goal as orientation reward and actor/critic heading features.
Cached velocity sampling is identical to the native TB-style rewards.

For example, at a .10 TB m/s command, a policy moving at .10 along its body
axis while facing60° away receives native XY reward1.0 before weighting.
Goal-frame tracking gives `exp(-1)=.368`, because the motion misses the
commanded direction. Exact commanded-world translation receives goal reward1
even when the body temporarily faces another direction.

`environment.reference_frame_xy_tracking` is false by default for existing
configs. The0.31.0 config enables it and requires the Paper recipe plus heading
observations. The override changes only `lin_vel_xy`; the vertical/angular
velocity rewards, leg imitation and other terms use their existing inputs.
The standard trainer prints the actual XY reward frame at startup. Config
snapshots record the flag for later reconstruction.

Heading observations remain1005 actor/1515 critic inputs, so the+120M
checkpoint needs no input expansion. Its actor, critic and normalizer restore
exactly. Native Brax starts new Adam state and a new step counter. Physics,
action mapping, ZMP reference, qualified landing500, command/randomization
distributions, PPO settings and the16-evaluation checkpoint schedule match
0.30.0. The output prefix is `wr2_paper_goal_xy`.

## Run

From the GPU WR2 checkout containing the new implementation:

```bash
uv run --locked --python 3.10 --extra training python -m wr2.locomotion.train \
  --config wr2/locomotion/configs/ppo_walking_goal_xy.yaml \
  --restore-checkpoint results/wr2_walking/wr2_paper_heading_feedback_20261025_201358_seed3/checkpoints/000120176640
```

This targets200M **additional** steps. The restore argument selects the
qualified source policy; `best_params` in that source run identifies+173M.
New results appear under `results/wr2_walking/wr2_paper_goal_xy_*`.

Optional GPU startup check using the same command:

```bash
uv run --locked --python 3.10 --extra training python -m wr2.locomotion.train \
  --config wr2/locomotion/configs/ppo_walking_goal_xy.yaml \
  --restore-checkpoint results/wr2_walking/wr2_paper_heading_feedback_20261025_201358_seed3/checkpoints/000120176640 \
  --smoke --smoke-steps 20
```

## Verification and review criteria

Local verification covers config delta/defaults/guards, analytic commanded
and misdirected motion, arbitrary global heading, torso pitch and preserved
non-XY rewards/observations. The existing TB numeric reward oracle and heading
feature/timing checks pass. Actual+120M parameters restore exactly, and a
native Brax restored PPO update passes.

The standard training entry point passes the full227-command reference
preflight and20 restored-policy physics steps on the Mac, with finite state,
actor1005/critic1515, no fall and root height.28254m. GPU execution of the
new reward is verified by the next run's startup and training metrics.

Compare the new retained policies with qualified+120M and the+173M control:

- Standard15-condition tracking, standing/transitions, finite state and falls.
- .10/.15/.20 requested-direction speed, physical forward/lateral travel,
  mean and tail net turn, and forward lean under matching GPU draws.
- Modeled foot self-contact, contact-phase agreement and alternating landing
  placement, including any speed-versus-gait tradeoff.

Native return now uses a different XY objective. Use independently measured
tracking/progression metrics to judge improvement and retain intermediates
for evaluation rather than assuming the final checkpoint is best.
