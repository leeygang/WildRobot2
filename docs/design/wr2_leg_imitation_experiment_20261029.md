# Leg-reference reward experiment — 2026-10-29

## Training focus

Test whether stronger moving leg-reference imitation makes PPO learn a useful
right hip swing and more even alternating placement while retaining commanded
progression. Reference generation, robot/model, actuator response, action
centers, reset and observations are identical across the two arms.

The [latest 200M comparison](wr2_hip_action_training_review_20261029.md) shows
that the expanded hip window recovers low-command progression but retains
nearly fixed right hip pitch. At .20, selected candidate target spans are
27.12° left /1.68° right, and landing leads are 8.23/2.01 TB cm. Its final
checkpoint has worse placement. Use selected **+173,588,480** as the common
starting policy.

## Why this setting first

Local TB **`paper-walking-baseline`** `paper.gin` sets XY tracking and leg
imitation coefficients to **5/5**. WR2 currently uses **15/5**. The executed
leg reward in both implementations is a negative mean squared error against
the moving reference, rather than departure from static home:

`r_leg(t) = -w_leg * mean((q_leg(t) - q_reference_leg(t))²)`.

Native WR2 component metrics already include control period `.02`. In the
selected model-home policy's native forward evaluation (1000 physical ticks):

| Component | Episode contribution |
|---|---:|
| XY tracking | +243.03 |
| Supported air time paid after sustained landing | +244.61 |
| Moving leg-reference imitation | −4.43 |
| Total native return | 665.16 |

Components reconstruct total return within .0006. Initial-to-selected return
increases 626.66→665.16 while leg imitation worsens −3.96→−4.43. The current
objective accepts progression gains alongside worse joint-reference tracking.
Reward magnitude alone does not prove component PPO gradient dominance; the
controlled training test measures whether changing this incentive helps.

**Change only `reward_weights.leg_motor_pos: 5 →15`.** Keep XY at 15.
This restores TB's **1:1 coefficient ratio** while preserving the WR2 XY
weight whose controlled fresh-training experiment substantially improved
progression and tracking. That earlier single-change experiment improved
held-out XY error about 39% and forward-.10 response .04467→.08787;
see the [XY15 result](wr2_progress_reward_result_20261025.md).

The absolute weight 15 is a declared WR2 experiment, not the exact TB recipe
or a demonstrated optimum. Coefficient ratio is not contribution ratio:
Gaussian speed reward and joint MSE have different units and numerical ranges.
On the same unchanged trajectory, leg contribution would become −13.29.
Only a learned policy can establish whether right swing and placement improve.

## Settings retained

| Setting | Both runs | Reason |
|---|---|---|
| XY tracking | 15, current desired-heading-frame formula | Preserve demonstrated commanded progression/heading behavior |
| Air time | 500, supported swing and ten-sample sustained landing | Retain the user's qualified-landing change |
| Model-home hip action centers | Enabled; other centers remain grounded home | Use the validated posterior window without another actuation change |
| Actual reset / observation home | Original grounded pose | Preserve checkpoint input coordinates and supported reset |
| PPO LR / entropy | 1e-4 / .0005 | Already match TB's effective optimizer settings; no evidence selects a simultaneous change |
| Rollout / PPO passes | 20 ticks, four epochs ×four minibatches | Existing TB-aligned trainer layout |
| Discount / GAE | .97 / .95 | Keep temporal learning settings identical across arms |
| Network / distribution | 512/256/128, SiLU, state-dependent tanh Gaussian | Existing TB-aligned network and distribution |
| Initialization | Same +173M actor/critic/normalizer, fresh Adam in both | Isolate reward from policy/exploration initialization |
| Budget / seed | 200M additional steps, seed 3 | Sufficient controlled learning budget |

Command, timing and length conversions remain `v_TB=v_WR/s`,
`t_TB=t_WR/s`, `length_TB=length_WR/s²`, with `s=.8240547565`.
Imitation uses joint angles in radians; the weight change is an objective
choice, not an additional robot-size scaling rule.

The existing moving reference supplies phase-dependent targets for both legs.

## GPU commands

Version 0.33.0 provides two complete matched configs. They differ only in
leg-reference coefficient and run labels. Both restore:

`results/wr2_walking/wr2_paper_hip_model_home_20261028_224138_seed3/checkpoints/000173588480`.

Checkpoint SHA256:
`e2f36e5bb3ef242da7d910e237073f29af5bc2662f9529f1c678c38992d4680f`.

From the updated GPU checkout, run sequentially:

```bash
uv run --locked --python 3.10 --extra training python -m wr2.locomotion.train \
  --config wr2/locomotion/configs/ppo_walking_leg_imitation5.yaml \
  --restore-checkpoint results/wr2_walking/wr2_paper_hip_model_home_20261028_224138_seed3/checkpoints/000173588480

uv run --locked --python 3.10 --extra training python -m wr2.locomotion.train \
  --config wr2/locomotion/configs/ppo_walking_leg_imitation15.yaml \
  --restore-checkpoint results/wr2_walking/wr2_paper_hip_model_home_20261028_224138_seed3/checkpoints/000173588480
```

The unchanged continuation is necessary: the preceding hip-center comparison
started from a different parent and cannot isolate the effect of this new
reward change. Output prefixes are `wr2_paper_leg_imitation5` and
`wr2_paper_leg_imitation15`. All intermediate checkpoints remain retained,
roughly every 13.35M steps; the rounded budget is 200,294,400 additional steps.

## Setup verification

Complete typed-config comparisons reproduce the saved parent recipe and
confirm the only treatment difference is the leg coefficient. Both effective
YAML round trips preserve every setting. Three existing reward/reference/
landing regression checks pass.

The saved +173M checkpoint restores without input conversion. A five-step
native MJX comparison using the same actor actions produces identical simulated
qpos and observations in both arms, zero falls/nonfinite states and identical
other reward components. Leg reward scales by three; total reward change agrees
with the expected extra leg penalty within 4.20e-8. This checks application of
the configured coefficient, not its effect on learning. The record is at
`results/wr2_walking/wr2_paper_hip_model_home_20261028_224138_seed3/leg_imitation_setup_mac/review.json`.

## Success and interpretation

Use common .10/.15/.20 TB commands and paired reset draws to compare actual
progress, heading, standing/transitions, falls, self-contact and native landing
geometry. Compare both arms against their common initial policy. The first
decision is whether the right hip actually gains useful cyclic motion and
the right foot lands farther forward, reducing the left/right placement gap.
Check both same-foot strides and inter-foot landing lead; they are different
measurements. Do not impose perfect numerical symmetry as a success rule.

The stronger imitation policy must preserve useful forward progression and
stability. Lower RMSE achieved by changing only a constant pose, or smaller
rocking because walking slowed, is insufficient. Inspect intermediate and
final policies independently; the existing automatic XY selector does not
measure geometric placement. Raw training return is not comparable across
different reward weights.

If reference error improves but right swing/placement does not, the proposed
reward balance has not solved the goal. Do not continue increasing weights
without evidence. The next training-setting investigation would then be
phase-specific PPO pressure and exploration/initialization of the retained
asymmetric actor, using the existing diagnostic rather than changing physics.

References: local TB Paper `paper.gin` lines 28/32 and `mjx_env.py` leg reward;
[TB paper v2, Tables VI–VII](https://arxiv.org/html/2502.00893v2);
[latest trained-policy review](wr2_hip_action_training_review_20261029.md).
