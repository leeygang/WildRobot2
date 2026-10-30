# Shoulder rocking review — 2026-10-29

## Confirmed motion

The latest viewer session is
`results/visual_eval/wr2_20261030_040444_434170/`: UTC October 30 04:04,
Pacific October 29 21:04. It loads weight15 `best_params`, confirmed as
checkpoint +186,941,440, at .20 TB-equivalent m/s. The checkpoint hash matches.
The session has 160 ticks/3.2 physical seconds, no reset or failure, and
98 samples after excluding the first 62 warmup ticks.

Integrated qpos was forwarded through the native model to recover shoulder
mount positions. All ranges below are post-warmup p95−p5:

| Measurement | Actual viewer |
|---|---:|
| Torso roll span | 9.49° |
| Root height span | 7.69 physical mm /11.33 TB mm |
| Left shoulder height span | 13.60 physical mm |
| Right shoulder height span | 17.07 physical mm |
| Left-minus-right shoulder height span | 27.60 physical mm |
| Mean shoulder height span | 7.42 physical mm |
| Largest actual arm-joint span | <.007° |

Shoulder height correlation is −.75: their alternating motion is primarily
torso roll, with common vertical root motion superimposed. The shoulder mounts
are fixed offsets from the torso; this is body movement. The longer nominal
20-TB-second replay independently gives similar roll/root-height spans,
9.37° and 11.25 TB mm. The short viewer session is not a robustness test.

The numerical record is the viewer folder's `rocking_review.json`.

## Why the current reward permits it

Both WR2 and local TB **`paper-walking-baseline`** use roll-range weight .5
and tolerance ±.1rad/±5.73°. Its value is flat throughout that band. All
post-warmup viewer samples have the maximum roll-range reward, .01 after
weight/control-period multiplication. This particular term provides no
additional benefit for reducing their roll amplitude.

The other torso terms still penalize motion: full orientation reward,
angular-XY velocity reward and vertical velocity reward. Thus this is not a
missing torso reward or evidence that all training rewards ignore rocking.
The learned policy accepts the remaining motion under their combined objective.

WR2's shoulder mount spacing is **167.2mm**, versus TB's **134.0mm**.
With the existing CoM-based length conversion `s²=.67906624`, its normalized
shoulder width is **1.84× TB's**. For nearly symmetric mounts,

`left_Z - right_Z ≈ shoulder_width * cos(pitch) * sin(roll)`.

Width determines how much a given roll moves the shoulders. It does not
establish why the policy chose that roll or why right-foot placement is uneven.
The current selected policy rolls less than the earlier native TB .20 sample;
TB also has visible body motion. Copying its raw angular tolerance does not
give the same tolerance for size-normalized shoulder displacement on WR2.

## Proposed training change

First target the dominant alternating component with **TB's existing roll
reward evaluated at a roll angle scaled for shoulder width**. Keep weight .5.
At small angles the equivalent TB roll is approximately `1.84 * WR2_roll`,
giving an effective WR2 tolerance of about **±3.1°**, versus current ±5.73°.

The exact zero-pitch tolerance that preserves TB's permitted normalized
shoulder-height difference is

`asin(s² * TB_width / WR2_width * sin(.1)) = .05436rad = 3.11°`.

This is a declared WR2 geometry adaptation of TB's angular reward. It creates
reward differences within the presently observed oscillation range. It has
not been implemented or shown to improve a trained policy. Body weight transfer
must still work; successful training must preserve progression and placement.

The proposed fresh-initialization study can supply a clean comparison:

| Arm | Initialization | Leg/XY weights | Roll tolerance |
|---|---|---|---|
| Fresh baseline | Same native fresh seed-3 initialization | 15/15 | Existing ±5.73° |
| Fresh body treatment | Same initial actor/critic/normalizer | 15/15 | Width-scaled, approximately ±3.1° |

Run 200M per arm with identical reference, dynamics, command distribution,
landing rule and PPO settings. The completed weight15 continuation remains
the comparison for initialization; these two fresh arms isolate the added
body-reward effect. This preserves the separate questions of policy
initialization and roll tolerance.

Judge torso roll, each shoulder's height span, shoulder differential/common
motion, root-height variation, actual commanded progress, placement and falls
at .10/.15/.20. Do not select lower rocking achieved by slowing or stopping
walking. Retain intermediate checkpoints.

This roll change targets the alternating shoulder component. Common root
bobbing is separate. If it remains excessive, investigate the existing
vertical-velocity term in a subsequent single-setting experiment. Its current
weight is 1; increasing it alongside the roll change would obscure attribution.

References: local TB Paper `walk_env.py::_reward_torso_roll`,
`mjx_env.py` torso/velocity rewards, and
[TB paper v2, Tables VI–VII](https://arxiv.org/html/2502.00893v2);
[trained-policy review](wr2_leg_imitation_result_20261029.md).
