# WR2 paper walking experiments

The original **0.28.0** transfer completed 300M steps with no qualifying
checkpoint and substantial hopping. The corrected **0.28.2** run also completed
300M: tracking improved 26.5%, nominal neither-foot support fell from about
61% to 0%, but approximately 5% of evaluation episodes still fall.
A further **300M fine-tune from 260M** completed: best tracking improved only
2.5%, then final error regressed 15.3% versus the parent. No checkpoint passes
the fall gate. See the [latest completed-run review](wr2_paper_finetune_result_20261017.md)
for the TB comparison and measured PPO update instability. The recipe retains
the native paper weights, with corrected landing normalization and TB's summed
foot-force support rule.

## Completed experiment: 300M from the retained 260M checkpoint

The completed seed-1 experiment used the regular trainer on the GPU host:

```bash
uv run --extra training python -m wr2.locomotion.train \
  --config wr2/locomotion/configs/ppo_walking_paper.yaml \
  --restore-checkpoint results/wr2_walking/wr2_paper_20261016_222755_seed0/checkpoints/000260116480 \
  --seed 1
```

This adds **300M requested transitions** to the policy trained through
260,116,480. Native restore loads the actor, critic and normalizer; Adam and
step counters start at zero. Its `wr2_paper_20261017_080025_seed1` directory
records the effective configuration and restore command, with checkpoints
about every 20M added transitions; final policy experience is **560,250,880**.
The existing 0.28.2 YAML supplies all 25
paper reward weights, full variation/pushes/noise and 128-environment evaluation.

The full checkpoint directory must be present on the GPU host at the path in
the command. It is a run artifact under ignored `results/`, not supplied by Git.

The trainer checks for a GPU. Its effective YAML,
source/package versions, model hash, fixed motor home, reference diagnostics
and geometry checks are recorded in each generated run directory.

For an optional restore/compilation check on the GPU host, use the **actual
1024-environment × 20-step batch**, four PPO passes and 128 evaluation environments:

```bash
uv run --extra training python -m wr2.locomotion.train \
  --config wr2/locomotion/configs/ppo_walking_paper.yaml \
  --restore-checkpoint results/wr2_walking/wr2_paper_20261016_222755_seed0/checkpoints/000260116480 \
  --seed 1 \
  --num-timesteps 20480 --num-evals 2
```

Start the full command from the original 260M checkpoint after this check.
The old WR2 RSL checkpoints have different observation/distribution contracts.

To reproduce the previous fresh seed-0 experiment, omit the restore and seed
option lines from the first command.

## Recommended next trial: smaller PPO updates from +60M

Retain seed 1's **+20M** checkpoint for best tracking and **+60M** for best
survival. The review recommends **fixed LR 1e−5** from +60M, another 300M and
seed 2. The LR differs from TB's 1e−4; large recorded KL and an identical-batch
update comparison support this one-parameter recipe change. Its long-run
tracking and survival effect remains to be verified.

```bash
uv run --extra training python -m wr2.locomotion.train \
  --config wr2/locomotion/configs/ppo_walking_paper.yaml \
  --restore-checkpoint results/wr2_walking/wr2_paper_20261017_080025_seed1/checkpoints/000060026880 \
  --learning-rate 1e-5 \
  --seed 2
```

The checkpoint directory must be present on the GPU host. Keep every ~20M
checkpoint, the full paper suite and the ≤1% falls gate. See the
[evidence and next decision](wr2_paper_finetune_result_20261017.md#7-recommended-next-trial).

## Sources and scope

The reference is the [ToddlerBot paper, Tables VI/VII](https://arxiv.org/html/2502.00893v2),
checked against the successful local TB control
`toddlerbot_walk_ppo_20261015_194347`, source **1982ef3** and Brax **1414780**.
Its frozen `locomotion/{mjx_env,walk_env,evaluate_xy,train_mjx}.py`,
`paper.gin`, and `algorithms/{zmp_planner,zmp_walk}.py` are the implementation
comparison. Preview control follows
[Tedrake et al., Humanoids 2015](https://doi.org/10.1109/HUMANOIDS.2015.7363473).
The [control result audit](toddlerbot_transfer_review_20261016.md) explains the
WR2 gap and why TB 180M/200M are stronger references than its final checkpoint.

This is a transfer of that recipe onto WR2 geometry and identified hardware.
It uses WR2's locked Brax 0.14 runtime, JAX 0.6.2 and MuJoCo/MJX 3.14.0.
The GPU dry run recorded **Python 3.10.19 / Brax 0.14.1**; local CPU checks
used Python 3.11 / Brax 0.14.2. The lock selects 0.14.1 below Python 3.11.
Versions are recorded at launch.

## Completed fine-tuning recipe

| Item | Completed WR2 experiment |
| --- | --- |
| PPO start/budget | Restore 260M parameters, seed 1; add requested 300M, native rounding 300,134,400 transitions |
| Networks | Actor and critic 512/256/128, `silu` = Swish |
| Distribution | Native state-dependent NormalTanh; native log probability/entropy |
| PPO batch | 1024 envs × 20 steps; 256 × 4 minibatches; 4 passes |
| Optimizer | LR 1e-4 fixed, no gradient clipping, entropy 0.0005 |
| Returns | Discount 0.97, GAE 0.95, clip 0.2; observations unnormalized, advantages normalized |
| Value loss | Actual 0.25 × MSE; adapter passes native `vf_loss_coefficient=0.5` |
| Rewards | All 25 Table VII terms; native coefficients, including leg imitation 5 and terminal survival 10, integrated at 20 ms |
| Policy target | Fixed `q_home + 0.25 rad × action`, 12 active leg joints |
| Actor | 65 × 15 = **975** values; no reference joint vector |
| Critic | 99 × 15 = **1485** values, including motor-reference error and stance |
| Training horizon | 1000 control ticks = 20 s |
| Checkpoints | Initial parameters plus 15 trained checkpoints about every 20M; retain all |

On zero network inputs, the native scale head gives
`softplus(0) + 0.001 ≈ 0.694147` before tanh. Actual observations change this
state-dependent scale; `init_noise_std` does not initialize the NormalTanh head.
The fixed LR approximates TB's released cosine schedule, which changes by only
0.004884% across this budget. The adapter maps schedules to Brax enum values
and records network/distribution/initializer metadata that its checkpoint
loader can reconstruct.

Leg imitation is `-mean((q_motor - q_ref)^2)` over the 12 legs, weight 5.
The actor learns it through rewards and PPO advantages. `q_ref` changes with
episode time and command. **`q_home` stays fixed** for action targets, motor
observations and the native standing regularizer. The conflicting old global
home-pose and prescribed foot-height-phase objective is absent from this recipe.
The per-foot slip mask is corrected; frozen TB broadcasts that mask across axes.

Landing air time remains in physical seconds; clearance divides the native
height sample sum by `s²`. Both robots use 50 Hz control, so these event
integrals must scale with cycle duration alongside continuous rewards.
The original `air_time/s` and `clearance/s³` conversions inflated their relative
reward rates by **21.35%**. Native coefficients 500 and 0.05 are retained.

Foot support applies **1 N after summing all solver forces on a floor/foot
pair**, matching frozen TB. The old paper path applied the threshold to each
point and also rejected positive-distance contacts with active solver forces.
Four 0.4 N points incorrectly produced no support despite 1.6 N total force.
The corrected rule affects contact rewards, landing detection and critic stance.

Actor frame order is phase, command, motor-minus-home, motor velocity, old
applied action, gyro and unwrapped Euler angles. Critic order matches native TB:
the same prefix through action, motor-minus-reference, linear velocity, gyro,
Euler, current/desired contacts and linear/angular pushes. Rewards and
observations read the same reference frame before incrementing the episode
clock. The one-tick actuator delay and old-applied-action history match TB.

White Gaussian noise matches native scaled observation units: motor position
0.05 rad, motor velocity 0.1 after its observation scale, gyro 0.25 and Euler
0.1 rad. Encoder error adds `0.02 tanh(torque/0.1)`. The critic is clean.
Model/reset/actuator variation and observation noise remain enabled in the
comparison; policy sampling is deterministic. Nominal evaluation explicitly
disables model/reset/actuator variation and pushes.

## Geometry and declared WR2 differences

The fixed paper home is the grounded reference crouch at torso height
**0.279260761 m**, 4 mm below the prior reference. Its measured CoM height is
**0.202233182 m**, compared with TB **0.297810684 m**. Therefore
`s = sqrt(h_WR2 / h_TB) = 0.824054757`: linear speed and time scale by `s`,
angular speed by `1/s`. This replaces the earlier proposal's nominal 0.838 scale.
TB's planner uses configured torso height as a CoM proxy. WR2 uses measured
CoM height in the preview dynamics: the fixed root-to-CoM gap is 77 mm,
compared with TB's 29 mm. Full geometry IK uses the actual torso height.

| Quantity | WR2 |
| --- | ---: |
| Training X | [-0.0824055, +0.1648110] m/s |
| Training Y | [-0.0824055, +0.0824055] m/s |
| Training yaw | [-0.606756, +0.606756] rad/s |
| Command mixture | 20% stand / 30% turn / 50% XY, native signed elliptical sampler |
| Comparison X / Y amplitudes | 0.0824055 / 0.0412027 m/s |
| Gait cycle / swing lift | 0.60 s (rounded to ticks) / 0.0339533 m |
| Command / push interval | 124 / 82 control ticks |
| Push velocity ranges | Each linear axis ±0.412027 m/s; angular axis ±1.213512 rad/s |
| Healthy torso height | [0.170959, 0.341917] m, native TB fractions scaled to the fixed torso height |

Walking references treat every nonzero command as motion. The separate
normalized 0.05 regularizer deadzone can still suppress air-time/stillness
shaping for small commands, matching TB. Lateral comparison commands therefore
receive moving references even below the physical deadzone.

The self-contained host planner generates **227 XY/turn commands × 1000
retained frames**, using TB preview/footsteps and WR2's full six-DOF leg IK.
The numerical preview agrees with native TB to below 0.15 micrometers on the
checked trajectory. Full-grid checks found two geometry-specific needs:

- Retry failed straight-knee warm starts from the canonical crouched leg seed.
  This resolves a reproduced wrong IK branch with 42 mm residual.
- Preview two extra gait cycles beyond retained frames. The finite stopping
  tail otherwise violates WR2 hip-roll range. Keep the full 20 s lookup and
  the grounded fixed crouch; joint limits stay at their hardware values.

All 227,000 frames pass the actual foot-box ground/penetration and desired-ZMP
support checks. Maximum IK residuals are approximately **37 micrometers /
98 microradians**. Native preview has a soft COP cost, with optimal COP
excursions reported separately; it does not impose a hard support constraint.
Soft motor imitation also exceeds the bounded action envelope: approximately
8.3% of uniformly sampled leg targets are beyond ±0.25 rad from the fixed home.
Requested reference steps reach about 33.4 rad/s, including the initial
static-to-walking transition. These are reference diagnostics; evaluate actual
motor RMSE, clipping/slew, torque and foot use before changing hardware limits.

WR2 retains Newton 10, 5 ms × 4 substeps, quantized commands, its identified
torque-speed model and command speed ceiling. TB used Newton 1, 4 ms × 5.
WR2's corrected `mj_setConst` refresh after mass/inertia/armature variation
remains active. Identified WR2 dynamics/randomization ranges remain explicit
in the YAML. Foot-distance shaping uses the actual WR2 stance width relative
to TB's configured 0.0728 m width. These differences reflect the robot and the
confirmed randomized-physics correction.

## Evaluation and next decision

Each checkpoint runs the control's **15 conditions × 128 episodes** on fixed
model/reset draws, selection seed **12102**: ten signed fixed walking points,
standing, and four cardinal start/stop schedules. Physical commands and timing
are normalized. The primary comparison lasts **824 ticks / 16.48 s**, with
transition boundaries **124, 371, 494, 700** and 21-tick recovery windows.
Keep the separate 1000-tick engineering profile for hardware-duration review.

`paper_evaluations/<step>.json` stores complete per-episode values and counts.
`paper_evaluation_history.jsonl` stores selection summaries. Measures include
instantaneous XY vector error/speed, leg RMSE, episode torque RMS, mean per-step
peak torque, physical target clipping, slew limitation, bounded-action
saturation, each foot's contact fraction, double support and XY path length.
Each episode is normalized before averaging episodes. Empty groups are null;
foot path is a sum, and RMSE is computed from each episode's mean square.
Finite terminal samples count; invalid transitions count toward length and
nonfinite totals, then stop contributing measurements.

Schema **2** also exports single support, neither-foot support and summed
vertical force for each foot. Support fractions partition every finite sample;
`no_foot_support_fraction = 1 − left − right + double`. A foot below the 1 N
threshold can still carry force, so this fraction alone does not establish
flight. Forces, torque and support are sampled at the end of each 20 ms control
step; torque RMS is over those samples, not every physics substep.

Candidates require the complete finite suite and at most 1% aggregate falls.
Among trained candidates, select the lowest ten-condition fixed walking XY
error. Native return is telemetry. Training then evaluates the selected
candidate on **held-out seed 22102**, writing `paper_confirmation_22102.json`.
This selects a review candidate; confirm individual directions, transitions,
foot use and hardware loads against TB 180M/200M before claiming parity.

To recheck a retained checkpoint with the existing evaluator:

```bash
uv run --extra training python -m wr2.locomotion.evaluate \
  --config results/wr2_walking/<run>/training_config.yaml \
  --checkpoint results/wr2_walking/<run>/checkpoints/<step> \
  --paper-benchmark --num-envs 128 --seeds 22102 \
  --output results/wr2_walking/<run>/held_out_<step>.json
```

Add `--engineering-horizon` for the separate 20 s profile. Inspect intermediate
180M/200M checkpoints and the selected candidate alongside the final checkpoint.

### Nominal diagnostic comparison

Use the regular evaluator's **`--nominal`** option to disable model/reset/actuator
variation, pushes and observation noise while preserving the saved policy and
normalized 15-condition schedule. It records evaluation source/package versions
separately from the saved training configuration's version.

For the old 200M control on the GPU host:

```bash
uv run --extra training python -m wr2.locomotion.evaluate \
  --config results/wr2_walking/wr2_paper_20261016_130552_seed0/training_config.yaml \
  --checkpoint results/wr2_walking/wr2_paper_20261016_130552_seed0/checkpoints/000200089600 \
  --paper-benchmark --nominal --num-envs 128 --seeds 22102 \
  --output results/wr2_walking/wr2_paper_20261016_130552_seed0/nominal_0282_200M.json
```

Use the same command with a fresh run's saved config/checkpoint when comparing
its gait. Fully deterministic nominal episodes repeat one trajectory per
condition; repeated episodes do not measure uncertainty across training seeds.
Use the ordinary randomized paper suite for candidate selection and confirmation.

The [completed-run review](wr2_paper_result_20261016.md#frozen-tb-checkpoint-support-comparison)
also records locally captured TB **180M/200M** nominal foot support, forces,
applied targets, torque and paths. Both controls maintain at least one foot
above 1 N in every forward/standing sample, versus about 61% neither-foot
support for old WR2 200M. TB preserves its native random heading and motor-home
offsets; WR2 nominal resets are fixed. The reports record these reset and
runtime differences. The prepared WR2 training command requires only this
checkout; the active TB checkout required no changes.

## Verification status

Local CPU verification passed **92 tests** across the new paper checks and
existing training/interface, XY, startup, randomized-physics and evaluation
regressions. It covers the full reference grid, actual JIT MJX
reset/physics, native delayed-action/AutoReset behavior, reward and sampler
oracles, policy log probabilities, a native PPO update and checkpoint
reload/parameter restoration. The completed GPU dry run exercised the full
1024 × 20 batch, 16 optimizer updates, both checkpoint evaluations and native
checkpoint saving. Its metrics and parameters are finite; checkpoint reload
also passed locally with Brax 0.14.1. Nine targeted trainer/evaluation tests
pass after the relative checkpoint path fix, including a regression rerun
with Brax 0.14.1. See the [dry-run audit](wr2_paper_dry_run_20261016.md).
The original 300M run also completed with finite metrics and checkpoints,
but its [walking results](wr2_paper_result_20261016.md) fail the candidate gate.
Four reward regressions pass for the 0.28.1 correction, including the new
whole-cycle test that reproduced the 21.35% bias before the fix. The
0.28.2 run completed 300M with finite metrics and checkpoints; its
[result audit](wr2_paper_result_20261017.md) records the remaining tracking and
recovery gap. No checkpoint passes the ≤1% falls gate.
For 0.28.2, **12 targeted tests pass**, covering the reproduced support bug,
reward oracles, episode accounting, real JIT MJX reset/step/AutoReset, support
partitions and force thresholds, actor/home contracts, checkpoint loading and
nominal/native evaluator settings. The standard `uv` evaluator also loaded the
real 200M GPU checkpoint with Brax 0.14.1 and completed **15 × 8** nominal
episodes with zero falls/nonfinite episodes and consistent support partitions.
The frozen TB 180M/200M controls each completed **15 × 8** nominal-model
episodes with zero falls/nonfinite episodes and consistent support partitions.

The seed-1 restore check loaded the **actual 260M GPU checkpoint** using
Brax **0.14.1** and seed **1**. Native PPO's step-0 callback exactly matched
the audited actor, critic and normalizer arrays; a small **4-env × 20-step**
batch with four PPO passes then changed the policy and kept parameters,
metrics and bounded actions finite. This CPU check uses the existing cheap
transition environment with the real 975/1485 observation dimensions and
12-action network. Its full configuration differs from the parent run only in
seed; the GPU batch remains 1024 × 20.
See the [restore verification](../../results/wr2_walking/wr2_paper_20261016_222755_seed0/review_20261017/next_run_restore_preflight.json).

The seed-1 fine-tune completed **300,134,400 added transitions** with finite
metrics and checkpoints. Independent auditing reconstructs **90,240** records
across it, its parent and TB, with zero failures; initial arrays exactly match
parent 260M. Actual nominal MJX replays show quieter final standing but worse
walking, and an identical-rollout PPO probe supports smaller updates.
The [fine-tuning review](wr2_paper_finetune_result_20261017.md) records the
protocols, numerical evidence and limits. No fine-tuned checkpoint passes the
≤1% falls gate.
