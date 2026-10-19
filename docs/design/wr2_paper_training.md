# WR2 paper walking experiments

The original **0.28.0** transfer completed 300M steps with no qualifying
checkpoint and substantial hopping. The corrected **0.28.2** run also completed
300M: tracking improved 26.5%, nominal neither-foot support fell from about
61% to 0%, but approximately 5% of evaluation episodes still fall.
A further **300M fine-tune from 260M** completed: best tracking improved only
2.5%, then final error regressed 15.3% versus the parent. No checkpoint passes
the fall gate. A further **0.28.3 / LR 1e−5 / seed 2** continuation completed
300M: reported epoch KL stays within 0.0051–0.0106, but best tracking improves
only 1.7% versus its start and remains 2.82× TB Paper 200M. No checkpoint meets
the fall gate. See the [WR2/Paper/Main comparison and next investigation](wr2_tb_progress_review_20261017.md)
and the [previous update-instability review](wr2_paper_finetune_result_20261017.md).
The recipe retains the native paper weights, corrected landing normalization
and TB's summed foot-force support rule.

The **0.28.4 fresh seed-3 300M** control also completed. Best tracking at
220M is **0.13063 TB-equivalent m/s / 55 falls (2.86%)**, versus TB Paper
200M **0.03702 / 8 falls (0.42%)**. Fewest new falls are at 140M; final tracking
and survival regress. Geometry replays confirm actual nominal flight.
The [completed result review](wr2_fresh_paper_result_20261018.md) records those
native profiles. The [deeper actuator review](wr2_actuator_gap_review_20261018.md)
confirms slower modeled WR2 response, a nonunique hardware inertia fit and
TB Paper's dropped physical pushes. It also corrects the earlier critic claim:
velocity normalization already works. The **0.28.5 no-push control completed
300M**: selected 260M has **0/1920 height failures on both evaluation seeds**,
but tracking error **.07684** is **2.08× TB Paper 200M** and exceeds the
stationary initial policy. The learned policy barely translates. See the
[completed no-push result and revised next plan](wr2_no_push_result_20261018.md).
The **GPU stance diagnostic completed**: reference hip rolls clear self-contact
and reduce hip tracking error, but worsen six-direction XY error **44.2%**.
The next measurement is the matching TB Paper 200M policy trace. See the
[GPU result and revised next plan](wr2_stance_result_20261018.md) and
[stance diagnostic guide](wr2_stance_diagnostic_20261018.md).
The canonical config retains the completed **0.28.5 / weight-500** baseline;
the [previously prepared air-time ablation](wr2_airtime_ablation_20261018.md)
is deferred. TB succeeds with the same air-time coefficient, so this term's
causal contribution remains unisolated.
TB's effective fixed sliding friction 1.0 remains a secondary candidate: its
0.58% frozen-policy benefit supplies weak propulsion evidence. The required
sole colliders and current foot-spacing/reference geometry are retained for
the stance comparison.

## Recorded stance diagnostic command

The GPU run below completed. A repeat requires a fresh output directory:

```bash
uv run --python 3.10 --extra training python -m wr2.tools.diagnose_walking \
  --paper-stance \
  --run-dir results/wr2_walking/wr2_paper_20261018_121025_seed3 \
  --output results/wr2_walking/wr2_paper_20261018_121025_seed3/stance_diagnostic \
  --seeds 22102
```

This evaluates the selected 260M checkpoint through three nominal controls,
without noise, randomization or pushes. The saved training configuration
supplies the reference, controller and limits. Python 3.10 selects the existing
locked GPU-control runtime. Full reference-table geometry checks and raw joint/
contact traces accompany `summary.json`. The
[diagnostic guide](wr2_stance_diagnostic_20261018.md) explains the controls,
matching TB measurements and the evidence required before another training run.

## Reproduce the completed no-push acquisition control

Use the regular trainer on the GPU host:

```bash
uv run --python 3.10 --extra training python -m wr2.locomotion.train \
  --config results/wr2_walking/wr2_paper_20261018_121025_seed3/training_config.yaml
```

This starts fresh, **seed 3 / fixed LR 1e−4 / 300M**, with physical push
amplitudes zero and the current actuator physics, TB rewards/reference and
PPO settings. Its saved 0.28.5 YAML preserves air-time weight 500. Each retained
checkpoint gets paired no-push and physical-push
evaluations. Selection uses the no-push score. See the
[experiment setup, commands and interpretation](wr2_no_push_acquisition_20261018.md).

## Reproduce the completed fresh seed-3 Paper run

The completed **0.28.4** configuration starts **fresh with seed 3**, **fixed LR
1e−4** and **300M requested transitions**. Its
[preparation and first-update check](wr2_fresh_paper_run_20261017.md) supported
that initial learning rate; the completed result above establishes its gait
and later training behavior.

Run the regular trainer from the WR2 checkout on the GPU host:

```bash
uv run --extra training python -m wr2.locomotion.train \
  --config results/wr2_walking/wr2_paper_20261017_195826_seed3/training_config.yaml
```

This initializes a new actor, critic, normalizer and Adam. It uses the completed
fixed grounded home, Paper rewards, normalized reference/timing, full
perturbations and about 20M checkpoint intervals. Config values match the
corrected fresh seed-0 control except for seed and version metadata.
The run is self-contained; the trainer checks for a GPU and records the
effective configuration and provenance.

For an optional GPU compilation/update check with the **actual training batch**:

```bash
uv run --extra training python -m wr2.locomotion.train \
  --config results/wr2_walking/wr2_paper_20261017_195826_seed3/training_config.yaml \
  --num-timesteps 20480 --num-evals 2
```

Then start the full command above with a fresh initialization. The short check
is not a trained walking candidate. All 15 trained checkpoints
from the completed full run remain available for comparison with WR2
+180M/+160M and TB Paper 180M/200M.

## Reproduce the completed low-LR continuation

Run the regular trainer from the WR2 checkout on the GPU host:

```bash
uv run --extra training python -m wr2.locomotion.train \
  --config results/wr2_walking/wr2_paper_20261017_115520_seed2/training_config.yaml \
  --restore-checkpoint results/wr2_walking/wr2_paper_20261017_080025_seed1/checkpoints/000060026880
```

The saved **0.28.3 YAML** sets **seed 2**, **fixed LR 1e−5** and **300M requested
added transitions**. The parent +60M policy has **320,143,360 cumulative
transitions**; native rounding adds 300,134,400, reaching **620,277,760**.
Native restore loads actor, critic and normalizer; Adam and step counters start
at zero. The timestamped `wr2_paper_*_seed2` directory records the effective
configuration and restore command, with checkpoints about every 20M added
transitions. All 25 paper reward weights, full variation/pushes/noise and
128-environment evaluation are retained.

The +20M parent has the best tracking; +60M has the best survival and better
tracking than the tied +220M checkpoint. The LR differs from TB's 1e−4;
large recorded KL and an identical-batch update comparison support this
one-parameter recipe change. The completed experiment reduced reported KL and
falls, with a small tracking gain. Retain native **+180,080,640** for tracking
and **+160,071,680** for survival; the final checkpoint has worse tracking.
The [loaded gait comparison and fresh-policy update check](wr2_fresh_paper_run_20261017.md)
documented preparation of the now-completed fresh Paper control above.

The saved configuration and full parent checkpoint directory must be present
on the GPU host at the command's paths. They are run artifacts under ignored
`results/`, not supplied by Git.

The trainer checks for a GPU. Its effective YAML,
source/package versions, model hash, fixed motor home, reference diagnostics
and geometry checks are recorded in each generated run directory.

For an optional restore/compilation check on the GPU host, use the **actual
1024-environment × 20-step batch**, four PPO passes and 128 evaluation environments:

```bash
uv run --extra training python -m wr2.locomotion.train \
  --config results/wr2_walking/wr2_paper_20261017_115520_seed2/training_config.yaml \
  --restore-checkpoint results/wr2_walking/wr2_paper_20261017_080025_seed1/checkpoints/000060026880 \
  --num-timesteps 20480 --num-evals 2
```

Start the full command from the original +60M checkpoint after this check.
The old WR2 RSL checkpoints have different observation/distribution contracts.

## Reproduce the completed seed-1 experiment

Use its saved **0.28.2 / seed 1 / LR 1e−4** configuration:

```bash
uv run --extra training python -m wr2.locomotion.train \
  --config results/wr2_walking/wr2_paper_20261017_080025_seed1/training_config.yaml \
  --restore-checkpoint results/wr2_walking/wr2_paper_20261016_222755_seed0/checkpoints/000260116480
```

That run added 300,134,400 transitions to 260,116,480, reaching **560,250,880**.
To reproduce the preceding fresh seed-0 control, use
`results/wr2_walking/wr2_paper_20261016_222755_seed0/training_config.yaml` and
omit the restore option. Saved configurations preserve each run's original
settings when the prepared YAML changes.

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

## Effective prepared recipe

| Item | Prepared WR2 experiment |
| --- | --- |
| PPO start/budget | Fresh actor/critic/normalizer/Adam, seed 3; requested 300M, native rounding 300,134,400 transitions |
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
| Acquisition exposure, 0.28.5 | Physical pushes disabled; other variation/noise retained; paired physical-push evaluations |

On zero network inputs, the native scale head gives
`softplus(0) + 0.001 ≈ 0.694147` before tanh. Actual observations change this
state-dependent scale; `init_noise_std` does not initialize the NormalTanh head.
TB's released cosine schedule changes by only 0.004884% from 1e−4 across this
budget. The prepared fixed 1e−4 matches that effective rate; its full fresh
first-update check passes. The mature-policy 1e−5 continuation is a completed
experiment described above. The adapter maps schedules to Brax enum values and records
network/distribution/initializer metadata that its checkpoint loader can
reconstruct. `desired_kl=0.01` is inactive under the fixed schedule; monitor
the actual logged KL.

Leg imitation is `-mean((q_motor - q_ref)^2)` over the 12 legs, weight 5.
The actor learns it through rewards and PPO advantages. `q_ref` changes with
episode time and command. **`q_home` stays fixed** for action targets, motor
observations and the native standing regularizer. The conflicting old global
home-pose and prescribed foot-height-phase objective is absent from this recipe.
The per-foot slip mask is corrected; frozen TB broadcasts that mask across axes.
WR2 also uses all 12 local actions in its smoothness rewards; frozen TB's global
actuator indexing omits four actions and repeats the last five times. The
[complete setup audit](wr2_tb_paper_setup_audit_20261018.md) verifies this and
compares effective friction, reset behavior, clocks, reference and metrics.

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
| Training pushes, 0.28.5 | Zero amplitudes; the 82-tick interval/key split remains active |
| Pushed comparison bounds | Each linear axis ±0.412027 m/s; angular axis ±1.213512 rad/s |
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
torque-speed model and additional derived setpoint slew cap. That command cap
is not a measured firmware rate limit; its same-policy nominal ablation gives
only 0.3–3% improvement. TB used Newton 1, 4 ms × 5.
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
`paper_evaluation_history.jsonl` stores selection summaries and physical-push
metadata. For 0.28.5 these primary reports have no physical pushes.
`paper_push_evaluations/<step>.json` and `paper_push_evaluation_history.jsonl`
record the paired physical-push diagnostic on identical model/reset draws.
Measures include
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
The same selected 0.28.5 checkpoint also gets
`paper_push_confirmation_22102.json`; pushed scores do not select the candidate.
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

Add `--engineering-horizon` for the separate 20 s profile. Compare every retained
~20M checkpoint with its parent, TB 180M/200M and the final checkpoint.
Add `--no-pushes` to explicitly disable only physical pushes, retaining
variation/noise; `--pushes` realizes the normalized TB bounds even when the
saved training amplitudes are zero. Omitting both uses the recorded configuration.

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

The **0.28.3** run passed a native **Brax 0.14.1** CPU restore/update
check using the **actual seed-1 +60M GPU checkpoint**, seed **2** and configured
fixed LR **1e−5**. The trainer's CLI/config path changes only version metadata,
seed and LR versus the completed run. Step-0 actor, critic and normalizer arrays
match the audited checkpoint exactly. One **4-env × 20-step** batch with four
PPO passes changes actor parameters; all parameters/metrics and 12 bounded
deterministic actions remain finite. The recorded LR is **1.00000016e−5**,
and native checkpoint save/reload preserves all updated arrays.
See the [next-run restore verification](../../results/wr2_walking/wr2_paper_20261017_080025_seed1/review_20261017/next_run_restore_preflight.json).

This uses the existing cheap transition environment with real **975/1485**
observation dimensions; the preceding identical-batch probe used actual MJX
robot physics at LR 1e−5. The full GPU compilation command is above.

For **0.28.4** preparation, a real randomized/noisy **1024 × 20** fresh-policy
batch at LR **1e−4** passes four minibatches × four PPO passes with finite
parameters, mean pre-gradient joint KL **0.002444**, post-batch KL **0.003794**
and **4.04%** post-batch ratio clipping. Native seed-3 initialization matches
that probe exactly; a separate native update on the existing cheap environment
saves/reloads its checkpoint exactly. Two native trainer/evaluator regressions
pass. Loaded WR2 +180M and frozen TB Paper 200M traces also complete eight
nominal trajectories / **7,296 ticks** with no falls/nonfinite samples and
consistent target, reference-lift and heading-frame measurements. These new
checks used CPU Brax 0.14.1 / JAX 0.6.2; WR2 physics uses MJX 3.14 and the
frozen TB control uses MJX 3.2.4. See the
[fresh run review and verification](wr2_fresh_paper_run_20261017.md).
