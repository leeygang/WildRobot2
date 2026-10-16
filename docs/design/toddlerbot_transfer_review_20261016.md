# ToddlerBot control result and WR2 transfer review — 2026-10-16

## Decision

Use the completed TB paper control to prepare a **fresh WR2 run with the
paper reward, reference and native policy semantics**. Use WR2's existing
Brax training entry point after fixing its adapters. Retain checkpoints
through a 200–300M budget and select candidates using tracking and survival.

The [prepared WR2 experiment](wr2_paper_training.md) implements this plan.
Full-grid geometry qualification uses a fixed crouch 4 mm lower than the prior
reference, recomputing the CoM scale to **0.824055** and evaluation horizon to
**824 ticks**. The nominal geometry values below document the earlier proposal.

The TB control supplies a useful walking reference at 180M/200M. Its final
checkpoint regresses in survival. The completed WR2 continuation still uses
the later TB code's different recipe, so another continuation of that policy
would not test transfer of the successful control.

This review reads the copied raw TB results, frozen training source, current
local TB/Brax code, WR2 code and the WR2 checkpoint tensors. The public recipe
was checked against the previously saved [paper text](../../results/wr2_walking/wr2_xy_physics_20261013_085712_seed0/review_20261013/toddlerbot_paper.txt),
particularly Tables VI/VII. It supplements the
[TB agent's report](../../../toddlerbot/PAPER_WALKING_RESULTS.md), including
the model-contract correction below.

## 1. Completed TB run and independent audit

Run: `toddlerbot_walk_ppo_20261015_194347`, source `1982ef3`, pinned Brax
`1414780`, fresh seed 0. Requested 300M; completed **300,134,400 transitions**.
The GPU runtime uses JAX 0.6.2, MuJoCo/MJX 3.2.4 and the separate Blackwell
lock. This is a paper-aligned control with a hardware compatibility runtime.

| TB checkpoint | Native return | Fixed walking vector MAE (m/s) | Fixed falls / 1,280 | Standing falls / 128 | Transition falls / 512 | All falls / 1,920 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 180,080,640 | 372.745 | 0.03838 | 2 | 0 | 1 | **3 (0.156%)** |
| 200,089,600 | 371.733 | **0.03702** | 5 | 0 | 3 | **8 (0.417%)** |
| 240,107,520 | **380.249** | 0.03823 | 39 | 2 | 16 | 57 (2.969%) |
| 300,134,400 | 369.533 | 0.03800 | 84 | 6 | 44 | **134 (6.979%)** |

Checks performed in this review:

- Reconstructed all 225 condition summaries from **28,800 episode-condition
  records**. Means, population SDs, eligibility, sample counts, lengths and
  fall counts agree; maximum rounding difference is **3.83e-8**.
- All 15 native metric rows are finite; all checkpoint finiteness records
  pass. Loaded every inference export independently and checked every actor
  and normalizer parameter leaf for finiteness. Full value parameters use
  the trainer's recorded checkpoint audit.
- Training/evaluation/checkpoint step lists match. The 11 frozen locomotion
  files match the training commit; all 12 recorded model/reference hashes
  and the dependency lock match the local artifacts.
- Verified `best_policy` maps to **240M** and `policy` to **300M** by SHA-256.

The recorded dirty-source flag remains: files outside the frozen/hash-listed
set were not fully captured at launch. Remote SSH access was unavailable in
this session; this audit covers the locally copied artifacts.

### Interpretation and candidate selection

Tracking improves mainly before 80M and then plateaus. Extending this TB run
from 180M to 300M produces many more falls while changing fixed-command error
only slightly. Native reward-best therefore provides a poor survival reference.

Keep **180M for survival and 200M for tracking**, and confirm both on new
evaluation draws. A paired bootstrap over the 128 model/reset indices, keeping
all conditions together, gives 200M minus 180M fixed-command error
**-0.001355 m/s**, with selection-set 95% interval
**[-0.001908, -0.000817]**. The all-condition fall-fraction difference is
**+0.002604**, interval **[-0.000521, +0.006771]**. This is uncertainty within
one training seed and a set already used for selection.

TB also has remaining gaps: reset-standing speed is **0.03085 m/s at 180M**
and **0.03196 m/s at 200M**. Neither is stationary control. The suite does not
contain sufficient foot-use, torque, clipping or displacement traces to certify
complete WR2 hardware qualification. The paper's 0.016 m/s square-trajectory
metric has a different protocol from these instantaneous vector errors.

Metric means normalize within each episode before averaging episodes. A failed
episode contributes its shorter measurements, so tracking error must be read
with falls and lengths. Transition results must separate walking, standing,
start and stop; their overall mean mixes commanded and zero-command periods.

## 2. WR2 comparison and a correction to the TB report

**The reviewed WR2 600M checkpoint already has twelve leg actions, including
both hip yaws.** The TB report's statement that it used a ten-action model is
incorrect.

Directly loaded `wr2_xy_physics_20261013_085712_seed0/checkpoints/000200007680.pt`:

- Actor output: `[12, 128]`; `log_std`: `[12]`.
- Actor/critic inputs: **1005 / 1740**.
- Recorded active actuator names include `left_hip_yaw` and `right_hip_yaw`.
- Saved MJX model hash is `741d1e613d487b34073a10fbc0365906c3364914f37e70a394c45bf1e1a696c0`,
  identical to current `wr2_mjx.xml`. No description files changed between
  training revision `93e4672` and reviewed WR2 revision `5924bde`.

The 400M source checkpoint has the same contract. They remain useful controls
under their recorded observation/recipe contracts. Fresh training is needed
to test the new recipe and initialization; adding hip yaw has already happened.

The following values were read from raw saved evaluations:

| Condition | WR2 corrected 600M velocity (X, Y), m/s | TB 200M velocity (X, Y), m/s | WR2 vector MAE | TB vector MAE | WR2 falls | TB falls |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| X +0.10 | (+0.0741, +0.0046) | (+0.0905, -0.0033) | 0.0623 | 0.0373 | 0/128 | 1/128 |
| X -0.10 | (-0.0838, +0.0043) | (-0.0736, +0.0008) | 0.0562 | 0.0423 | 5/128 | 1/128 |
| Y -0.05 | (-0.0126, -0.0029) | (-0.0018, -0.0320) | 0.0604 | 0.0373 | 0/128 | 1/128 |
| Y +0.05 | (-0.0087, +0.0091) | (-0.0013, +0.0350) | 0.0570 | 0.0354 | 1/128 | 0/128 |
| Standing | (-0.0079, +0.0026) | (-0.0022, +0.0027) | 0.0353 | 0.0320 | 0/128 | 0/128 |
| Backward transitions | (-0.0517, +0.0047) | (-0.0420, +0.0014) | 0.0513 | 0.0401 | 13/128 | 3/128 |

TB responds to lateral commands at roughly **64–70%** of their magnitude;
WR2 responds at roughly **6–18%**. WR2's backward mean speed is closer to the
command, while its vector error and backward-transition survival are worse.
Mean forward/backward speed alone therefore misses a substantial part of the gap.

Both implementations measure linear velocity in the full torso orientation
frame and use episode-first vector-error means. Equal numeric commands/base
seeds do not make this a controlled robot comparison: morphology, disturbances,
training budget, policy, reference, rewards and initialization differ. The
new comparison should use normalized commands/timing and each robot's recorded
variation distribution. This result identifies a recipe to transfer; it does
not isolate a causal reward or servo defect.

## 3. Source-confirmed recipe differences

| Item | Completed TB control | Current WR2 XY recipe |
| --- | --- | --- |
| Trainer / start | Native Brax, fresh networks/Adam/counters | RSL-RL; 600M run continued a 400M policy and optimizer |
| Network / distribution | Swish; state-dependent Normal followed by tanh | ELU; unbounded Normal with independent learned log std |
| Action origin | Native home + 0.25 rad bounded residual | WR2 walk-home + 0.25 rad unbounded residual, physical target clipping/slew |
| Leg imitation | Negative mean motor-reference squared error, weight **5** | Weighted pose penalty to home; reference error supplied to critic |
| XY tracking | `5 exp(-100 * error²)` | `2 exp(-1000 * error²)` |
| Survival | Negative height-terminal indicator, weight **10**, integrated at dt | Positive alive rate, weight **1** |
| Foot shaping | Contact-reference match, air time, clearance and regularizers from Table VII | Strong prescribed foot-height-phase reward and different regularizers |
| Commands | X [-0.10, +0.20], Y [-0.10, +0.10], yaw [-0.50, +0.50]; 20% stand / 30% turn / 50% XY | X [-0.10, +0.10], Y [-0.05, +0.05]; 20% stand / 80% XY |
| Reference | TB ZMP planner, footstep trajectories and analytic IK | Periodic LIPM trajectory and WR2 six-DOF numerical IK |
| Observations | Motor pose/velocity, previous action, gyro and Euler orientation; critic adds motor-reference errors, velocity, stance and pushes | Different gravity/heading representation; critic also includes actuator force; no corresponding push protocol |
| Physics | MJX 3.2.4, 4 ms × 5, Newton 1 | MJX 3.14, 5 ms × 4, Newton 10, corrected randomized constants |

At a 0.05 m/s vector error, the TB velocity reward rate is **3.894**;
WR2's is **0.164**. These objective differences are large. Their individual
effects still require a matched training experiment.

Leg imitation trains the actor through rewards and PPO advantages. The
reference joint vector remains critic information; the TB control does not
require adding that vector to actor observations. Match the reference phase,
motor indexing and averaging over the **12 active leg motors**.

### Existing Brax route needs adapter fixes

Using the existing route keeps the first transfer close to the successful
TB trainer. It needs concrete changes before launching:

1. `make_network_factory` currently rejects `tanh_normal`, although the config
   loader and installed Brax support it. Use the native distribution, including
   its log-probability/entropy semantics, with `silu` (Swish) activation.
2. WR2 passes `fixed`/`adaptive` to Brax's `LRSchedule` enum. Both raise
   `ValueError`. Map them to **NONE / ADAPTIVE_KL**; select NONE for this control.
3. TB native value loss is **0.25 × MSE**. Brax 0.14 implements
   `0.5 × vf_loss_coefficient × MSE`. Keep WR2's config coefficient **0.25**
   defined against actual MSE and convert it to **0.5** in the Brax adapter.
   Passing the config value directly would halve the loss on the Brax route.
4. Native TB does not clip gradient norms. The WR2 loader currently requires
   a positive norm; permit the native Brax `None` setting for this recipe.

The released cosine schedule changes 1e-4 to **9.9995116e-5** over this run:
its counter is optimizer updates. A fixed 1e-4 differs by at most **0.004884%**;
use and record this small approximation rather than adding a new scheduler.

Source inspection and numerical checks found identical **GAE returns and
advantages**, and identical **NormalTanh log probabilities, entropy and
postprocessing**, between pinned TB Brax and installed WR2 Brax 0.14.2 for
nontrivial test inputs. This supports using the existing route. Full rollout,
PPO update, checkpoint and GPU compatibility still need an integration check
after implementation. If choosing RSL-RL instead, establish the equivalent
TB recipe under RSL-RL before attributing the result to WR2.

### Confirmed native TB feet-slip bug

TB multiplies squared foot XY velocities shaped `(2 feet, 2 axes)` by a
stance mask shaped `(2,)`. Broadcasting applies the mask to **axes**.
For velocities `[[1, 2], [3, 4]]` and left-only stance `[1, 0]`, its function
returns **-10**, while the per-foot mask gives **-5**. Right-only stance gives
**-20 versus -25**. This was reproduced by executing the native function.

WR2's existing per-foot reduction already uses the correct mask axis. Preserve
that behavior when adding the paper slip weight. The effect of the TB bug on
learning or late falls has not been measured; it is not a demonstrated cause
of this training gap.

## 4. Geometry, timing and reference feasibility

Measured with `mj_resetDataKeyframe` / `mj_forward` on the actual run/current
models, with COM from `data.subtree_com[torso_id, 2]`:

| Quantity | TB run model, home | WR2, walk-home |
| --- | ---: | ---: |
| Total mass | 3.35818 kg | 2.05833 kg |
| Torso height | 0.326700 m | 0.288261 m |
| COM height | 0.297811 m | 0.209278 m |
| Mean hip-pitch anchor to sole-site vertical distance | 0.273117 m | 0.238203 m |

The earlier approximately 1% leg-size argument compared WR2 with **TB 2xc**.
The successful control uses the original **toddlerbot** model. Use that model
for this transfer comparison.

For the LIPM dynamics, preserve `v / sqrt(g h)` and `T / sqrt(h / g)`.
The nominal COM scale is **s = sqrt(h_WR2 / h_TB) = 0.838286**:

| Transfer quantity | Proposed WR2 value |
| --- | ---: |
| TB +0.10 / -0.10 comparison commands | +0.08383 / -0.08383 m/s |
| TB +0.05 / -0.05 lateral comparison commands | +0.04191 / -0.04191 m/s |
| TB full training X range | [-0.08383, +0.16766] m/s |
| TB full training Y range | [-0.08383, +0.08383] m/s |
| Gait cycle | 0.60357 s, approximately 0.60 s |
| XY reward width, TB equivalent 0.10 m/s | **0.08383 m/s** |
| Command resampling / periodic push interval | 2.515 / 1.677 s, rounded to control ticks |
| Full-turn command range, when reference-qualified | [-0.59646, +0.59646] rad/s |

Keep the 20 ms hardware control period. Root-height thresholds and swing
clearance need their own posture/kinematic checks. TB's healthy torso-height
fractions correspond nominally to **[0.17647, 0.35294] m** on WR2; the current
WR2 terminal range is [0.20, 1.00]. Declare the definition used in survival
comparisons. TB's planner uses 50 mm foot steps; WR2's knee range/IK limits
justify a separate feasible clearance choice. Multiplying every length by
the COM ratio would mix mass distribution with leg reach.

### What the reference checks establish

Four cardinal commands plus standing pass the existing ground, orientation,
IK and support gates at both 0.72 s and normalized 0.604 s. IK position errors
are below **10 micrometers**, with approximately **9 mm** minimum walking
support margin. The actual WR2 reference crouch has COM height **0.205281 m**;
the nominal scale above is a starting value.

The same reference has motor offsets up to **1.27 rad from walk-home** and
approximately **24–27%** of sampled active-motor targets outside ±0.25 rad.
Its sampled target speeds are **12.2–13.1 rad/s at 0.72 s** and
**14.2–14.8 rad/s at 0.604 s**, exceeding the WR2 servo's **5.8178 rad/s**
no-load/command ceiling. Ground/support validity alone therefore does not
establish command-target or actuator feasibility.

Native TB also uses a soft reference: its 20-second reference samples reach
**1.064 rad** offsets, **11–15%** outside its bounded target envelope, and
**14.4–16.3 rad/s** requested steps. These are requested reference movements,
not achieved motor speeds. Soft imitation need not reproduce every target.
This evidence supports measuring representability, actual clipping/slew and
leg RMSE; it does not justify widening the actor or raising servo limits.

Before activating WR2 imitation, use TB's reference formulation with WR2's
measured geometry and six-DOF IK, and verify its posture, phase/contact/motor
mapping and dynamic tracking. Keep the corrected mass/armature constant
refresh and identified servo model. The prior WR2 lateral traces measured
no hip-roll/yaw slew or overspeed hits, providing no basis for a servo retune.

## 5. Concrete next experiment

### A. Confirm the TB reference candidates

Use the native `XYEvaluator` with new held-out model/reset draws, for example
seed 22102, on 180M and 200M. Record the same 15 conditions plus foot use,
actual torque, target clipping/slew and representative trajectories. Keep
the original 12102 measurements as the checkpoint-selection set.

### B. Prepare one paper recipe in the regular WR2 trainer

- Fix the existing Brax adapters above; use 512/256/128 Swish networks,
  native state-dependent NormalTanh, fresh initialization and 1e-4 LR.
- Match complete Table VII reward definitions, including leg imitation,
  terminal survival, contact, air time, clearance, stillness and action
  acceleration. Set the current conflicting home-pose/foot-phase shaping
  weights consistently with that recipe. Use the correct per-foot slip mask.
- Generate the TB-style ZMP/footstep reference for WR2 geometry, preserving
  WR2 motor ranges, controller limits and full orientation IK. Use one
  consistent reference time/frame for rewards, critic context and metrics.
- Match the native observation information/scales with WR2's motor count;
  record any retained WR2 sensing difference as an explicit experiment
  variant. The actor needs no reference joint vector.
- Match normalized command sampling and pushes. The current XY sampler
  always returns zero yaw; the current reference has no turning footsteps.
  Qualify turning before enabling TB's 30% turning branch. An initial XY-only
  run must be described as a task-restricted transfer.
- Separate sampling deadzone from zero-command reference classification.
  The current `_is_walking` threshold 0.05 would classify a normalized
  0.04191 m/s lateral command as standing. Native TB still generates walking
  references for nonzero commands below its regularizer deadzone.
- Specify comparison commands separately from training endpoints. The
  current evaluator derives lateral commands from full training bounds,
  which would test ±0.08383 instead of the TB-equivalent ±0.04191 m/s.
- Support the normalized evaluation schedule: approximately **838 steps /
  16.76 s**, phase boundaries **126, 377, 503, 713**, recovery window **21
  steps**. The current schedule requires 1000 steps. Keep the paper's 1000
  training-step horizon and report the 20-second engineering suite separately.

Keep TB source/checkpoints in the TB repository and use
`wr2.locomotion.train` directly. This preparation needs no launcher wrapper
or vendored TB repository.

### C. Verify, then train fresh for 200–300M

Before the long run, verify reward values/indexing, contact-mask axes,
reference/phase/command consistency, bounded policy log probabilities, one
native PPO update and checkpoint restoration, and actual GPU compilation
and memory fit with **1024 environments × 20 steps**. A two-environment smoke
does not establish the full batch's memory fit.

Train seed 0 fresh with Table VI settings. Retain checkpoints approximately
every 20M transitions, including 180M/200M candidates and later checkpoints
if extending to 300M. Compare to the existing compatible WR2 controls under
their recorded contracts. Use held-out draws before promoting a checkpoint.

Report dimensionless errors and signed command response, with falls/lengths,
standing speed, start/stop phases, leg RMSE and hardware diagnostics. For
example, the TB 200M reference translates nominally to:

| WR2 normalized condition | TB-equivalent signed velocity | TB-equivalent vector MAE |
| --- | --- | ---: |
| X +0.08383 | (+0.07589, -0.00280) m/s | 0.03126 m/s |
| X -0.08383 | (-0.06167, +0.00068) m/s | 0.03547 m/s |
| Y -0.04191 | (-0.00153, -0.02686) m/s | 0.03130 m/s |
| Y +0.04191 | (-0.00105, +0.02933) m/s | 0.02970 m/s |
| Standing | instantaneous speed 0.02680 m/s | 0.02680 m/s |

These are reference measurements scaled for comparison, not measured WR2
outcomes. Keep the 0.02 m/s engineering objective distinct from measured
TB parity. A high reward or a low mixed transition error cannot establish
survival or reliable lateral walking.

## Evidence and references

Independent review artifacts, all local and ignored by Git:

- [Metric reconstruction / input hashes](../../results/wr2_walking_diagnostics/tb_transfer_20261016/metric_audit.json)
  and [inference-export finiteness](../../results/wr2_walking_diagnostics/tb_transfer_20261016/policy_export_audit.json).
- [Checkpoint contract / reproduced Brax adapter errors](../../results/wr2_walking_diagnostics/tb_transfer_20261016/wr2_contract_audit.json),
  [GAE/distribution equivalence](../../results/wr2_walking_diagnostics/tb_transfer_20261016/brax_contract_audit.json)
  and [native feet-slip reproduction](../../results/wr2_walking_diagnostics/tb_transfer_20261016/feet_slip_audit.json).
- [Geometry / WR2 reference checks](../../results/wr2_walking_diagnostics/tb_transfer_20261016/geometry_reference_audit.json),
  [native TB reference requests](../../results/wr2_walking_diagnostics/tb_transfer_20261016/tb_reference_audit.json),
  [normalized comparison values](../../results/wr2_walking_diagnostics/tb_transfer_20261016/normalized_benchmark.json)
  and [paired candidate uncertainty](../../results/wr2_walking_diagnostics/tb_transfer_20261016/paired_selection_audit.json).

Source/data:

- TB frozen [paper configuration](../../../toddlerbot/results/toddlerbot_walk_ppo_20261015_194347/locomotion/paper.gin),
  [environment/rewards](../../../toddlerbot/results/toddlerbot_walk_ppo_20261015_194347/locomotion/mjx_env.py),
  [walking rewards](../../../toddlerbot/results/toddlerbot_walk_ppo_20261015_194347/locomotion/walk_env.py)
  and [evaluator](../../../toddlerbot/results/toddlerbot_walk_ppo_20261015_194347/locomotion/evaluate_xy.py).
- WR2 [environment](../../wr2/locomotion/walking_env.py),
  [Brax network factory](../../wr2/locomotion/ppo.py),
  [trainer](../../wr2/locomotion/train.py),
  [configuration](../../wr2/locomotion/configs/ppo_walking_xy_physics.yaml),
  [reference](../../wr2/reference/walk_zmp.py) and
  [saved comparison](../../results/wr2_walking/wr2_xy_physics_20261013_085712_seed0/review_20261013/candidate_comparison.json).

Public references:

- [ToddlerBot paper, Tables VI/VII and Figure 5](https://arxiv.org/html/2502.00893v2).
- [ToddlerBot project](https://toddlerbot.github.io/).
- [Tedrake et al., closed-form ZMP planning](https://doi.org/10.1109/HUMANOIDS.2015.7363473).

This review records analysis and the proposed experiment. Training-result
CHANGELOG changes await user confirmation.
