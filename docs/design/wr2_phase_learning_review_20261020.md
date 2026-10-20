# Why WR2 loses outward right-leg coordination — 2026-10-20

The approved next training run is the
[collision-weight experiment](wr2_collision_experiment_20261020.md), retaining
20M/40M checkpoints to compare the acquisition path described here.

## Revised conclusion

**The evidence does not establish geometry as the primary cause.** WR2 can
produce outward right-hip movement during left-foot unloading with the
existing model. Its learned 20M checkpoint already does so on some steps.
The expected coordination disappears by 40M as the policy earns more reward
through angular stability and loses most commanded net translation. Later,
its right-hip request saturates inward throughout the cycle.

This revises the geometry-first priority in the
[paired stance review](wr2_tb_stance_result_20261018.md) and the
[whole-episode right-hip review](wr2_right_hip_incentives_20261019.md).
The contact approximation remains a measured model issue. Its contribution
to learning this stance has not been isolated.

## 1. The expected support pattern is present in TB

Compare the copied GPU forward `.10 m/s` nominal traces after the first
**1.5 TB-equivalent seconds**. These phases use **actual native floor
support**, not the desired reference contact. Both native implementations
identify support using a floor force above 1 N; this agrees with every saved
contact flag in both traces.

| Actual support phase | WR2 right hip | TB Paper 200M right hip |
| --- | ---: | ---: |
| Right supports; left unloaded | **5.51° inward** | **6.51° outward** |
| Left supports; right unloaded | 5.86° inward | 2.93° outward |
| Right-support samples after warmup | 426 | 332 |
| Right target clipping during right support | **100%** | 0% |

TB therefore makes the stronger outward right-hip movement in the support
phase the user describes. WR2's delivered right target stays at **−10°** in
both support phases, despite the actor receiving phase sine/cosine and
15 frames of joint, action, gyro and orientation history.

Physical signs were checked against each native model: **positive right hip
is outward in WR2; negative is outward in TB**. This difference is a coordinate
convention. WR2 trained a fresh policy; it did not reuse TB policy weights.

Neither native reference directly teaches this large outward support angle.
During actual right support, the mean right-hip references are only
**.055° inward for WR2 / .046° inward for TB**. TB's successful policy learns
to deviate from that soft reference. Increasing hip imitation would therefore
not directly teach the measured TB support pattern.

Native desired contact and actual support must remain separate in this
comparison. TB's actual gait has a substantial phase offset from its
reference. An unloaded foot also does not establish a complete geometric
swing or a successful forward step.

## 2. WR2 partially learned outward movement before losing it

Seven retained policies were evaluated with the same reset/key, unchanged
model and controller, normalized forward command `.10 m/s`, no pushes,
randomization or observation noise. Each nominal horizon is **20 TB seconds**.

| Checkpoint | Right hip during desired left swing | Net forward velocity, TB m/s | Native return |
| --- | ---: | ---: | ---: |
| Initial | .07° inward | −.00017 | 231.00 |
| **20M** | **.64° outward** | **.05181** | 303.17 |
| **40M** | **4.62° inward** | **.00290** | **346.92** |
| 80M | 4.29° inward | .00611 | 377.56 |
| 180M | 5.56° inward | .00361 | 388.04 |
| Selected 260M | 5.47° inward | .00553 | 384.94 |
| Final 300M | 5.03° inward | .00906 | 385.93 |

The 20M observation also holds when requiring **both desired left swing and
actual right-only floor support**: 166 ticks, mean actor request **1.85°
outward**, mean achieved hip **.16° outward**, maximum **5.27° outward**, and
outward achieved angles on **44.0%** of those ticks. Thus these are movements
from the learned actor with the native plant, rather than replaced reference
actions. They establish partial ability, not TB-quality walking.

By 40M, none of its actual right-support ticks has an outward right angle.
Yet its total nominal return increases **43.75**, while net forward velocity
falls **94.4%**. Its large 20M XY error, `.13335 m/s`, also shows that the
earlier movement is oscillatory and poorly directed; forward displacement
alone must not be treated as successful tracking.

### What rewards change from 20M to 40M

Same command, seed and full-duration horizon; all 25 native weighted,
dt-scaled terms were recorded and summed:

| Component change | Native return delta |
| --- | ---: |
| Yaw-rate tracking | **+46.19** |
| Roll/pitch-rate tracking | **+21.44** |
| Vertical-velocity stability | +8.28 |
| Torso orientation | +5.21 |
| Reference foot contacts | +4.12 |
| XY velocity tracking | **+1.70** |
| Air time | **−44.00** |
| Collision | −2.23 |
| Total, including other terms | **+43.75** |

**The first loss of outward coordination is accompanied mainly by angular
stability gains. Air time decreases across this transition.** Air time then
adds **+32.60** between 40M and selected 260M while translation stays small.
The earlier air-time-only hypothesis does not explain both stages.

The XY reward also improves slightly despite lost forward progress. Both
robots use `5 * exp(-100 * ||velocity-command||²)`: at `.10 m/s`, a stationary
robot still earns `5*exp(-1) = 1.84` per physical second from this term.
Irregular forward motion with substantial sideways motion can score worse.
The measured reward ranking can therefore favor a stable, nearly stationary
policy over the unstable 20M movement. This formula matches TB; it is not a
newly discovered Brax trainer bug.

## 3. Phase-specific outward intervention survives

The selected 260M actor continues to control every other joint. After
1.5 TB seconds, only the right hip is changed during **native desired left
swing/right support**, using a smooth sine-squared envelope over that phase.
The native one-tick delay, quantization, joint limits and slew remain active.

| Intervention | Survives 20 TB s | XY error, TB m/s | Net forward speed, TB m/s | Native return |
| --- | --- | ---: | ---: | ---: |
| Learned policy | Yes | .099103 | .00553 | 384.94 |
| Add up to 3° to raw actor right request | Yes | .099044 | .00607 | 388.23 |
| Move bounded target outward by up to 1° | Yes | .098435 | .00656 | 387.77 |
| Move bounded target outward by up to 3° | Yes | .097404 | .00712 | 384.23 |
| Blend bounded target to reference + 3° outward | **Yes** | .096554 | .01082 | 358.07 |

In the last condition the delivered target reaches **2.88° outward** and
the actual hip **1.41° outward**. The current geometry therefore permits
outward motion in this phase, including with the frozen policy's other
actions. Tracking improves only **2.6%** and net progress remains about 11%
of the command: this is a diagnostic result, not a walking fix. The larger
phase change loses angular-stability/orientation return and reduces total
return by **26.87**.

The earlier whole-episode `reference + 3°` override fell after 1.36 TB seconds.
It changed the hip during standing, transfer and opposite support as well.
That experiment did not answer whether a timed outward movement during left
swing was possible. This narrower experiment does.

## 4. Exploration becomes strongly biased inward

The native tanh-normal distribution was evaluated on each policy's own
nominal states. During desired left swing:

| Checkpoint | Probability of an outward right target | Probability of a target above the inward limit |
| --- | ---: | ---: |
| Initial | 49.4% | 89.3% |
| 20M | 59.5% | 99.98% |
| 40M | **2.48e−11** | Approximately 100% |
| Selected 260M | **2.42e−12** | **2.46e−5** |

These are mean **single-action conditional probabilities** under the saved
policy on nominal deterministic trajectories, not measured frequencies in
the randomized training state distribution.

Outward exploration has already disappeared at 40M **while the phase's
delivered target remains inside the joint limits**. Saturation develops
later and further limits effective exploration. At selected 260M, adding
up to 3° to the raw inward request still delivers **−10° on every steady
tick**. Changing the bounded physical target does move the joint. Raw action
changes can still affect action history and later policy responses; these
measurements do not imply a zero PPO gradient through all learning paths.

TB's training entropy loss is similar to WR2's, and selected WR2's right
latent standard deviation is larger than the mean of its other joints.
The specific problem is the strongly inward mean and the resulting physical
action probabilities; a generic claim that WR2 needs more entropy is not
established.

## 5. Comparison with TB's learning progress

The native training evaluations corroborate different learning paths. Their
component logging differs: TB logs weighted terms before multiplying by dt,
whereas WR2 logs weighted, dt-scaled terms. Divide each by its logged dt,
weight and mean episode length before comparing the reward kernels:

| Native 40M evaluation mean kernel | WR2 | TB Paper |
| --- | ---: | ---: |
| XY tracking | .369 | **.755** |
| Roll/pitch-rate tracking | **.508** | .072 |
| Yaw-rate tracking | **.958** | .709 |
| Torso orientation | **.725** | .436 |

TB learns translation while tolerating considerably more torso angular
motion under the shared objective. WR2 improves stability much more than
command tracking. These native evaluations use different command/reset
profiles and are corroboration; the nominal intervention comparisons above
are the controlled evidence.

Code was read from local ToddlerBot **`paper-walking-baseline`**, revision
`a4f693d7cc4544f011dad23d6c353e752a6819ab`. Its `MJXEnv.step`, `_get_obs`,
`_reward_leg_motor_pos`, `WalkEnv._reward_feet_air_time` and
`_reward_feet_distance` match the frozen diagnostic source by Python AST.
The actor phase input, fixed-home action origin, soft reference imitation,
old applied-action history and tanh-normal distribution are present in both
implementations. In particular, native TB updates `last_act` with
**`delayed_action`**, after observations/rewards; WR2 uses the same convention.
The previously documented native TB global-index issue in action penalties
remains separate from this phase investigation.

Sources: [`_paper_observation`, `_policy_target`, `_paper_step`](../../wr2/locomotion/walking_env.py),
[`paper_reward_terms`](../../wr2/locomotion/paper_rewards.py),
[`make_network_factory`](../../wr2/locomotion/ppo.py), and native TB
`mjx_env.py`, `walk_env.py`, `paper.gin` and `brax/training/distribution.py`.
Public recipe: [ToddlerBot, Tables VI–VII](https://arxiv.org/html/2502.00893v2).

## 6. Revised next priority

1. **Diagnose the angular instability of the early moving policy.** Make the
   next matched comparison WR2 **20M/40M** and TB Paper **20M/40M**, alongside
   their selected checkpoints. Use actual support phases and record both
   hips' requested/delivered/achieved angles, joint velocities/torques,
   torso roll/pitch/yaw rates, action probabilities and commanded net
   displacement. This distinguishes the loss of coordinated translation
   from the later saturating stance.
2. **Isolate the physical/control cause of that instability before choosing
   a training change.** Compare the phase and delivered-to-achieved response
   of the supporting and swinging legs, using native axes/transmissions and
   normalized clocks. The current evidence establishes the reward tradeoff
   and partial outward ability, but does not isolate which plant difference
   makes the early WR2 coordination much less stable than TB's.
3. **Choose the fresh 200–300M TB-aligned run from that result**, retaining
   intermediate checkpoints. Keep the established TB weights and action
   distribution during the diagnostic. Geometry validation remains a
   separate measured physics issue; it should not be presented as the
   established leading walking fix. Simply continuing the saturated 260M
   policy, prescribing a permanent outward hip target or disabling air
   time is not supported by this comparison.

## Verification and local artifacts

Thirteen nominal CPU traces pass finite/termination, native target delivery,
action-history, floor-contact, air-time, collision, imitation, XY tracking,
action-rate/acceleration and reward-sum audits. Maximum full-reward sum error
is **9.54e−7 per tick**. All selected-policy interventions have identical
states before warmup. No probe fell or produced nonfinite data.

Runtime: CPU JAX .6.2, Brax .14.2, MuJoCo/MJX 3.14.0, Flax .11.2, NumPy
2.4.6, high matmul precision; the training GPU used Brax .14.1 and Flax .10.7.
The selected CPU baseline reproduces copied GPU XY error within
**7.60e−6 m/s** and self-contact fraction within one of 824 ticks. New phase
interventions and historical replays have not been repeated on the GPU.

The probe generates only the two native reference commands it needs. Every
retained forward reference angle exactly matches the copied full-grid GPU
reference, maximum difference **0 rad**. Seed is 22102; control dt `.02 s`;
normalization `s=.8240547565`; 824 ticks = 16.48 physical seconds = 19.99867
TB seconds. Selected checkpoint SHA256:
`3d0fef64a99924474cd46ff5fac2c8e19fad7a115bf64390ba11c2871f51bf52`.

- [Probe summary and checkpoint action probabilities](../../results/wr2_walking/wr2_paper_20261018_121025_seed3/review_20261020/phase_hip/summary.json).
- [Native support comparisons, reward deltas and independent audit](../../results/wr2_walking/wr2_paper_20261018_121025_seed3/review_20261020/phase_hip/analysis.json).
- [Probe source](../../results/wr2_walking/wr2_paper_20261018_121025_seed3/review_20261020/phase_hip/probe.py)
  and [analysis source](../../results/wr2_walking/wr2_paper_20261018_121025_seed3/review_20261020/phase_hip/analyze.py).

Generated traces and analysis scripts remain under ignored `results/`; this
review records the evidence in Git. These measurements cover one nominal
forward command and reset seed. The original event that selected the right
side, and its sensitivity to training seed/randomization, remain unresolved.
