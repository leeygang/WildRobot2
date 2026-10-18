# WR2 and TB Paper training setup audit — 2026-10-18

## Scope and conclusion

This reviews the prepared **WR2 0.28.5 / seed 3 / fresh 300M / no physical
pushes** experiment at source **5f58c13**, while the user runs training. Its GPU
run manifest and progress logs have not been copied locally. WR2 result numbers
below belong to the completed **0.28.4** control, not the running experiment.

The comparison code is local TB branch **`paper-walking-baseline`**, HEAD
**a4f693d**, with run `toddlerbot_walk_ppo_20261015_194347`, training source
**1982ef3**, and released Brax fork **14147806**. Branch changes since the run
are documentation/results only. Direct branch-object checks confirmed that
**20 inspected source/model files** match the frozen checkout used here,
including all **11 captured locomotion files**. All **12 provenance
model/reference hashes** also match. The run recorded a dirty tree; these
checks cover its captured files. Active TB `main-walking-compare` is a different
development recipe. The local Paper run reconstructs the published recipe; it
does not establish the exact implementation used for the authors' results.

**WR2 already matches Paper's main PPO settings, network/distribution, fixed
action origin and moving leg imitation. Matching all 25 reward coefficients
does not make the complete training problems equivalent.** The remaining
differences include the physical actuator, control clock, effective
randomization, several reward implementations and episode reset behavior.

The current experiment is useful: it removes a confirmed disturbance mismatch
without changing the optimizer, rewards or plant. Its treatment also zeros
critic push labels, whereas frozen TB retained random labels for unrealized
pushes. Interpret the result as this declared acquisition recipe comparison.

### Verified differences that matter

| Item | Frozen TB Paper | Current WR2 | Evidence / interpretation |
| --- | --- | --- | --- |
| Physical pushes | Configured, but dropped by the `qd`/`qvel` alias bug | Disabled during acquisition; realized in the paired diagnostic | Earlier same-state physics probe and complete paired suites; current run addresses this exposure difference |
| Ground friction | Configured randomization changes `geom_friction`; explicit pairs keep effective sliding friction **1.0** | Effective sliding friction varies **0.7–1.3** | New native MuJoCo 3.2.4 contact probe plus MJX contact parameter source |
| Leg action smoothness | Global actuator indices incorrectly index a 12-value action; four actions omitted and the last action counted five times | All 12 leg actions weighted evenly | New calls to actual frozen reward methods with the model's indices |
| Foot slip | Two-foot mask broadcasts across XY axes | Each foot masks its own XY speed | Actual reward bodies; a confirmed TB indexing bug |
| Automatic reset | Actor/pipeline reset; critic history and action buffers carry; native reference clock survives a timeout | Actor and critic reset together; environment resets action history, counters and landing accumulators | New actual TB timeout probe and WR2 reset code |
| Reference table | 960 ticks / 19.2 s, including a finite stopping tail; wraps inside a 20 s episode | 1000 ticks with extra preview beyond retained frames; counter resets at the training timeout | Hashed original TB cache, lookup methods and WR2 generator |
| Body-relative control clock | 20 ms at TB size | 20 ms physical = **24.27 ms in TB time units** | Measured model CoM heights and source clocks |
| Plant response | Different Dynamixel families and transmission mechanics | HTD45H inertia/friction, continuous motoring taper, active derivative feedback and host target slew limit | Previous matched waveform and exact WR2 input-replay review |

These are confirmed implementation differences. Their individual effects on
learning require controlled experiments; the table does not assign all of the
remaining tracking gap to any one row.

## 1. PPO, networks and initialization

Source: WR2 [`train.py`](../../wr2/locomotion/train.py),
[`ppo.py`](../../wr2/locomotion/ppo.py), the installed Brax 0.14.1/0.14.2 selected
loss paths, and TB's captured
[`train_config.json`](../../../toddlerbot/results/toddlerbot_walk_ppo_20261015_194347/train_config.json),
[`train_mjx.py`](../../../toddlerbot/results/toddlerbot_walk_ppo_20261015_194347/locomotion/train_mjx.py)
and pinned local Brax fork.

| Setting | TB Paper | WR2 0.28.5 |
| --- | --- | --- |
| Initialization | Fresh, seed 0 | Fresh, seed 3; same seed as the 0.28.4 control |
| Requested / rounded transitions | 300,000,000 / **300,134,400** | Same |
| Actor / critic hidden layers | 512 / 256 / 128; Swish | Same; `silu` is Swish |
| Distribution | State-dependent Gaussian, then tanh | Same NormalTanh path |
| Initial scale on zero inputs | `softplus(0) + 0.001 = 0.694147` | Same; real observations change the scale |
| Rollout batch | 1024 environments × 20 ticks = **20,480** | Same |
| PPO passes / minibatches | 4 × 4 = **16 Adam updates per batch** | Same; each minibatch has 256 sequences of 20 ticks |
| Whole-run batches / Adam updates | **14,655 / 234,480** | Same |
| Learning rate | Starts 1e-4; cosine ends approximately **9.999512e-5** | Fixed **1e-4** |
| Discount / GAE lambda | 0.97 / 0.95 | Same |
| Ratio clipping / entropy coefficient | 0.2 / 0.0005 | Same |
| Advantage / observation normalization | Advantages normalized; observations unnormalized | Same |
| Value loss | **0.25 × mean squared error** | Same; adapter passes 0.5 because native Brax multiplies by another 0.5 |
| Gradient / value clipping | No gradient clipping; no value clipping | Same |
| Timeout treatment | Truncation-masked GAE | Same selected path; `bootstrap_on_timeout` remains false |
| Mirror loss / distributional critic | Absent | Disabled |

The TB cosine counter counts optimizer updates against a 50M-update decay
duration. At this budget its LR decreases only **0.004884%**. The WR2 fixed
schedule therefore matches its effective LR. `desired_kl=0.01` is a diagnostic
reference under the fixed schedule, not an enforced limit. PPO ratio clipping
also does not guarantee a small distribution update.

The prior native update/restore checks confirm correct actor/critic routing,
GAE and value-loss coefficient conversion. This review finds **no demonstrated
WR2 Brax adapter bug that explains the walking gap**. Equal mathematical PPO
settings do not imply identical trajectories across different observations,
plants, rewards, seeds and library versions.

Actor inputs are 975 for WR2 and 1245 for TB; critic inputs are 1485 and 1890.
With the same hidden layers, actor/critic trainable parameter counts are
**667,032 / 925,185** for WR2 and **805,272 / 1,132,545** for TB. The capacity
difference comes from each robot's motor observations, rather than a smaller
WR2 hidden network.

### Runtime and numerical settings

The completed WR2 control recorded Python 3.10.19, Brax 0.14.1, JAX 0.6.2,
MuJoCo/MJX 3.14.0, Flax 0.10.7 and Optax 0.2.8. TB recorded Python 3.10.19,
Brax 0.10.5 at the pinned fork, JAX 0.6.2, MuJoCo/MJX 3.2.4, Flax 0.10.6 and
Optax 0.2.5. The WR2 lock selects Brax 0.14.2 on Python 3.11 or newer; the new
GPU run's provenance determines its actual runtime.

TB explicitly selects JAX matmul precision `high` and
`--xla_gpu_triton_gemm_any=true`. WR2's entry point does not set either; host
settings can supply them, and the completed WR2 manifest did not record those
values. This is a reproducibility gap, not evidence of a numerical failure.
Record actual device, precision and XLA flags when preparing a subsequent run.

## 2. Physical scale, duration and clocks

Measured nominal model values are WR2 mass **2.05833 kg**, CoM height
**0.202233 m**; TB mass **3.35818 kg**, CoM height **0.297811 m**. Define:

- Length ratio `lambda = h_WR / h_TB = 0.679066`.
- Gravitational time ratio `s = sqrt(lambda) = 0.824055`.
- Mass ratio `mu = m_WR / m_TB = 0.612930`.
- TB-equivalent velocities: `v_WR / s`, angular velocities: `omega_WR * s`;
  TB-equivalent duration: `t_WR / s`.

This follows gravitational dynamic similarity. The robots are not geometric
copies: actual foot separations are approximately 81.4 mm WR2 and 74.0 mm TB,
and their feet are similarly long despite WR2's lower CoM.

| Clock / duration | TB physical | WR2 physical | WR2 in TB time units |
| --- | ---: | ---: | ---: |
| Policy tick and one-tick action delay | 20 ms | 20 ms | **24.27 ms** |
| Physics tick / substeps | 4 ms × 5 | 5 ms × 4 | 6.07 ms × 4 |
| Gait period | 0.72 s / 36 policy ticks | 0.60 s / 30 ticks | **0.72811 s**, 1.13% longer |
| Observation history, span between first/last of 15 samples | 0.28 s | 0.28 s | **0.33978 s** |
| 20-tick rollout | 0.40 s | 0.40 s | **0.48540 s** |
| Training episode ceiling | 20 s / 1000 ticks | 20 s / 1000 ticks | **24.27023 s** |
| Primary benchmark ceiling | 20 s / 1000 ticks | 16.48 s / 824 ticks | **19.99867 s** |
| Command resampling | 3 s / 150 ticks | 2.48 s / 124 ticks | **3.00951 s** |
| Discount e-fold time, `-dt/log(0.97)` | 0.65662 s | 0.65662 s | **0.79681 s** |
| GAE trace e-fold time, `-dt/log(0.97*0.95)` | 0.24464 s | 0.24464 s | **0.29687 s** |

Thus the gait and command schedules are scaled closely, but **the control,
history, rollout and discount clocks are not a complete time-scaled replica**.
WR2 has 30 decisions per gait cycle versus TB's 36. Each rollout covers about
two-thirds of a WR2 cycle versus 0.56 TB cycles. The 20 ms clock reflects WR2's
current hardware/control contract; changing it also changes actuator sampling,
delay, slew limits and reward event accounting.

For illustration, preserving TB's discount/trace time constants at the current
WR2 clock would give `gamma = 0.97**(1/s) = 0.963712` and
`lambda_GAE = 0.95**(1/s) = 0.939653`. These are equivalence calculations, not
a proposed optimizer change for the running experiment.

At 300,134,400 transitions each run generates **6,002,688 aggregate physical
simulation seconds** across its training environments. WR2's corresponding
TB-equivalent time is **7,284,331 s**. The same budget gives WR2 more normalized
elapsed time, with coarser decisions. It does not demonstrate a shortage of
training time.

## 3. Complete reward comparison

Source: WR2 [`paper_rewards.py`](../../wr2/locomotion/paper_rewards.py) and
[`_paper_step`](../../wr2/locomotion/walking_env.py), TB captured
[`mjx_env.py`](../../../toddlerbot/results/toddlerbot_walk_ppo_20261015_194347/locomotion/mjx_env.py),
[`walk_env.py`](../../../toddlerbot/results/toddlerbot_walk_ppo_20261015_194347/locomotion/walk_env.py)
and [`env_config.json`](../../../toddlerbot/results/toddlerbot_walk_ppo_20261015_194347/env_config.json).

All **25 coefficients match**, with **21 nonzero terms**. Both sum weighted
terms and multiply by **0.02 s**. There is no clipping of total reward to zero.
For the table, `v`, `omega` and `command` use TB-equivalent units; `q` is motor
position unless stated otherwise; `R(x,lo,hi)` is the mean of the two
exponentials penalizing deviation beyond each range boundary.

| Term | Weight | WR2 computation / role | Difference from actual frozen TB |
| --- | ---: | --- | --- |
| `torso_pos` | 0 | Disabled | TB position-to-path formula is also disabled |
| `torso_quat` | 1 | `exp(-20 * orientation_angle_error²)` | Same upright/yaw-path objective with zero waist command |
| `lin_vel_xy` | 5 | `exp(-100 * ||v_xy - command_xy||²)` | Same after velocity scaling; torso-origin velocity in both |
| `lin_vel_z` | 1 | `exp(-100 * v_z²)` | Same after scaling; target zero |
| `ang_vel_xy` | 2 | `exp(-25 * ||omega_xy||²)` | Same after scaling; target zero |
| `ang_vel_z` | 5 | `exp(-25 * (omega_z-command_yaw)²)` | Same after scaling |
| `leg_motor_pos` | 5 | `-mean((q_leg-q_ref_leg)²)` | Same soft imitation objective; references and physical execution differ |
| `neck_motor_pos` | 0 | Disabled | Disabled |
| `arm_motor_pos` | 0 | Disabled | Disabled |
| `waist_motor_pos` | 0 | Disabled | Disabled |
| `feet_contact` | 1 | Number of feet matching reference contact, 0–2 | Same rule; contact and reference schedules must be checked separately |
| `feet_air_time` | 500 | Sum old air-time counters at filtered first landing, gated by command | Same uncapped landing-counter logic; physical seconds retained |
| `feet_clearance` | 0.05 | Sum accumulated foot heights at filtered landing, divided by `s²`, gated by command | TB also sums height samples, without `dt`; WR2 normalizes height |
| `feet_distance` | 1 | `R(lateral_foot_distance/r_foot, 0.05, 0.13)` | Stance-width normalization uses `r_foot=1.114559`, rather than uniform length ratio |
| `feet_slip` | 0.05 | `-sum(contact_foot * ||foot_v_xy/s||²)` | TB broadcasts `[left,right]` contact across the XY columns; WR2 fixes the mask |
| `align_ground` | 1 | Negative mean squared `abs(hip_pitch)+abs(ankle_pitch)-abs(knee)` | TB reads physical joint positions; WR2's corresponding pitch motors directly represent these joints |
| `stand_still` | 1 | Negative squared sum of absolute leg displacement from home, only below deadzone | TB uses physical leg joints; its geared hip-yaw displacement differs from motor displacement |
| `torso_roll` | 0.5 | `R(roll,-0.1,0.1)` | Same |
| `torso_pitch` | 0.5 | `R(pitch,-0.2,0.2)` | Same |
| `collision` | 0.1 | Negative count of active non-floor collider pairs, both directions | Same count intent; TB sums force by pair before threshold, WR2 thresholds contact points before deduplicating; collider sets also differ |
| `leg_action_rate` | 0.05 | Negative mean squared new action minus old applied action | WR2 uses all 12 local actions; TB's global indexing produces asymmetric weights |
| `leg_action_acc` | 0.05 | Negative mean squared new action minus twice old applied action plus its predecessor | Same history convention, with the same TB global-index bug |
| `motor_torque` | 0.01 | `-mean(torque²)` over all 21 motors | TB averages all 30 motors; physical torque units are retained |
| `energy` | 0.05 | `-mean((torque * motor_velocity * s)²)` | TB uses squared mechanical power; torque remains physical in WR2 |
| `survival` | 10 | Negative terminal indicator | **Death penalty**, not +10 per living tick; TB also suppresses it at its 100 s native-clock boundary |

### Landing reward can favor flight

Both implementations allow **both feet's counters to grow during flight**.
There is no opposite-foot support requirement, cap at the planned swing time,
or direct no-support penalty in this term. TB defines `max_feet_air_time` in its
constructor but does not use it in the reward.

At the shared control tick:

- One foot landing after 0.1 s airborne receives **1.0 return** from this term.
- Both feet landing after 0.1 s receive **2.0 return**.
- Perfect XY tracking yields at most **0.1 return per tick**.
- Standing at a +0.10 m/s command yields **0.03679 per tick** from XY tracking.
- A height termination has an explicit survival penalty of only **-0.2**;
  termination also removes future rewards.

For an ideal alternating cycle with aggregate airborne foot-time equal to
two-thirds of elapsed time, the air-time contribution averages **6.67 reward
units/s**, versus at most 5 from XY tracking. Reward from airborne counters
can approach 20 units/s when both feet spend nearly all elapsed time airborne,
provided there are subsequent landings to realize the counters. This is a
reward accounting limit, not a physically feasible gait prediction.

The previously audited WR2 220M nominal forward trajectory received **66.9%**
of its total return from air time. Its flight and poor tracking were also
measured from geometry and velocity, rather than inferred from this reward.
Those findings support checking the running no-push policy for exploitation;
they do not yet establish that a reward change is the best next intervention.

The deadzone uses the norm of the three command components, as TB does. A
nonzero small command still requests a walking reference even when it earns no
air-time/clearance reward. The strict inequalities can leave commands exactly
at the 0.05 threshold outside both landing shaping and standing regularization.

### Newly confirmed TB smoothness indexing bug

TB's actual leg actuator IDs are **4–15** because neck and waist precede the
legs. Its policy action has **12** elements, corresponding to those legs.
`action[self.leg_actuator_indices]` therefore indexes that local action as:

```text
[4, 5, 6, 7, 8, 9, 10, 11, 11, 11, 11, 11]
```

JAX clamps these positive out-of-range gathers to the last element. Both
smoothness rewards consequently omit **left hip pitch, left hip roll, left
hip yaw and left knee**, and count **right ankle pitch five times**. In an
actual-method single-action pulse probe, a left-hip pulse costs 0, a right
ankle-pitch pulse costs -5/12; WR2's corresponding terms each cost -1/12.

This corrects the earlier description that slip was the only reward indexing
difference. The existing numerical reward fixture uses local action indices,
so it checks formulas without reproducing this frozen TB bug. WR2's even
indexing reflects the intended leg regularizer. Replicating this asymmetry
would require evidence from training, rather than an assumption that every
detail of the successful frozen control is beneficial.

Both codes compare the new request against the **old applied action**. With a
one-tick delay, this is `a_t - a_(t-2)`, and the acceleration term is
`a_t - 2*a_(t-2) + a_(t-3)`. Neither is a derivative of delivered physical
targets in rad/s. Compare target rate and acceleration independently when
studying actuator bandwidth.

### Effort and range rewards need careful units

`energy` is a squared-power penalty, not energy consumed or cost of transport.
Torque and energy averages include held upper-body motors as well as legs.
For otherwise identical leg loads and zero upper-body load, changing the
denominator from 30 to 21 increases the mean by **30/21 = 1.43**.

Dynamic similarity would scale torque by `mu*lambda = 0.416220`. For a
hypothetically similar motion, its squared torque penalty would scale by
**0.173239**. The current effort terms therefore are not completely
dimensionless. Actual HTD45H torques do not follow that hypothetical scaling;
it does not justify multiplying a weight by 5.77 without further evidence.

The range rewards tend toward **0.5**, rather than zero, far outside the allowed
interval. Roll/pitch and foot-distance terms are soft shaping. The termination
condition independently uses torso height; there is no direct tilt or no-foot
support termination. A healthy-height policy can still hop or fail to track.

## 4. Actions, observations, reference and reset

### Action origin and reference roles

Both use a **fixed home + 0.25 rad × delayed bounded leg action**. The home
does not follow the walking reference. The moving reference affects **leg
imitation reward and critic motor-reference error/contact features**. It is
absent from the actor's direct joint input. PPO advantages transmit the
imitation objective to the actor.

This is the current WR2 recipe. The old walking home-pose regularizer has been
replaced by moving leg imitation; home regularization now applies only while
standing. The ZMP/COP trajectory itself is not a direct actor feature or a
direct COP-tracking reward.

| Input / control item | TB Paper | WR2 |
| --- | --- | --- |
| Active actions / observed motors | 12 legs / 30 motors | 12 legs / 21 motors |
| Actor frame / stack | 83 × 15 = 1245 | 65 × 15 = 975 |
| Critic frame / stack | 126 × 15 = 1890 | 99 × 15 = 1485 |
| Actor features | Phase sin/cos, command, motor-minus-home, motor velocity, old applied action, local gyro, unwrapped Euler | Same feature roles, normalized velocity units |
| Critic additions | Motor-minus-reference, local linear velocity, actual/reference contacts, six push labels | Same roles; push labels zero during acquisition |
| Linear velocity critic scale | 2 × physical m/s | **2/s × physical m/s**, already correct |
| White actor noise in final feature units | Position 0.05, velocity 0.1, gyro 0.25, Euler 0.1 | Same |
| Backlash feature | `0.02*tanh(torque/0.1)` | Same in the noisy Paper path |
| Simulation target filtering | `none`, plus motor limits | Motor limits, encoder quantization and extra host slew cap |

Paper noise is white Gaussian added in observation units. WR2's legacy colored
gyro noise, amplitude multipliers and projected-gravity noise are not applied
by `_paper_observation`. The critic excludes white noise/backlash, but can
contain the configured push labels. Motor force is not a direct critic feature
in either Paper observation vector.

TB adds its torque-dependent backlash feature even with `add_noise=False`;
it also samples yaw and +/-0.02 rad leg target bias even with domain
randomization disabled. WR2 gates those features/draws with its noisy/randomized
paths. A TB "nominal" run with white noise and domain variation disabled
therefore retains these effects. The current primary profiles both retain the
full noisy/randomized paths apart from WR2's zero physical pushes.

### Reference construction and indexing

Both use preview ZMP planning, alternating footsteps, single/double support
ratio 2 and nearest-command lookup indexed by episode time. TB swing height is
0.05 m; WR2 uses **0.033953 m = 0.05*lambda**. WR2 generates six-DOF leg IK on
its actual model. The full lookup passes the existing position/orientation
geometry gate; kinematic reachability does not ensure actuator executability.

TB's planner uses **0.3267 m torso height as a CoM proxy**, 9.7% above measured
whole-body CoM. WR2 uses its measured **0.202233 m**. Scaling TB's proxy by
`lambda` instead would give 0.221851 m. The actual-CoM choice is an explicit
physical difference supported by the model measurement.

The hashed original TB cache has **732 commands and 960 frames per command**.
It drops the first planning cycle from a finite trajectory. Its lookup wraps
at 19.2 s, before the 20 s evaluation/training horizon; 19.2 s is not an integer
multiple of the 0.72 s gait. At the first wrap, the phase signal and retained
reference indexing differ by **4.18879 rad**. The +0.10 forward cached joint
trajectory jumps by up to **0.27215 rad** from its last to first sample.

WR2 has **227 commands and 1000 retained frames**, with three additional
planning cycles of preview horizon. The training counter resets at 1000, so
it does not wrap within an episode. This avoids exposing WR2 to the finite
planner's stopping tail. Its table covers the configured training bounds;
TB's cache extends beyond its training command bounds.

Float construction also breaks exact ties differently at benchmark commands
halfway between 0.02-spaced entries:

| Requested component in TB units | TB nearest reference | WR2 nearest reference in TB units |
| --- | ---: | ---: |
| +0.05, forward or lateral | approximately +0.06 | approximately +0.04 |
| -0.05, forward or lateral | approximately -0.04 | approximately -0.06 |
| +0.10 forward | approximately +0.10 | approximately +0.10 |

The WR2 reconstruction used in this probe exactly matches its saved
227-command runtime grid. This affects imitation/teacher diagnostic
comparability at those points. For a frozen deterministic actor, the moving
leg reference is not an actor input or an applied leg target, so the tie alone
does not directly change its walking trajectory during evaluation. It is not
evidence that the reported XY tracking error has the wrong command target.

### Automatic reset is a real implementation difference

The pinned TB wrapper restores `first_pipeline_state` and `first_obs` on done.
It does not restore `privileged_obs` or environment-owned action/history
fields. A timeout does not reset TB's native `info['step']`; height failure
does. Thus a 20 s physical restart can retain a later reference clock, command,
pending delayed action and landing accumulators.

The actual native timeout probe at tick 1000 produced:

```text
done = 1; truncation = 1
pipeline and actor difference from initial = 0
critic maximum difference from initial = 2.560292
native reference step = 1000; pending action buffer contains 0.4
```

WR2 stores actor and critic in the same observation tree, so native AutoReset
restores both. `_paper_step` explicitly clears environment history and resets
its own counter at height failure or timeout. This protects the physical reset
contract. Both recipes reuse each environment's initial model/yaw/controller
draws across automatic resets; this is not newly sampled domain variation
every episode. Do not interpret a finite TB training run as proof that its
critic/history reset semantics were correct.

TB's reset also converts its uniform 0–2*pi yaw draw to degrees before a
radian-input quaternion conversion. It still covers all headings, but its
wrapped distribution is slightly nonuniform. WR2 samples radians directly.
There is no evidence assigning the walking gap to this small yaw difference.

## 5. Commands, randomization and actuator physics

The signed ellipse sampler, shared radial draw and branch probabilities match:
**20% standing, 30% pure turning, 50% XY walking**. Commands in TB units are
X **[-0.10,+0.20] m/s**, Y **[-0.10,+0.10] m/s**, yaw **[-0.50,+0.50] rad/s**;
WR2 scales linear commands by `s` and yaw by `1/s`. There is no command or
push curriculum in the current acquisition run.

| Variation | Frozen TB configured / effective | Current WR2 |
| --- | --- | --- |
| Sliding friction | Configured uniform coefficient 0.5–2.0; explicit contact pairs effectively keep **1.0** | Relative scale 0.7–1.3 on nominal 1.0; **effective 0.7–1.3** |
| Passive damping / armature | Per-DOF factors 0.8–1.2 | Per-DOF factors 0.8–1.2; different base values |
| Dry joint friction | Per-DOF factors 0.8–1.2 | 0.7–1.3 |
| Body mass/inertia | Torso additive -0.2..+0.2 kg; each hand 0..+0.1 kg; others unchanged, host-permuted grids | Independent 0.8–1.2 factors for each non-world body, inertia scaled with mass |
| Controller gains | Kp/Kd factors 0.9–1.1 | Kp/Kv factors 0.9–1.1; nominal TB active Kd is zero |
| Torque / speed envelope | Controller torque, plateau speed and maximum speed factors 0.8–1.2 | Motoring cap, maximum speed and braking factors 0.9–1.0; plateau remains zero |
| Target bias | Uniform +/-0.02 rad on leg home, sampled at reset | Same leg target-bias bounds |
| Initial joint / velocity noise | Reference-derived pose, zero velocity | Grounded reference pose, zero configured joint/velocity noise |
| Model constant refresh | Mass/inertia constants prepared first; armature varied later without another refresh | Host `mj_setConst` after mass/inertia/**armature** changes |

Equal percentage ranges do not imply equal physical uncertainty: TB nominal
ankle armature is 0.0012 kg m²; WR2 is 0.020951 kg m². The accepted WR2
low-load identification does not uniquely establish that armature or validate
the whole torque-speed envelope. A more credible physics model needs hardware
measurement, rather than replacing it with TB numbers to improve simulation.

TB also has MuJoCo control limits on its motor classes. For example, hip-yaw
and hip-pitch classes cap torque at 1 and 2 Nm respectively, potentially
trimming the top of their randomized controller torque range. Read effective
actuator force when comparing torque; a sampled controller limit is not itself
proof of delivered torque.

### Ground friction finding

TB's model has **18 explicit contact pairs**. Its randomizer modifies
`geom_friction`, while MJX 3.2.4 takes pair contact parameters from
`pair_friction`. With the actual native model, setting geom sliding friction
to **0.5, 1.0 or 2.0** produces the same floor-contact friction vector:

```text
[1.0, 1.0, 0.005, 0.0001, 0.0001]
```

WR2 has no explicit pairs. Its geometry friction changes do reach contact
parameters: the corresponding sliding entries become **0.7 or 1.3** at the
configured endpoints. Thus the current no-push experiment still has variable
ground friction while TB's successful control had fixed ground friction.
That makes one exposure axis more demanding; the other different ranges
prevent declaring all of WR2 randomization categorically harder.

### Plant differences remain substantial

The [actuator review](wr2_actuator_gap_review_20261018.md) isolates these mechanisms:

- WR2 ankle reflected armature is about **61.8 times** dynamically scaled TB;
  total damping about **24.5 times**, dry friction about **8.3 times**.
- WR2's motoring cap falls from 4 Nm at zero speed to about **1.94 Nm at
  3 rad/s**. TB ankle/knee retain a plateau to 4.22 rad/s. Braking and passive
  drag need separate treatment.
- WR2 quantizes targets at **0.004189 rad** and limits delivered setpoint
  change to **0.113097 rad per tick**. TB simulation has no equivalent host
  slew cap. The cap is a derived WR2 restriction, not a measured firmware limit.
- Same scaled 0.035 rad ankle excitation at 1.373 TB Hz: WR2 tracking RMSE
  **0.01861 rad**, TB **0.00400 rad**. This establishes a modeled plant
  response difference independently of the learned policy.
- The newer WR2 policy requests much larger/faster target excursions than
  the retained older policy. Exact target replays with half armature or a
  constant motoring cap modestly improve joint tracking and fail to recover
  TB-like locomotion. They are not closed-loop retraining predictions.

The compiled WR2 model enables collision geometry only for the floor and its
two feet. TB enables the floor and 16 body colliders, including torso, waist,
hips, ankles, arms and hands. WR2's other rendered parts cannot produce
contacts or self-collision penalties. This changes the physical constraints;
its contribution to the learning gap has not been isolated.

WR2 uses Euler, Newton 10 and 5 ms substeps; TB overrides its model to Newton
1 and 4 ms substeps. Both disable implicit Euler damping. Collider geometry,
transmissions and integration/constraint behavior also differ. Blindly
transplanting TB's low inertia into the current WR2 controller/timestep can
produce braking oscillations in the scalar diagnostic. Use a validated
controller/physics identification before a training physics change.

## 6. Metrics, checkpoint selection and training progress

### Keep the metrics distinct

1. **Training objective:** stochastic policy, sampled standing/turning/XY
   commands, weighted reward rates and terminal loss.
2. **Native PPO return evaluation:** TB uses random commands and a stochastic
   actor; WR2 uses a fixed forward command and deterministic actor. These
   returns are progress telemetry, not a common cross-robot score.
3. **Shared task benchmark:** both use deterministic actors, ten fixed signed
   moving commands, standing and four start/stop conditions. The primary score
   is the mean, over episodes and ten fixed commands, of each episode's mean
   **instantaneous XY vector error**. WR2 errors divide by `s` for TB units.

TB's native reward-component logs accumulate weighted rates **before** the
0.02 multiplication. WR2 Paper component logs include it. Summing raw component
columns across repos therefore introduces a **50-fold** accounting error.
Neither convention changes the reward passed to its learner.

The task benchmark uses **128 models/resets × 15 conditions**, seed **12102**.
WR2's normalized schedule has boundaries 124/371/494/700 ticks and recovery
windows 21 ticks; TB uses 150/450/600/850 and 25 ticks. The nominal ceiling is
16.48 physical seconds WR2 and 20 seconds TB. The separate WR2 engineering
profile uses 20 physical seconds and should not replace the scaled profile.

Important interpretation rules:

- Error is neither RMSE nor error of mean velocity. Oscillation can increase
  instantaneous error despite reasonable average forward speed.
- A finite terminal sample counts, then the episode stops contributing.
  Invalid transitions count toward episode length/nonfinite totals but are
  excluded from measurement values. High failure can artificially lower a
  censored tracking score; inspect lengths and survival with it.
- "Fall" means torso-height termination: TB 0.2–0.4 m, WR2
  **0.170959–0.341917 m**, scaled by home torso-origin height. It does not count
  every undesirable gait, stumble or geometric flight event.
- Foot support is a thresholded summed upward force. Less than 1 N is not
  proof of no geometric contact. Compare force and actual foot geometry.
- Torque RMS uses control-step force samples; the primary report does not
  integrate all four/five physics substeps. Compare physical units and motor
  population, and use a separate substep trace for saturation/load analysis.
- Leg-reference RMSE is relative to different kinematic teachers, including
  the midpoint tie and stopping-tail differences. It is not a standalone
  cross-robot walking-quality measure.
- The 15-condition suite contains **no pure yaw-turn benchmark**, despite
  turning occupying 30% of training commands. Its selection score covers XY
  walking and transitions, not the entire trained command distribution.
- Fixed model/reset keys support paired comparisons. Episode-condition
  results sharing a model/reset are correlated; confidence intervals should
  cluster by that draw. They do not replace independent training seeds.

### Checkpoint retention and selection

Both retain 15 trained checkpoints about every **20,008,960 transitions**.
WR2 also evaluates initial parameters. TB's native reward-best marker selects
240M; its useful task controls are **180M for survival and 200M for tracking**.
WR2 selects the lowest ten-command XY error among complete finite primary
suites with at most 1% height terminations (**at most 19 of 1920**).

For 0.28.5, primary selection uses **no physical pushes**. Each checkpoint also
has the paired full physical-push suite on identical model/reset draws. That
secondary profile does not choose the checkpoint. The selected checkpoint is
confirmed under both profiles on held-out seed **22102**. Passing the 1% gate
alone does not establish TB parity: check directions, transitions and foot use.

The extra evaluations do not count toward the 300M training transitions or
perform PPO updates. Before censoring, current WR2 primary/secondary suites
cover about **50.63M evaluation ticks**, plus about **3.16M** for held-out
confirmation and **2.05M** for native return telemetry. TB's 15 task suites
cover **28.8M ticks**. Expect added evaluation overhead in the running experiment.

### Measured reference progress

These are saved GPU metrics. WR2 columns refer to the completed pushed 0.28.4
seed-3 control, so the comparison retains unequal disturbance exposure.
Errors are TB-equivalent m/s; height terminations are out of 1920.

| Fresh transitions | TB Paper error / height terminations | WR2 0.28.4 error / height terminations |
| --- | ---: | ---: |
| 20.009M | 0.062016 / 9 | 0.193075 / 329 |
| 80.036M | 0.039653 / 17 | 0.143076 / 56 |
| 180.081M | **0.038379 / 3** | 0.132171 / 60 |
| 200.090M | **0.037025 / 8** | 0.131349 / 45 |
| 300.134M | 0.037999 / 134 | 0.149946 / 130 |

TB reaches useful tracking by about 80M. Its final policy has worse survival
than its intermediate controls. WR2's best 0.28.4 tracking is at 220M:
0.130627 / 55 terminations. Preserve intermediate policies and compare
acquisition at 80M and mature behavior at 180–220M; longer training is useful
evidence, but final checkpoint selection is not automatic.

The previous complete paired CPU suites reduce WR2 220M error from **0.130581
to 0.076575** and height terminations from **54 to 0** when physical pushes are
zeroed. TB's native CPU control is **0.037069 / 11**. Remaining WR2 error is
**2.07 times TB**, with flight still present. The analytic zero-XY-velocity
baseline for these ten commands is **0.071623 m/s**; it is not a simulated
standing controller. These observations make stable survival insufficient
evidence of useful acquisition.

Completed native learner timers were **8892.8 s / 2.47 h TB** and
**6795.7 s / 1.89 h WR2 0.28.4**. Last metric callback elapsed times were
**11243.5 s / 3.12 h TB** and **8707.5 s / 2.42 h WR2**, including intervening
work such as evaluations. Different plants, runtimes, diagnostics and device
recording prevent attributing that timing difference to an intrinsically more
efficient learning algorithm. The current paired evaluation adds overhead.

## 7. What the running experiment establishes, and the next decision

Keep the running **fresh 300M no-push** recipe as prepared. Compare its retained
checkpoints against the same-seed 0.28.4 control and TB Paper 180M/200M.
Evaluate **learning speed, directional tracking, height survival, geometric
flight, target/delivered/actual motion and reward composition** together.

| Outcome | Supported conclusion | Next investigation |
| --- | --- | --- |
| No-push training reaches TB-like tracking and gait, confirmed on held-out draws | The prior acquisition exposure was a major obstacle | Develop robustness from the retained acquisition checkpoint; use a TB control trained with realized pushes before comparing published robustness |
| Height survival improves but XY error/flight remain poor | Removing physical pushes alone does not close acquisition; existing frozen-policy evidence predicts this is possible | Compare nominal versus randomized no-push execution, then separate actuator execution error from policy requests and airborne-counter reward |
| Nominal walking is useful, randomized no-push walking is poor | Variation/execution sensitivity is a concrete remaining gap | Paired friction and actuator-variation evaluations; a subsequent single-variable acquisition control needs the measured sensitivity, not a broad DR rewrite |
| Actual joints lag feasible delivered targets strongly | Plant response needs a better identified explanation | Resolve inertia versus delay with known fixtures and logged command timing; validate torque-speed/load response before modifying physics |
| Plant follows feasible targets but the policy prefers flight for landing reward | Objective exploitation becomes the leading measured mechanism | Recheck opposite-foot-supported landing credit against TB/WR2 traces, then consider the previously deferred supported-swing reward experiment |
| Selection seed looks good but held-out draws fail | Candidate generalization is insufficient | Retain and compare nearby checkpoints; confirm multiple evaluation and training seeds before claiming parity |

The 0.28.5 treatment changes physical pushes **and** the associated critic
features. It does not isolate friction, reset history, reference-table changes,
action indexing or actuator identification. The no-push/push evaluations
isolate exposure at evaluation for each frozen policy; they cannot by
themselves identify the effect of training under that exposure.

The current evidence supports completing this run before changing several
parts of the recipe. In particular, there is no demonstrated need to swap
Brax forks, increase imitation weight, expand the action envelope or reduce
LR again to obtain a valid comparison.

### Configuration fields whose labels can mislead

Read the selected Paper implementation when tuning: `init_noise_std` and
`noise_std_type` do not initialize the NormalTanh scale head; fixed
`desired_kl` does not enforce a bound; Paper reward exponents use the hardcoded
100/25 constants rather than the legacy velocity-sigma fields. Legacy gyro
amplitude/colored-noise knobs are not used in the Paper observation method.
The Paper command grid and support ratio are implemented in its reference
generator, rather than controlled by the legacy `command_samples` and
`phase_samples` lookup knobs. Torque exposure is a diagnostic, not a Paper
reward term. Changing one of these labels can leave the intended learning
behavior unaffected.

## 8. Verification artifacts and references

New read-only source/runtime probes are under the completed control's ignored
[`review_20261018/setup_audit`](../../results/wr2_walking/wr2_paper_20261017_195826_seed3/review_20261018/setup_audit):

- `captured_source_hashes.json`: 11 captured TB locomotion files match;
  12 provenance model/reference hashes were also rechecked successfully.
- `paper_branch_source_hashes.json`: direct `paper-walking-baseline` branch
  checks for 20 source/model files, all 11 captured sources, exact frozen HEAD,
  the Brax gitlink and the documentation-only changes since training source.
- `timing_contact_probe.json`: physical clocks, batch/update counts and
  common MuJoCo contact probes on both actual models.
- `compiled_contact_probe.json`: enabled collider sets and WR2's actual
  sliding-friction endpoints, with its walking pose lowered 2 mm to create
  controlled 1 mm penetration for the static contact-parameter test.
- `tb_native_probe.json`: MuJoCo 3.2.4 friction probe, actual wrapped TB
  timeout transition and original cache dimensions.
- `tb_action_index_probe.json`: actual frozen reward methods, model indices,
  all 12 single-action pulses and MuJoCo actuator control limits.
- `tb_lookup_boundary_probe.json`: hashed original cache wrap and joint jumps.
- `nearest_command_probe.json`: both JAX float32 nearest-command selections;
  the WR2 grid was checked against the saved runtime grid with exact equality.
- `reward_timing_arithmetic.json`: all 25 loaded reward weights, actor noise
  feature scales, event reward accounting and effective TB cosine LR.
- `TB_progress_reference.json`: reference milestone metrics reconstructed
  from individual episodes in `xy_evaluation.jsonl`.

No GPU training or TB source changes were made for this review. Prior full
suite, substep load, hardware-fit and finite-difference CoM evidence is linked
from the [actuator review](wr2_actuator_gap_review_20261018.md).

Public references:

- [ToddlerBot: Open-Source ML-Compatible Humanoid Platform for Loco-Manipulation,
  paper v2, Tables VI/VII](https://arxiv.org/html/2502.00893v2).
- [Tedrake et al., ZMP-based planning with terminal conditions for legged
  locomotion, Humanoids 2015](https://doi.org/10.1109/HUMANOIDS.2015.7363473),
  the preview formulation used by both local implementations.
- [MuJoCo contact parameter rules](https://mujoco.readthedocs.io/en/stable/modeling.html#contact-parameters).
  The friction conclusion above was established from actual models and local
  MJX 3.2.4 `collision_driver.py`, not inferred from a configured range.
- [PPO](https://arxiv.org/abs/1707.06347) and
  [GAE](https://arxiv.org/abs/1506.02438), for the distinction between clipped
  objectives, discount clocks and an actual KL constraint.

Local numerical claims use captured code, configurations and measured artifacts.
The public reference pointers are not a claim that the authors' unpublished
runtime had the bugs observed in this reconstructed TB control.
