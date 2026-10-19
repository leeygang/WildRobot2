# WR2 no-push Paper acquisition result — 2026-10-18

## Conclusion

The **0.28.5 fresh seed-3 run completed 300,134,400 transitions**. Removing
physical pushes produced reliable height survival, but the policy learned
very little translation. The selected **260M** checkpoint has **0/1920 height
failures** on both evaluation seeds and **0.07684 TB-equivalent m/s** tracking
error. That error is **2.08× TB Paper 200M** and is worse than the untrained
stationary policy's **0.07162**. No checkpoint qualifies as useful walking.

The native forward return increased from **280 to 459**, driven by added
air-time credit. Posture/yaw rewards recovered from an early unstable phase
while translation nearly vanished. Return does not establish progress toward
the commanded speed. A paired fixed-friction replay improves tracking
only **0.58%** and preserves the poor directional response.

This result revises the [setup audit's decision tree](wr2_tb_paper_setup_audit_20261018.md).
The next step focuses on **narrow stance, inward hip-roll clipping and persistent
foot-to-foot contact**, using paired learned/reference controls through the
native actuator pipeline. See the
[prepared stance diagnostic and decision criteria](wr2_stance_diagnostic_20261018.md).
The earlier weight-0 air-time ablation is deferred; the canonical configuration
retains the completed TB-aligned weight-500 baseline.
Required sole colliders remain active. TB's effective fixed sliding friction
**1.0** remains a secondary candidate with weak current evidence for improving
propulsion; it is not established as the cause of the large tracking gap.
WR2's wider feet receive maximum spacing reward while colliding with each
other, but the proposed global 70 mm bound also penalizes valid rightward
reference poses. That geometry finding is an investigation target, rather
than a validated training fix. The frozen friction replay does not test how
a fresh policy learns with fixed friction throughout acquisition.

## 1. Why missing body collisions can affect training

Collision geometry determines where contact forces and physical constraints
exist. Without a knee collider, that knee can pass through the floor without
receiving a ground reaction. Missing self-collision geometry can similarly
allow motions that the real robot cannot execute. If a policy visits those
states, both its dynamics and rewarded behavior change.

Additional colliders also cost simulation time. MJX groups eligible geometry
pairs and allocates contact capacity from their possible contact points;
active contacts add constraint work. Extra geometry can therefore reduce
transitions per second even when it rarely contacts anything. The actual cost
depends on shapes, masks, eligible pairs and constraint capacity. There is no
WR2 timing measurement supporting a particular slowdown percentage.

The compiled WR2 model enables the **floor and two foot boxes**. Frozen TB
enables **the floor and 16 body colliders**, with **18 explicit pairs**:
16 body-floor pairs and two left/right ankle/foot pairs. Both models already
permit foot-to-foot collision. Their collision masks do not reveal a WR2-only
foot-pair filtering mistake.

WR2's **`left_foot_collision` and `right_foot_collision` are required walking
colliders**, derived from the sole mesh. Both have compiled contact masks
**1/1**, as does the floor. The Paper environment sums floor/foot vertical
forces before applying the **1 N** support threshold. Frozen TB uses
`foot_name=ank_roll_link` to identify its main support-foot boxes. The deferred
body additions concern other links; the sole contacts already provide the
ground reactions and support measurements for walking.

This physical coverage difference is real, but there is **no evidence that
missing torso/arm/knee colliders caused the current tracking failure**.
More geometry does not directly reward propulsion. Defer its addition to the
acquisition experiment. If contact coverage later needs correction, use WR2
geometry and relevant collision exclusions, and measure simulation cost with
the same training batch before adopting the model.

Sources: the two compiled scenes, frozen TB `config_collision.json`, and
MuJoCo/MJX `collision_driver.py`, especially `geom_pairs` and `make_condim`.
See [MuJoCo collision detection](https://mujoco.readthedocs.io/en/stable/computation/index.html#collision-detection)
and [MJX](https://mujoco.readthedocs.io/en/stable/mjx.html).

## 2. Run identity and metric audit

Run: `results/wr2_walking/wr2_paper_20261018_121025_seed3`.
Clean training source: **5f58c138503f16c28731eaf53916cff76867282d**.
Fresh initialization, seed **3**, fixed LR **1e-4**, 300M requested transitions;
native batching finishes at **300,134,400**. All 15 trained checkpoints load
and contain finite arrays. `best_params` matches 260,116,480 and `params`
matches 300,134,400. Initial actor/critic/normalizer arrays exactly match the
preceding fresh seed-3 control.

Relative to **0.28.4**, saved training configuration changes only the two push
amplitudes and version/name. Model, actuator controller, rewards, reference,
PPO implementation and physical normalization are unchanged. The 82-tick
push schedule retains its random-key splits. Zero amplitudes also zero the
six critic push labels; that observation change accompanies the exposure
change and cannot be isolated by this experiment.

The comparison uses **TB `paper-walking-baseline`**, commit
**a4f693d7cc4544f011dad23d6c353e752a6819ab**. Its training sources match the
captured files from `toddlerbot_walk_ppo_20261015_194347`, source **1982ef3**,
with Brax fork **1414780**. The active TB checkout was read without changing
its branch or files.

Independent reconstruction checked **124,800 episode-condition records**
across new primary/pushed histories, the preceding WR2 control, TB Paper and
both held-out confirmations. It verified schedules, finite terminal samples,
eligible episode means/standard deviations, failure counts, support
partitions, histories and selection. All checks pass. Maximum aggregate
rounding difference is **2.93e-6**; maximum support partition error is
**8.06e-8**. All 16 native callbacks and 149 rollout records are finite;
the 25 reward components reconstruct native return within **0.000978**.
Another **3,840 records** independently validate the paired friction reports.

Tracking is mean instantaneous XY velocity error over the ten fixed nonzero
commands, averaged within episodes and then across conditions. WR2 velocities
are divided by **s = 0.8240547565**. Each WR2 benchmark episode has **824 ×20 ms
=16.48 physical seconds**, approximately **20 TB-equivalent seconds**;
TB has 1000 ×20 ms. Height failures use each robot's existing torso-height
criterion. Zero such failures does not prove satisfactory gait, contact
coverage or robustness.

## 3. Learning progress against TB Paper

All errors below are **TB-equivalent m/s**; each failure count is out of 1920
episode-condition samples on selection seed **12102**. New primary reports
have no pushes. TB's configured pushes were physically absent in its frozen
implementation. The preceding WR2 native reports contain realized pushes and
serve as exposure controls, rather than equivalent acquisition scores.

| Transitions | New WR2 no-push error | Height failures | TB Paper error | Height failures |
| --- | ---: | ---: | ---: | ---: |
| Initial | .07162 | 0 | — | — |
| 20M | .14596 | 71 | .06202 | 9 |
| 40M | .08480 | 0 | .04771 | 20 |
| 80M | .07834 | 0 | .03965 | 17 |
| 180M | .07791 | 0 | .03838 | 3 |
| 200M | .07859 | 0 | **.03702** | 8 |
| 260M | **.07684** | **0** | .03713 | 96 |
| 300M | .07871 | 0 | .03800 | 134 |

Every new trained checkpoint is worse than the initial policy on tracking.
The paired 260M-minus-initial error difference is **+.00522**, with a 95%
cluster-bootstrap interval **[+.00480, +.00564]**. Final error is **2.44%**
worse than selected 260M. The plateau is already apparent around 80M; another
continuation of the same objective is not supported as the next step.
Intervals resample 128 shared model/reset indices, preserving all conditions;
they quantify evaluation variation for this one training seed.

### Directional response

Velocities below are measured along the commanded axis, in TB-equivalent m/s.
The aggregate score alone hides how little WR2 actually translates.

| Command | Requested | WR2 260M | TB Paper 200M |
| --- | ---: | ---: | ---: |
| Forward | +.100 | +.00573 | +.09053 |
| Backward | −.100 | −.00324 | −.07357 |
| Forward slow | +.050 | +.00414 | +.04401 |
| Backward slow | −.050 | −.00074 | −.04111 |
| Left | +.050 | +.00131 | +.03499 |
| Right | −.050 | −.00019 | −.03204 |

Standing mean instantaneous XY speed is **.03050** for WR2 versus **.03196**
for TB. Similar standing speed does not compensate for missing commanded
translation. Start/stop groups remain available in the audited full reports;
the existing 15-condition benchmark does not test pure yaw commands.

### Selection and held-out robustness

The selector excludes initial parameters and chooses the best **trained**
checkpoint with a finite suite and ≤1% height failures. Thus `best_params`
is correctly selected under the current rule, but its name does not imply
walking qualification. It loses to the stationary baseline.

Here, **qualification means selecting a checkpoint to carry into a later
training stage**. Minimum evidence of learned walking is a complete finite
suite, ≤1% height failures, tracking better than the initial stationary
baseline on selection and held-out draws, and commanded motion in all six
cardinal cases. Standing/stopping are assessed after demonstrated translation;
a stationary policy can have low stopping speed without having acquired
walking. These checks affect selection and interpretation. The full 300M run
and intermediate checkpoint retention remain necessary.

The **TB parity target** is stronger: approximately **.037–.038 TB-equivalent
m/s** tracking error with ≤1% height failures, TB-comparable directional
response and satisfactory contact/target traces. The 260M WR2 candidate does
not pass even the stationary-baseline comparison. Its `best_params` label
establishes the current height/selection gate, rather than walking acquisition.

At 260M on held-out seed **22102**:

| Exposure | XY error | Height failures | Mean physical episode duration |
| --- | ---: | ---: | ---: |
| No pushes | .07703 | 0/1920 | 16.48 s |
| Realized TB-scaled pushes | .17710 | 918/1920 =47.81% | 11.83 s |

Both reports hash the same selected export. No matching held-out TB evaluation
was performed here. Under pushes on selection seed 12102, new 260M fails
**693/1920** episodes versus **84/1920** for the preceding push-trained WR2
260M. The new policy sacrifices robustness while acquiring survival without
disturbances. Pushed tracking error is censored by early termination and must
be interpreted with episode duration and failures.

## 4. Return increased without useful walking

These are weighted native deterministic forward-evaluation returns over
**20 physical seconds**, with native variation/noise and a **+.10 TB-equivalent
m/s** command. This measures one forward condition over a longer horizon than
the ten-condition, normalized 824-tick selection score.

| Checkpoint | Total return | Air-time contribution | XY tracking contribution | Foot-collision contribution |
| --- | ---: | ---: | ---: | ---: |
| Initial | 280.16 | 0 | 36.67 | 0 |
| 20M | 336.03 | 205.77 | 28.94 | −.93 |
| 40M | 399.46 | 144.96 | 36.90 | −3.77 |
| 80M | 440.87 | 184.09 | 37.10 | −3.64 |
| 260M | 458.92 | 190.30 | 37.23 | −3.76 |
| 300M | 453.47 | 190.67 | 38.33 | −3.70 |

At 260M, air time supplies **41.5%** of native return, while XY reward is only
**1.5%** above its initial value. From 20M to 40M, air-time credit decreases
while yaw, tilt and vertical-motion rewards recover and forward speed nearly
vanishes. The evidence therefore concerns the combined reward tradeoff;
it does not isolate air time as the sole cause of failed acquisition.

WR2 and TB both increment each foot's air counter while ground-support force
is below the one-newton threshold, then credit its accumulated time on filtered
landing. The term requires an active command, but requires neither translation
nor the planned support schedule. WR2's selected fixed-command evaluations
have **94.93% single support**, **5.03% double support** and **0.040% neither-foot
support**, compared with about **two-thirds planned single support**.
These are force classifications, not geometric swing/flight measurements.

The previously proposed opposite-foot-supported counter primarily removes
credit during neither-foot support. That condition is now rare, so the guard
would preserve most of the current in-place behavior's credit. Do not implement
it as the presumed solution to this completed run.

### Fixed-friction execution probe

The exact selected native parameters were evaluated with paired model/reset
keys and all other native variation/noise retained:

| Ground friction | XY error | Height failures | Fixed-command single support |
| --- | ---: | ---: | ---: |
| Native .7–1.3 | .076843 | 0 | 94.92% |
| Fixed 1.0 | .076400 | 0 | 94.92% |

The change is **−.000443**, 95% paired interval **[−.000666, −.000223]**,
only **0.58%** improvement. Forward speed remains about **.0056** against
requested **.1**; its mean decreases from **.005714 to .005633**.
Ground-friction variation does not explain the current
policy's poor execution. This intervention does not establish how a fresh
policy would learn with fixed friction throughout training.

The randomizer changes sliding friction on every collision geometry, including
the feet. A fixed-1.0 treatment therefore changes foot-to-floor and foot-to-foot
friction together. Treat it as a contact-friction control. The recorded nominal
traces retain maximum force magnitudes, not tangential/normal force components,
and cannot establish whether a friction limit is binding. A different learning
trajectory is possible, but no measured early-training slip or force-limit
mechanism currently supports predicting a large gain. A run that changes only
friction is easier to interpret; that alone does not establish it as the most useful next
300M experiment. A null result in one training seed also cannot exclude every
learning effect of friction variation.

The CPU native baseline differs from the saved GPU aggregate by **+.00000492**;
both have zero failures. Brax **0.14.1**, JAX **0.6.2** and MJX **3.14.0** match
the training physics path. The read-only CPU projection uses Flax **0.11.2**,
NumPy **2.4.6**, Orbax **0.12.6** and explicit high matmul precision; the GPU
recorded **0.10.7 /2.2.6 /0.11.39**, and did not record matmul precision.
This is a close measured replay, rather than a claim of identical numerics.

Saved native timing records **6,679 s (1.86 h)** of learner wall time and
**9,740 s (2.71 h)** through the final callback; the subsequent held-out
confirmations are outside that callback timestamp. Additional paired
evaluations contribute to total elapsed time. These timings do not measure
the incremental cost of body colliders.

### Nominal joint and contact diagnosis

The static saved grounded home has **22.36 mm** signed foot-to-foot clearance.
There is no overlapping-foot initialization in that pose. The following
nominal diagnostics use the same 260M policy with model/local variation and
observation noise disabled; results describe one reset per command.

| Command | Commanded-axis speed, TB-equivalent m/s | Single support | Foot-to-foot contact | Geometric flight |
| --- | ---: | ---: | ---: | ---: |
| Forward +.10 | +.00549 | 99.03% | **95.75%** | .12% |
| Backward −.10 | −.00216 | 99.39% | 91.99% | .12% |
| Left +.05 | +.00102 | 96.36% | 89.32% | .12% |
| Right −.05 | **+.00120** | 94.42% | 85.07% | .12% |
| Standing | — | 95.39% | 89.20% | .12% |

Each trajectory has **824 finite samples**, no height failure and just one
flight tick at index **2**, during startup. There is no recurrent nominal
flight. Flight here requires both oriented foot boxes above the plane by
10 μm, neither foot classified as supported, and no active floor-pair force
above 10 μN. Forward internal foot-contact force reaches **7.73 N**, with
maximum penetration **0.70 mm**. Foot-to-foot contact is already modeled;
adding torso or arm colliders would not remove this measured contact mode.

The forward unloaded foot's box bottom is a median **1.82 mm** above the plane;
only **2.08%** of unloaded-foot samples geometrically reach the floor. Thus the
dominant behavior is low-clearance foot unloading with feet touching each
other, rather than an entirely force-threshold artifact on two grounded feet.
That trajectory receives **161.4** air-time return, **32.05** XY return and
only **−3.16** collision return. Air-time return is zero in standing and both
±.05 lateral traces, but the same foot-contact strategy persists. Its presence
across commands does not establish that air-time reward alone caused it.

Requested targets already diverge from the walking reference. In the forward
trace, 12-leg-motor RMSE is:

| Comparison | RMSE |
| --- | ---: |
| Current request versus reference | .1682 rad |
| Delivered target versus reference | .1589 rad |
| Actual motor position versus reference | .1474 rad |
| Actual motor position versus delivered target | .0464 rad |
| Delayed request versus delivered target | .0200 rad |

The **right hip-roll target clips on 99.88%** of samples, near its **−10°**
inward limit; left hip roll clips on **29.25%**, near its **+10°** inward limit.
Their actual-versus-delivered RMSE is **.0755/.1123 rad**, respectively.
Other forward joints are **.0117–.0457 rad**. This identifies a persistent
request/stance problem alongside substantial roll tracking error. It does not
prove that the plant follows every required walking target. The physical joint
limits should remain; increasing them to accommodate a clipped policy would
require hardware evidence. Frozen TB hip-roll limits are ±45°.

### What makes the observed behavior rewarding

The frozen TB and WR2 implementations pay accumulated foot air time at
recontact when a command is active. The term requires no actual translation
or minimum clearance. Moving leg-reference imitation and contact matching
are soft objectives. The policy therefore receives substantial air-time
credit with an unloaded foot only millimeters above the floor.

An additional audit of the same **824-tick nominal forward trace** identifies
a specific contact-timing pattern:

- All **549/549 planned single-support ticks** have exactly the requested
  supporting foot.
- Of **275 planned double-support ticks**, the policy uses double support on
  **7**, single support on **267**, and neither-foot support on **1** at startup.

The next table applies the actual reward formulas to the recorded contact
sequence and, separately, the planned reference contact sequence. This is
reward arithmetic over **16.48 physical seconds**, not a simulated alternate
gait. Air counters start at zero and unpaid air time at the horizon is excluded,
as in the environment. Other reward terms and dynamics are not predicted.

| Weighted episode component | Recorded contacts | Reference contact schedule |
| --- | ---: | ---: |
| Air-time credit | **161.40** | **108.00** |
| Contact-reference matching | **27.58** | **32.96** |

Following the reference contact schedule would lose **53.40** air-time credit
and gain **5.38** contact credit. The actual XY component is **32.05**; its
perfect instantaneous-tracking upper bound is **82.40**, leaving at most
**50.35** additional XY reward. These quantities establish a substantial
competing incentive between unloaded-foot duration and the planned support
schedule. They do not establish that the stationary strategy has greater total
return than every feasible translating gait. A good walker could retain long
single support while translating, and different landing timing affects PPO's
discounted advantages.

The measured reference-imitation loss is only **−1.79** and the foot-collision
loss **−3.16**. The existing spacing term awards its maximum despite the
observed foot contact, as detailed below. These scores explain why the found
behavior earns considerable return. The causal explanation for why training
converges to it remains unisolated: TB has the same air-time/contact formulas
and weights and succeeds. The next diagnostic tests the stance contribution;
WR2's geometry, actuator response and other declared differences remain
possible contributors requiring matched measurements.
The zero-air-time standing/lateral traces also retain foot contact, so air-time
credit alone does not explain every command's behavior.

The shared forward benchmark further reports mean episode leg-reference RMSE
of **.2340 rad for TB Paper 200M**, versus **.1487 rad for WR2 260M**, while
their commanded-axis speeds are **.0905/.0057 TB-equivalent m/s**. Both benchmark
implementations take the square root of each episode's mean leg MSE before
averaging episodes. These are different robot/reference trajectories and are
not actuator-delivery errors. The larger TB reference error alongside successful
translation shows why aggregate reference-error magnitude alone cannot identify
the propulsion gap or justify stronger imitation.

### A concrete geometry-dependent reward difference

The primitive WR2 foot boxes are derived from its CAD foot mesh, and their
centers/sizes are checked in [post-processing](../../wr2/tools/post_process.py).
This is a broader physical footprint than the TB model's ankle-roll boxes:

| Static model quantity | WR2 saved grounded home | Frozen TB home |
| --- | ---: | ---: |
| Foot-box world lateral width | **59.1–59.4 mm** | **42.0 mm** |
| Foot-body lateral spacing | 81.14 mm | 74.00 mm |
| Signed foot-box separation | 22.36 mm | 32.00 mm |
| Lower body-spacing bound for full reward | **55.73 mm** | **50.00 mm** |

The existing WR2 normalization uses a stance-width ratio **1.11456**, based
on TB's configured reference width **2 ×foot_to_com_y =72.8 mm**. The actual
TB model home is **74.0 mm** wide. WR2's lower bound for full spacing reward
is narrower than its actual foot box, whereas
TB's 50 mm minimum leaves about **8 mm** between parallel 42 mm feet. Stance
scaling therefore does not preserve TB's positive geometric clearance for
WR2's wider footprint. Foot orientation and offsets also affect actual contact;
a center-distance range is not a universal collision-free constraint in either
robot. A geometry-only lateral translation to the existing reward minimum
produces **−3.06 mm** WR2 main-foot separation, versus **+8.00 mm** for TB's
ankle-roll foot boxes. TB's additional ankle-pitch pair has **−1.04 mm**
separation in that translated pose, reinforcing that its minimum is a soft
spacing objective rather than a guarantee against every self-collision.

In the nominal forward trace, median world-Y body spacing is **60.40 mm**,
and the environment awards the **maximum feet-distance return, 16.48**, despite
foot-to-foot contact on **789/824 ticks**. This is a measured reward/geometry
mismatch that warrants investigation. It does not by itself establish that
a different spacing bound will recover locomotion.
No corresponding TB policy foot-contact trace was measured here; the TB
comparison in this subsection concerns actual geometry and reward code.

### Follow-up check of the proposed 70 mm bound

A kinematic probe reconstructed all **4,120 recorded nominal reference
samples** using the saved home and the exact 12-leg-joint order. It measures
relative foot geometry, rather than a dynamic rollout. Root yaw is zero for
these conditions; this is not the full training command-grid validation.

| Reference command | Minimum body-Y spacing | Minimum foot separation during planned double support | Planned double-support samples below 70 mm |
| --- | ---: | ---: | ---: |
| Forward +.10 | 81.14 mm | 22.31 mm | 0% |
| Backward −.10 | 81.14 mm | 22.31 mm | 0% |
| Left +.05 | 71.29 mm | 12.50 mm | 0% |
| Right −.05 | **66.37 mm** | **7.59 mm** | **50.55%** |
| Standing | 81.14 mm | 22.36 mm | 0% |

None of these reference samples has foot-box overlap. A global **70 mm**
bound, derived only from width scaling at parallel home geometry, would
penalize about half the rightward reference's double-support samples despite
their positive clearance. It is therefore **not a validated next change**.
Compare actual WR2 and frozen TB policy/reference contact geometry before
choosing a spacing or clearance intervention. A soft lateral center-distance
reward does not directly encode signed surface clearance.

## 5. Updated next plan

The first investigation now targets the **stance/geometry interaction**. The
reward timing calculation is evidence of competing incentives, but TB's
successful weight-500 training prevents identifying that coefficient as the
cause. The [air-time ablation](wr2_airtime_ablation_20261018.md) is deferred.

1. **Audit reference clearance and reachability.** Check signed sole-to-sole
   separation, joint limits and the home-centered action range for every
   nominal training lookup pose. A proposed global 70 mm body-spacing bound
   already conflicts with valid rightward reference poses.

2. **Test the observed hip-roll stance through the existing controller.** Replay
   the selected 260M policy unchanged, then replace only its hip-roll targets
   with the walking reference. Add full leg-reference execution as a teacher/
   plant diagnostic. Use paired resets, fixed directional commands, nominal
   physics and the real action delay, quantization, limits and slew. Measure
   signed clearance, self-contact force, floor support, clipping, joint tracking
   and actual translation. Prescribed reference controls can fall without
   proving the plant cannot walk.

3. **Obtain the same measurements from TB Paper.** Use the local
   `paper-walking-baseline` source and its 200M policy, with the retained 180M
   survival checkpoint. Compare normalized velocities and durations, support
   timing, signed clearance and requested/delivered/actual hip-roll positions.
   Different robots' aggregate leg-reference errors alone cannot identify
   servo tracking quality.

4. **Select a training correction from the measured mechanism.** A reference
   collision or action-reach failure requires reference/action validation
   before training. The new diagnostic confirms both also occur in TB's
   cached teacher, so neither property automatically calls for changing the
   shared action/reference design. Reduced self-contact plus restored
   command tracking under the hip-roll control would support a stance-specific
   acquisition change. If
   contact improves without translation, stance clearance is insufficient;
   investigate reference execution and delivered-to-actual joint response.
   Use a fresh 200–300M TB-aligned run for a justified correction, retain all
   intermediate checkpoints and require improvement over the initial policy
   and the completed WR2 control on both evaluation seeds.

See the [standard diagnostic command, controls and metric definitions](wr2_stance_diagnostic_20261018.md).
The completed CPU control eliminates self-contact and improves net propulsion,
but increases drift/oscillation and worsens six-direction tracking. GPU
confirmation and matching TB policy traces are the prepared next measurement;
hard reference hip-roll commands have not qualified as a training correction.
Native epoch-mean KL rises to **.213** at the final callback;
the rollout log's recent-100-minibatch windows stay below **.031** and can miss
larger updates. This late instability supports retaining intermediates, but
the much earlier propulsion plateau does not justify an LR-only continuation.

## 6. Evidence and references

All raw results, replay reports, arrays and plots remain under ignored local
`results/`; they are not committed binary assets. In the new run's
`review_20261018/` directory:

- `result_audit.json` and `audit_result.py`: reconstructed new/previous WR2 and
  TB progress, all condition metrics, confirmations and paired intervals.
- `checkpoint_audit.json`: 15 native loads, finite arrays, selected/final export
  equality and exact initial-array equality against the preceding control.
- `native_random_friction.json`, `fixed_friction_1.json`, `friction_audit.json`
  and `audit_friction.py`: paired friction intervention and independent audit.
- `home_collision_probe.json`: static saved-home foot separation.
- `foot_geometry_comparison.json`: actual WR2/frozen TB home footprints,
  center spacing and distance-reward bounds.
- `reference_spacing_probe.json`: compiled required-contact masks and 4,120
  nominal-reference geometry samples; the 70 mm bound conflicts with valid
  rightward reference poses.
- `nominal_gait.json`, `nominal_trace_*.npz`, `nominal_audit.json`,
  `diagnose_result.py` and `audit_nominal.py`: five nominal trajectories,
  request/delivery/reference comparisons, contact geometry and reward checks.
- `reward_timing_audit.json`: source/trace hashes, reconstructed air-time,
  contact and imitation scores, actual contacts grouped by reference phase,
  alternate contact-schedule arithmetic and the XY reward upper bound.
- [Local progress plot](../../results/wr2_walking/wr2_paper_20261018_121025_seed3/review_20261018/training_comparison.png)
  and its PDF companion: checkpoint tracking, height failures, commanded speeds
  and support partitions.

Implementation references:

- WR2 [environment](../../wr2/locomotion/walking_env.py),
  [Paper rewards](../../wr2/locomotion/paper_rewards.py),
  [randomization](../../wr2/locomotion/domain_randomization.py),
  [evaluation](../../wr2/locomotion/paper_evaluation.py),
  [PPO](../../wr2/locomotion/ppo.py),
  [reference](../../wr2/reference/paper_walk.py) and
  [MJX robot](../../wr2/descriptions/wr2/wr2_mjx.xml).
- Captured TB [base environment](../../../toddlerbot/results/toddlerbot_walk_ppo_20261015_194347/locomotion/mjx_env.py),
  [walking environment](../../../toddlerbot/results/toddlerbot_walk_ppo_20261015_194347/locomotion/walk_env.py)
  and [Paper configuration](../../../toddlerbot/results/toddlerbot_walk_ppo_20261015_194347/locomotion/paper.gin),
  verified against **`paper-walking-baseline`**.
- [ToddlerBot paper, Tables VI/VII](https://arxiv.org/html/2502.00893v2),
  [preview-control reference](https://doi.org/10.1109/HUMANOIDS.2015.7363473),
  [complete setup audit](wr2_tb_paper_setup_audit_20261018.md) and
  [no-push preparation](wr2_no_push_acquisition_20261018.md).
