# Training configuration size audit — 2026-10-29

## Conclusion

WR2 already scales commanded velocities, swing height, turning radius,
velocity observations and their active white noise correctly. Actual geometry
also justifies its torso-height and foot-spacing thresholds. **The main
remaining mismatch is learning time:** episode, rollout, observation history
and discount/GAE horizons retain TB's tick counts despite a coarser control
tick in TB-equivalent time.

For a recipe preserving TB's normalized learning durations at WR2's current
20ms control tick, the practical candidate is:

| Field | Current | TB-time equivalent candidate |
|---|---:|---:|
| Training `episode_length` | 1000 | **824** |
| `unroll_length` | 20 | **16** |
| `discounting` | .97 | **.9637121762** |
| `gae_lambda` | .95 | **.9396526368** |

These are equivalence calculations, not proof of improved walking. They are
four coupled recipe changes, not a single reward ablation. They should not be
silently inserted into the proposed fresh-initialization/roll comparison:
doing so would also change its comparison with the completed continuation.

Contact detection and effort signals have additional dimensional differences,
discussed below. Joint angles, policy action amplitude, PPO clipping and
probabilities do not need a uniform robot-size multiplier.

## Source and measured normalization

Inspected current `ppo_walking_leg_imitation15.yaml`, `walking_env.py`,
`paper_rewards.py`, `paper_walk.py`, `domain_randomization.py`, robot history/
timing contract and the selected Brax network path. TB source is local
**`paper-walking-baseline`** at `a4f693d7cc4544f011dad23d6c353e752a6819ab`,
including `paper.gin`, `mjx_env.py`, `walk_env.py` and captured run configs.
The local frozen model XML/scene bytes were checked against that branch.

Native model forwarding gives:

| Quantity | WR2 grounded reference standing | TB model home |
|---|---:|---:|
| Total body mass | 2.05832762kg | 3.358176486kg |
| Whole-body CoM height | .202233182m | .297810684m |
| Torso/free-root height | .279260761m | .3267m |

The existing normalization is still reproduced exactly:

- Dynamic length ratio `L = h_WR/h_TB = .6790662418`.
- Gravitational time ratio `s = sqrt(L) = .8240547565`.
- Actual mass ratio `M = m_WR/m_TB = .6129301508`.

Mass is not `L³`: these robots have different proportions and mass distribution.
Use measured mass when normalizing load. The fixed nominal CoM ratio provides
stable command/metric units; do not recompute it from instantaneous gait height.

Dimensions and source hashes are recorded in
`results/wr2_walking/wr2_paper_leg_imitation15_20261029_173955_seed3/review_size_20261029/dimensions.json`.

## Dimensional rules

For gravitational similarity at the same gravity, a TB quantity maps into
WR2 units with the following multiplier:

| Quantity | Multiplier |
|---|---:|
| Length | `L = .67907` |
| Time | `s = .82405` |
| Linear velocity | `s` |
| Angular velocity | `1/s` |
| Angle | **1** |
| Linear acceleration | **1** |
| Angular acceleration | `1/L` |
| Force | `M = .61293` |
| Characteristic torque | `M*L = .41622` |
| Characteristic power | `M*s = .50509` |
| Inertia | `M*L² = .28264` |

These are dynamic-equivalence rules, not instructions to replace WR2's measured
actuator limits, stiffness, friction or inertia with scaled TB hardware values.
Geometry-specific clearances need the actual relevant WR2 dimensions.

Network layer widths/activation, PPO ratio clipping.2, advantage normalization,
entropy coefficient.0005, learning rate1e-4, branch probabilities and relative
randomization factors have no automatic dimensional size conversion. Keep
their TB-aligned values unless a learning experiment supports a change.

## Settings already aligned

| Active setting | TB value | WR2 value / code confirmation |
|---|---:|---|
| Forward command range | −.10…+.20m/s | −.0824055…+.164811m/s; multiplies by `s` |
| Lateral command range | ±.10m/s | ±.0824055m/s |
| Yaw command range | ±.50rad/s | ±.606756rad/s; divides by `s` |
| Linear command deadzone | .05 | .0412027; reward/sampler convert back to TB units |
| Command resample period | 3s | 124 ticks/2.48s =3.0095 TB s |
| Push interval | 2s | 82 ticks/1.64s =1.9902 TB s; training amplitudes zero |
| Push linear/angular amplitudes when enabled | .5m/s /1rad/s | Existing evaluator uses `.5*s` /`1/s` |
| Reference swing height | .05m | .0339533m =`.05*L` |
| Reference turn radius | .10m | `.10*L` in `footstep_plan` |
| Reference support ratio | 2 | 2; dimensionless |
| Actor/critic linear velocity | TB units | Physical velocity divided by `s` |
| Actor/critic angular velocity | TB units | Physical angular velocity multiplied by `s` |
| Joint-velocity observation scale | .05 | `.05*s`, converting physical qdot into TB units |
| Joint velocity white noise | Effective raw 2rad/s | 2.427023rad/s; multiplied by `s` gives2 |
| Gyro white noise | Effective .25rad/s | .303378rad/s; multiplied by `s` gives.25 |
| Joint position / Euler noise | .05rad /.10rad | Same angles; no length/time multiplier |
| Residual action gain | .25rad | .25rad; angle scaling is unnecessary |

`paper_rewards.py` converts linear/angular velocities before its 100/25
tracking exponents. The legacy YAML velocity/yaw sigma fields are not the
active Paper reward widths; changing them would not change these rewards.

White noise here is per-sample augmentation. Matching continuous sensor-noise
spectral density under different sampling rates would be a different model;
it does not justify an additional noise multiplier in this experiment.

## Learning horizons and update counts

Current WR2 policy tick is **20ms physical /24.27ms TB time**, versus TB20ms.
The ideal scaled tick would be16.481ms physical. The actual20ms tick is part
of the WR2 controller/deployment contract and should remain a declared exception.
One-tick action delay is likewise actual20ms latency, not a freely scalable
training integer.

Physics integration also remains coarser in normalized time: WR2 uses5ms×4
substeps, versus TB4ms×5; a WR2 substep is6.07ms in TB units. Altering this
changes simulated dynamics/solver sampling and compute cost, so it is a
separate integration study. Solver iteration counts are numerical settings,
not quantities to multiply by the robot-size factor.

| Duration | TB | Current WR2 in TB units | Closest candidate at20ms WR tick |
|---|---:|---:|---:|
| Training episode | 20s | **24.270s** | 824 ticks ≈19.999s |
| PPO rollout | .400s | **.4854s** | 16 ticks ≈.3883s |
| 15-frame first-to-last history span | .280s | **.3398s** | 13 frames ≈.2912s |
| Discount e-fold time | .65662s | **.79681s** | `.97**(1/s)` restores.65662s |
| GAE e-fold time | .24464s | **.29687s** | Gamma/lambda exponent correction restores.24464s |

Gamma and lambda are dimensionless per-tick parameters but encode time
constants. Preserving normalized discount/trace duration gives
`gamma_WR = gamma_TB**(dt_WR/(s*dt_TB))` and the same exponent for GAE lambda.
This is a deliberate continuous-time interpretation of TB's settings.

History13 is **not currently an ordinary training YAML change**. History15 is
owned by the robot observation contract. With current67/101 frame features,
13 frames would change actor/critic inputs from1005/1515 to871/1313. It needs
a versioned inference/deployment contract and compatible loading. Keep15 until
that work is explicitly part of the task.

Rollout16 preserves1024 trajectories per batch but reduces transitions per
batch20480→16384. At requested200M/16 evaluations, rounded transitions become
200,048,640 and Adam updates195,360, versus current200,294,400/156,480. This is
about25% more updates per transition. It also brings updates per TB-equivalent
simulation second close to TB: .04024 versus TB.03906, rather than current.03219.
Report transitions, normalized elapsed time and optimizer updates separately.
Do not reduce the affordable200M budget merely to make raw counts equal.

Changing episode horizon needs timeout/autoreset and final-sample verification.
The Paper reference retains at least1000 frames, so episode824 does not shrink
the table. Native diagnostic checks using reset `previous_target` must recognize
the timeout reset at the final sample rather than report a false target mismatch.

## Gait timing and landing credit

Exact scaled gait period is `.72*s=.593319s`; current.60s is only1.13% longer.
Keep.60s: native support/swing segments are rounded to policy ticks, and this
gives a consistent30-tick cycle. Simply inserting.593319 would change the
continuous phase clock while rounded support segments still tend to30 ticks.

The user-requested minimum stance is `.20*s=.164811s`. Ten consecutive samples
span nine intervals/.18s physical =**.21843 TB s**. This is the correct upward
quantization for a minimum-duration requirement; nine samples would fall short.
Keep the qualified-landing rule and report its effective duration.

Gamma correction automatically changes the delayed-payment compensation:
1.31539→1.39468 for nine ticks. It preserves the chosen discounted value at
the original landing time, but changes the undiscounted credit. A gamma study
must acknowledge that derived coupling.

Air reward intentionally retains physical swing seconds. At the shared50Hz
tick, both air/clearance cycle returns and continuous-term cycle returns scale
with physical cycle duration. An extra division of rewarded air time by `s`
would change their relative balance. Clearance already divides heights by `L`.

## Geometry-specific thresholds: preserve them

Torso heights use the actual reference-root ratio **.854793**, not the CoM
length ratio. Current termination bounds **.1709585… .3419171m** exactly equal
TB.2… .4 times that ratio. Uniform CoM scaling would incorrectly replace them
with.135813… .271626m; that would alter what counts as a fall.

Current foot-spacing limits **.0557279… .1448926m** exactly equal TB.05… .13
times the measured foot-body spacing ratio **1.114559**. The reward also scales
its slope by that ratio. Feet/stance dimensions are not uniformly smaller;
using the CoM length factor would permit materially narrower spacing.

Joint ranges and home angles follow actual kinematics. Roll/pitch tolerances
are angles and are dimensionless. The proposed width-scaled roll band of about
±3.1° is an **optional visible-displacement objective**, not a mandatory size
correction to TB's angular posture reward. Preserve ±5.73° for the baseline.

This distinction also matters for balance. WR2 foot-body spacing after CoM
length conversion is about1.64× TB's, while shoulder width is about1.84×.
The robots are not uniform scaled copies. Tightening torso roll to reduce
visible shoulder motion can change the learned weight-transfer strategy;
verify progression and support as well as appearance. Geometric safety limits
and a visual-motion objective should not be treated as one universal scale rule.

## Force, effort and physical uncertainty

Support threshold1N is **4.95% of nominal WR2 weight**, versus **3.04% of TB
weight**. Matching that nominal load fraction gives **.612930N**. This is a
valid force-normalization candidate, but both collider aggregation and force
measurement definitions must remain explicit. It changes detection/rewards/
critic inputs, not the physical contact solver.

Reclassifying all18 existing nominal gait traces with.61293N changes at most
**.364% of foot samples**; some traces are unchanged. This is not evidence
that the threshold explains the large asymmetric gait. The calculation is in
`review_size_20261029/contact_sensitivity.json`.

The non-floor collision-force cutoff also remains an absolute.1N. Its load-
fraction counterpart would be.0613N; pair aggregation differs from TB. Treat
it separately from support detection and from the dimensionless collision
reward coefficient.

Torque reward uses physical Nm; power reward normalizes angular velocity but
not torque. Characteristic torque normalization by `M*L=.41622` would change
squared-effort terms by about**5.77×**. This is not a prescription to multiply
weights or motor limits: actual HTD45H/Dynamixel capacity, inertia, friction,
transmissions and load distribution are different. Averages also include21
versus30 motors. Start with normalized effort metrics before choosing a reward
change. Keep measured WR2 gains, rotor inertia, torque/speed caps and thermal
time constants tied to its hardware.

Friction coefficients and relative randomization factors are dimensionless.
TB's additive mass ranges would scale by measured `M`, while WR2's percentage
ranges need no size multiplier. Their distributions still differ: TB perturbs
torso/hands; WR2 perturbs every non-world body. This is not fixed by one factor.

The actor's load-conditioned position bias uses `.02*tanh(torque/.1Nm)`.
Its angle amplitude needs no size scaling; its torque activation is an absolute
hardware/noise-model assumption. It is not normalized by body torque scale.
Evaluate its hardware meaning separately; the legacy `backlash_rad` field is
not this active Paper bias.

## Smoothness and inactive knobs

Action-rate shaping is a difference, not rad/s. Under current one-tick delay
both implementations compare `a_t-a_(t-2)`. At equal normalized trajectory
rates, WR2 differences are larger because its normalized tick is longer.
If interpreted as rate regularization, coefficient.05 would become
`.05*s²=.0339533`. That is a separate objective interpretation, not an
automatic scalar correction to every action penalty.

Current acceleration stencil is `a_t-2*a_(t-2)+a_(t-3)`. Its expansion contains
a first-derivative term; it is not a pure second difference. Therefore blindly
multiplying its weight by `s⁴` is unjustified. TB shares the stencil and has
an additional global-indexing bug; preserve intended local12-action weighting.

The selected Tanh network path also ignores `init_noise_std`/`noise_std_type`
as scale-head initializers. A local Brax.14.2 probe with configured std.1/.694/1
always gives actual zero-input scale.694147. Default fresh behavior matches the
native Tanh recipe; changing those labels alone is not a useful size adjustment.
The separate Normal-policy std branch should not be confused with this path.

Other legacy-only knobs include colored gyro cutoff/amplitude, legacy velocity
sigma and foot-phase sigma. Lookup `phase_samples`/`command_samples` do not
control the native Paper grid; torque exposure is diagnostic; actuator-force
observation scale is absent from the Paper critic. Audit the active code path
before interpreting their units or tuning them.

The six-DOF IK positional tolerance is an absolute1mm, while its angular
tolerance is.001rad. Equal relative length accuracy would suggest.679mm,
but TB uses different IK and WR2 already measures roughly.04mm residual.
Changing that gate is not a meaningful gait intervention. Damping/iteration
settings likewise require numerical conditioning evidence, not a simple size
multiplier.

## Recommended next steps

1. Preserve verified command/noise scaling and actual geometric/hardware limits.
2. Make the learning-time candidate explicit: episode824, rollout16,
   gamma.963712 and lambda.939653, with unchanged history contract.
3. Decide its comparison scope before training. A fresh current-clock baseline
   versus a fresh corrected-time recipe isolates the coupled temporal change.
   Applying it to both fresh/roll arms keeps the roll comparison controlled,
   but comparison against the old continuation then changes initialization
   **and** learning-time recipe. Do not label that an initialization-only result.
4. Keep contact/effort changes separate unless explicitly making an integrated
   recipe comparison. Their unit equivalences do not prove gait improvements.

References: local TB **`paper-walking-baseline`** source/model and captured
config; [TB paper v2, Tables VI–VII](https://arxiv.org/html/2502.00893v2);
[prior full setup audit](wr2_tb_paper_setup_audit_20261018.md). Time rules follow
gravitational dimensional similarity, `time ∝ sqrt(length/gravity)`.
