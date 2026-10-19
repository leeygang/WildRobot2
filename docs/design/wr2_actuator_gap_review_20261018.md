# WR2 / TB actuator and training-gap review — 2026-10-18

This review follows the completed fresh seed-3 300M run and the request to
understand the physical causes before changing the implementation. It replaces
the previous review's ordering of the proposed next experiment. Training code,
actuator parameters, reward functions and the canonical run configuration have
not been changed during this review.

## Findings

1. **A policy's target-tracking RMSE is not an actuator identification test.**
   The newer WR2 policy requests larger motion and much higher angular rates
   than the retained WR2 policy under the identical robot/controller model.
   Controlled waveform tests nevertheless confirm a substantial difference
   between the modeled WR2 and TB knee/ankle responses.
2. **WR2's modeled inertia, damping, friction and torque-speed behavior explain
   slower response.** They are different actuators on different robots. WR2 is
   lighter and has higher stall torque; a shortage of static lifting torque is
   not the explanation supported by these data.
3. **The physical interpretation of the fitted inertia is uncertain.** A fit
   with 20 ms more delay and about 18 times less armature predicts the held-out
   hardware captures almost equally well. Internal firmware response and
   mechanical inertia have not been separated by the current experiment.
4. **The frozen TB Paper code does not realize its configured pushes.** It
   changes Brax `qd`, while native MJX integrates `qvel`. WR2 changes both.
   This affects training difficulty and the interpretation of the native
   robustness comparison. A controlled evaluation is recorded below.
5. **Better tracking of the recorded targets still leaves flight.** Same
   WR2 inputs with half the armature or a constant motoring cap still produce
   extensive flight and worse velocity tracking. The observed failure combines
   policy selection, actuator response and the training/evaluation conditions.
6. **The previous critic normalization diagnosis was incorrect.** The actual
   Paper observation already reads the constructor's `linear_velocity_scale/s`.
   A direct source/runtime probe confirms it; no correction is needed there.

## Sources and measurement rules

WR2 training source: `09b9805`, robot model SHA256
`741d1e613d487b34073a10fbc0365906c3364914f37e70a394c45bf1e1a696c0`.
The retained WR2 +180M and newer 220M controls use the same robot physics.
TB control: Paper run `toddlerbot_walk_ppo_20261015_194347`, checkpoint
`200089600`, source `1982ef3`, local frozen review checkout. The controller,
reference and identification files in that checkout match `1982ef3`.
The active TB development checkout was used for read-only source checks.
Paper provenance reports a dirty tree. Its copied locomotion sources,
model/reference hashes and dependency lock match the frozen revision; those
checks establish the inspected paths rather than a complete historical tree.

Primary WR2 source:

- [Controller and target processing](../../wr2/locomotion/walking_env.py).
- [Canonical servo parameters](../../wr2/descriptions/default.yml).
- [Accepted fitting model](../../wr2/tools/servo_sysid/fit.py) and
  [known-inertia fixture](../../wr2/tools/servo_sysid/fixtures/htd45h_2650g.xml).
- [Packet duration](../../wr2/actuation/htd45h.py),
  [capture clocks](../../wr2/tools/servo_sysid/capture.py),
  [servo evidence](servo_model.md).
- [Paper reference](../../wr2/reference/paper_walk.py),
  [reward functions](../../wr2/locomotion/paper_rewards.py).

Primary TB source at the frozen Paper revision:
`toddlerbot/sim/mujoco_control.py`, `tools/run_sysID.py`,
`reference/walk_zmp_ref.py`, `locomotion/mjx_env.py` and
`descriptions/toddlerbot/config.json`. The captured
[environment source](../../../toddlerbot/results/toddlerbot_walk_ppo_20261015_194347/locomotion/mjx_env.py)
and local
[Brax MJX pipeline](../../../toddlerbot/toddlerbot/brax/brax/mjx/pipeline.py)
make the push issue directly inspectable.

Response RMSE compares the motor angle at the **end of a 20 ms tick** with the
target held during that tick. It excludes the earlier action delay and setpoint
processing. Target spectra discard the first physical second and express time
in TB units. Full WR2 torque/clipping measurements cover all four 5 ms substeps;
earlier diagnostic tables measured only the first substep.

All same-input WR2 controls use native Brax 0.14.1 / MJX 3.14.0. TB's paired
control uses its Brax 0.10.5 fork `14147806` / MJX 3.2.4. The two original WR2
recorded forward trajectories reproduce **exactly**, with zero maximum motor
angle and velocity difference. Scalar actuator tests use MuJoCo and controllers
checked against actual source on 2,867 state/error combinations: maximum torque
difference is 2.09e−7 N m for WR2 and 2.22e−16 N m for TB.
The scalar tests use a common MuJoCo 3.14 engine with each controller's native
substep period; they isolate model mechanisms without claiming equivalence to
the complete older TB MJX solver.

Both canonical models use torque actuators with gain 1, bias 0 and gear 1.
Their explicit controllers produce joint-axis torque; a hidden MuJoCo position
actuator does not multiply the gains again. WR2's .5 active plus .173622 passive
damping matches the fitted .673622 below saturation. The selected PPO/GAE,
NormalTanh and value-loss paths also match TB, as checked in the preceding
source comparison. The present evidence does not identify a PPO adapter bug.

The attempted TB whole-body fixed-target replay did not meet its trajectory
reconstruction gate and is excluded from causal comparisons. TB's earlier
native nominal measurements, source-checked scalar controls and the paired
push evaluation are the TB evidence used here.

## 1. Correct the physical normalization

The actual CAD model masses are **2.0583 kg WR2** and **3.3582 kg TB**. Measured
model CoM heights are .20223 m and .29781 m. Therefore:

| Quantity | WR2 / TB ratio |
| --- | ---: |
| Length, `lambda` | .679066 |
| Gravitational time, `s = sqrt(lambda)` | .824055 |
| Mass, `mu` | .612930 |
| Dynamically similar torque / position gain, `mu*lambda` | .416220 |
| Dynamically similar inertia, `mu*lambda^2` | .282641 |
| Dynamically similar viscous damping, `mu*lambda^(3/2)` | .342988 |

These are comparison units, not proposed replacements for measured servo
parameters. WR2's .60 s gait becomes .7281 s in TB units, close to TB's .72 s.
Its .0824 m/s forward command becomes .10 m/s. **The gait clock is already
scaled; the actuator's response does not acquire that scaling automatically.**

WR2 is also not a uniform geometric copy. Its foot-center separation is about
81.4 mm versus TB's 74.0 mm despite its smaller CoM height. Both collision feet
are approximately 120 mm long; WR2's lateral width is about 58.8 mm versus
42 mm TB. The [matching stance/CAD audit](wr2_tb_stance_result_20261018.md)
corrects this lateral comparison: 49 mm is TB's other local box dimension;
its native-home lateral projection is 42 mm. Leg geometry, body CoM and contact
lever arms must be measured on each model. A scalar height ratio cannot remove
those differences.

### Velocity point matters, but does not remove the remaining gap

Both native rewards/benchmarks measure velocity of the **torso body origin**.
That point is about 77.0 mm above whole-body CoM in WR2's home, versus 28.9 mm
in TB. Torso rotation and internal leg motion can therefore contribute different
amounts of point velocity.

As a secondary check, reconstruct whole-body CoM from the existing nominal
forward poses and use the **same 20 ms pose-difference method** for both points
and robots, discarding the first physical second:

| Single nominal trajectory | Torso-point XY error | Whole-CoM XY error |
| --- | ---: | ---: |
| New WR2 220M | .06594 | .05719 |
| Retained WR2 +180M | .05738 | .05587 |
| TB Paper 200M | .02777 | .02995 |

Errors are TB-equivalent m/s. The newer WR2 CoM error remains **1.91× TB**,
and its actual flight remains. These interval-average diagnostics do not replace
the primary native returned-pipeline velocity metric. They show that motion of
the torso point explains part of the nominal fluctuation gap.

## 2. What physically slows the modeled actuator

For motion below saturation, the useful relationship is approximately

`Kp*(q_target - q) = I_eff*q_acc + D_total*q_vel + friction + external_load`.

Inertia requires torque to accelerate; damping and friction consume effort as
the joint moves; gravity and contact add load. When required torque exceeds
the speed-dependent limit, increasing position error no longer increases
motoring torque. This creates attenuation and phase lag.

| Parameter | WR2 HTD-45H | TB knee / ankle pitch |
| --- | ---: | ---: |
| Position gain, N m/rad | 24.1574 | 21.09375 |
| Active velocity gain, N m s/rad | .5 | 0 |
| Passive damping, N m s/rad | .173622 | .08 |
| Effective damping below saturation | .673622 | .08 |
| Reflected armature, kg m² | .020951 | .0012 |
| Friction loss, N m | .465471 | .134 |
| Peak torque, N m | 4.0 | 1.86 |
| Speed envelope endpoint, rad/s | 5.8178 | 10.0 |
| End of constant-torque plateau, rad/s | 0 | 4.22 |
| Added target slew | 5.655 rad/s | None |

TB selects different actuator models for its hips, yaw, knees and ankles. Its
hip/yaw friction and inertia differ from the knee/ankle values in this table;
not every TB joint responds better at every frequency.

### Inertia and damping

A geared motor exchanges speed for output torque: approximately
`torque_out = efficiency*G*torque_motor`, `speed_out = speed_motor/G`, and
`reflected_inertia = G^2*rotor_inertia`. A high stall-torque rating therefore
does not establish fast position response. Gear/rotor inertia, transmission
loss and internal velocity feedback are physical mechanisms that can produce
the modeled WR2 behavior. The current evidence does not identify WR2's gear
ratio or rotor inertia separately, so it cannot assign the fitted .020951
to one specific component.

With root and other motors held, and passive linkages included through the
equality-constraint tangent, WR2's ankle-pitch effective inertia is
**.021427 kg m²**, of which **97.8% is armature**. Its knee is **89.6% armature**.
TB ankle-pitch effective inertia is **.001478 kg m²**. The constrained model
reduction tolerates the micrometre rounding of TB's exported loop geometry;
it is a local mechanical comparison, not a measured walking bandwidth.

After mass/length normalization, WR2's ankle armature is about **61.8 times**
the value that would dynamically match TB; its effective damping is **24.5
times** the corresponding TB value. That damping figure includes .5 of active
velocity feedback; the passive component alone is about **6.3 times** the
size-normalized TB ankle value. Its higher position gain and torque do
not compensate for that response difference.

Simple unsaturated natural-frequency estimates in TB time units are roughly
**4.4 Hz WR2 ankle pitch versus 19.0 Hz TB**. They omit saturation, friction
and contact. The controlled response tests below include the controller's
nonlinear torque limits.

### Friction and the torque-speed curve

The static friction/gain ratio is about **.0193 rad WR2 versus .00635 rad TB
knee/ankle pitch**, before gravity or contact. That is a measurable source of
deadband and tracking error. It does not account for most of the newer policy's
error: halving friction barely improves its recorded-input tracking.

At 3 rad/s, WR2's provisional motoring cap is only **1.94 N m**. Its braking
cap remains 4 N m. TB knee/ankle pitch retains full torque up to 4.22 rad/s.
WR2's normalized speed endpoint is **4.79 rad/s versus TB's 10**.

There is an additional modeling ambiguity: a gross torque curve ending at the
vendor no-load speed, followed by fitted passive drag, does not retain that
literal output no-load endpoint. With a sufficiently large positive position
error, this model's unloaded steady speed is about

`(4 - .465471) / (4/5.817764 + .173622) = 4.1043 rad/s`.

That is **3.3822 rad/s in TB units**. This is an algebraic consequence of the
current model, not a new hardware measurement. The curve and drag need to be
identified together before changing the endpoint or calling the loss terms
double-counted.

### Command processing and delay

Both Paper environments delay the policy by one 20 ms tick. WR2 additionally
quantizes targets to .24° and limits each tick's target change to 6.48°. The
limit is derived from mechanical no-load speed. The actual HTD packet accepts
a position **and a move duration**; a motor speed limit does not by itself
establish that a separate host setpoint limiter represents its firmware.

The physical tick is the same, but its normalized duration differs: WR2's
20 ms is **24.27 ms in TB time units**, so one-tick delay is 21.35% longer
relative to the scaled body dynamics. Its .60 s gait has **30 policy updates**
versus TB's .72 s / 36 updates. Matching the gait clock therefore does not
match controller sampling or latency. The current 20 ms choice follows the
physical command period; changing it requires actuator timing evidence.

For the newer forward policy, delayed request / delivered target RMSE is
**.07235 rad**, versus **.00424** for the retained WR2 policy and zero in the
TB simulator. Quantization alone has at most .002094 rad error per in-range
target. The large preprocessing difference therefore comes from the active
slew limiter. Its hardware equivalence at large commands has not been tested.

The active/passive damping split is also equivalent only below saturation.
`clip(Kp*error - .5*velocity) - .173622*velocity` differs from
`clip(Kp*error) - .673622*velocity` when clipping occurs. TB's identification
tool fits the controller/envelope and passive dynamics jointly; WR2's current
fit does not identify this decomposition.

### Integration clock and effort units

WR2 uses 5 ms ×4 / MJX 3.14; the Paper control uses 4 ms ×5 / MJX 3.2.4.
The scalar native WR2 response changes modestly with 0.5 ms substeps: the .25 rad
ankle RMSE changes from .04982 to .05419 at gait frequency. Blindly substituting
TB-scaled inertia produces overspeed/braking oscillations at the current 5 ms
clock; the same diagnostic model behaves differently at 0.5 ms. This is a
numerical limitation of that hypothetical parameter substitution, not evidence
of overspeed in the current walking trace. Identify dynamics and resolve them
with a consistent controller/integration clock.

The existing torque penalty uses physical N m, and energy uses
`torque*velocity*s`, matching TB's formula after angular-time scaling. They
are not fully dimensionless effort costs: a dynamically similar torque scales
by `mu*lambda = .41622`, so these squared penalties scale by .17324 before
differences in the motor-count mean. This is a remaining cross-robot objective
choice, not a demonstrated cause of this gait. Retain those native formulas in
the first disturbance control to isolate its effect; any later effort
normalization needs its own physical rationale and training comparison.

## 3. Separate plant response from the policy's requested motion

Forward targets, after the first physical second:

| Measurement | WR2 newer 220M | WR2 retained +180M | TB Paper 200M |
| --- | ---: | ---: | ---: |
| Mean joint-target standard deviation, rad | .1138 | .0412 | .0596 |
| Target angular-rate RMS in TB units, rad/s | 3.431 | 1.814 | .891 |
| Dominant hip-pitch frequency in TB units, Hz | 3.94 | 5.27 | 1.37 |
| Fraction of total target variance above 3 Hz | 98.6% | 98.8% | 9.8% |
| End-of-tick actual / delivered RMSE, full horizon, rad | .10750 | .04855 | .02380 |

The retained WR2 policy also uses rapid motion, but with much smaller
amplitude. The newer policy's large, rapid targets stress the plant more.
TB's target frequency follows its planned gait clock.
After settling, time-varying error accounts for about 97.8% of the newer WR2
tracking MSE; per-joint mean bias accounts for the remaining 2.2%.

A .22 rad sinusoid at the newer WR2 hip's approximately 4.78 physical Hz would
require about **6.4 N m of inertial torque alone** under the local held-motor
inertia, with peak speed about **6.6 rad/s**. Both exceed the corresponding
model limits before damping and external load. This is an illustrative smooth
motion calculation; the actual delivered waveform is clipped by the slew
limiter and is not an exact sine.

### Matched waveform test

Same sinusoid, matched amplitude and frequency in TB time units, scalar native local
inertia/damping/friction, native controller and substep clock, no policy,
gravity, noise, pushes or contact. The targets are supplied after host setpoint
processing, isolating the plant response:

| Amplitude | Equivalent frequency | WR2 ankle RMSE | TB ankle RMSE | WR2 gain | TB gain |
| ---: | ---: | ---: | ---: | ---: | ---: |
| .035 rad | 1.373 Hz | .01861 rad | .00400 rad | .628 | .955 |
| .035 rad | 4.0 Hz | .02434 rad | .00250 rad | .394 | .990 |
| .25 rad | 1.373 Hz, intended WR2 gait | .04982 rad | .00644 rad | 1.020 | 1.002 |
| .25 rad | 4.0 Hz, near newer requested motion | .18648 rad | .02842 rad | .531 | 1.052 |

The small-amplitude WR2 cases do not saturate; friction, damping and inertia
already reduce their response. At 4 Hz / .25 rad, WR2's sampled response lags
its held target by about 81°, while TB
ankle pitch follows it much more closely. This validates a **modeled plant
response difference** independently of either learned policy. It does not
identify the internal hardware component causing that difference.

### Exact recorded-input WR2 interventions

Same newer WR2 target sequence, reset and command; change one physical-model
choice at a time. All runs complete the 824-tick horizon without a height
termination or nonfinite state.

| Model intervention | Joint RMSE, rad | Equivalent XY error, m/s | Exact aerial samples |
| --- | ---: | ---: | ---: |
| Current controller | .10750 | .06732 | 56.92% |
| Constant 4 N m motoring cap | .10396 | .08356 | 59.71% |
| Half armature | .09691 | .08279 | 50.36% |
| Half friction | .10673 | .08133 | 57.04% |
| Same total damping, all passive | .11114 | .09629 | 51.46% |

Thus half armature improves tracking **9.8%**, and removing the taper improves
it **3.3%**; neither recovers good walking. The retained WR2 policy's half-
armature RMSE improves from .04855 to .03908 while its XY error barely changes.
These are fixed-input plant controls. A closed-loop policy or a newly trained
policy can respond differently to changed physics; those effects are not
established by this replay.

The retained policy's original nominal forward trajectory has **zero geometric
aerial samples** under the same WR2 controller/model. That does not establish
full command parity, but it shows this plant does not make flight inevitable.

The newer policy clips **26.89% of all joint/substep torque commands**, mostly
motoring (25.55%). Its nominal forward trajectory never reaches the special
overspeed full-braking branch. That branch cannot explain this trajectory's
failure, although its hardware model remains unmeasured.

Removing slew for the **same recorded delayed requests** gives .13354 rad
request / actual RMSE, versus .14388 with the current cap: a **7.2% improvement
against the common original request**. Aerial samples remain 57.52%; XY error
rises from .06732 to .08884. Removing the slew limit does not remove
the nonlinear plant limits or the learned flight strategy.
The earlier frozen-policy, closed-loop cap-removal experiment also retained
flight and worsened forward tracking. Neither experiment is a retraining test.

Slowing the recorded newer targets to the planned gait frequency terminates
after about **1.86 s**. This rules out adopting a simple time-stretched policy
as a walking fix. Any timing change must preserve the intended speed/stride
relationship and be learned with the actual plant dynamics. The current
reference's gait clock is already normalized.

![Controlled actuator and input comparison](../../results/wr2_walking/wr2_paper_20261017_195826_seed3/review_20261018/actuator_review/actuator_comparison.png)

## 4. What the hardware evidence actually identifies

Replaying all six accepted fit/held-out traces through the training controller
reproduces the accepted fit:

| Replay model | Mean held-out position RMSE |
| --- | ---: |
| Accepted native-position fitting model | .85460° |
| Explicit training PD + same constant cap | .85462° |
| Explicit training PD + current torque-speed taper | .85462° |
| Same training controller at 5 ms | .85397° |

None of these model replays reaches torque saturation; predicted speeds stay below .331 rad/s and
torque below 1.531 N m. Their small motions cannot validate the region where the
new walking policy repeatedly saturates. The copied servo archive contains 84
completed nonempty captures; none has measured telemetry speed above 1 rad/s.
The largest such finite-difference value is .85 rad/s and is quantized telemetry.

The fixture's rigid moving inertia is **.04302 kg m²**, versus the fitted
.02095 armature. All accepted unit-A dynamic captures use that same fixture.
Changing fixture inertia or packet duration is necessary to distinguish a
mechanical inertia from firmware response absorbed into the fit.

An existing alternate fit provides direct evidence of that ambiguity:

| Parameterization | Armature | Additional delay | Held-out mean RMSE |
| --- | ---: | ---: | ---: |
| Accepted | .020951 kg m² | 0 | .85460° |
| Alternate candidate | .001152 kg m² | 20 ms | .86478° |

Held-out error is only **1.19% worse**, with about **18.2 times less armature**.
The reported armature interval is conditional on the selected delay. Unit B/C
fits also vary substantially (.04291/.03666 kg m²); the fit is not a direct
rotor-inertia measurement.

On the heavy fixture, those two armatures change total moving inertia from
.06397 to .04417 kg m². On WR2's ankle, the armature contributes 97.8% of the
modeled inertia. The fixture therefore hides much of the uncertainty that
matters on the robot. The current ±20% armature randomization around unit A
also excludes the B/C point estimates; their nonunique fits need to be resolved
before claiming a hardware population envelope.

Mechanical inertia changes MJX's mass matrix and contact response. Firmware
delay changes when effort responds to a command. Confusing the two can preserve
a fixture's angle fit while producing different whole-body dynamics.

Possible actual contributors include reflected motor/gear inertia, internal
velocity feedback, transmission losses and firmware position profiles. The
current captures identify their **combined output response**, not each part.
There is no measurement of the current robot's shaft torque/speed curve under
walking loads. Supply sag and heating are relevant hardware effects, but they
are not simulated causes demonstrated by these completed training logs.

## 5. A confirmed TB push bug changes the comparison

Frozen TB `MJXEnv.step()` writes the sampled push only to
`pipeline_state.qd`. Its Brax MJX pipeline calls `mjx.step(sys, state)` using
`state.qvel`, then rebuilds `qd` from `qvel`. The controller reads motor velocity
indices after the six root coordinates, so changing only the root `qd` does
not even alter the motor torques. The requested root push is dropped.

Controlled one-substep test from the same native TB state and zero torque:

| Mutation before physics | Requested forward change | Result relative to unpushed physics |
| --- | ---: | ---: |
| `qd` only, frozen TB | .5 m/s | **Exactly zero qpos/qvel change** |
| `qd` and `qvel` | .5 m/s | **+.4833 m/s forward qvel** after constraint reaction |

WR2 explicitly updates both fields. Its pushes therefore impose a physical
training challenge absent from this frozen TB Paper code. This is a code-level
delta from the published intent, not an argument that the published recipe
should omit disturbance training.

The sampled pushes include **all three translation and all three rotation
components**, starting at episode step zero and recurring every two TB-equivalent
seconds. The per-component bounds are .5 m/s and 1 rad/s for TB, .4120 m/s and
1.2135 rad/s for WR2. The WR2 bounds and interval are already size/time scaled;
the missing physical application in TB is the difference.

The paired local TB 200M evaluation uses the native 128 ×15 conditions, seed
12102, 1000 ticks, identical native model/reset/actuator/noise variation, and
identical push requests. The treatment synchronizes `qvel=qd` immediately before
physics in memory. No TB source file or training checkpoint is modified.

The WR2 treatment instead zeros the two physical push amplitudes while keeping
the interval and active random-key split unchanged. Its 128 model/reset draws,
noise draws, selected 220M policy and 824-tick horizon are otherwise identical
within the pair. All four complete suites pass the independent aggregate audit:

| Policy / physical exposure | Fixed XY error, TB-equivalent m/s | Height terminations / 1920 | Mean duration, TB-equivalent s |
| --- | ---: | ---: | ---: |
| TB Paper 200M, native dropped pushes | .03707 | 11 / .57% | 19.955 |
| Same TB, configured pushes actually applied | .44159 | 1920 / 100% | 1.579 |
| WR2 220M, native realized pushes | .13058 | 54 / 2.81% | 19.728 |
| Same WR2, zero physical pushes | .07658 | 0 / 0% | 19.999 |

Error averages the ten fixed moving conditions; termination/duration summaries
cover all 15 conditions. Both use the native torso-height criterion: .2–.4 m
TB, .17096–.34192 m WR2, with the WR2 bounds scaled by home torso-origin height.
These are **height terminations**, not an independently classified count of
tipovers. Every suite has zero nonfinite episodes.

These are CPU reproductions. The original GPU-native records are .03702 /
8 terminations for TB and .13063 / 55 for WR2; the small numerical differences
do not alter this conclusion. Cross-robot model/reset/actuator distributions
remain native, rather than identical. The physical exposure is now explicit.

Realizing TB's configured pushes gives a median episode of **.89 s**; 1443/1920
terminate before the second push at 2 s. Its .44159 error is **censored by early
termination** and must not be treated as a 20-second tracking comparison or
evidence that TB could not learn the correctly applied published recipe. The
frozen policy was trained without those physical perturbations. A benchmark
for the published disturbance recipe would require a separately trained TB
control with the implementation corrected, owned by the active TB project.

Removing WR2's physical evaluation pushes reduces error **41.36%**, with zero
height terminations. The paired error difference is −.05401 m/s
(95% bootstrap interval −.05689 to −.05142). The bootstrap resamples the 128
model/reset indices, keeping each index's conditions paired; it does not cover
training-seed or retraining uncertainty. The remaining unpushed error is
**2.07× TB's native result**. In WR2's forward condition, neither-foot contact
samples remain **53.76%**, versus 52.01% with pushes; unpushed physical forward
velocity is .05722 m/s, or .06943 in TB units against a .10 target. Removing
evaluation disturbances does not recover a TB-like gait from this frozen policy.

## 6. Revised order before implementation

1. **Use the physically correct benchmark.** Separate the recorded native
   Paper result from evaluations with realized pushes. Continue comparing
   noise-free/no-push gait motion as well; WR2's flight and tracking gap persist
   there. An unchanged native fall rate is not proof of equal disturbance
   robustness.
2. **Test acquisition under TB's effective conditions.** A fresh long WR2
   control with physical pushes disabled would match this frozen TB code's
   effective disturbance exposure while keeping its Paper rewards, reference,
   network, PPO layout and measured WR2 actuator model. This is preferable to
   immediately attributing the failure to a WR2-specific reward change. It is
   an acquisition experiment, not a final deployment recipe. Setting the two
   push amplitudes to zero also zeros WR2's six critic push features; TB's
   frozen critic still receives its unrealized random push labels. Record that
   observation difference rather than calling this an exact training replica.
3. **Identify the complete actuator response before editing its physics.**
   Use multiple known fixture inertias and logged packet durations; collect
   responses that exercise the relevant velocity/load region under the hardware
   test procedure. Fit the same controller/envelope/command processing used by
   training, as TB's identification tool does. Preserve separate held-out
   sessions. Adopt gains, inertia, damping split, braking or slew only as a
   consistent identified model, not individually because they improve a replay.
4. **Keep the already-correct critic velocity feature.** The earlier review
   misread this path. The constructor divides the cached scale by `s`, and
   `_paper_observation` uses that cached field at entries 80:83. A physical
   .0824055 m/s input produces .2000000, exactly the intended `2*linear/s`.
   Removing that proposed fix keeps the next experiment focused on its actual
   physical difference.
5. **Evaluate a complete long training curve, retaining every ~20M checkpoint.**
   Prefer a 200–300M TB-aligned acquisition control. If the wrong flight strategy
   remains after physical conditions are aligned, test the previously proposed
   supported-swing air-time counter separately with the same critic
   and physical settings. Offline credit preservation on TB is useful evidence;
   it is not a prediction of the retrained policy.

No existing evidence proves WR2 cannot reach the .10 m/s TB-equivalent target.
It also does not justify copying TB actuator numbers, increasing gains,
stretching a learned policy in time, or assuming a reward guard alone provides
parity. The next experiment should first answer whether the harder acquisition
conditions explain the policy's choice of motion.

## Artifacts and references

All raw captures, replay arrays, plots and analysis programs are local ignored
results under
`results/wr2_walking/wr2_paper_20261017_195826_seed3/review_20261018/actuator_review/`.
Core artifacts: `static_and_excitation.json`, `controlled_servo.json`,
`loaded_replay.json`, `capture_inventory.json`, `critic_velocity_probe.json`,
`tb_push_same_state_probe.json`, and paired `tb_push_eval*.json` /
`wr2_push_eval*.json` results. `actuator_audit.json` reconstructs 102 measurements
with maximum aggregate difference 2.98e−9. `push_eval_audit.json` checks all
7680 episode-condition records and 14,813 aggregates, with maximum difference
2.01e−6. `com_motion_comparison.json` records the secondary pose-difference check.
The standalone figure is also
available as `actuator_comparison.pdf`.

- [ToddlerBot paper](https://arxiv.org/html/2502.00893v2) and the frozen local
  implementation/run cited above.
- TB's local [system-identification guide](../../../toddlerbot/docs/tools/02_sysID.rst)
  and `toddlerbot/tools/run_sysID.py`: optimize damping, armature, friction and,
  for the actuator fixture, torque/speed parameters in the actual controller.
- [MuJoCo DC motor model](https://mujoco.readthedocs.io/en/latest/_static/dcmotor.pdf),
  public background reference; the numerical conclusions here come from local
  code, captures and controlled simulations.
- [Completed WR2/Paper training comparison](wr2_fresh_paper_result_20261018.md).
