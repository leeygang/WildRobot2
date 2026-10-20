# WR2 collision-weight result — 2026-10-20

## Conclusion

**Increasing collision weight from 0.1 to 1.0 substantially changes the
learned stance:** foot self-contact is almost eliminated, the right hip is
outward, and physical joint-target clipping disappears. Command tracking
improves **10.1% /10.5%** on the selection/held-out evaluation seeds, with
zero height failures in both primary suites.

**Walking remains incomplete.** Error is still **1.87× TB Paper 200M** and
only **3.6% /3.8%** better than WR2's own initial policy. Forward motion is
slow, rightward commands still produce slightly leftward mean velocity, and
the policy favors left-foot support. The result supports a reward contribution
to the unwanted contact strategy; it does not establish the full cause of the
remaining propulsion and weight-transfer gap.

Run: `results/wr2_walking/wr2_paper_collision1_20261020_072851_seed3`.
Completed **300,134,400 transitions**; selected checkpoint is **300M**, whose
`best_params` and final `params` are identical. This is the measured best
among retained eligible checkpoints, rather than an assumption about the
final checkpoint.

## 1. Configuration and source verification

Saved configurations differ from completed WR2 0.28.5 seed 3 only in collision
**0.1→1.0**, version/name and output prefix. Both start fresh; their
`initial_params` files are byte-identical. All recorded actuator-model,
physics, home/action-scale and reference-validation values agree. Python
**3.10.19**, Brax **0.14.1**, JAX **0.6.2**, MuJoCo/MJX **3.14.0**, Flax
**0.10.7** and other recorded package versions match.

Baseline training source is `5f58c13`; new source is clean `a93c4f8` on an
RTX 5070. The intervening environment edits add measurements. Trainer edits
add provenance and initial-policy confirmation. The reference command grid
was extracted into a shared helper with unchanged values; controller,
physical model, native reward formulas and PPO construction are unchanged.
The baseline did not record GPU identity or matmul precision, so those
runtime fields cannot be independently matched.

Initial evaluation contact labels, clipping and sample counts agree exactly.
Floating measurements have small differences: largest per-episode velocity
difference is **3.23e−5 physical m/s**, and the aggregate initial XY-error
difference is **1.17e−8 TB m/s**. Identical parameter files therefore do not
imply bitwise-identical entire evaluation trajectories.

Native TB source was read from local **`paper-walking-baseline`**, revision
`a4f693d`. TB uses collision **0.1**, air time **500**, leg-reference imitation
**5**, and the same fixed-home action origin and tanh-normal policy family.
Its collision term excludes the floor and counts both directions of an
active collider pair. WR2's stronger coefficient is the declared experimental
departure. One WR2 foot pair costs **−.04 per contact tick**, versus −.004
in the old control.

## 2. Primary benchmark and held-out confirmation

Each WR2 suite is **15 conditions ×128 episodes**, 824 physical control ticks
at 20 ms, approximately **20 TB seconds**. The tracking score averages each
episode's instantaneous XY error over the ten fixed nonzero-command
conditions. WR2 velocities divide by `s=.8240547565`; TB uses its native units
and 1000 ticks. Total rewards are not the comparison criterion.

| Policy | Selected transitions | Selection XY error, TB m/s | Held-out XY error, TB m/s | Primary height failures |
| --- | ---: | ---: | ---: | --- |
| WR2 initial | 0 | .071623 | .071623 | 0 /1920 on both seeds |
| WR2 collision .1 | 260M | .076838 | .077025 | 0 /1920 on both seeds |
| **WR2 collision 1.0** | **300M** | **.069066** | **.068908** | **0 /1920 on both seeds** |
| TB Paper | 200M | .037025 | Not in this copied suite | 8 /1920 on selection seed |

The WR2 difference is stable across the 128 shared model/reset groups. A
paired bootstrap resampling these groups, preserving repetition across the
ten commands, gives a 95% interval for new-minus-old error of:

- Seed **12102**: **−.00830 to −.00725 TB m/s**, mean −.00777.
- Seed **22102**: **−.00866 to −.00758 TB m/s**, mean −.00812.

These intervals describe evaluation variation for this pair of training
policies. There is only one paired training seed; they do not quantify
variation across independently trained seeds. TB retains its native
friction, noise/reset and actuator implementation; the comparison measures
native normalized performance, not identical robot physics.

### Foot contact and actuator effort

| Primary measure | Result |
| --- | --- |
| All-condition foot self-contact, seed 12102 | **134 /1,582,080 valid ticks = .00847%**; 15 episodes have any contact |
| All-condition foot self-contact, seed 22102 | **71 /1,582,080 valid ticks = .00449%**; 19 episodes have any contact |
| Fixed-walking target clipping | **10.28%→0%** |
| Fixed-walking torque RMS | **.778→.563 Nm**, 27.6% lower |
| Fixed-walking target slew limiting | **.19%→3.81%** |
| Fixed-walking leg-reference RMSE | **.148→.168 rad**; TB Paper is .232 rad |

Baseline schema 2 does not contain `foot_self_contact_fraction`. Its absence
is not treated as zero or compared directly with the new schema-3 fraction.
The paired nominal replay below supplies a direct contact comparison.
The model boxes remain conservative relative to CAD; these counts measure
model contact, not a proof of physical foot-on-foot loading in every case.

## 3. Directional walking gap

Mean body-frame commanded-axis velocities in TB-equivalent m/s, from the
same primary benchmark. Positive Y is leftward:

| Command | WR2 collision .1 | WR2 collision 1.0 | TB Paper 200M |
| --- | ---: | ---: | ---: |
| Forward +.10 | +.00573 | **+.01696** | +.09053 |
| Backward −.10 | −.00324 | **−.03764** | −.07357 |
| Forward +.05 | +.00414 | +.00525 | +.04401 |
| Backward −.05 | −.00074 | −.01375 | −.04111 |
| Left +.05 | +.00131 | +.01552 | +.03499 |
| Right −.05 | −.00019 | **+.00151: wrong sign** | −.03204 |

The gain is real but uneven: forward +.10 reaches about **17%** of its
command, backward −.10 about **38%**, and left +.05 about **31%**. Rightward
XY error worsens from **.05721 to .05945 TB m/s**, despite the improved
aggregate score. The forward command also has mean cross velocity
**+.01190 TB m/s**, producing leftward drift.

These benchmark velocities are body-frame means. The separate nominal CPU
replay measures signed world displacement under zero commanded yaw:
new forward net velocity is **(+.02000, +.01095)** versus old
**(+.00555, −.00010) TB m/s**. Translation improves, but it is far below the
command and includes substantial cross motion.

## 4. What changed in the learned coordination

Twelve nominal CPU probes use the native model/controller, deterministic
learned actions, seed **22102**, no pushes/variation/noise, and a 20-TB-second
horizon. All seven selected-policy commands survive with **zero foot
self-contact**. The old selected forward policy has **95.87%** contact in
the same CPU batch. There are no replaced reference actions in these probes.

Right-hip physical signs are opposite in the robots; the following table
expresses both as **positive outward**. Phases use actual floor support,
after 1.5 TB seconds, not the desired reference contact:

| Actual phase | Old WR2 right hip | New WR2 right hip | TB Paper right hip |
| --- | ---: | ---: | ---: |
| Left supports; right unloaded | **−5.87° inward** | **+6.90° outward** | +2.93° outward |
| Right supports; left unloaded | **−5.51° inward** | **+6.94° outward** | +6.51° outward |

**The right hip now goes outward, but its phase means barely differ.** TB
changes its mean right angle by **3.58°** between these actual support phases;
WR2 changes by **.03°**. Left-hip means likewise differ only **.14°** in WR2,
versus **6.99°** in TB's native joint convention. These are conditional mean
differences, not full within-cycle motion amplitudes.

Across fixed walking commands, the left foot has floor contact on **79.5%**
of ticks and the right on **24.8%**, with **4.3%** double support. These are
contact fractions, not percentages of body weight. In the forward condition,
mean floor forces are **16.6 N left /4.8 N right**. The policy favors one
supporting leg while the other repeatedly unloads; this differs from TB's
stronger phase-dependent lateral coordination.

Nominal forward hip execution RMSE falls **5.50°→1.39°**. Hips contribute
**5.2%**, rather than **71.4%**, of leg delivered-to-achieved squared error.
The other ten actuators' execution RMSE increases **1.56°→2.67°**, alongside
more target motion/slew limiting. This does not isolate a faulty actuator
model: greater error can also result from the policy's faster or poorly
coordinated target requests. Tightening reference imitation is not supported
by TB's larger successful reference deviation.

The replay therefore establishes a change from inward, contacting coordination
to an outward stance with incomplete phase-dependent transfer. It does not
prove which physical/control difference causes the remaining limited transfer
and propulsive motion.

## 5. Learning progress compared with TB

XY error in TB-equivalent m/s; parentheses give height failures /1920:

| Transitions | WR2 collision .1 | WR2 collision 1.0 | TB Paper |
| --- | ---: | ---: | ---: |
| 20M | .14596 (71) | .18138 (5) | .06202 (9) |
| 40M | .08480 (0) | .16658 (3) | .04771 (20) |
| 80M | .07834 (0) | .13092 (0) | .03965 (17) |
| 140M | .07726 (0) | .07848 (0) | .03806 (20) |
| 200M | .07859 (0) | .07801 (0) | **.03702 (8)** |
| 240M | .07722 (0) | .07344 (0) | .03823 (57) |
| 300M | .07871 (0) | **.06907 (0)** | .03800 (134) |

Stronger collision weighting slows early tracking acquisition, even though
model foot contact is already rare at 20M. The new run only approaches the
old tracking level around 140M and beats its stationary initial policy around
280M. It then improves about **6.0% over 240M→300M**. TB acquires useful
translation much earlier and deteriorates in survival late in its run.

In the new nominal forward replay, the right hip remains outward at both
20M and 40M. The earlier inward collapse does not repeat. However, net motion
at those stages is in the wrong direction. At 140M the left foot supports
almost continuously and net progress is small; by 240M/300M right support
occurs more often and forward progress increases. Thus avoiding contact and
learning propulsion are distinct parts of acquisition.

Late reported PPO KL is stable around **.023–.024**, and latent mean standard
deviation is around **.099** at 280M/300M. There is no recorded late numerical
failure or abrupt aggregate tracking collapse. These statistics do not prove
that further training must improve.

## 6. Robustness tradeoff

Neither WR2 acquisition run trained with physical pushes. The companion
profile applies the same size-scaled TB push bounds and is separate from
checkpoint selection:

| Pushed profile | WR2 collision .1 | WR2 collision 1.0 |
| --- | ---: | ---: |
| Seed 12102 height failures | 693 /1920 (**36.1%**) | 1115 /1920 (**58.1%**) |
| Seed 22102 height failures | 918 /1920 (**47.8%**) | 1218 /1920 (**63.4%**) |

Contact reduction and modest tracking gains therefore accompany worse pushed
survival. The no-push zero-failure result is not a deployment robustness claim.
Pushed tracking scores are fall-censored and are not the primary shared
20-second walking comparison. Frozen TB Paper's configured pushes were
physically absent, as established in the earlier source/runtime audit.

## 7. Proposed next step

1. **Retain collision weight 1.0.** The contact objective is essentially met;
   increasing it again has little direct incentive to improve translation.
2. **Test an additional 200M from the selected 300M checkpoint**, preserving
   the TB-aligned PPO/remaining objective and retaining approximately 20M
   checkpoints. The final checkpoint is the measured best and recent tracking
   still improves. This is a justified budget test, not a promise of parity.
   The current Brax restore interface restores parameters and starts new Adam
   state; label the experiment as parameter continuation and preserve the
   original 300M candidate as a control.
3. **Judge progress by directions and phase-dependent transfer.** Require
   improved signed rightward/forward response, cross-drift, hip-roll phase
   means, floor-load distribution and knee/ankle execution alongside the
   aggregate error and survival. Confirm nominal phase findings on the GPU.
   If the longer run still produces a largely fixed, one-leg-biased stance,
   use those measurements to isolate control response and lateral transfer
   before choosing a further WR2-specific reward or physics change.

The current evidence supports the stronger collision coefficient and a longer
learning test. It does not support disabling air time, prescribing permanent
hip angles, strengthening imitation or retuning actuator physics from these
frozen-policy traces alone.

## Audit, artifacts and sources

**85 reports /66,570 mean-and-standard-deviation checks** passed. Primary
scores were recomputed from per-episode values and checkpoint selection was
rechecked. Maximum group-statistic discrepancy is **2.93e−6**; recomputed
primary aggregate scores match exactly. Contact/support fractions and
conditional denominators were checked; missing old-schema measures remain
explicitly missing.

All twelve CPU traces pass finite/termination, action mapping/delay, native
target delivery, old-applied-action history, floor threshold, collision,
air-time, imitation, XY reward and component-sum audits. Maximum full reward
sum error is **9.54e−7 per tick**. The reduced reference-command set uses the
full native lookup's nearest rows; all seven command references agree with
copied full-grid GPU traces within **1.82e−12 rad**.

CPU probes use JAX .6.2, Brax .14.2, MuJoCo/MJX 3.14.0, Flax .11.2, NumPy
2.4.6 and high matmul precision. Primary conclusions use the copied GPU
reports. New nominal phase/angle probes have not been repeated on the training
GPU; the CPU old-policy forward XY score differs from its copied nominal GPU
score by about **5.7e−5 TB m/s**. TB nominal phase values come from its copied
Paper GPU diagnostic, with its native runtime and backlash behavior.

- [Audited comparison and checkpoint curves](../../results/wr2_walking/wr2_paper_collision1_20261020_072851_seed3/review_20261020/comparison.json).
- [Nominal traces and summary](../../results/wr2_walking/wr2_paper_collision1_20261020_072851_seed3/review_20261020/nominal_cpu/summary.json).
- [Native nominal audit and phase statistics](../../results/wr2_walking/wr2_paper_collision1_20261020_072851_seed3/review_20261020/nominal_cpu/audit.json).
- [Learning/direction/phase figure](../../results/wr2_walking/wr2_paper_collision1_20261020_072851_seed3/review_20261020/collision_comparison.png)
  and [exportable PDF](../../results/wr2_walking/wr2_paper_collision1_20261020_072851_seed3/review_20261020/collision_comparison.pdf).

Selected checkpoint SHA256:
`d1a4eca667db1d80de327768231b3c4a0a38124fa3981570168491160aee83cb`.
Analysis programs/traces/figures remain under ignored `results/`.

Sources: WR2 [rewards](../../wr2/locomotion/paper_rewards.py),
[environment](../../wr2/locomotion/walking_env.py),
[per-episode evaluator](../../wr2/locomotion/paper_evaluation.py),
[trainer](../../wr2/locomotion/train.py), local
`paper-walking-baseline:toddlerbot/locomotion/{paper.gin,mjx_env.py,walk_env.py}`,
native saved TB Paper evaluation records, and
[ToddlerBot, Tables VI–VII](https://arxiv.org/html/2502.00893v2).
