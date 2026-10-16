# WR2 paper transfer: completed 300M review — 2026-10-16

## Decision

**The transfer has not reached TB parity.** Retain **200,089,600** as the best
trained diagnostic checkpoint: it has both the lowest tracking error and
fewest falls among trained checkpoints. None passes the complete suite's
**≤1% falls** gate. Extending this policy from 200M to 300M worsened both measures.

The learned motion contains frequent synchronized hops, including under zero
command. A WR2-specific landing-reward normalization error was reproduced and
fixed in recipe **0.28.1**. Test that correction with fresh training; capture
matching TB foot measurements before choosing additional reward or physics
changes. The correction removes a measured bias; its learning effect is unknown.

Run: [`wr2_paper_20261016_130552_seed0`](../../results/wr2_walking/wr2_paper_20261016_130552_seed0).
Sources: clean **0a63ae9**, recipe **0.28.0**, fresh seed 0, no argument overrides
or restore. The saved YAML exactly matches the prepared 300M configuration at
that revision. The copied artifacts contain structured logs and checkpoints;
console stdout/stderr was not included.

## 1. Accuracy and completeness audit

- Completed **300,134,400 transitions**. Initial parameters, 15 distinct native
  checkpoints, final parameters, 16 native evaluations and 16 paper reports
  are present. All evaluation/checkpoint/history step lists agree.
- Reconstructed means, population standard deviations, counts, episode
  lengths, fall/nonfinite totals and selection summaries from **30,720
  episode-condition records**. Maximum rounding discrepancy: **6.90e−7**.
  Every group's sample count matches its schedule and recorded episode length.
- All native metrics and all **149** rollout-log rows are finite. All 25
  weighted reward terms are present; their sum agrees with return within
  **0.000504**. No paper evaluation contains a nonfinite episode.
- Loaded every native checkpoint locally with the GPU's **Brax 0.14.1**
  package. All **25 leaves / 1,599,600 values** per checkpoint are finite.
  The final native checkpoint exactly matches the `params` export, SHA-256
  `05522d388f7c126a0029184efc219227fc1de0d8cad4dc542874c4dcf6b9482f`.
  The 200M policy also ran through actual MJX inference/physics locally.
- Model hash matches current/prepared `wr2_mjx.xml`; all **227 × 1000**
  recorded reference validation samples pass. Newton 10, 5 ms × 4 substeps,
  fixed grounded home and measured CoM scale **0.824054757** match the recipe.

The suite uses 15 conditions × 128 episodes, selection seed **12102**,
deterministic policy, observation noise and model/reset/actuator variation.
WR2 lasts **824 ticks / 16.48 s** with normalized commands and start/stop timing;
TB lasts 1000 ticks / 20 s. Tracking is the per-episode mean instantaneous
vector error, then averaged across episodes. Failed episodes contribute their
shorter measurements. It is not the error of the mean velocity.

Audit data: [`metrics_audit.json`](../../results/wr2_walking/wr2_paper_20261016_130552_seed0/review_20261016/metrics_audit.json).
Figures: [training curves](../../results/wr2_walking/wr2_paper_20261016_130552_seed0/review_20261016/training_overview.png)
and [nominal forward trace](../../results/wr2_walking/wr2_paper_20261016_130552_seed0/review_20261016/nominal_forward_trace.png).

## 2. Checkpoint and ToddlerBot comparison

| WR2 transitions | Physical fixed XY error (m/s) | TB-equivalent (m/s) | Falls / 1,920 |
| --- | ---: | ---: | ---: |
| 0 | 0.103833 | 0.126002 | 786 (40.94%) |
| 20,008,960 | 0.159435 | 0.193476 | 190 (9.90%) |
| 80,035,840 | 0.139874 | 0.169739 | 61 (3.18%) |
| 140,062,720 | 0.125372 | 0.152140 | 58 (3.02%) |
| 180,080,640 | 0.122396 | 0.148529 | 86 (4.48%) |
| **200,089,600** | **0.120675** | **0.146441** | **37 (1.93%)** |
| 300,134,400 | 0.128171 | 0.155538 | 77 (4.01%) |

| Successful TB paper control | Fixed XY error (m/s) | Falls / 1,920 |
| --- | ---: | ---: |
| 180,080,640 | 0.038379 | 3 (0.156%) |
| 200,089,600 | 0.037025 | 8 (0.417%) |
| 300,134,400 | 0.038002 | 134 (6.979%) |

WR2's best trained error is **3.96× TB 200M**. WR2 final survival is better
than TB final survival, but TB 180M/200M remain much stronger controls.
Initial WR2 error is lower than its trained errors while 41% of episodes fall;
that unresponsive initialization is not a useful walking candidate.

Paired bootstrap of WR2 final minus 200M, retaining all conditions together
for each of 128 model/reset indices, gives **+0.00750 m/s physical error**,
selection-set 95% interval **[+0.00447, +0.01158]**, and **+2.08 percentage
points falls**, interval **[+0.47, +3.91]**. These are conditional comparisons
on one training seed and the selection set, not held-out guarantees.

## 3. Mean command response hides the gait failure

The following rows use the WR2 200M checkpoint; WR2 velocities/errors are
divided by `s` for this table. TB values come from its raw 200M episode exports.

| Condition | WR2 mean commanded-axis velocity | TB mean commanded-axis velocity | WR2 vector error | TB vector error | WR2 falls | TB falls |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Forward +0.10 | +0.0869 | +0.0905 | 0.1552 | 0.0373 | 4 | 1 |
| Backward −0.10 | −0.0762 | −0.0736 | 0.1469 | 0.0423 | 2 | 1 |
| Right −0.05 | −0.0217 | −0.0320 | 0.1443 | 0.0373 | 4 | 1 |
| Left +0.05 | +0.0289 | +0.0350 | 0.1426 | 0.0354 | 1 | 0 |

Lateral response is **43–58%** of the command versus TB's **64–70%**.
Forward/backward mean response is close to TB, while instantaneous error is
much worse. Reset standing speed is **0.1160 m/s physical / 0.1407 TB-equivalent**,
versus TB **0.0320 m/s**. Near-zero mean standing velocity hides oscillation.

Forward leg-reference RMSE is **0.173 rad for WR2 / 0.234 rad for TB**.
Both implement moving leg-reference imitation. Raw joint RMSE depends on each
robot's kinematics and motor coordinates; compare actual foot motion and
applied targets to locate the transfer failure.

At WR2 200M, forward support fractions are **17.74% single support / 26.88%
double support / 55.38% neither foot supported**. Standing has **55.95%**
neither-foot support. These percentages are 50 Hz control samples, not
integrals over all physics substeps. The derived fraction is exactly
`1 − left_contact − right_contact + double_support`.

The fixed-forward native return is **408.72**, of which **306.11 (74.9%)** is
air time and **32.86 (8.0%)** is XY tracking. Leg imitation contributes **−2.95**.
The objective rewards landing after any flight; the native formula imposes
no explicit prohibition on both feet being airborne. Its declared maximum
air-time variable in TB is not used to cap this reward.

Forward actual torque RMS is **1.46 Nm**, mean per-step peak **2.89 Nm**.
Command slew limitation affects **37–41% of leg-target samples** across
walking conditions; joint-limit clipping affects only about **0.10–0.29%**.
Increasing the action or speed envelope is not supported by these measurements.

## 4. Physics and observation diagnostics

Ran two canonical conditions, 8 environments, seed **22102**, 824 ticks using
the frozen 200M policy. These are local CPU diagnostics with Brax 0.14.1,
JAX 0.6.2 and MuJoCo/MJX 3.14.0; they are not full qualification suites.
The two references use the same canonical lookup entries as training.

| Nominal replay | Forward physical error | Standing physical speed | Forward neither-foot fraction | Standing neither-foot fraction |
| --- | ---: | ---: | ---: | ---: |
| No variation, pushes or observation noise | 0.06866 | 0.05517 | 61.17% | 60.68% |
| Same nominal model, observation noise enabled | 0.06882 | 0.05824 | 59.98% | 59.85% |
| Exact frozen TB symmetric torque law, WR2 parameters, quiet | 0.06285 | 0.05913 | 60.44% | 60.80% |

Hopping persists in nominal physics and with the native TB torque law.
Observation noise has a small effect in these nominal replays. Randomization
and pushes amplify tracking error in the full suite, but the gait issue is
already present without them. The torque-law comparison alone does not resolve it.

An additional quiet one-environment forward trace has **500/824** unsupported
samples. At every such sample both foot vertical forces are **zero**, and
both foot contact distances are **positive**, with minimum **14.7 mm**.
The model has only floor and two foot collision geoms. This rules out hidden
knee/body support and a light-contact threshold explanation for that trace.
The trace contains **147 flight bursts**, median **60 ms**, about **8.9 Hz**,
versus the **1.67 Hz** reference gait cycle. Sampled mean foot vertical force
is 20.82 N versus model weight 20.19 N; individual sampled peaks reach 122 N.

The trace's free-joint velocity produces **0.07367 m/s** error versus the
reported torso velocity's **0.07387 m/s**. Their mean velocities agree closely;
the different within-step snapshots do not explain the tracking gap.
Motor speeds remain below the vendor ceiling in this trace, while requested
target steps frequently hit the command limiter.

WR2's hardware model is identified at low loads and retains quantization,
slew, Newton 10 and asymmetric braking. Frozen TB uses symmetric torque-speed
clipping, no additional target slew limit and Newton 1. The low-load fit is
validated only through **1.0157 Nm for 3 s**; the observed policy loads lie
beyond that validation range. High-load/braking behavior and this rapid motion
need hardware validation before deployment or changing the physical model.

## 5. Confirmed normalization error and correction

Both robots use **20 ms control**, despite their different normalized cycle
times. The original WR2 helper divided landing air time by `s` and the sampled
clearance accumulator by `s³`. These are event impulses, while tracking is
a continuous rate. A complete normalized gait cycle has fewer control samples
for continuous rewards; the old conversion preserves the event impulse instead
of scaling it with duration. Its relative reward rate is inflated by **1/s**.

Executed the actual helper over the native 36-tick / 0.72 s cycle and WR2's
rounded 30-tick / 0.60 s cycle, with two triangular swings and geometrically
scaled heights. The independent known cycle totals expose the discrepancy:

| Reward rate in the gait-cycle check | TB | WR2 0.28.0 | Corrected WR2 0.28.1 |
| --- | ---: | ---: | ---: |
| Air time, weight 500 | 6.66667 | 8.09008 | 6.66667 |
| Clearance, weight 0.05 | 0.00083333 | 0.00101126 | 0.00083333 |
| Perfect XY tracking, weight 5 | 5 | 5 | 5 |

The new regression failed before the fix with **21.351%** relative error.
Correction: use **physical air time** and divide the height accumulator by
**`s²`**. The 25 native coefficients are retained. Four reward regressions pass,
including the unchanged scale-1 TB numerical oracle and the full-cycle check.
The prior pointwise normalization test missed this event-frequency effect.

Re-scoring the same 200M policy after the correction gives return **354.86**;
air time still accounts for **71.1%**. Re-scoring does not change its actions
or gait. The correction removes a proven transfer bias, but it cannot establish
that fresh learning will close a roughly fourfold tracking gap.

## 6. Next experiment and source comparison

1. Capture TB **180M/200M** with the same foot-contact, flight, applied-target,
   torque and path measurements, keeping its frozen model/source and normalized
   evaluation protocol. Its existing XY export lacks those measurements. This
   distinguishes a native recipe loophole from a robot/controller-specific
   failure before adding any new gait term.
2. Run a **fresh 200–300M corrected paper comparison**, retaining all intermediate
   checkpoints and recording flight and slew alongside XY error and falls.
   Keep native reward coefficients, distribution, PPO settings and identified
   WR2 hardware limits. Use the 200M old policy as a diagnostic control; its
   hopping solution is unsuitable as the initialization for this comparison.
3. If hopping remains after the correction and adequate budget, use the matched
   TB foot evidence and high-load WR2 servo replay to choose the next change.
   More steps on the old policy, larger hardware limits or optimizer changes
   are not supported as solutions by this run.

The run uses moving **leg-reference imitation**, weight 5, with fixed
`q_home` action targets and motor observations. Reference errors remain critic
inputs; the actor receives phase/commands and learns reference tracking through
rewards. These contracts, Swish, NormalTanh, GAE and the value-loss coefficient
match the successful TB paper control. The intended Brax trainer and native
actor/reference contract were used in this completed run.

Source comparison used WR2's `walking_env.py`, `paper_rewards.py`, `ppo.py`,
`paper_evaluation.py`, reference implementation and saved run configuration;
TB's frozen `mjx_env.py`, `walk_env.py`, `evaluate_xy.py`, `paper.gin`, and
`1982ef3` reference/controller code. Current local TB main uses a different
reward configuration and is not the source of the successful paper control.

References: [ToddlerBot paper, Tables VI/VII](https://arxiv.org/html/2502.00893v2),
[ZMP preview control](https://doi.org/10.1109/HUMANOIDS.2015.7363473),
[TB control audit](toddlerbot_transfer_review_20261016.md), and the
[updated WR2 recipe](wr2_paper_training.md).
