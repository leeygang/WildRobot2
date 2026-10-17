# WR2 corrected paper recipe: completed 300M review — 2026-10-17

## Decision

**The corrections improved tracking and eliminated measured nominal loss of
foot support, but WR2 has not reached ToddlerBot parity.** Best fixed-command
tracking is at
**260,116,480** transitions; fewest falls are at **280,125,440**. Retain both
and the final checkpoint. None passes the complete suite's **≤1% falls** gate.

Compared with the previous run's best checkpoint, tracking improves **26.5%**,
while falls increase from **1.93% to 5.31%**. Best normalized tracking error
remains **2.91× TB 200M**. Controlled replays identify poor push recovery and
an underlying nominal tracking gap. This review finds no new confirmed trainer,
reference, support-accounting or checkpoint bug.

Run: [`wr2_paper_20261016_222755_seed0`](../../results/wr2_walking/wr2_paper_20261016_222755_seed0).
It trained fresh from clean **c1150ed**, recipe **0.28.2**, seed 0, with no
restore or CLI overrides. The two reproduced fixes were landing-reward
normalization and summing foot forces before applying TB's 1 N threshold.
Their individual learning contributions were not isolated in this run.

## 1. Metric, provenance and checkpoint audit

- Completed **300,134,400 transitions** from the requested 300M budget.
  There are 16 native evaluations, 16 paper reports, 149 rollout records,
  15 trained native checkpoints, initial parameters and the final export.
  Checkpoint, evaluation and history step lists agree.
- Independently reconstructed **90,240 episode-condition records** across
  this run, the previous WR2 run and the TB control. Checked means, population
  standard deviations, group counts, truncated schedules, episode lengths,
  falls and nonfinite totals. **Zero audit failures**; maximum aggregate
  rounding discrepancy **2.35e−6**. All native and rollout metrics are finite.
- Every new support partition satisfies `single + double + neither = 1`
  and `neither = 1 − left − right + double`, within **8.20e−8**. All 25 native
  weighted rewards sum to return within **0.000465**. No paper report has a
  nonfinite episode.
- Saved YAML, recorded effective configuration and prepared YAML agree.
  All **227 commands × 1000 reference frames** pass recorded validation.
  The nominal replay independently matches model SHA-256
  `741d1e613d487b34073a10fbc0365906c3364914f37e70a394c45bf1e1a696c0`.
- Loaded all 15 native checkpoints with the training run's **Brax 0.14.1**.
  Each contains **25 finite leaves / 1,599,600 values**. Final native arrays
  exactly match the `params` export; array-content SHA-256 is
  `6308906a359c5acbab1361f03eb6017c7092b272db56c0b0e0f1143153f48fc6`.
  Initial network/normalizer arrays exactly match the previous run's initial
  arrays. These hashes describe arrays, not checkpoint-file bytes.

GPU training recorded Python **3.10.19**, Brax **0.14.1**, JAX/JAXlib **0.6.2**,
MuJoCo/MJX **3.14.0**, Flax **0.10.7**, Optax **0.2.8** and Orbax **0.11.39**.
Elapsed time was **2.33 h**, including **1.83 h** of native training wall time,
about **45,584 transitions/s**. Console stdout/stderr was not copied; the
structured logs and checkpoint set are complete.

The primary suite is **15 conditions × 128 episodes**, selection seed **12102**,
with deterministic policy, model/reset/actuator variation, pushes and observation
noise enabled. WR2 lasts **824 ticks / 16.48 s**, versus TB's 1000 ticks / 20 s.
Commands and timing are scaled by
`s = sqrt(h_WR2 / h_TB) = 0.824054757`; divide physical WR2 velocity/error by
`s` for the TB comparison. Tracking is each episode's mean instantaneous XY
vector error, then averaged across episodes. It is not velocity RMSE or the
error of mean velocity. Falls shorten measurements, so error must be read
alongside survival.

Support, forces and torque use the **end of each 20 ms control step**.
Neither foot exceeding 1 N does not alone establish geometric flight.
Absence of `best_params` and held-out confirmation is expected: no trained
candidate qualifies. The gate permits at most **19 falls / 1920**.

## 2. Checkpoint comparison

| Policy / transitions | Physical fixed XY error (m/s) | TB-equivalent error (m/s) | Falls / 1920 |
| --- | ---: | ---: | ---: |
| Previous WR2 0.28.0, best 200,089,600 | 0.120675 | 0.146441 | 37 (1.93%) |
| New WR2 200,089,600 | 0.098807 | 0.119904 | 120 (6.25%) |
| New WR2 240,107,520 | 0.091208 | 0.110682 | 164 (8.54%) |
| **New WR2 260,116,480: best tracking** | **0.088750** | **0.107699** | **102 (5.31%)** |
| **New WR2 280,125,440: fewest falls** | **0.089136** | **0.108168** | **95 (4.95%)** |
| New WR2 300,134,400: final | 0.088856 | 0.107827 | 96 (5.00%) |
| TB 180,080,640 | 0.038379 | 0.038379 | 3 (0.156%) |
| **TB 200,089,600** | **0.037025** | **0.037025** | **8 (0.417%)** |
| TB 300,134,400: final | 0.037999 | 0.037999 | 134 (6.98%) |

TB 180M/200M remain the useful controls; its final checkpoint has worse
survival. WR2's last 40M shows little further improvement. Native forward return
continues rising, from **372.27 at 260M to 384.80 at 300M**, without improved
benchmark tracking. It is insufficient for choosing the checkpoint.

A 20,000-resample paired bootstrap retains all conditions together for each
of 128 model/reset indices. New 260M minus previous 200M gives physical error
**−0.03193 m/s**, 95% interval **[−0.03396, −0.02992]**, and falls **+3.39
percentage points**, interval **[+1.46, +5.63]**. New 300M minus 260M gives
error **+0.000106 m/s**, interval **[−0.000934, +0.001108]**, and falls
**−0.313 points**, interval **[−1.719, +1.094]**. These conditional selection-set
comparisons use one training seed and do not establish held-out performance.

![Training, survival, nominal support and direction comparisons](../../results/wr2_walking/wr2_paper_20261016_222755_seed0/review_20261017/training_comparison.png)

## 3. Directional response and standing

All velocities/errors below are **TB-equivalent m/s**. Commanded-axis means
use X for forward/backward and Y for lateral motion. Standing uses total XY
speed; signed standing mean velocity is near zero despite substantial motion.

| Condition | New WR2 final axis velocity / standing speed | TB 200M axis velocity / standing speed | WR2 vector error | TB vector error | WR2 / TB falls |
| --- | ---: | ---: | ---: | ---: | ---: |
| Forward +0.10 | +0.0687 | +0.0905 | 0.1142 | 0.0373 | 16 / 1 |
| Backward −0.10 | −0.0503 | −0.0736 | 0.1157 | 0.0423 | 12 / 1 |
| Right −0.05 | −0.0091 | −0.0320 | 0.1051 | 0.0373 | 7 / 1 |
| Left +0.05 | +0.0090 | +0.0350 | 0.1088 | 0.0354 | 5 / 0 |
| Standing 0 | 0.0925 speed | 0.0320 speed | 0.0925 | 0.0320 | 2 / 0 |

WR2 final lateral response is only **18% of command**, versus TB's **64–70%**.
At WR2 260M it is **31% right / 12% left**, still weak. Forward response improves
late in training, but instantaneous error remains high. Priorities are lateral
control, stopping/standing motion and recovery, alongside aggregate tracking.

## 4. The learned gait changed substantially

Replayed forward and standing at 260M/280M/300M with nominal model, fixed reset,
no pushes and no observation noise: **2 conditions × 8 episodes × 824 ticks**
per checkpoint, with zero falls/nonfinite episodes. These are CPU diagnostics,
not qualification suites. Runtime was Python **3.11.14**, Brax **0.14.1**,
JAX **0.6.2**, MJX **3.14.0**, Flax **0.11.2** and Orbax **0.12.6**.

| Nominal policy | Forward error, TB-equivalent (m/s) | Standing speed, TB-equivalent (m/s) | Forward single / double / neither support |
| --- | ---: | ---: | ---: |
| Previous WR2 200M, corrected support evaluator | 0.08332 | 0.06695 | 9.71% / 29.13% / 61.17% |
| New WR2 260M | 0.05447 | 0.05038 | 99.27% / 0.73% / 0% |
| New WR2 300M | 0.05143 | 0.04407 | 99.39% / 0.61% / 0% |
| TB 200M | 0.02673 | 0.02471 | 84.98% / 15.03% / 0% |

Both new WR2 conditions maintain at least one foot above 1 N in every sampled
nominal step. The previous run's independent zero-force/positive-distance trace
established actual flight; the corrected evaluator still measured about 61%
neither-foot support. This comparison therefore controls the changed support
definition. Old randomized percentages using per-point thresholds should not
be directly compared with new summed-force percentages.

New WR2 is almost always in single support, with much less double support
than TB. Its nominal forward error remains **1.92× TB 200M**, and standing
speed **1.78×**. Ground contact alone does not establish the desired gait.

Within WR2, nominal forward sampled torque RMS drops from **1.517 to 0.785 Nm**;
slew-limited leg targets drop from **41.1% to 6.0%**. In the new randomized
forward suite, torque RMS is **0.802 Nm**, mean per-step peak torque **2.080 Nm**,
slew limitation **8.09%**, physical joint-limit clipping **0.103%** and bounded
action saturation **0.973%**. At final native forward evaluation, air time is
**49.9% of return**, versus 74.9% previously; XY tracking contributes 12.4%.
Reward shares describe this fixed-forward evaluation, not TB's different
native mixed-command evaluation.

TB nominal diagnostics retain native random heading and unconditional
per-episode motor-home offsets **±0.02 rad**; WR2 nominal reset is fixed.
TB uses its frozen Brax fork and MJX **3.2.4**. Repeated WR2 nominal episodes
repeat one trajectory, not eight independent training outcomes. Raw torque
also averages over **21 WR2 / 30 TB motors** with different gearing/hardware;
its cross-robot ratio cannot establish a physics error.

## 5. Controlled push and observation-noise ablation

For the final policy, retained the primary 128-draw model batch and selected
eight difficult indices **[80, 57, 97, 0, 102, 28, 64, 38]**. Replayed forward
and standing while keeping model/reset/actuator variation. Model arrays were
hash-identical and reset keys identical across the three profiles. Only push
magnitudes, then observation noise, were disabled.

| Profile | Forward falls / 8 | Standing falls / 8 | Forward physical error (m/s) | Standing physical speed (m/s) |
| --- | ---: | ---: | ---: | ---: |
| Full variation, pushes, noise | 3 | 4 | 0.10567 | 0.09925 |
| Same draws, no pushes | 0 | 0 | 0.05925 | 0.04678 |
| Same draws, no pushes or observation noise | 0 | 0 | 0.05203 | 0.04226 |

Removing pushes eliminates **7/16 falls** on these draws and substantially
reduces error; removing noise gives a smaller additional improvement. All
no-push episodes finish 824 ticks. This supports a push-recovery problem on
tested cases, while the residual error confirms a separate nominal gait gap.
It does not justify disabling pushes in the comparison or deployment policy.

This sample was deliberately selected for difficult cases: **it is not a
population fall-rate estimate**. CPU and GPU long trajectories can diverge
numerically despite the same protocol. The diagnostic full profile is not
an exact reproduction of every recorded GPU fall time.

In the primary final suite, 96 falls occur across 43/128 model/reset indices;
index 80 fails all 15 conditions. Several clusters occur after scheduled
pushes. The ablation provides stronger evidence than those timings alone.

## 6. Reference imitation and actuator tracking

A separate final-policy nominal forward trace records 824 end-of-control
samples using the actual reference, delivered target and simulator joint state.
Across the 12 leg motors:

| Measured mismatch | RMSE (rad) |
| --- | ---: |
| Actual motor position versus moving reference | 0.18214 |
| Delivered motor target versus moving reference | 0.19126 |
| Actual motor position versus delivered target | 0.05896 |

The requested targets themselves differ substantially from the reference;
servo lag alone cannot explain this trace. The RMSE components are not additive.
Actual foot sole-site heights peak at **6.7 / 8.0 mm**, versus a reference
peak of **34.0 mm**. The policy maintains support but does not reproduce the
reference swing lift. This is evidence about one nominal trajectory, not a
measurement of every randomized gait.

The fixed-home action range prevents exact reference targets in both robots.
An independent scan of frozen TB's native forward reference, using its actual
12 motor coordinates and 1000-tick schedule, gives:

| Native fixed-home ±0.25 rad target range | WR2 forward reference | TB forward reference |
| --- | ---: | ---: |
| Reference samples outside range | 8.25% | 12.44% |
| Minimum possible target-reference RMSE under those bounds | 0.08964 rad | 0.12923 rad |

TB succeeds despite this limitation and has higher raw forward leg-reference
RMSE than WR2. Native imitation is a soft reward; exact trajectory playback
is not required. These measurements do not support widening WR2's action range,
moving `q_home` with the reference, or changing imitation weight solely to
reduce joint RMSE. Different motor coordinates also limit cross-robot RMSE
interpretation. The TB scan is reference-only, not a new policy rollout.

## 7. Code-confirmed TB alignment and remaining physical differences

The comparison uses successful local TB run
`toddlerbot_walk_ppo_20261015_194347`, frozen locomotion source **1982ef3**,
Brax fork **14147806f6c25724ba830ba36764df3243889b0d** and actual episode exports.
The original TB provenance records a dirty source tree; the prior audit
matched the captured 11 locomotion files, 12 model/reference files and lock.
Files outside that captured set are not a complete historical snapshot.
The actively developed TB checkout is a different revision; it required no
changes for this review.

Confirmed against WR2 `walking_env.py`, `paper_rewards.py`, `paper_evaluation.py`,
`ppo.py` and `train.py`, and frozen TB `mjx_env.py`, `walk_env.py`, `paper.gin`,
`walk_zmp_ref.py` and `train_mjx.py`:

- Both use **moving leg-reference imitation, weight 5**, with static motor home
  for actions/observations and the native standing regularizer. Actor inputs
  include phase/command; reference errors enter critic/reward. The actor learns
  reference imitation through PPO advantages. Reference motor positions are
  not required as direct actor observations in this native TB implementation.
- WR2 uses native **Swish, state-dependent NormalTanh, 1024 × 20 rollouts,
  256 × 4 minibatches × 4 passes**, LR 1e−4 and the correct actual 0.25 × MSE
  value coefficient. The published 25 reward weights and 300M budget remain.
  The current Brax adapter checks do not show a new mismatch causing this gap.
- Reference/reward clocks, delayed action and old-action observation agree with
  frozen TB. Nonzero lateral commands get moving references. Native normalized
  deadzone boundaries were checked against TB; they do not explain the weak
  lateral response. Foot support now sums solver forces before the threshold.

Hardware and solver differences remain explicit rather than being replaced
with TB values:

| Parameter | WR2 HTD45H | TB hip pitch/roll | TB knee/ankle pitch |
| --- | ---: | ---: | ---: |
| Position gain | 24.15736 | 16.40625 | 21.09375 |
| Controller derivative gain | 0.5 | 0 | 0 |
| Joint damping | 0.17362248 | 0.075 | 0.08 |
| Armature | 0.02095098 | 0.0044 | 0.0012 |
| Friction loss | 0.46547120 | 0.252 | 0.134 |
| Torque ceiling (Nm) | 4.0 | 1.74 | 1.86 |
| Maximum motor speed (rad/s) | 5.81776 | 10 | 10 |
| Additional target slew ceiling (rad/s) | 5.81776 | None | None |

WR2 retains asymmetric braking and a motoring torque taper beginning at zero
speed; TB uses symmetric clipping with a constant-torque speed region. WR2
uses Newton 10 with 5 ms × 4 substeps; TB uses Newton 1 with 4 ms × 5.
Geometric time normalization gives WR2 a **0.60 s** reference cycle versus
TB's **0.72 s**, while WR2 motors have a lower speed ceiling. Matching body
scale therefore does not match actuator response. These differences are
plausible mechanisms to investigate, not established causes of the current gap.

WR2's identified fit was validated at loads through **1.0157 Nm for 3 s**;
the policy's mean per-step peak is about 2.1 Nm. Loaded actuator response and
braking need measurements before changing the identified model or gains.
Armature alone cannot establish complete robot bandwidth.

## 8. Recommended next experiment

**Keep recipe 0.28.2 and fine-tune the retained 260M checkpoint for another
300M transitions with seed 1.** This preserves the TB objective, full pushes,
noise, model variation and hardware limits while testing whether additional
experience on new training draws improves the supported gait's recovery.
The last 40M plateau means improvement is uncertain; this is a budget/seed
experiment, not a demonstrated fix.

From the WR2 checkout on the GPU host:

```bash
uv run --extra training python -m wr2.locomotion.train \
  --config wr2/locomotion/configs/ppo_walking_paper.yaml \
  --restore-checkpoint results/wr2_walking/wr2_paper_20261016_222755_seed0/checkpoints/000260116480 \
  --seed 1
```

Native restore loads actor, critic and observation-normalizer parameters;
**Adam state and training counters start fresh**. Frozen TB uses the same
checkpoint-parameter restore route. This is fine-tuning, not an exact optimizer
resume or an independent initialization seed. The new run logs 0–300M and
adds about 300M experience to the policy trained through 260M.

Retain every 20M checkpoint and evaluate the unchanged full suite. At 200M
and 300M added experience, compare fixed error, per-direction response,
standing motion, push recovery and support with the retained control. A useful
candidate first needs **≤1% falls**; TB parity additionally requires approaching
TB 200M's **0.0370 m/s normalized error** and **0.42% falls**, with satisfactory
directional/transition behavior. Confirm a qualifying candidate on held-out
seed **22102**. Higher native return alone is insufficient.

If an adequate additional budget preserves the same recovery/tracking
plateau, prioritize loaded actuator validation and matched response/foot-motion
diagnostics before introducing a WR2-specific reward or curriculum. No new
training was launched as part of this analysis.

## Evidence and references

- [Metric audit and all checkpoint/direction tables](../../results/wr2_walking/wr2_paper_20261016_222755_seed0/review_20261017/metrics_audit.json),
  [checkpoint audit](../../results/wr2_walking/wr2_paper_20261016_222755_seed0/review_20261017/checkpoint_audit.json).
- [Nominal 260M](../../results/wr2_walking/wr2_paper_20261016_222755_seed0/review_20261017/nominal_260116480.json),
  [280M](../../results/wr2_walking/wr2_paper_20261016_222755_seed0/review_20261017/nominal_280125440.json),
  [300M](../../results/wr2_walking/wr2_paper_20261016_222755_seed0/review_20261017/nominal_300134400.json).
- [Variation control audit](../../results/wr2_walking/wr2_paper_20261016_222755_seed0/review_20261017/variation_control_audit.json),
  [full variation](../../results/wr2_walking/wr2_paper_20261016_222755_seed0/review_20261017/variation_full.json),
  [no pushes](../../results/wr2_walking/wr2_paper_20261016_222755_seed0/review_20261017/variation_no_push.json),
  [no pushes/noise](../../results/wr2_walking/wr2_paper_20261016_222755_seed0/review_20261017/variation_no_push_no_noise.json).
- [Actual target/reference trace](../../results/wr2_walking/wr2_paper_20261016_222755_seed0/review_20261017/nominal_target_trace.json),
  [TB native reference range check](../../results/wr2_walking/wr2_paper_20261016_222755_seed0/review_20261017/tb_reference_action_box.json).
  Reproduction scripts, the raw trace and PNG/PDF figures are in the ignored
  run's `review_20261017` directory; the Git change contains text documentation.
- [Previous WR2 review and reproduced fixes](wr2_paper_result_20261016.md),
  [TB control/source audit](toddlerbot_transfer_review_20261016.md),
  [training and evaluation guide](wr2_paper_training.md).
- [ToddlerBot paper, Tables VI/VII](https://arxiv.org/html/2502.00893v2),
  [Tedrake et al., ZMP preview control, Humanoids 2015](https://doi.org/10.1109/HUMANOIDS.2015.7363473).
