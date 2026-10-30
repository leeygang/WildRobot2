# Leg-reference imitation result — 2026-10-29

## Decision

**Increasing leg-reference weight 5→15 did not solve uneven placement.** It
preserved tracking better than the unchanged continuation and reduced some
contact/heading errors. It did not improve joint-reference RMSE, produce a
meaningful right hip swing, or make right-foot passing comparable to the left.
Retain the starting model-home +173M policy as the placement comparison;
do not promote the stronger-reward final checkpoint as a gait fix.

The next training-setting experiment should test **fresh initialization of
the same current recipe**, rather than another imitation-weight increase.
All four continuations examined so far inherit the nearly fixed right-hip
strategy. A policy-parameter audit confirms its persistent posterior mean
bias and attenuated action-map sensitivity. That supports an initialization
experiment; it does not prove fresh training will fix the gait.

## Verified experiment

| | Weight 5 control | Weight 15 treatment |
|---|---|---|
| Folder under `results/wr2_walking/` | `wr2_paper_leg_imitation5_20261029_105135_seed3` | `wr2_paper_leg_imitation15_20261029_173955_seed3` |
| Added transitions | 200,294,400 | 200,294,400 |
| Automatically selected added steps | 13,352,960 | 186,941,440 |
| Time through final training evaluation | 1.90 hours | 1.98 hours |

Both record clean source **13a6229**, seed 3 and the same RTX 5070/Python
3.10.19/JAX .6.2/Brax .14.1/MJX 3.14.0 runtime with default matrix precision.
Complete saved configs differ only in leg-reference weight and run labels.
Reference, model hash, actuator settings, action centers, observations, reset,
command distribution and PPO settings match. Initial actor, critic and
normalizer tensors equal the shared model-home **+173,588,480** checkpoint.
Both start with fresh Adam/counters.

All fifteen retained checkpoints per arm are finite. `best_params` and final
`params` match their declared checkpoints. Recomputed **72 reports /1,080
conditions /138,240 attempts** reproduce selector decisions, finite flags,
lengths, command identities, sample partitions and per-episode statistics.
Maximum group mean/std rounding differences are 3.06e-6/5.43e-7. All recorded
training and evaluation metrics are finite.

Native reward application is correct: selected weight-15 leg contribution is
−13.94 versus weight-5 −4.48, with corresponding native forward evaluation
returns 655.57/660.28. Returns cannot be compared as policy quality because
their objective coefficients differ. The formula remains executed joint MSE
against the moving reference; no static-home imitation or action override was
introduced.

The audit artifacts are in the treatment folder's `review_20261029/`:
`analysis.json`, `nominal_gait/`, `policy_exploration/report.json`, and
`comparison.png`/`.pdf`. These are ignored local results, not Git additions.

## Standard GPU comparison

The standard suite has ten fixed nonzero XY commands through .10 TB m/s,
standing and four stop/restart schedules, all zero yaw. Deterministic policies
use observation noise and model/reset/actuator variation, without pushes.
Values below use held-out seed 22102. Errors/speeds are **TB-equivalent m/s**;
forward speed uses native cached velocity in the desired-heading frame over
the full episode.

| Metric | Common starting policy | Weight 5 selected | Weight 15 selected |
|---|---:|---:|---:|
| Desired-frame XY error | .04295–.04297 | .04475 | **.04258** |
| Body-frame XY error | .04339–.04341 | .04483 | **.04259** |
| Forward .10 speed | **.0923–.0924** | .09016 | .09057 |
| Forward .10 desired-frame error | .04348–.04358 | .04636 | **.04240** |
| Forward transition walking speed | **.0913** | .09057 | .08720 |
| Fixed-walking heading error | 5.73° | 4.87° | **4.49°** |
| Fixed-walking self-contact | 3.09–3.11% | 2.37% | **1.58%** |
| Fixed-walking leg-reference RMSE | **.2106rad** | .2145rad | .2171rad |
| Fixed-walking reference contact-phase agreement | **52.6%** | 50.92% | 43.47% |
| Falls, selection + held-out seeds | 0/3,840 | 2/3,840 | 1/3,840 |

The small initial ranges reflect separately compiled GPU repeats of identical
starting tensors; these are not distinct initial policies.

Against selected control, weight 15 lowers held-out desired-frame error 4.8%:
paired change **−.002167**, bootstrap 95% [−.002616,−.001731]. Body-frame error
also improves, and self-contact falls .78 percentage points. Forward speed
is essentially unchanged; its paired interval includes zero. Reference
contact-phase agreement is 7.45 points lower and joint RMSE is .00255rad higher.

Against its own starting policy, weight 15's desired-frame gain is only
**−.000371** [−.000764,+.000028], including zero. Body-frame error improves
.000803 [−.001190,−.000411]; forward-.10 speed decreases .00184
[−.00286,−.00079]. Joint RMSE increases **.00649rad**, about 3.1%, and contact
phase agreement falls 9.13 points. Thus the apparent advantage over control
partly comes from the unchanged continuation regressing.

All control checkpoints have worse aggregate tracking than the starting
policy. Its `best_params` is the best *learned* checkpoint, not the best
including the parent: `beats_initial_tracking` is explicitly false. The
trainer keeps initial tensors separately. Do not treat the control's automatic
selection as a promotion over its parent.

At equal final budgets, weight15/weight5 desired-frame errors are .04316/.04751
on selection seed 12102, but forward speed is **.0804/.0867**. Selected and
final policies must remain separate. Final checkpoints have only inline
seed-12102 reports; held-out confirmation refers to the selected checkpoints.

Recorded training-rollout MSE also fails to improve with weight 15: its early
0–20M mean is .0400 and late 160–200M mean .0420, versus control .0398→.0396.
These are averages of logged rollout statistics, not a substitute for the
fixed-command benchmark. The failure is not confined to benchmark joint RMSE.
Native logged KL ranges .031–.074 for treatment and .029–.086 for control;
learning rate remains 1e-4. There is no observed nonfinite optimizer failure.

Push stress confirmation has 855/1,920 treatment and 1,032/1,920 control falls.
Training is push-free; these stress results do not establish push robustness.

## Actual gait: native Mac replay

Eighteen nominal episodes replay the parent, control selected/final, and
treatment +133M/+187M/final at .10/.15/.20. All finish 824 ticks without falls
or nonfinite states. Production policies/mappings are used with no action
replacement, noise, randomization or pushes. Mac uses Python 3.11/Brax .14.2,
high matrix precision and one episode per command; this is local gait evidence,
not GPU higher-command qualification. Motion means exclude 1.5 TB seconds.

At .20:

| Policy | Integrated forward speed | Left/right landing lead, TB cm | Left/right hip target span | Eligible same-foot cycle frequency, TB Hz |
|---|---:|---:|---:|---:|
| Starting model-home +173M | .1623 | 8.23/**2.01** | 27.12°/1.68° | 1.53/1.53 |
| Weight5 selected +13M | .1673 | 12.33/−1.05 | 27.36°/.24° | 1.42/1.42 |
| Weight5 final | **.1805** | 8.16/1.29 | 25.92°/.96° | 1.83/1.79 |
| Weight15 +133M | .1733 | 10.60/−.57 | 24.24°/.72° | 1.65/1.65 |
| Weight15 selected +187M | .1665 | 9.16/**.23** | 24.71°/1.68° | 1.72/1.72 |
| Weight15 final | .1562 | 8.90/−.18 | 22.56°/.96° | 1.72/1.72 |

Spans are p95−p5, not full min/max. Lead compares the landing foot center
against the other foot center. The same-foot cycle frequency counts successive
eligible landings of one foot; regular alternating step frequency is about
twice that value. Eligibility requires supported air, >1mm sole clearance
and ten sustained contact samples. Contact and geometry share the native
cached final-substep sampling point; root progress uses integrated qpos.

The selected weight15 policy's right lead decreases **2.01→.23cm**, while
left lead increases 8.23→9.16cm. Right target span is unchanged at 1.68°;
executed right span decreases **1.44→1.15°**, versus left 24.21°. Mean
same-foot strides shorten about 10.48→9.66cm and cycle frequency rises.
This is more frequent, shorter stepping with persistent unequal placement.

At .10, selected weight15 right lead remains negative, −1.17cm, versus parent
−1.02cm. Its final right leads are negative at all three commands:
−2.81/−1.11/−.18cm. Both feet still advance and make sustained ground contact;
negative lead does not mean a stationary foot or a foot hanging permanently.

There are secondary improvements. At comparable .20 achieved speeds,
selected weight15 torso roll span decreases 12.25→9.37° and self-contact
5.12%→.66%; normalized root-height span remains about 11.1–11.3mm. The +133M
policy is particularly straight in this nominal .20 sample (.22° net turn),
but its right foot lands behind the left. None passes the primary placement
goal merely by improving stability, heading or aggregate tracking.

## Policy bias and TB comparison

Exact Gaussian logits were captured alongside nominal native .20 physics for
parent, control final and weight15 selected/final. Those physics traces agree
with their original replay within 1e-6. The deterministic target is
`q_center + .25*tanh(mu)`; local target sensitivity to raw mean is
`.25*(1 - tanh(mu)²)`.

| Right hip-pitch statistic | WR2 weight15 selected | Native TB Paper +200M forward samples |
|---|---:|---:|
| Mean tanh sensitivity factor | **.0495** | .7621 |
| Mean raw Gaussian standard deviation | .3116 | **.0370** |
| Mean physical forward residual from action center | **−13.95°** | +5.52° |

TB joint signs are opposite WR2's; the last row uses each robot's verified
physical forward convention. TB data are 25,761 native GPU stochastic-training
samples with varied positive forward commands. WR2 data are 762 nominal
post-warmup ticks at fixed .20. This is a qualitative policy-strategy
comparison, not matched-runtime causal evidence.

The right mean sits close to the posterior action endpoint throughout the
reference cycle. Its four phase-bin means span only about **.07°**, versus
about **6.83°** in TB's forward sample set. TB has a coordinated phase-dependent
mean despite smaller raw noise. WR2's local action-map sensitivity is about
15 times smaller; the inherited mean strategy remains the concrete difference.

This does **not** prove exploration is absent. Analytic single-tick probability
of a requested change of at least 5° is .0555% for WR2 selected right hip,
versus .0259% for TB right hip. These probabilities are conditional on sampled
states, not probabilities of discovering a coordinated gait. They do not
justify simply increasing raw entropy/noise, and the dataset contains no
actual multi-epoch on-policy PPO-gradient measurement for these new checkpoints.

Local **`paper-walking-baseline`** at `a4f693d` confirms moving leg imitation,
fixed model-home residual targets, and Paper XY/leg weights 5/5. Current WR2
uses 15/15; the experiment restores a coefficient ratio, not exact TB weights.
TB +200M body-frame error is .03702, versus new WR2 selection .04259, still
about **15% higher**. TB joint RMSE is .2318rad, above WR2 .2171rad; that scalar
joint metric alone cannot establish gait parity. Dynamics and randomization
remain different. TB's successful run starts fresh; these WR2 comparisons
all continue a policy that already has the asymmetric strategy.

Leg imitation is an RL reward on executed joints, not a supervised actor-angle
loss. Multiplying it changes sampled returns; it does not directly replace the
actor's right target with the reference. The total return, critic, exploration
and shared network update determine the learned mean. The observed failure
therefore rejects this weight change as a sufficient fix, not the relevance
of reward design generally.

## Next training-setting test

Test **fresh actor/critic/normalizer initialization with the existing weight15
recipe for 200M**, using TB's ordinary fresh-network initialization. Keep
reference, physics, rewards, commands, network/distribution and optimizer
settings identical. The completed weight15 continuation provides its 200M
comparison; another unchanged continuation is unnecessary.

This tests whether the asymmetric starting policy prevents discovery of a
balanced strategy. It changes initialization, including critic initialization;
it is not an actor-only reset. The current recipe has not been trained fresh:
the earlier fresh XY15 run used grounded hip centers and leg weight5.
Fresh initialization creates zero-biased mean heads and the configured
`.694147` initial scale on zero inputs, rather than retaining learned corner
means/scales. The current factory's fresh zero-input mean/scale were checked
numerically. Actual nonzero observations can change those initial outputs.

Use the existing regular trainer without a restore argument:

```bash
uv run --locked --python 3.10 --extra training python -m wr2.locomotion.train \
  --config wr2/locomotion/configs/ppo_walking_leg_imitation15.yaml \
  --run-id wr2_paper_leg_imitation15_fresh_seed3
```

This is the next recommended experiment, not a launched run. Retain
intermediate checkpoints and evaluate actual right swing, landing lead,
bilateral stride, commanded progress and falls. If fresh training also retains
the same strategy, initialization is insufficient and the next training
investigation should measure phase-dependent PPO credit/updates before another
reward or exploration adjustment.

References: [TB paper v2, Tables VI–VII](https://arxiv.org/html/2502.00893v2),
local TB Paper `paper.gin`, `mjx_env.py` and archived PPO diagnostic;
[experiment setup](wr2_leg_imitation_experiment_20261029.md).
