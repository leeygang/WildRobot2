# Size-aligned continuation: 400M total — 2026-10-30

## Decision

**The additional200M did not improve the walking priorities. Keep the accepted
fresh200M policy as the walking baseline.** Standing and self-contact improve,
but aggregate walking tracking regresses, cadence stays fast and placement
remains uneven. Final is not the automatically selected checkpoint.

The longer-budget experiment has answered its intended question: another
unchanged continuation is not the next recommended intervention. This does
not prove that every possible longer run would fail.

Code review also identifies a specific information omission: the qualified
landing reward depends on internal air-time/credit counters which are absent
from the privileged critic input. A controlled probe confirms that omission.
It is **not yet established as the cause of fast cadence**. Test exposing this
reward memory to the critic before changing another reward coefficient.

## Verified implementation and result accounting

Run: `results/wr2_walking/wr2_paper_size_continue_20261030_153953_seed3`.

- 200,048,640 added transitions /195,360 added Adam updates; cumulative
  **400,097,280 transitions /390,720 updates**.
- Correctly restores the fresh run's `000200048640` actor, critic and
  normalizer. Initial parameter tensors equal the parent exactly.
  `restore_report.json` is null because no input expansion is needed.
- Complete saved config matches the parent's environment, rewards, network,
  PPO settings and randomization. Only version/name/output labels differ.
  Physics/model and actuator descriptions match exactly.
- Clean source **a624283**, same RTX5070/Python3.10.19/JAX.6.2/Brax.14.1/
  MuJoCo/MJX3.14.0 runtime and native GPU matrix precision as the parent.
  Adam, environment/RNG state and counters restart, following native restore.
- All15 retained checkpoints and all16 training/99 rollout records are finite.
  Audited36 reports /540 conditions /69,120 attempts. Episode flags, command
  vectors, sample partitions and summary recomputation agree. Maximum group
  mean/std rounding differences are2.38e-6/4.26e-7.
- Native evaluation reward components reconstruct total return within.000567.
  Time through final training evaluation is6,694.8s /1.86h.

The selected checkpoint is **+53,346,304**, cumulative253,394,944.
`best_params` equals that checkpoint, while final `params` equals
`000200048640` in this continuation, cumulative400,097,280. Every learned
checkpoint has worse selection-seed tracking than its own initial parent.

This is not an incorrect selector decision: source `receive_policy_params`
ranks learned checkpoints separately from `initial_params`. The selected
file records **`beats_initial_tracking: false`**. It is the least-error learned
candidate, rather than a promotion over the starting policy.

## Standard randomized GPU comparison

Use this continuation's re-evaluated parent and selected policy on held-out
seed22102: same128 attempts per condition, ten fixed nonzero commands through
.10 TB-equivalent m/s, standing and four transitions, with model/reset/actuator
variation and observation noise, without pushes. Both use824-tick episodes.
Small original-versus-recompiled parent differences are recorded separately;
the paired comparison uses the parent evaluation within this run.

| Held-out metric | Parent200M | Selected +53M |
|---|---:|---:|
| Desired-frame walking XY error, TB m/s | **.03964** | .04050 |
| Body-frame walking XY error, TB m/s | **.03975** | .04031 |
| Forward.10 desired-direction speed, TB m/s | **.09434** | .09084 |
| Forward.10 desired-frame error, TB m/s | **.03647** | .03835 |
| Forward-transition walking speed, TB m/s | **.09314** | .08544 |
| Fixed-walking leg-reference RMSE, rad | .16309 | **.16210** |
| Fixed-walking contact-phase agreement | **52.74%** | 52.37% |
| Fixed-walking heading error | **4.75°** | 4.88° |
| Fixed-walking self-contact | 4.14% | **2.03%** |
| Standing desired-frame XY error, TB m/s | .02096 | **.00857** |
| Standing double support | 58.79% | **92.12%** |
| Falls, held-out seed | **0/1,920** | 1/1,920 |

Selected desired-frame walking error worsens **2.17%**. Episode-index clustered
bootstrap change+.000859 TB m/s,95% [.000501,.001217]. Body-frame error worsens
.000564 [.000209,.000916]. Joint RMSE improves only.000991rad, about.61%.
Intervals describe evaluation variation for these policies, not training-seed
uncertainty. They do not show that better standing caused worse walking.

Selected no-push falls across both seeds are1/3,840; nonfinite count is zero.
Held-out push stress also worsens: **1,158/1,920 falls(60.31%)**, versus the
re-evaluated parent's1,065(55.47%). Pushes remain absent from training.

Final has selection-seed12102 reports, but no separate held-out confirmation.
On the same selection draws, parent→final:

- Desired-frame walking error.03970→.04086, **+2.92%**; paired change+.001158
  [95%+.000684,+.001635].
- Body-frame error.03980→.04105; leg RMSE.16332→.17666, about+8.2%.
- Forward.10 speed.09516→.08875; heading error4.40°→7.07°.
- Self-contact3.33%→1.37%; standing error.02112→.00634.

Final therefore has useful standing/contact gains but is not the walking
upgrade sought here. Its zero falls in this selection suite do not establish
held-out higher-command robustness.

## Original-policy Mac gait comparison

Generated12 nominal original-policy episodes at+13.3/+53.3/+106.7/+160/+200M,
commands.10/.20, plus.15 for selected/final. All **9,888 transitions are finite**,
with zero falls and exact target reconstruction through the timeout tick.
No learned action is replaced with a reference or a prescribed target.

The comparison preserves CPU/high precision, exact native reference rows,
seed22102,20 TB-equivalent seconds and1.5 TB-second warmup. Distances use
cached control-boundary collider centers; body motion uses integrated root
poses and actual shoulder-anchor geometry. Force classification is the same
.612930N threshold, with ten-sample qualified landings.

At.20 TB command, parent /selected /final:

| Nominal .20 metric | Parent200M | Selected253M | Final400M |
|---|---:|---:|---:|
| Desired-direction speed, TB m/s | .19129 | .18771 | .1916 |
| Median landing lead L/R, TB cm | **5.98/2.77** | 6.78/1.96 | 1.49/6.70 |
| Same-foot cadence, TB Hz | 2.060 | 2.060 | **2.169** |
| Executed hip span L/R | 10.82°/12.61° | 10.61°/11.72° | 13.06°/7.72° |
| Absolute net turn | **1.57°** | 7.39° | 6.59° |
| Torso roll5–95% span | 4.99° | 5.02° | **3.91°** |
| Root Z span, TB mm | **7.14** | 9.52 | 9.10 |
| Shoulder Z span L/R, TB mm | 14.53/12.16 | 16.98/13.86 | **14.50/8.82** |

Root bobbing worsens despite lower final roll. Shoulder motion should be
separated into common vertical translation and differential rocking when
judging visual improvement. All cases still alternate qualified landing events,
but alternation alone does not imply equal forward placement.

Lead is measured ahead of the opposite foot along the initial command heading.
It is different from same-foot stride and from a rotating torso-axis distance.
Heading changes can affect this projection. The+160M policy also has right-
biased1.68/5.25cm leads with only1.01° net turn, so the side change is not
explained solely by final heading drift. Its root yaw at qualified landings
ranges−1.57°…+2.60°. Actual root trajectories are retained.
Different policies on identical physics produce different placement biases;
the evidence does not identify an immutable right-leg limitation.

The+106.7M policy is a secondary nominal candidate: speed.2103, leads3.61/5.57cm,
net turn2.59°, roll4.41° and shoulders14.58/12.12mm. It overshoots the command,
still uses2.169Hz cadence, and has worse aggregate tracking than the parent.
It is not automatically promoted on the basis of that single nominal sample.

### Cadence settles close to the minimum landing qualification

Reference is30 physical ticks/.60s, **1.373 TB Hz**. Current actual same-foot
cycles are19–20 ticks, **2.06–2.17 TB Hz**. Complete final.20 support intervals
are10 or11 samples on both feet: left22×10/18×11; right16×10/24×11.
They meet the ten-sample reward condition with little extra supported time.

The qualified-air rule requests a minimum stable contact duration and pays
supported flight seconds. It does not request a30-tick cycle or equal forward
placement. The existing reference-contact term requests phase matching, but
match remains about52–53% and the faster rhythm survives another200M.
These observations support investigating timing incentives; they do not prove
that air-time weight500 alone is the cause.

## Source-confirmed critic reward-state omission

`qualified_air_time_step` maintains three two-foot arrays:

- `supported_air_time`: accumulated supported swing duration;
- `pending_air_time`: landing credit awaiting sustained support;
- `stance_steps`: progress toward the required contact samples.

They determine future landing payments. However, `_paper_observation` sees
motor/reference errors, velocities, current/reference contact, push signals
and heading; **none of those three reward-memory arrays is passed to it**.
Actor1005 and critic1515 are15-frame histories of67/101 features.

Original TB Paper air time pays on first filtered contact, without this
pending-credit stage. Its reference swing is about12 ticks, within a15-frame
history. WR2's roughly9–10 flight ticks plus9 payment-delay ticks can put the
beginning of the rewarded swing outside the14-interval history span. Histories
and reference phase can provide indirect estimates, so missing explicit
memory does not imply that every value prediction is wrong.

A controlled probe uses the200M parent, identical current physics/history and
otherwise matching state, but pending credit.18 versus.24s. Rebuilding the
observations through the production observation builder gives identical actor
and critic inputs. The next physical transition and next observations match
exactly, while next reward differs **.8368068**, matching
`(.24-.18)*1.3946778*500*.02`. The critic returns the same value14.6187744.

This is an **information-availability probe**, with deliberately varied
internal reward memory. It does not claim that two exactly identical complete
production histories with those values were observed, quantify alias frequency,
or establish the cause of the cadence. It confirms that the critic has no
direct access to a state variable controlling deferred payment. That can make
value/GAE estimation harder and merits a focused correction/test.

## TB comparison and next priority

Local **`paper-walking-baseline`** confirms phase-clock actor input, moving
leg-reference reward, contact-reference matching and Paper XY/leg/contact
weights5/5/1. WR2 retains15/15/1, its qualified-landing extension and normalized
learning horizons. Reference-time indexing uses rounded20ms samples; actor
phase remains `sin(phase),cos(phase)`. This review finds no cadence factor
error in those clocks or a wrong checkpoint/parameter restore.

TB200M body-frame error is.03702, with eight falls in its native15-condition
suite. WR2's selected.04031 remains about8.9% higher; models/randomization
differ, so this is not a controlled robustness ranking. TB300M has234,480
updates versus WR2 cumulative390,720. The remaining gap is not a simple lack
of raw training transitions or optimizer updates.

TB constructs a cosine schedule but its counter is Adam updates against50M
updates: at300M it ends near9.999512e-5. WR2's fixed1e-4 is effectively matched.
Continuation KL averages.0226–.0303; metrics/weights remain finite.
There is no evidence here requiring a new trainer/distribution/activation.

**Recommended next experiment:** retain the parent, expose the current six
reward-memory values to the privileged critic, and repeat a matched200M
continuation with the current reward formulas/weights preserved. Keep actor
inputs, deployment contract, reference, physics and command ranges unchanged.

Use TB-equivalent seconds for the four duration/credit values and bounded
qualification progress for the two counters. Six current values can be appended
to the existing critic history, making critic input1521 while actor stays1005.
Initialize new critic input weights to zero and verify unchanged initial actor
and critic predictions. The completed v0.35.0 run supplies the unchanged
comparison from the same parent; a repeated control is unnecessary.

Check actual on-policy value/TD residuals around qualified payouts as well as
tracking, cadence, placement and body motion. Improved information is not a
guarantee of slower or more symmetric walking; the unchanged objective may
still prefer the faster strategy.

If cadence persists after that correction, the next scalar reward probe is
existing `feet_contact`1→3: it restores TB's contact/XY and contact/leg coefficient
ratios while retaining XY/leg15. This is a relative-emphasis experiment, not
an automatic size scaling rule or complete TB weight parity. Its contact/air
ratio becomes three times TB's. Keep it separate from the critic-input test
so the cause of any change can be interpreted.

References: local TB **`paper-walking-baseline`** at `a4f693d`, `paper.gin`,
`walk_env.py`, `mjx_env.py`, and pinned Brax `1414780` optimizer construction;
[TB paper v2, Tables VI–VII](https://arxiv.org/html/2502.00893v2);
[Asymmetric Actor Critic for Image-Based Robot Learning](https://arxiv.org/abs/1710.06542),
whose full-state critic rationale motivates exposing simulator reward memory;
[accepted200M review](wr2_size_aligned_result_20261030.md).

## Artifacts and policy to retain

Ignored `review_20261030/` contains audited `analysis.json`, `gait_analysis.json`,
12 nominal traces with index/provenance, contact-run distributions,
`landing_critic_probe.json`, reproducible scripts and comparison PNG/PDF.
The proposal was not implemented or trained as part of this review. Its
[implementation and matched 200M experiment are now prepared](wr2_landing_critic_experiment_20261030.md).

Continue viewing the accepted200M baseline on the Mac:

```bash
uv run --no-sync mjpython -m wr2.tools.view_walking \
  --run-dir results/wr2_walking/wr2_paper_size_aligned_20261030_085111_seed3 \
  --command .2 0 0 --playback-speed 1
```
