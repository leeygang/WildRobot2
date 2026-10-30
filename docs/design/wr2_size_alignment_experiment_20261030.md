# Fresh size-alignment comparison — 2026-10-30

## Question and scope

Does matching TB's normalized learning durations and nominal foot-support
load fraction improve WR2's forward progression and alternating placement?
Prepare two **fresh 200M runs**, seed 3, using the current leg15 recipe.
Compare the existing settings against an explicit size-aligned candidate.

The [size audit](wr2_training_size_audit_20261029.md) verifies the active WR2
and local TB **`paper-walking-baseline`** implementations. The existing
continuations retain an almost fixed right-hip mean, and stronger leg imitation
did not fix placement; [the result review](wr2_leg_imitation_result_20261029.md)
supports testing fresh learning before another reward specialization.

The paired fresh control is necessary. Comparing the candidate only against
the completed continuation would change both initialization and recipe.
The candidate changes **four temporal settings plus one contact threshold**;
it is an integrated recipe comparison, not a single-variable attribution test.
Improved walking is a hypothesis, not a consequence established by dimensional
equivalence. The support-threshold audit changed at most .364% of existing
nominal foot samples, so it is unlikely to explain the large placement gap alone.

## Exact changes

Measured grounded WR2/TB CoM-height ratio is `L=.67906624178424`, giving
gravitational time ratio `s=sqrt(L)=.8240547565448791`. Native total masses
are 2.05832762/3.358176486kg, giving `M=.6129301508068508`.

| Setting | TB Paper | Fresh control | Fresh aligned |
|---|---:|---:|---:|
| Training episode ticks | 1000 | 1000 | **824** |
| Episode in TB-equivalent seconds | 20 | 24.270 | **19.999** |
| PPO rollout ticks | 20 | 20 | **16** |
| Rollout in TB-equivalent seconds | .400 | .4854 | **.3883**, nearest integer |
| Discount per tick | .97 | .97 | **.9637121761767793** |
| GAE lambda per tick | .95 | .95 | **.9396526367723493** |
| Foot-support threshold | 1N | 1N | **.6129301508068508N** |
| Threshold / nominal body weight | 3.035% | 4.952% | **3.035%** |

At the retained WR2 20ms tick, `gamma_WR=gamma_TB**(1/s)` and the same
exponent for lambda preserve discount and GAE time constants in TB-equivalent
seconds. Gamma's e-fold horizon becomes .65662 TB seconds, and the combined
gamma/lambda horizon becomes .24464 TB seconds, equal to TB's.

Support detection still sums upward solver forces on each foot/floor pair.
The threshold affects contact labels, rewards and critic features; it does
not change the contact physics or friction. It is a fixed nominal threshold,
as in TB, rather than a threshold recomputed for each randomized body mass.

Qualified air-time credit remains paid only after ten consecutive contact
samples. Nine intervals span .18 physical seconds/.21843 TB seconds.
The existing environment derives payment compensation from PPO gamma;
it changes **1.31539→1.39468**, preserving the credit's discounted value at
the original landing tick. The undiscounted payment therefore increases.
No extra air-time or clearance size multiplier is introduced.

## Shared recipe and declared exceptions

Both complete v0.34.0 configs preserve:

- Model-home hip-pitch action centers, grounded physical reset/observation home,
  and the fixed-center `.25rad * action` map.
- Desired-heading feedback and reference-frame XY tracking; XY and moving
  leg-reference weights **15/15**. TB uses 5/5; the WR2 XY increase has prior
  measured progression benefit, while absolute leg15 remains experimental.
- Sustained-landing air time500, collision1 and all other current weights.
  These are declared WR2 departures from the Paper recipe, not size conversions.
- SiLU512/256/128 actor and critic, state-dependent tanh Gaussian, LR1e-4,
  entropy.0005, clipping.2, four PPO epochs/four minibatches and batch size256.
- Existing scaled command ranges, command/push intervals, white noise,
  .60s gait, .0339533m swing height and relative domain randomization.
- 1024 environments, 16 evaluations ×128 episodes and all intermediate
  learned checkpoints; retain initial parameters separately.

Settings that should **not** receive a uniform multiplier:

| Setting | Reason for retaining it |
|---|---|
| Actual joint limits, gains, torque/speed caps, friction and inertia | Measured WR2 hardware; the robots are not geometric copies |
| 20ms control, one-tick latency, 5ms×4 physics | Controller/deployment contract; changing integration is a separate dynamics study |
| History15 / actor1005 / critic1515 | Existing inference contract; history13 requires a versioned deployment change |
| Torso height and foot-spacing limits | Already follow actual root/stance geometry; uniform CoM scaling would misclassify valid postures |
| Angular targets/noise and roll±.1rad, pitch±.2rad | Angles are dimensionless; narrower shoulder-motion tolerance changes the objective |
| Non-floor collision cutoff.1N | Local pair aggregation/effective masses differ; do not equate it with body-support load |
| Raw torque/power penalties and torque-conditioned observation bias | Hardware effort assumptions; geometric scaling would materially change their influence |
| Action rate/acceleration penalties | TB's delayed stencil is not a pure acceleration; no justified blanket time multiplier |

The [shoulder-width roll proposal](wr2_shoulder_rocking_review_20261029.md)
remains separate. A ±3.1° band matches visible size-normalized shoulder
displacement, whereas TB's existing reward specifies an angular band.
It can influence weight transfer and should be evaluated as an objective
change after this comparison. Neither fresh arm implements it.

## Diagnostic correction

At falls/timeouts, WR2 clears controller history before native Brax autoresets
the physical state. `previous_target` then contains reset home, rather than
the target just delivered. The hip-action and stance diagnostics formerly
compared against that reset value at finite timeouts, creating false failures.

Environment info now records `last_applied_target` on every tick, including
terminal ticks. Both diagnostics compare reconstruction against that value.
Controller history still resets normally. This metadata does not enter the
actor/critic, controller or reward. Finite terminal samples are retained;
invalid/padded samples remain excluded by the diagnostics' existing masks.

## Run commands

From the updated GPU WR2 checkout, run sequentially using the regular trainer:

```bash
uv run --locked --python 3.10 --extra training python -m wr2.locomotion.train \
  --config wr2/locomotion/configs/ppo_walking_size_control.yaml

uv run --locked --python 3.10 --extra training python -m wr2.locomotion.train \
  --config wr2/locomotion/configs/ppo_walking_size_aligned.yaml
```

Omitting `--restore-checkpoint` creates fresh actor, critic and Adam state in
both arms. Architecture, observation dimensions and seed are shared. Existing
GPU backend and full-reference preflight gates apply. Run folders use
`wr2_paper_size_control_*` and `wr2_paper_size_aligned_*`; the saved complete
configuration and provenance record the settings.

## Comparison and selection

Shorter rollouts change optimizer exposure. At requested200M/16 evaluations,
the native trainer rounds the budget to:

| Exposure | Control | Aligned |
|---|---:|---:|
| Actual transitions | 200,294,400 | 200,048,640 |
| Adam minibatch updates | 156,480 | 195,360 |

Use recorded transition counts and report updates and `transitions*.02/s`
TB-equivalent aggregate simulation seconds. The candidate has about25% more
updates per transition; it approximately restores TB's optimizer exposure per
normalized simulation second. This can increase compute cost. It is not an
LR change or a promise of faster wall-clock training.

Use the same independent evaluator, seeds12102/22102, both nominal and
randomized no-push profiles, and .10/.20/.30 TB-equivalent forward commands.
Its standard horizon is already824 physical ticks in **both** arms, independent
of the training episode horizon, ensuring equal comparison duration. The .30
condition remains outside the training command range and is a generalization
check. Evaluate retained intermediate policies rather than assuming final is best.

Priorities: requested-direction speed/distance and heading, per-foot alternating
landing leads and executed hip excursion, then stability/contact and root/shoulder
motion. Compare actual landing events and size-normalized placement, not cadence
or reward alone. Reclassify contact/landing events from recorded vertical forces
using the **same .612930N threshold for both arms**, alongside native labels,
so a detector change is not mistaken for a gait improvement.
With different gamma-derived air payments, episode lengths and
contact definitions, native returns are particularly unsuitable for ranking arms.

If both fresh arms improve right swing, that supports initialization/continued
learning as a contributor. If aligned wins over fresh control, that supports
the integrated normalized recipe, without identifying which setting caused it.
If neither closes the gap after200M, use their checkpoint trajectories and
executed joint/landing metrics to select a targeted objective study.

## Verification

All15 focused checks pass on the Mac: seven new checks plus eight existing
regressions. They cover complete-config isolation/round trips; TB-time constants;
nominal support load fraction and summed forces; once-only, discounted landing
payment; native824-tick timeout/autoreset; retained final target samples; and a
native16-tick PPO rollout, optimizer update and saved-checkpoint reload.
The holding diagnostic retains all824 finite samples with zero falls and zero
target-reconstruction error, including the timeout tick. The native PPO check
collects64 real WR2 transitions, performs four epochs/four minibatches, changes
actor parameters and reloads saved parameters exactly. Existing Paper action-delay,
autoreset, diagnostic masking and terminal-accounting checks, plus legacy
timeout/fall wrapper regressions, also pass. The new test module passes Ruff.
CPU integration uses actual WR2 physics/reference/IK with a reduced exact
standing/forward command grid. It verifies implementation, not GPU throughput
or walking quality; production training builds and checks the full command grid.

References: local TB **`paper-walking-baseline`** at
`a4f693d7cc4544f011dad23d6c353e752a6819ab`, `paper.gin`, `ppo_config.py`,
`mjx_config.py`, `mjx_env.py` and its pinned native Brax GAE default;
[ToddlerBot paper v2, Tables VI–VII](https://arxiv.org/html/2502.00893v2);
[measured size audit and source hashes](wr2_training_size_audit_20261029.md).
