# WR2 reward direction: short taps versus useful steps — 2026-10-22

## Findings

**The reward permits the observed tapper to earn substantial return with little
commanded translation.** Its air-time term rewards unloaded foot-time collected
at contact, while reference imitation and phase matching provide weak incentives
to complete a normal step. This supports prioritizing the reward investigation.

Repeating the existing learned and reference controls under the same WR2 model
also establishes a limit: the reference control makes long steps but has large
lateral motion and lower stability reward. It is not a balanced walking policy.
Removing its penalties would not establish TB parity.

The strongest candidate direction is **credit for a completed single-support
step**, including a sustained landing. Offline qualification rejects most
current right-foot tap credit and preserves the copied TB policy's credit.
However, changing payment timing matters: the simple candidate reverses raw
episode ranking but not discounted return from reset. It is not ready to select
as a training correction. Training weights, observations and physics remain
unchanged in this review.

## 1. Source and comparison scope

Read WR2 [reward terms](../../wr2/locomotion/paper_rewards.py) and
[`_paper_step`](../../wr2/locomotion/walking_env.py), and local
`paper-walking-baseline:toddlerbot/locomotion/{walk_env.py,mjx_env.py,paper.gin}`,
revision `a4f693d`. Both use air coefficient **500**, XY **5**, contact **1**,
leg imitation **5**, action rate/acceleration **.05** and control `dt=.02`.
Current WR2 retains its approved collision coefficient **1**, versus TB **.1**.
TB's known action-index and slip-mask bugs remain declared differences.

The accounting below repeats the already recorded WR2 nominal controls:
learned, reference hip rolls, and reference all leg angles. It does not change
reward coefficients, model parameters or invent a new controller. Native
reference substitution, action delay, quantization, bounds and slew remain.
All three complete **824 ticks /16.48 physical seconds /19.999 TB seconds**.

The repeated controls run on the existing Mac CPU profile (Brax .14.2,
JAX .6.2, MuJoCo/MJX 3.14.0, Flax .11.2, high matmul precision). Their net
speeds and contact patterns reproduce the copied GPU results closely. The
repetition adds all 25 reward components; sums reconstruct native total reward
within `1e−5` per valid tick. Native GPU data remain the runtime confirmation;
CPU/GPU trajectories are not claimed bitwise identical.

## 2. What air time actually rewards

With a fixed eligible walking command, each foot accumulates unloaded time,
then collects that credit when contact returns. The native counter and payment
are audited directly against the recorded contact sequence.

For the learned forward recording:

- Aggregate unloaded foot-time: **16.44 foot-seconds**.
- Paid credit: **16.34 foot-seconds**.
- Remaining unpaid at the recording end: **.10 foot-seconds**.
- Weighted air return: `500 × .02 ×16.34 = 163.40`.

Thus raw air return approximately rewards the amount of unloaded foot-time,
with finite-horizon credit still waiting for contact. It requires no sustained
stance, horizontal placement or appropriate opposite-foot support. A foot that
never lands earns no landing credit; brief taps realize the accumulated credit.

Increasing contact count alone does not multiply raw air reward for the same
unloaded time. With full eventual payment, the average raw contribution is
approximately `500*dt*(single_support_fraction + 2*no_support_fraction)`.
A reference-like 2/3 single-support, 1/3 double-support signal yields about
**6.67 reward/s**; continuous single support yields **10/s**. These are native
physical-second accounting rates, not predictions of feasible full gaits.

The existing 20M trace has **33.9% no support**. Its raw air return is **194.6**;
crediting only unloaded time while the opposite foot is loaded would yield
**83.0** on that same trace. The corresponding 40M values are **206.2 /95.8**.
This is direct evidence that early reward includes flight credit, not just
normal alternating swings. At 300M, no support is zero and almost all support
is single: the policy has changed to tapping while largely eliminating double
support.

## 3. Reference incentives are weak at the achieved angle scale

The learned forward return includes:

| Component | Native episode contribution |
| --- | ---: |
| Air time | **+163.40** |
| Angular XY + Z stability | **+102.80** |
| XY tracking | +39.88 |
| Contact/reference matching | +16.94 |
| Lateral foot spacing | +16.48, at its maximum |
| Leg-reference imitation | **−2.74** |
| Action rate + acceleration | **−.185** |
| Energy | −.547 |
| Foot slip | −.0011 |

The leg imitation coefficient 5 multiplies a **mean squared radian error**,
not a value of the same scale as the bounded XY reward. For example, a 10°
error on every leg coordinate costs about **.152 reward/s** before the final
`dt` multiplication. It is soft guidance rather than enforced reference motion.

The binary contact reward averages **1.028 matches/tick** for the tapper.
Always keeping only the left foot loaded would give **1.011** on the same
reference schedule. The actual tapper therefore gains only about **.28 episode
return** beyond that fixed-support contact pattern. The coefficient and matching
rule do not strongly distinguish useful alternating stance here.

Spacing is already saturated and measures lateral separation. Slip and energy
costs are small. None of these terms directly rewards forward foot placement.

### Illustrative competition among air, XY and contact

Score exact reference contact plus perfect commanded velocity as synthetic
signals on the same timeline. This illustration includes **only three terms**;
it is not a feasible robot trajectory or a global optimum of the full reward.

| Three-term contribution | Current tapper | Exact reference contacts + perfect XY |
| --- | ---: | ---: |
| Air time | 163.40 | 108.00 |
| XY | 39.88 | 82.40 |
| Contact | 16.94 | 32.96 |
| Sum | **220.22** | **223.36** |

Perfect tracking and exact reference contact gain only **3.14** in these terms:
the air-time loss consumes almost all their benefit. Additional movement/stability
costs can erase that small margin. At air coefficient 250, the same illustrative
margin is **30.84**, but this arithmetic does not establish a successful new
policy or select that coefficient for training.

## 4. The model can make long steps, but the reference control sways

| Measure | Learned | All-leg reference control | TB Paper learned GPU |
| --- | ---: | ---: | ---: |
| Net forward speed, TB m/s | .0200 | **.1159** | approximately .09 |
| Median right load, TB seconds, after warmup | **.049** | **.388** | .340 |
| Median right foot-center advance, TB mm | **2.95** | **85.20** | 63.53 |
| Body lateral velocity std, TB m/s | .037 | **.234** | .021 |
| Native total WR2 return | **367.31** | **308.13** | Not compared across models |

This establishes that the current WR2 model can execute recurring long contacts
and significant commanded-axis translation through reference control. It does
not prove that a learned policy will discover a balanced version of that gait.

The full-reference control loses **59.18** return against the tapper:

- Air time: **−15.60**.
- XY: **−21.35**.
- Angular XY + Z: **−43.92**.
- Contact alignment: **+11.42**.
- Leg imitation: **+1.30**.
- Remaining terms partially offset those losses.

The larger XY loss is specifically dominated by the lateral factor. The
native Gaussian is separable:

`5*exp(-100*(vx-command_x)^2)*exp(-100*(vy-command_y)^2)`.

| XY accounting | Learned | All-leg reference |
| --- | ---: | ---: |
| Forward factor alone | 43.62 | **73.24** |
| After the lateral factor | **39.88** | 18.53 |
| Forward reward removed by lateral factor | 3.75 | **54.71** |

Increasing the existing combined XY coefficient would increase the current
score gap in favor of the tapper on these frozen trajectories. Air coefficient
zero would also leave the tapper about **43.58** ahead. Consequently, this
particular comparison cannot justify an air-only or XY-weight-only fix.

The lateral motion is not purely a torso-origin measurement artifact. An
independent kinematic reconstruction of whole-body CoM and finite differences
of its position gives lateral velocity std approximately **.171 TB m/s** for
the reference control, versus **.030** for the tapper and **.026** for copied
TB walking. These finite-difference values are supplementary motion diagnostics,
not exact replacements for native end-of-substep reward velocities. A weaker
lateral penalty could accept a substantially swaying gait rather than produce
TB-like walking.

## 5. Candidate direction: qualify the completed step

Offline, retain the air coefficient and other components but change which
recorded air credit qualifies:

1. Accrue eligible foot-air-time only while the opposite foot is loaded.
2. Save that credit at landing; pay it once the landing foot has sustained
   consecutive sampled contact for a minimum interval.
3. Cancel unpaid credit if contact breaks before that interval.

This is a candidate reward definition, not current training code. Single-support
qualification addresses the early flight credit; sustained landing addresses
the current short taps. A lower bound on air duration alone can still reward
long hovering followed by a brief touch. Sustained support is the missing
property identified by the visual/GPU review.

For an illustrative **.10 TB-second** stance interval, WR2 physical duration
is `.10*s=.0824 s` (five recorded 20 ms ticks). The copied TB policy uses `.10 s`.
This is a discrimination example, not a selected final threshold.

| Offline rescore | Learned tapper | All-leg reference |
| --- | ---: | ---: |
| Original air return | 163.40 | 147.80 |
| Qualified air return | **40.60** | **146.20** |
| Original total | **367.31** | 308.13 |
| Qualified total | 244.51 | **306.53** |

The right foot's **122.8** air return is eliminated at this interval, while the
reference control retains almost all credit. The copied TB learned trace
retains **all 169.40 air return** at .10 and .20 TB-second qualification, and
166.8 at .30. This discrimination is grounded in the actual load intervals.
It does not validate a new policy, threshold population or robustness profile.
A qualified stance still need not include useful horizontal placement;
commanded motion remains an evaluation requirement.

### Discounting is a required gate before implementation

PPO uses `gamma=.97`. Original discounted finite return from the shared reset
is **14.77 learned /9.86 reference**. The .10 TB-second candidate gives
**10.39 /9.10**. **Its raw rank reverses, but the discounted reset rank does not.**
Air time contributes **2.27** of the original **4.91** discounted gap; tracking
and angular stability account for most of the rest. These deterministic return
calculations are not PPO gradient attribution or training-randomization estimates.

Payment is delayed, so raw TB credit preservation also does not imply unchanged
TB learning incentives. A correction must account for timing and must be
checked against both discounted return and stance/movement quality. The current
diagnostic does not justify launching a new run with this simple candidate.

Raw equal-duty timing also deserves attention: two synthetic patterns with
75% unloaded time, periods 8 versus 32 ticks, both collect 19.2 raw air reward
in 128 ticks. Discounted from a common lift onset, they collect **4.53 /3.64**.
Earlier realization can matter, but uniform stationary phase averaging gives
the same average raw air reward. This illustration does not establish an
arbitrary-state frequency preference or predict the observed 6 Hz cadence.

## 6. TB and public implementations

The native TB code has the same unqualified landing-credit formula and weight
500. Its successful policy combines substantial air return with good tracking:
its copied forward trace earns **169.4 air /92.9 XY**, versus WR2 **163.4 /39.9**
over their respective nominal horizons. These native reward totals are not
cross-model quality scores. They show that high air credit can coexist with
useful walking, and that the common formula does not guarantee that outcome.

Public implementations help distinguish the objective's intended properties:

- [Legged Gym `_reward_feet_air_time`](https://github.com/leggedrobotics/legged_gym/blob/master/legged_gym/envs/base/legged_robot.py)
  pays `(air_time - .5)*first_contact`, so short flights can be penalized rather
  than rewarded. The .5-second quadruped default must not be copied to WR2.
- [Isaac Lab v2.2 biped rewards](https://github.com/isaac-sim/IsaacLab/blob/v2.2.0/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/mdp/rewards.py)
  include single-stance gating and duration bounds. That exact biped term does
  not require a completed landing, so it is not the proposed delayed-payment
  definition and does not by itself exclude permanent one-leg balance.
- [ToddlerBot, Tables VI–VII](https://arxiv.org/html/2502.00893v2) supplies the
  present PPO and coefficient baseline. A stance qualification would be a
  declared WR2-specific implementation change, motivated by measured taps.

## Next work and artifacts

Prioritize step-credit validity and its interaction with body tracking/stability.
Keep physical parameters and the training recipe fixed while finishing the
qualification/timing analysis. Retain useful stability penalties when judging
the prescribed reference control. Choose a controlled reward change
only after its discounted incentive agrees with the intended movement and
stance requirements.

The existing [Paper diagnostic](../../wr2/tools/diagnose_walking.py) now retains
all native weighted per-step reward components and the total. Its schema-3
summary includes censored component returns and their reconstruction error.
This adds direct GPU accounting; it does not alter the trainer or rewards.
Tests verify valid-terminal inclusion, invalid-padding exclusion and sums.

Artifacts: [reward audit](../../results/wr2_walking/wr2_paper_collision1_20261020_072851_seed3/review_20261022/reward_audit.json),
[control repetition](../../results/wr2_walking/wr2_paper_collision1_20261020_072851_seed3/review_20261022/summary.json),
with adjacent source scripts and raw NPZs in the ignored review directory.
The inherited `phase_envelope` field in the repetition script is unused;
reference substitutions apply at every tick.
