# WR2 fresh 300M result versus TB Paper — 2026-10-18

## Conclusion and recommended direction

**The fresh seed-3 control completed, but did not improve the retained WR2
policy or reach TB Paper parity.** Its best tracking checkpoint is **220,098,560**;
fewest falls are at **140,062,720**. All 15 trained checkpoints fail the ≤1%
fall gate. Retain these intermediate checkpoints for diagnosis and the previous
seed-2 +180M policy as the WR2 tracking control.

This seed learns stronger lateral motion than the previous WR2 policy, together
with a hopping gait and large instantaneous velocity fluctuations. Geometry and
force measurements independently confirm actual flight in nominal replays.
The outstanding problems include the learned hopping gait, fluctuating velocity
and incomplete command tracking.

**Recommended next experiment:** correct the critic's missing velocity
normalization, then train a fresh **300M seed-3** policy with air-time credit
restricted to a swing supported by the opposite foot. Retain TB's PPO settings,
all 25 reward coefficients, fixed action home, reference, timing, controller,
variation and checkpoint intervals. This is an explicit WR2 adaptation to the
measured flight failure. An offline calculation preserves all air-time credit
on the tested successful TB/retained WR2 trajectories and removes about **96%**
of that credit from the new hopping trajectories. It is a proposed experiment;
its learning and robustness effects require training.

The requested work here is analysis and a proposal. The canonical YAML still
describes completed 0.28.4; the proposed experiment has not been implemented or
started. This review is recorded separately from CHANGELOG pending confirmation.

## 1. Run, metric and checkpoint verification

Run: `results/wr2_walking/wr2_paper_20261017_195826_seed3`, recipe **0.28.4**,
clean source **09b9805**, fresh seed 3, fixed LR **1e−4**, requested 300M,
actual **300,134,400 transitions**. The restore argument is empty.
Saved YAML, recorded effective configuration and current canonical YAML agree.
Compared with corrected fresh seed 0, configuration differs only in seed and
version metadata; model, physical controller and motor home agree.

- Reconstructed **63 suites / 120,960 episode-condition records** from this
  run, corrected fresh seed 0, the seed-2 continuation and TB Paper. Means,
  population standard deviations, group counts, shortened transition schedules,
  episode lengths, falls, nonfinite totals and support partitions pass.
  **Zero audit failures**; maximum aggregate difference **2.55e−6**.
- There are **16 native metric/evaluation boundaries**, **149 rollout records**,
  **15 trained native checkpoints**, initial parameters and a final export.
  Step lists agree and native/rollout metrics are finite. All 25 native weighted
  rewards sum to return within **0.000781**. TB's logged reward components need
  multiplication by 0.02; WR2's components already include that timestep.
- Loaded every new native checkpoint with **Brax 0.14.1**. Each has **25 finite
  leaves / 1,599,600 values**. Final export exactly matches the final native
  arrays, SHA-256
  `b4930a8905be768c6500e72819b4b971ed5147d4e999fcb7d96ce8fc9553a00e`.
  Fresh initial arrays differ from seed 0. Recreating seed-3 initialization on
  CPU matches the preflight exactly and the GPU arrays within **1.49e−8**;
  raw CPU/GPU hashes differ at this floating-point precision.
- Recorded **227 commands × 1000 reference frames** pass geometry validation.
  Model SHA-256 is
  `741d1e613d487b34073a10fbc0365906c3364914f37e70a394c45bf1e1a696c0`.
  These are reference geometry checks, not a loaded execution test.
- Verified the TB Paper control's **11 captured source files**, **12 frozen
  model/reference hashes** and copied dependency lock against source **1982ef3**.
  The active TB checkout is in development; the successful Paper run's frozen
  source is the comparison implementation.

The shared primary metric is the mean of each episode's mean instantaneous XY
vector error across ten fixed moving conditions. It is not velocity RMSE or
error computed from aggregate mean velocity. The selection suite has **15
conditions × 128 episodes**, seed **12102**, deterministic policy, native model/
reset/actuator variation, pushes and observation noise. WR2 uses **824 control
ticks**, TB **1000**, at 20 ms each. Divide WR2 velocity/error by
`s = 0.8240547565`; lengths scale by `s² = 0.6790662418` and time by `s`.
Native robot/controller and fall-height definitions still differ, so survival
is a comparison under the named native profiles. Falling shortens measurements;
tracking and survival must be considered together.

The unchanged ideal zero-XY-velocity baseline is **0.071623 m/s** for these
commands. It is an analytic reference, not a simulated standing policy.
The untrained seed-3 error of 0.12816 accompanies **45% falls** and is not a
successful walking control. No trained candidate qualifies, so absence of a
promoted best policy or held-out confirmation is expected.

Training records Python **3.10.19**, Brax **0.14.1**, JAX/JAXlib **0.6.2**,
MuJoCo/MJX **3.14.0**, Flax **0.10.7**, Optax **0.2.8**. Native training wall time
is **1.89 h**, final throughput **44,740 transitions/s**, callback elapsed
**2.42 h**. TB Paper records **2.47 h** of native training wall time and about
34,400 transitions/s. Robot, physics/runtime and evaluation differences prevent
attributing that throughput difference to the PPO implementation alone.
The copied console log is absent; structured metrics and checkpoints are complete.

Artifacts: [metric/source audit](../../results/wr2_walking/wr2_paper_20261017_195826_seed3/review_20261018/comparison_audit.json),
[checkpoint audit](../../results/wr2_walking/wr2_paper_20261017_195826_seed3/review_20261018/checkpoint_audit.json).
Raw results, diagnostic arrays and figures remain local under ignored `results/`.

## 2. Progress and retained checkpoints

All errors below are **TB-equivalent m/s**. Continuation steps are added to an
inherited policy; they are not independent fresh training budgets.

| Policy / checkpoint | Fixed walking XY error | Falls / 1920 |
| --- | ---: | ---: |
| WR2 fresh seed 3, 140M: fewest falls | 0.13216 | **30 (1.56%)** |
| WR2 fresh seed 3, 220M: best tracking | **0.13063** | 55 (2.86%) |
| WR2 fresh seed 3, 300M: final | 0.14995 | 130 (6.77%) |
| Corrected WR2 fresh seed 0, 260M | 0.10770 | 102 (5.31%) |
| Retained WR2 seed-2 continuation, +180M | **0.10453** | 44 (2.29%) |
| TB Paper 180M: retained survival control | 0.03838 | **3 (0.16%)** |
| TB Paper 200M: retained tracking control | **0.03702** | 8 (0.42%) |
| TB Paper 300M: final | 0.03800 | 134 (6.98%) |

The new best is **25.0% worse** than retained WR2 tracking and **3.53×** TB
Paper 200M. Its final checkpoint is **14.8% worse** than its own best and
**4.05×** Paper 200M. Relative to fresh seed 0, seed 3 improves early survival
and later best survival but has worse final tracking. The seed control exposes
two different unsatisfactory gaits; it does not establish a training-seed success
rate. The retained seed-2 +180M policy has **500,224,000 ancestral transitions**.

Paired bootstrap over 128 WR2 model/reset indices, keeping conditions clustered:
final minus 220M error is **+0.01932**, 95% interval **[+0.01662, +0.02211]**;
falls increase **3.91 percentage points**, interval **[2.40, 5.42]**. New best
minus retained WR2 error is **+0.02609**, interval **[+0.02361, +0.02874]**.
These intervals concern fixed evaluation draws, not variation across training
seeds or held-out checkpoint performance.

| Fresh-run transitions | WR2 seed-3 error / falls | TB Paper error / falls |
| --- | ---: | ---: |
| 20M | 0.19308 / 17.14% | 0.06202 / 0.47% |
| 40M | 0.17132 / 5.89% | 0.04771 / 1.04% |
| 80M | 0.14308 / 2.92% | 0.03965 / 0.89% |
| 140M | 0.13216 / 1.56% | 0.03806 / 1.04% |
| 200M | 0.13135 / 2.34% | 0.03702 / 0.42% |
| 220M | 0.13063 / 2.86% | 0.03898 / 2.19% |
| 300M | 0.14995 / 6.77% | 0.03800 / 6.98% |

TB's useful tracking arrives mainly by **20–80M**; its 180M/200M policies
combine tracking and survival. WR2 seed 3 improves principally before 140M,
changes little through 240M, then regresses. Both runs show why retained
intermediate checkpoints matter. A longer budget by itself does not select a
walking gait or preserve a good checkpoint.

![Verified progress, response, nominal support and optimizer diagnostics](../../results/wr2_walking/wr2_paper_20261017_195826_seed3/review_20261018/training_comparison.png)

[PDF](../../results/wr2_walking/wr2_paper_20261017_195826_seed3/review_20261018/training_comparison.pdf),
[full progress CSV](../../results/wr2_walking/wr2_paper_20261017_195826_seed3/review_20261018/progress.csv),
[condition CSV](../../results/wr2_walking/wr2_paper_20261017_195826_seed3/review_20261018/conditions.csv).

## 3. Stronger directional response, excessive fluctuating motion

Velocities and errors are in TB-equivalent m/s. Standing reports total XY speed.

| Command | WR2 seed-3 220M axis velocity / standing speed | Retained WR2 +180M | TB Paper 200M | New WR2 / TB vector error |
| --- | ---: | ---: | ---: | ---: |
| Forward +0.10 | +0.0794 | +0.0566 | +0.0905 | 0.1365 / 0.0373 |
| Backward −0.10 | −0.0608 | −0.0431 | −0.0736 | 0.1374 / 0.0423 |
| Right −0.05 | −0.0303 | −0.0115 | −0.0320 | 0.1260 / 0.0373 |
| Left +0.05 | +0.0411 | +0.0086 | +0.0350 | 0.1255 / 0.0354 |
| Standing 0 | 0.1220 | 0.0909 | 0.0320 | 0.1220 / 0.0320 |

New lateral means are approximately **61–82% of command**, versus the retained
WR2 policy's **17–23%**. However, errors remain much larger than TB's.
For forward at 220M, error of the aggregate mean velocity is only **0.0214**,
while the mean instantaneous error is **0.1365**. Good net displacement can
coexist with large motion around that mean. Standing signed mean also hides
motion; total speed is **3.82×** Paper's.

In forward transitions, new WR2's start-window error is **0.1800** versus
Paper **0.0579**; stop-window error is **0.0990** versus **0.0545**.
At final, 91 falls are in fixed moving conditions, 11 in standing and 28 in
transitions. Failure coverage grows from 22/128 model/reset indices at 220M
to 69/128 at final. The new failure is broader than one isolated bad draw.

### Nominal replay

Replayed 140M/220M/300M for forward, left and standing: **72 episode-condition
replays / 59,328 control ticks**, zero falls or nonfinite states, independently
reconstructed statistics. WR2 reset is fixed; TB retains native random heading
and per-reset home offset. CPU runtimes use each training implementation's
Brax/MJX generation. The eight nominal WR2 episodes repeat a deterministic
trajectory rather than representing independent training outcomes.

| Nominal measurement | New WR2 220M | New WR2 final | Retained WR2 +180M | TB Paper 200M |
| --- | ---: | ---: | ---: | ---: |
| Forward error | 0.07088 | 0.10331 | 0.05947 | 0.02673 |
| Left error | 0.05563 | 0.06817 | 0.07034 | 0.02606 |
| Standing speed | 0.05397 | 0.07110 | 0.04306 | 0.02471 |
| Forward single support | 11.89% | 11.29% | 99.64% | 84.98% |
| Forward double support | 30.95% | 29.61% | 0.36% | 15.03% |
| Forward neither foot >1 N | **57.16%** | **59.10%** | **0%** | **0%** |

The gap persists without pushes or observation noise. The corrected seed-0
lineage learned contact-preserving but weak lateral motion; independent seed 3
rediscovers hopping under the same corrected implementation.

### Actual flight and controller trace

Additional individual trajectories measure the exact lowest point of each
oriented foot collision box relative to the floor, alongside all active contact
forces. A sampled aerial state requires **both boxes >10 μm above the floor**
and **all active contact-force components <10 μN**. At 220M this holds for
**56.92% forward, 54.00% right, 55.70% left and 51.70% standing** ticks.
It persists after the first second. Longest sampled aerial streaks are
**80–100 ms**; these are control-boundary observations, not continuous substep
measurements. All eight geometry/slew-control trajectories finish without falls.

The loaded forward trace has actual-motor versus delivered-target RMSE
**0.1075 rad**, compared with **0.0486** for retained WR2 and **0.0238** for
the TB Paper trace. At the first physics substep, **32.18%** of new leg samples
are limited by the torque/speed envelope, versus **4.93%** retained WR2 and **1.18%** TB.
Setpoint slew modifies **36.99%** of new leg samples, versus **3.11%** retained
WR2. These are individual nominal trajectories; they do not identify a corrected
hardware parameter or characterize all four/five physics substeps.

Native fixed-forward evaluation at 220M returns 356.82, of which air time
contributes **238.80 (66.9%)**, XY tracking **38.59 (10.8%)**, and leg imitation
**−3.06**. The nominal geometry trace similarly gives air-time reward **205.80**
and XY reward **52.41**. TB and WR2 native returns use different command/policy
sampling and are not shared performance scores. Standing and exact-deadzone
lateral commands can receive **zero** air-time reward while still hopping;
the shared learned gait persists outside the reward-active moving cases.

## 4. Source-confirmed alignment and remaining deltas

Compared WR2 `walking_env.py`, `paper_rewards.py`, `paper_walk.py`, `ppo.py` and
`train.py` with captured TB Paper `mjx_env.py`, `walk_env.py`, `paper.gin`,
`train_mjx.py`, frozen `walk_zmp_ref.py`/`mujoco_control.py` and Brax fork
**14147806**. Native WR2 Brax **0.14.1** is also checked directly.

| Item | Confirmed comparison |
| --- | --- |
| PPO | 1024 × 20 rollout, batch 256 × four minibatches × four passes; 512/256/128 Swish, state-dependent NormalTanh; γ=.97, GAE=.95, clip=.2, entropy=.0005, advantage normalization, no global gradient clipping |
| Value loss | Both apply **0.25 × MSE**; WR2 correctly passes twice its configured coefficient to native Brax's extra 0.5 multiplier |
| GAE / action density | Fork and native use the same GAE, PPO ratio clipping and NormalTanh density/entropy formulas on the selected path; the newer native diagnostic KL is additional logging |
| Observations | Both use 15 history frames; WR2 actor/critic **975/1485**, TB Paper **1245/1890**, reflecting 21 vs 30 motors with 12 leg actions |
| Action home / reference | Both use static home +0.25×delayed action; time-zero reference is static. Moving leg reference affects reward/critic. Actor learns imitation through advantages |
| Rewards | All 25 Paper coefficients, including leg imitation 5 and air time 500; corrected landing dt/length normalization and sum-before-1N support remain active |
| Geometry/timing | WR2 CoM 0.20223 m vs TB 0.29781 m, `s=.82405`; WR2 cycle .60 s vs TB .72 s (TB-equivalent .7281 s), physical swing .03395 m vs .05 m |
| Critic linear velocity | **Normalization omission:** WR2 appends `2*linear` at newest-frame indices **80:83**, while normalized TB equivalence requires `2*linear/s`. Angular velocity and commands already normalize. This feature is 17.6% smaller than intended; it is a concrete correction, not evidence of a failed PPO algorithm |
| Target delivery | WR2 additionally quantizes targets and caps each change at **.113097 rad/20 ms**. TB has no setpoint slew filter (`filter_type=none`). WR2's fitted position-servo replay also delivers recorded setpoints directly |
| Actuator dynamics | WR2's fitted total damping .673622 is correctly split into .5 active +.173622 passive. Armature .020951, friction loss .465471 and provisional torque envelope differ materially from TB hardware |
| Runtime / geometry | WR2 uses MJX 3.14, Newton 10, 5 ms ×4; Paper uses MJX 3.2.4, Newton 1, 4 ms ×5. Motor/observation counts and gearing also differ |

WR2 also retains its corrected per-foot slip mask; frozen TB broadcasts the
two-foot mask across XY axes. This is a documented implementation correction.

The new mean epoch joint KL is **0.0156–0.0280**, usually around **0.020**.
Fixed LR does not enforce `desired_kl=.01`. Mean latent standard deviation
falls to **0.1384** at 100M, then rises to **0.2672** at final. This is not the
previous continuation's collapsing-exploration pattern. These metrics support
monitoring update drift, but do not isolate its contribution to hopping.
TB Paper did not log KL or standard deviation; its late fall regression cannot
be assigned the same cause. TB's cosine decay is effectively fixed 1e−4 over
300M because decay steps count optimizer updates.

Full-reference imitation is a soft target for both robots. TB's forward
reference exceeds its ±.25 rad action box in **12.44%** of leg samples and has
an unavoidable projected-reference RMSE of **0.1292 rad**, yet its policy walks
successfully. New WR2's delivered/reference RMSE is smaller than TB's in the
loaded comparison. These facts do not support increasing action range or
imitation weight solely to reproduce the entire reference exactly.

WR2's motor speed in TB-equivalent angular units is **5.8178*s = 4.7942 rad/s**,
versus TB's nominal 10. Its larger identified armature and damping also change
bandwidth. Size-normalized geometry and cycle time therefore do not establish
dynamic equivalence. The unit-A fit covers low-load captures; loaded torque-speed
and braking curves remain provisional. All six accepted fit/held-out captures
have maximum recorded target changes **.03491 rad**, below the training slew
cap, so that fit cannot validate the extra cap at the new gait's command rates.
See [servo evidence and limitations](servo_model.md).

## 5. Controlled tests and the improvement plan

### Target slew is not an immediate repair

Replay the same 220M policy/reset/model while changing only the setpoint cap
from .113097 rad to infinity. Physical torque/speed limits remain active.

| Nominal condition | Current error | No-setpoint-slew error | Change | Current / no-slew actual aerial samples |
| --- | ---: | ---: | ---: | ---: |
| Forward | 0.06732 | 0.12227 | +81.6% | 56.92% / 57.16% |
| Right | 0.05696 | 0.05365 | −5.8% | 54.00% / 54.13% |
| Left | 0.06137 | 0.08885 | +44.8% | 55.70% / 58.13% |
| Standing | 0.05851 | 0.07095 | +21.3% | 51.70% / 52.43% |

This control shows that disabling the cap does not repair the learned gait.
It does not establish the outcome of retraining without the filter. The cap's
hardware basis still needs verification. Keep this physical mapping constant
in the proposed reward experiment; review it separately with measured command
delivery and a matched controller replay.

### A targeted air-time change has direct behavioral evidence

TB rewards accumulated air time on landing. Its current counter increments
whenever a foot is unsupported, including ticks when **both** feet are airborne.
The same implemented reward can therefore credit hopping in WR2.

Proposed update: accumulate a foot's air time only while it is unsupported
**and the opposite foot is supported**. Clear that counter when both feet lose
support. Retain the native landing filter, coefficient 500, timestep and all
other reward functions. This uses the existing counter and contact measurements;
it introduces no additional observation or tunable reward weight.

Offline replay of recorded contact sequences, with the active-command gate
enabled, gives:

| Recorded trajectory | Original landing-air-time credit (s) | Proposed credit (s) | Retained |
| --- | ---: | ---: | ---: |
| WR2 fresh seed 3, 220M forward | 20.58 | 0.88 | **4.28%** |
| WR2 fresh seed 3, final forward | 21.10 | 0.98 | **4.64%** |
| Retained WR2 +180M forward | 16.40 | 16.40 | **100%** |
| TB Paper 200M forward | 16.54 | 16.54 | **100%** |
| TB Paper 200M right / left | 17.14 / 17.18 | 17.14 / 17.18 | **100%** |

This confirms the proposed rule discriminates the observed flight strategy
while preserving credit for the tested successful TB behavior. It is a
**deliberate difference from frozen TB's reward implementation**, justified by
two completed fresh 300M WR2 controls, actual geometric flight and the credit
reconstruction. It does not prove a future policy will stop hopping, track
better or recover from pushes. Standing has no direct air-time reward, so its
behavior must also improve through the complete objective and learned policy.

### Concrete next-run proposal

1. **Correct critic linear velocity to `linear/s`.** Verify physical states
   related by `s` produce the same intended critic velocity feature; retain
   actor dimensions and deterministic checkpoint evaluation behavior.
2. **Test the supported-swing counter above.** Verify identical landing credit
   on single/double-support sequences and reduced credit for both-foot flight.
   Implement it as an explicit recipe change, with the rationale recorded.
3. **Train fresh seed 3 for 300M**, fixed LR 1e−4, current Swish/NormalTanh and
   full 1024 ×20/four-pass layout, unchanged reference/action range/controller,
   all 25 TB coefficients, variation and timing. Use the regular WR2 trainer.
   Record both the feature correction and counter change; improvement would
   be evidence for this combined treatment, not an isolated causal estimate.
4. **Retain/evaluate every ~20M checkpoint.** Compare the full curve with
   seed-3 140M/220M, retained WR2 +180M/+160M and TB Paper 180M/200M. Inspect
   instantaneous error, signed response, standing/stopping, support and actual
   nominal flight together. Retain the final hopping policy as a diagnostic
   and use fresh initialization for this experiment.
5. **Qualification remains ≤19 falls /1920**, zero nonfinite episodes and
   held-out seed **22102** for qualifying candidates. TB parity also requires
   approximately **.0370 equivalent fixed error**, its directional response and
   contact-preserving nominal motion. A low fall rate alone is insufficient.

For a strict causal estimate of the counter change, a control training run
must use the same corrected critic feature and seed. The proposed single next
run prioritizes a correctness fix plus the measured gait failure; it cannot
separate their individual learning effects.

Continue loaded actuator work alongside this experiment: replay reachable
action-projected references and learned targets, measure lag/clipping through
all physics substeps, and compare with measured servo motion at relevant loads
and frequencies. Any cycle/command adjustment should preserve geometry and
explicitly scale time, velocity and stride together. Adopt new gains, armature,
braking or torque-speed values only when those measurements identify them.

The completed TB Main control remains a weaker recipe: no leg imitation,
stronger action-change penalty and weak signed translation. Its result supplies
no evidence for replacing the successful Paper objective.

Artifacts: [nominal/trace audit](../../results/wr2_walking/wr2_paper_20261017_195826_seed3/review_20261018/nominal_and_trace_audit.json),
[geometry and slew control](../../results/wr2_walking/wr2_paper_20261017_195826_seed3/review_20261018/flight_and_slew.json),
[geometry reconstruction audit](../../results/wr2_walking/wr2_paper_20261017_195826_seed3/review_20261018/flight_and_slew_audit.json),
[air-time credit reconstruction](../../results/wr2_walking/wr2_paper_20261017_195826_seed3/review_20261018/unilateral_air_time_credit.json).

## References

- [ToddlerBot, Tables VI/VII](https://arxiv.org/html/2502.00893v2), with the
  successful local Paper reconstruction at source `1982ef3`, run
  `toddlerbot_walk_ppo_20261015_194347` as the numerical control.
- [Schulman et al., Proximal Policy Optimization](https://arxiv.org/abs/1707.06347).
- [Tedrake et al., ZMP preview planning](https://doi.org/10.1109/HUMANOIDS.2015.7363473).
- [Previous WR2/Paper/Main progress review](wr2_tb_progress_review_20261017.md)
  and [fresh-run preparation and loaded-motion evidence](wr2_fresh_paper_run_20261017.md).
