# WR2 and ToddlerBot progress and next decision — 2026-10-17

## Assessment

**Large PPO updates describe the size of an optimization change, not an
excessive number of training transitions.** The completed WR2 0.28.3 continuation
now provides the missing experiment: fixed LR 1e−5 keeps reported epoch KL
between 0.0051 and 0.0106, but tracking improves only 1.7% at its best checkpoint.
Optimizer stabilization has helped survival without closing the walking gap.

Use **TB Paper 180M/200M** as the walking controls. The new TB Main run learns
survival with very weak translation, including nominally. Its recipe changes
do not establish an improvement direction for WR2. Current WR2 already has
Paper's moving leg-reference imitation, fixed action home, Swish/NormalTanh
networks and rollout/update layout.

The next priority is a measured comparison of reference, policy target,
delivered target and loaded gait execution. WR2's nominal lateral response
remains weak after 500M transitions along the best policy's ancestry. One
controlled replay finds that removing its extra target slew limit improves
forward error by 3% and lateral error by 0.3%; that restriction alone does not
account for the gap. A fresh 200–300M Paper experiment should follow the
diagnostic, retaining every intermediate checkpoint and making one justified
change.

## 1. Data and verification

| Dataset | Result folder | Recorded source / learner |
| --- | --- | --- |
| WR2 corrected fresh | `results/wr2_walking/wr2_paper_20261016_222755_seed0` | `c1150ed`, recipe 0.28.2, Brax 0.14.1 |
| WR2 continuation, LR 1e−4 | `results/wr2_walking/wr2_paper_20261017_080025_seed1` | `dd18051`, 0.28.2, Brax 0.14.1; restore fresh 260M |
| WR2 continuation, LR 1e−5 | `results/wr2_walking/wr2_paper_20261017_115520_seed2` | Clean `1cc55a0`, 0.28.3, Brax 0.14.1; restore prior +60M |
| TB Paper | `~/projects/toddlerbot/results/toddlerbot_walk_ppo_20261015_194347` | `1982ef3`; Brax fork `14147806`, package 0.10.5 |
| TB Main | `~/projects/toddlerbot/results/toddlerbot_2xc_walk_brax_20261016_191310` | `a8637a1`, `main-walking-compare`, Brax 0.14.2 |

Paper is the local reconstruction of the published recipe; the curves here
are measured local training results. Reviewed the TB agent's
[`MAIN_PAPER_WALKING_RESULTS.md`](../../../toddlerbot/MAIN_PAPER_WALKING_RESULTS.md)
and independently checked its raw data and captured source.

The [comparison audit](../../results/wr2_walking/wr2_paper_20261017_115520_seed2/review_20261017/comparison_audit.json)
reconstructs **95 suites / 182,400 episode-condition records** across these
runs, including both Main profiles. Means, population standard deviations,
falls, lengths, nonfinite counts, eligible counts and truncated transition
schedules reconcile with **zero failures**. Maximum summary discrepancy is
3.09e−6; WR2 support partitions reconcile within 8.20e−8. Every logged native
and rollout metric checked is finite.

Reward components also reconcile. TB's recorded episode components are
**weighted rates accumulated before multiplication by 0.02 s**; WR2's
`episode_paper_reward/*` components already include that multiplication.
Applying each implementation's convention reconstructs its return. Native
returns have different command sampling, reward definitions and metric sources,
so they are unsuitable as a common cross-recipe score.

Verified all **72 archived Main source hashes and 28 model/reference hashes**.
Paper's **11 copied locomotion sources** match frozen `1982ef3`; its **12
model/reference hashes** and dependency lock match the previously prepared
local frozen checkout. Paper provenance records a dirty tree; these checks
cover the captured files rather than a complete historical tree. Reading the
captured source avoids confusion with active TB development.

For new WR2, saved/effective/prepared configurations agree. Changes from the
previous continuation are version metadata, seed and LR. The restore path and
parameter-only/new-Adam semantics match the plan; all 227,000 recorded reference
frames pass validation. Loaded all **15 new native checkpoints**: each has
25 finite leaves / 1,599,600 values. Initial arrays exactly match the selected
+60M parent; final native arrays exactly match `params`. See the
[checkpoint audit](../../results/wr2_walking/wr2_paper_20261017_115520_seed2/review_20261017/checkpoint_audit.json).
Independently loaded all **16 Main exports**, checked their manifest hashes
and finite arrays, and verified `model_best` = 220M / `model_last` = 300M.

Paper has 15 metric records. Main has 30: an episode-window callback and an
epoch callback at each trained boundary. Its duplicate step numbers represent
different metric sources. Paper's empty console copy and Main's missing W&B
copy do not prevent reconstructing these structured results.

### Comparison units and limits

Each primary suite has ten fixed moving conditions, standing and four transition
conditions, with 128 episodes per condition. Tracking is the per-episode mean
**instantaneous XY vector error**, averaged over the ten fixed moving conditions.
Falls truncate measurements, so read error together with survival and lengths.

WR2 uses `s = sqrt(h_WR2 / h_TB) = 0.824054757`: divide its physical velocities
and errors by `s`. Its benchmark is 824 control ticks / 16.48 s, versus TB's
1000 ticks / 20 s. Linear speeds and time scale by `s`; angular speeds by `1/s`.
Main numbers below retain its physical m/s on the same numeric commands.
Its `toddlerbot_2xc` model differs from Paper's `toddlerbot` model.

Paper/WR2 include pushes and their native model/reset/actuator/noise variation.
Main's native profile has **no pushes**, different noise/randomization and a
shared benchmark height stop criterion separate from its native training bound.
Survival comparisons therefore describe the named profiles, rather than a
controlled common robustness experiment. Support and torque measurements use
the end of each 20 ms control step.

## 2. What a large policy update means

All three recipes collect **1024 environments × 20 ticks = 20,480 transitions**
per batch and take **4 passes × 4 minibatches = 16 Adam updates** on it.
Training budget determines how many such batches we process. Learning rate,
gradients and the policy's current distribution determine how far an update
moves the policy. Longer training can remain useful with suitably sized updates.

WR2's native `training/kl_mean` averages the analytic joint Gaussian
`KL(old rollout policy || current policy)` across states/minibatches and the
training epoch. It sums the 12 action dimensions before averaging. It is a
distribution-change diagnostic, rather than a checkpoint weight distance or
percentage change. These are epoch averages, not maxima over individual updates.

In the earlier 1e−4 continuation, epoch KL reaches **1.252 at +100M, 5.318 at
+120M and 1.543 at +240M**. The fixed schedule maps to `LRSchedule.NONE`;
configured `desired_kl: 0.01` is inactive. PPO ratio clipping at 0.2 does not
strictly constrain the policy distribution's movement. TB Paper's cosine
schedule changes its LR negligibly over these optimizer counts and has no
KL stop; its copied loss does not log KL.

The previously recorded
[identical actual-rollout batch probe](../../results/wr2_walking/wr2_paper_20261017_080025_seed1/review_20261017/ppo_update_probe.json)
isolates LR at the +60M parent: lowering it 10× reduces post-batch KL from
**0.2885 to 0.01684** and deterministic action-change RMS from 0.00926 to
0.00231. That supports smaller updates at this mature checkpoint. It does not
prove that KL caused every performance regression or the original nominal gait gap.

The new long run's maximum reported epoch KL is **0.01060**, final **0.01037**.
Main's new epoch KL rises gradually from **0.00829 to 0.03142**, with finite
parameters throughout. The historical Main archive's numerical failure is a
different experiment. Paper's late falls cannot be assigned a KL cause from
its unlogged KL.

## 3. Fresh training progress

Error columns are TB-equivalent m/s for WR2 and physical m/s for TB.
Falls are out of 1,920; profiles differ as described above.

| Fresh transitions (M) | TB Paper error / falls | TB Main native error / falls | WR2 corrected error / falls |
| --- | ---: | ---: | ---: |
| 20.009 | 0.062016 / 9 | 0.391617 / 1920 | 0.196249 / 658 |
| 80.036 | 0.039653 / 17 | 0.075357 / 4 | 0.171695 / 528 |
| 200.090 | **0.037025 / 8** | 0.081486 / 143 | 0.119904 / 120 |
| 300.134 | 0.037999 / 134 | 0.083954 / 28 | 0.107827 / 96 |

- **Paper learns useful tracking by 80M.** Best survival is 180M: 3 falls /
  1,920, error 0.038379. Best tracking is 200M: 8 falls, error 0.037025.
  Late training mostly preserves tracking while degrading survival. The
  native-return-best 240M policy is not the best tracking/survival checkpoint.
- **Main learns survival by 60–80M, then weak translation persists.** None of
  its trained checkpoints beats the analytic zero-XY-velocity error baseline
  **0.071623 m/s** on these ten commands. Final nominal error is 0.079021,
  despite no nominal falls. This baseline is analytic rather than a simulated
  standing controller.
- **WR2 learns much more slowly**, with substantial improvement through 260M
  and a persistent gap thereafter. The low-LR continuation confirms that
  longer training alone has not resolved it from the inherited policy.

![Verified training progress and signed response](../../results/wr2_walking/wr2_paper_20261017_115520_seed2/review_20261017/wr2_tb_progress.png)

The standalone [PDF](../../results/wr2_walking/wr2_paper_20261017_115520_seed2/review_20261017/wr2_tb_progress.pdf),
[all-checkpoint progress CSV](../../results/wr2_walking/wr2_paper_20261017_115520_seed2/review_20261017/progress.csv)
and [optimizer CSV](../../results/wr2_walking/wr2_paper_20261017_115520_seed2/review_20261017/optimizer.csv)
retain the complete curves. Native final training timers are 8,892.8 s for
Paper, 7,006.5 s for Main and 6,660.0 s for the new WR2 run. Different models,
physics versions, learners and callback timing prevent assigning runtime
differences to one feature.

## 4. Completed WR2 0.28.3 continuation

The parent is the previous continuation's +60,026,880 checkpoint, itself
restored from fresh 260,116,480. This experiment adds **300,134,400**, reaching
**620,277,760 transitions along the final policy's ancestry**. The two
continuations start from different checkpoints and training seeds; the full-run
comparison alone does not isolate LR.

| Added transitions | Error, TB-equivalent m/s | Falls / 1920 | Policy ancestry transitions |
| --- | ---: | ---: | ---: |
| 0 | 0.106388 | 61 (3.18%) | 320,143,360 |
| +20,008,960 | 0.106298 | 42 (2.19%) | 340,152,320 |
| **+160,071,680: retain for survival** | 0.105444 | **42 (2.19%)** | 480,215,040 |
| **+180,080,640: retain for tracking** | **0.104534** | 44 (2.29%) | 500,224,000 |
| +300,134,400: final | 0.107269 | 52 (2.71%) | 620,277,760 |

Best tracking improves **1.74% versus this run's start**, but only **0.43% versus
the previous best-ever 0.104990**. A 20,000-resample paired bootstrap keeps all
conditions for each model/reset index together. Best-minus-start TB-equivalent
error is −0.001855, interval [−0.002882, −0.000897]. Best-minus-previous-best is
−0.000457, interval [−0.001260, +0.000344]. The latter tracking gain is not clear
on this selection set. Final-minus-best is +0.002735, interval
[+0.002007, +0.003503]. These post-selection comparisons use one evaluation
seed and do not establish held-out or independent-training-seed performance.

No checkpoint meets **≤1% falls**: at most 19/1920 are permitted. Absence of a
qualified `best_params` export and held-out confirmation is expected. Use the
retained native +160M/+180M checkpoints for diagnosis. TB Paper 200M has
**0.037025 error / 0.417% falls**; WR2 best error is still **2.82×** that control.

Native forward return increases from **389.21 to 402.57** while benchmark error
ends near its start. Optimizing the full reward does not ensure monotonically
improving command tracking. Mean latent standard deviation narrows to about
0.03; this observation alone does not establish an exploration failure.

### Directional response under the native profiles

| Measurement | TB Paper 200M | TB Main final | WR2 new +180M |
| --- | ---: | ---: | ---: |
| Forward X, command +0.10 | +0.090528 | +0.006191 | +0.056619 |
| Backward X, command −0.10 | −0.073570 | −0.001106 | −0.043119 |
| Right Y, command −0.05 | −0.032036 | +0.001440 | −0.011538 |
| Left Y, command +0.05 | +0.034992 | +0.003468 | +0.008649 |
| Standing XY speed | 0.031964 | 0.019769 | 0.090942 |

WR2's lateral endpoint response is **20.2%** of command span, versus Paper's
**67.0%** and Main's **2.0%**. WR2 moves forward/backward appreciably but has
large instantaneous errors and unwanted standing motion. Main's smaller error
than WR2 does not demonstrate better commanded walking: its signed translation
is weak even in the nominal profile.

## 5. Code-confirmed recipe comparison

| Feature | TB Paper | TB Main run | Current WR2 Paper |
| --- | --- | --- | --- |
| Leg-reference reward | `−5 mean((q − q_ref)²)` | Motor-reference imitation weight 0 | Same Paper term, weight 5 |
| Action home | Fixed episode home + 0.25 residual | Fixed home + 0.25 residual | Fixed grounded crouch + 0.25 residual |
| Moving-command home penalty | No separate Main pose term | `−0.5 sum(pose_weights × (q − q_home)²)` | Paper imitation; standing-only home term |
| XY reward | `5 exp(−100 e²)` | `2 exp(−1000 e²)` | Paper formula on normalized velocities |
| Action-change penalty | `−0.05 mean(Δa²)` | `−2 sum(Δa²)` | Paper term |
| Second action difference | `−0.05 mean(Δ²a²)` | Absent | Paper term |
| Activation / distribution | Swish / state-dependent NormalTanh | ELU / unbounded Normal, independent log std | Swish / state-dependent NormalTanh |
| Learning rate | Effectively 1e−4 over this run | Fixed 3e−5 | Fixed 1e−5 in the new continuation |
| Global gradient norm clip | Absent | 1.0 | Absent |
| Rollout / update layout | 1024 × 20; 256 × 4 × 4 | Same | Same |

At equal numeric action changes over 12 motors, Main's action-change coefficient
is **480×** Paper's. At XY error 0.05 m/s, its weighted velocity reward is
**0.164**, versus Paper's **3.894**. Main's final training return receives about
**+123.24 from foot phase/height** and **+9.88 from XY tracking**. These establish
changed objective balance and are candidate causes of undertracking; their
individual effects have not been isolated.

Confirmed in captured Paper `mjx_env.py` / `paper.gin` / `train_mjx.py`, Main
`source/toddlerbot/locomotion/mjx_env.py` / `train_mjx.py`, WR2
[`paper_rewards.py`](../../wr2/locomotion/paper_rewards.py),
[`walking_env.py`](../../wr2/locomotion/walking_env.py),
[`ppo.py`](../../wr2/locomotion/ppo.py) and
[`train.py`](../../wr2/locomotion/train.py).
WR2's adapter maps its value coefficient to the native Brax factor correctly;
actual value loss is 0.25 × MSE, matching the Paper fork. Its static home is
correct. Moving reference errors enter reward/critic; the actor learns through
PPO advantages using phase, command and proprioception. This review finds no
new confirmed adapter or reference-grid bug causing the gap.

## 6. Nominal gait and the tested actuator restriction

Replayed new +180M and final with Brax 0.14.1 / MJX 3.14, deterministic policy,
nominal model, fixed WR2 reset, no observation noise or pushes, seed 22102.
Three conditions × eight environments per policy all finish 824 ticks with
zero falls, nonfinite values or sampled neither-foot support. Independent
[nominal reconstruction](../../results/wr2_walking/wr2_paper_20261017_115520_seed2/review_20261017/nominal_metrics_audit.json)
also checks the controlled restriction ablation: 96 recorded episode-conditions,
zero accounting failures. Fixed nominal repetitions are not independent model
draws.

| Nominal measurement, TB-equivalent m/s | TB Paper 200M | WR2 new +180M | WR2 final |
| --- | ---: | ---: | ---: |
| Forward command +0.10: measured X | 0.090890 | 0.071001 | 0.061629 |
| Forward vector error | 0.026727 | 0.059466 | 0.064854 |
| Left command +0.05: measured Y | 0.040675 | 0.005100 | 0.001136 |
| Left vector error | 0.026059 | 0.070337 | 0.072266 |
| Standing XY speed | 0.024714 | 0.043056 | 0.044372 |

The earlier TB nominal control uses its native reset heading/home offsets and
MJX 3.2.4, so this comparison does not isolate a physical parameter. Nevertheless,
WR2's lateral undertracking persists with its variation removed. Forward single
support is 99.76% at +180M, versus TB's 84.98%; neither-foot support is zero in
both. The observed problem is a weak gait, rather than the old nominal hopping.

WR2 has an extra target slew restriction absent from TB. It is derived from
the vendor no-load speed; quantization makes the actual cap **0.113097 rad per
20 ms = 5.655 rad/s**. Hardware angular speed and setpoint slew are distinct.
The [servo evidence table](servo_model.md) calls this command limit derived;
it is not a measured firmware setpoint-rate ceiling.

Tested **the same +180M policy, model, reset keys, quantization, joint bounds and
physical torque/speed envelope**, changing only this additional target restriction.
The baseline exactly reconstructs the earlier nominal replay.

| Condition | Error with slew, TB-equivalent m/s | Error without slew | Change |
| --- | ---: | ---: | ---: |
| Forward | 0.059466 | 0.057700 | −2.97% |
| Left | 0.070337 | 0.070097 | −0.34% |
| Standing | 0.043056 | 0.042573 | −1.12% |

All ablation episodes finish with zero falls/nonfinite/neither-foot support.
Left velocity remains approximately 0.005 m/s. This small diagnostic does not
establish the effect of retraining without the limit; it shows that removing
it from this retained policy does not close the nominal lateral gap. See the
[controlled results](../../results/wr2_walking/wr2_paper_20261017_115520_seed2/review_20261017/slew_ablation_summary.json).

Other source-backed physical differences remain relevant: WR2 armature is
0.02095 versus TB hip/knee values 0.0044/0.0012, effective damping is 0.67362
versus about 0.075/0.08, and its vendor maximum speed is 5.818 versus 10 rad/s.
Its 0.60 s cycle is geometrically normalized from TB's 0.72 s; angular speed
capacity normalized into TB units is about 4.79 rad/s. Geometry normalization
does not establish similar actuator bandwidth. WR2's damping decomposition
remains correct, and its fitted low-load model differs from a measured full
loaded torque-speed/braking characterization. These are concrete quantities
to investigate, rather than established causes of the training gap.

## 7. Recommended next work

1. **Retain +180M and +160M as diagnostic controls.** The final checkpoint has
   worse tracking. Stable updates and a 620M final ancestry provide sufficient
   evidence to advance the previous plan's gait/actuator investigation.
2. **Measure the location of the nominal execution mismatch against TB Paper
   200M.** For forward, both lateral directions and standing, trace reference
   angles, policy-requested targets, delivered targets, actual motor motion,
   foot lift/slip, contact timing, torque/speed limits and CoM/support motion.
   Compare normalized time/speed and within-robot tracking errors. This
   distinguishes a policy requesting weak motion from a controller failing
   to execute its requests, or a dynamically unsuitable reference. Static IK
   and desired-ZMP validation alone do not validate loaded execution.
3. **Use a controlled correction before the next long run.** The prior
   [forward trace](wr2_paper_result_20261017.md#6-reference-imitation-and-actuator-tracking)
   already shows delivered-target/reference RMSE 0.191 rad versus
   actual/delivered-target RMSE 0.059 rad. Target choice deserves attention as
   well as actuator lag. If the loaded model or timing is wrong, correct that
   measured difference using WR2 dimensions/actuator evidence. If delivered
   motion is adequate and the inherited policy requests a weak gait, a fresh
   Paper initialization is the appropriate control for that inherited
   strategy. Joint-reference RMSE alone is insufficient justification for
   changing imitation weight: TB succeeds with larger raw reference RMSE.
4. **Then train a fresh 200–300M Paper experiment with one justified change.**
   Keep the fixed crouched home, leg imitation, bounded distribution, rollout
   layout and full perturbation protocol. Retain a suitable update regime;
   the mature-policy 1e−5 experiment does not by itself identify the best
   initial LR for a fresh policy. Use actual rollout KL to check the selected
   optimizer setting. Retain/evaluate every ~20M checkpoint, including signed
   velocities, standing and transitions. Advancement requires tracking and
   survival improvement; parity is measured against Paper 200M's 0.0370
   equivalent error and directional response, with ≤1% falls eligibility and
   held-out seed 22102 for qualifying candidates.

The immediate work is the loaded gait comparison. Current evidence establishes
the residual gap and rules out two simple complete explanations; it does not
yet identify a single physical correction guaranteed to improve training.

### Follow-through and prepared run

The [loaded gait comparison and fresh Paper setup](wr2_fresh_paper_run_20261017.md)
now complete this investigation step. WR2 +180M and TB Paper 200M traces cover
forward, both lateral directions and standing. They confirm weak requested
motion and greater execution lag without identifying a measured physics fix.
A real **1024 × 20** fresh-policy batch at TB's **1e−4** LR passes with
post-batch joint KL **0.003794**, finite arrays and 4.04% ratio clipping.
Prepared **0.28.4 / fresh seed 3 / 300M** repeats the corrected Paper recipe with
an independent initialization and retains every intermediate checkpoint.

## References

- [ToddlerBot paper, Tables VI/VII](https://arxiv.org/html/2502.00893v2): the
  published recipe used by the local Paper reconstruction.
- [Schulman et al., PPO](https://arxiv.org/abs/1707.06347) and
  [Spinning Up's PPO explanation](https://spinningup.openai.com/en/latest/algorithms/ppo.html):
  clipped surrogate updates and their limits.
- [Previous WR2 continuation review](wr2_paper_finetune_result_20261017.md),
  [corrected fresh review](wr2_paper_result_20261017.md) and
  [servo measurement evidence](servo_model.md).
- All new raw audit outputs, diagnostic scripts and figures are local under
  [`review_20261017`](../../results/wr2_walking/wr2_paper_20261017_115520_seed2/review_20261017).
  Public references retain the previously reviewed v2 recipe; external page
  bodies could not be refreshed through this session's connector. Code and
  measurement assertions above are verified from the local captured sources.
