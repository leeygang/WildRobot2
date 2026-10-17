# WR2 paper walking: completed fine-tuning review — 2026-10-17

## Decision

**Another 300M transitions gave a small early improvement, followed by regression.
WR2 has not reached ToddlerBot parity.** Retain **+20,008,960** for best tracking
and **+60,026,880** for best survival. None passes the complete suite's **≤1%
falls** gate. The final policy has worse walking despite quieter standing.

The best tracking checkpoint improves error only **2.5%** over the restored
parent and remains **2.84× TB 200M**. Final error is **15.3% worse than the
parent**, with **8.07% falls**. The policy has now seen **560,250,880 transitions**.
Large measured PPO updates justify a controlled trial at **fixed LR 1e−5** from
the retained +60M checkpoint. This changes one recipe parameter; its effect on
long-run walking performance still needs verification.

Run: [`wr2_paper_20261017_080025_seed1`](../../results/wr2_walking/wr2_paper_20261017_080025_seed1).
Parent: [`wr2_paper_20261016_222755_seed0`, 260M](wr2_paper_result_20261017.md).
**Every new checkpoint number below counts added transitions**, not the policy's
total experience.

## 1. Metric, provenance and restore audit

- Completed **300,134,400 added transitions** from the requested 300M budget,
  restoring parent `checkpoints/000260116480`. Source was clean **dd18051**,
  recipe **0.28.2**, seed **1**. Saved YAML and recorded effective configuration
  match the prepared YAML with seed 1; the parent configuration differs only
  in seed. There were no training implementation or dependency changes.
- Independently reconstructed **90,240 episode-condition records** across this
  run, its parent and the frozen TB control: means, population deviations,
  eligibility/counts, truncated schedules, lengths, falls and nonfinite totals.
  **Zero audit failures**. Maximum aggregate rounding discrepancy is **3.10e−6**;
  support identities agree within **8.20e−8**. All 25 weighted native rewards
  sum to return within **0.000409**.
- All **16 native evaluations, 16 paper reports and 149 rollout records** are
  finite. Checkpoint/history step lists agree. All **227 commands × 1000
  reference frames** pass recorded validation. The model hash is unchanged:
  `741d1e613d487b34073a10fbc0365906c3364914f37e70a394c45bf1e1a696c0`.
- Loaded all **15 trained native checkpoints** with Brax **0.14.1**: each has
  **25 finite leaves / 1,599,600 values**. Initial actor, critic and normalizer
  arrays **exactly equal the parent 260M checkpoint**. Final native arrays
  exactly equal the `params` export. Array hashes and runtime are recorded in
  the [checkpoint audit](../../results/wr2_walking/wr2_paper_20261017_080025_seed1/review_20261017/checkpoint_audit.json).
  Parameter-only restore resets Adam and step counters, as in native TB.

GPU training used Python **3.10.19**, Brax **0.14.1**, JAX/JAXlib **0.6.2**,
MuJoCo/MJX **3.14.0**, Flax **0.10.7**, Optax **0.2.8**, Orbax **0.11.39** and
NumPy **2.2.6**. Elapsed time was **2.36 h**, including **1.86 h** of native
training wall time; final reported training throughput was **45,365 transitions/s**.

The primary suite uses **15 conditions × 128 episodes**, selection seed
**12102**, deterministic policy, and enabled model/reset/actuator variation,
pushes and observation noise. WR2 runs **824 ticks / 16.48 s**, TB 1000 ticks /
20 s. Speeds and timing use `s = sqrt(h_WR2 / h_TB) = 0.824054757`.
**TB-equivalent WR2 velocity/error is physical velocity/error divided by `s`.**
Tracking means each episode's mean instantaneous XY vector error, averaged
across the ten fixed walking conditions. Falls shorten measurements; read
error alongside survival. It is neither RMSE nor the error of mean velocity.

Support, force and torque samples are taken at the end of each **20 ms** control
step. A sample with neither foot exceeding 1 N alone does not establish flight.
No `best_params` or held-out confirmation was produced because no candidate
qualifies: **≤1% permits at most 19 falls / 1920 episodes**.

## 2. Checkpoints and the TB gap

| Policy | Cumulative policy transitions | Physical fixed XY error (m/s) | TB-equivalent error (m/s) | Falls / 1920 |
| --- | ---: | ---: | ---: | ---: |
| Parent WR2 260M | 260,116,480 | 0.088750 | 0.107699 | 102 (5.31%) |
| Restored WR2 +0 | 260,116,480 | 0.088771 | 0.107725 | 103 (5.36%) |
| **WR2 +20M: best tracking** | **280,125,440** | **0.086518** | **0.104990** | **80 (4.17%)** |
| **WR2 +60M: early best survival** | **320,143,360** | **0.087762** | **0.106500** | **64 (3.33%)** |
| WR2 +220M: tied survival | 480,215,040 | 0.097536 | 0.118361 | 64 (3.33%) |
| WR2 +240M: major regression | 500,224,000 | 0.128592 | 0.156047 | 426 (22.19%) |
| WR2 +300M: final | 560,250,880 | 0.102346 | 0.124198 | 155 (8.07%) |
| TB 180M | 180,080,640 | 0.038379 | 0.038379 | 3 (0.156%) |
| **TB 200M** | **200,089,600** | **0.037025** | **0.037025** | **8 (0.417%)** |
| TB 300M: final | 300,134,400 | 0.037999 | 0.037999 | 134 (6.98%) |

The +60M and +220M policies tie at 64 falls; +60M has substantially better
tracking. Final error is **18.3% worse than +20M** and **3.35× TB 200M**.
The tiny +0 evaluation difference does not indicate a failed restore: arrays
are identical, while long numerical trajectories can differ across evaluations.
TB's final checkpoint also loses survival; its 180M/200M controls remain the
appropriate references.

A **20,000-resample paired bootstrap** clusters all 15 conditions by each of
128 model/reset indices. Physical error and fall-rate differences are:

| Comparison | Error difference, 95% interval (m/s) | Falls difference, 95% interval (percentage points) |
| --- | ---: | ---: |
| +20M minus parent 260M | −0.002232 [−0.003051, −0.001485] | −1.146 [−2.500, +0.208] |
| +60M minus parent 260M | −0.000987 [−0.002041, +0.000013] | −1.979 [−3.385, −0.677] |
| Final minus +20M | +0.015828 [+0.011371, +0.021385] | +3.906 [+1.823, +6.250] |
| Final minus parent 260M | +0.013596 [+0.009143, +0.019183] | +2.760 [+0.521, +5.156] |

Thus the early tracking gain and later regression exceed this selection set's
sampling variation. These intervals use one fine-tuning seed and the selection
set; they do not establish held-out or independent-seed performance.

![Added training, tracking, survival, PPO updates and direction errors](../../results/wr2_walking/wr2_paper_20261017_080025_seed1/review_20261017/finetune_overview.png)

## 3. Directional response and recovery

The following are **TB-equivalent m/s**: mean X for forward/backward, mean Y
for lateral motion, total XY speed for standing.

| Command / measure | WR2 +20M | WR2 +60M | WR2 final | TB 200M |
| --- | ---: | ---: | ---: | ---: |
| Forward +0.10: mean X | +0.0590 | +0.0619 | +0.0508 | +0.0905 |
| Backward −0.10: mean X | −0.0467 | −0.0362 | −0.0156 | −0.0736 |
| Right −0.05: mean Y | −0.0116 | −0.0100 | **+0.0020** | −0.0320 |
| Left +0.05: mean Y | +0.0101 | +0.0047 | +0.0111 | +0.0350 |
| Standing: XY speed | 0.0913 | 0.0922 | 0.0679 | 0.0320 |

Final backward motion reaches only **16%** of command versus TB's **74%**;
mean rightward response has the wrong sign. Even the retained +20M policy
reaches only **20–23%** of lateral commands. A near-zero signed standing mean
would hide its actual motion. Final standing speed improves, but standing falls
rise from **3/128 at the parent to 12/128**; TB 200M has zero.

At +60M, falls split **45 fixed / 2 standing / 17 transition**, versus final
**102 / 12 / 41**. Mean length falls from **813.2 to 785.1 of 824 ticks**.
The +240M spike is broad: **426 falls across 72/128 model/reset indices**, with
seven indices failing all 15 conditions. Final has **155 falls across 45 indices**.
This is more than a single difficult draw.

The parent's earlier controlled ablation showed a push-recovery problem on
selected draws. The new nominal replays cannot determine which perturbations
cause the new falls.

## 4. Nominal walking also regresses

Replayed forward, left and standing at parent 260M, +20M, +60M and final:
**3 conditions × 8 episodes × 4 policies = 96 episode-condition replays**,
824 ticks, nominal model/fixed reset/no pushes/no noise. All finish with
**zero falls, nonfinite episodes or neither-foot support samples**.

| Nominal policy | Forward error | Left error | Standing speed | Forward mean X | Left mean Y |
| --- | ---: | ---: | ---: | ---: | ---: |
| Parent 260M | 0.05447 | 0.07384 | 0.05038 | +0.06950 | +0.00167 |
| WR2 +20M | **0.04992** | **0.06773** | 0.04203 | +0.07385 | +0.00709 |
| WR2 +60M | 0.05285 | 0.07139 | 0.04403 | +0.07230 | +0.00126 |
| WR2 final | **0.07617** | **0.08166** | **0.02075** | +0.05214 | +0.00099 |
| TB 200M | **0.02673** | **0.02606** | 0.02471 | +0.09089 | +0.04068 |

All values are TB-equivalent m/s. Final nominal forward error is **52.6%
worse than +20M** and **2.85× TB**. The nominal lateral gap exists without
pushes or noise: +20M achieves **14%** of the left command, TB **81%**.

Within WR2, final nominal standing changes to **88.96% double support**, versus
about **0.7%** for the parent/+20M/+60M. It is quieter than TB nominal standing,
but survives the primary variations less reliably. Final nominal walking remains
**98.42% single support forward / 84.95% left**. These traces provide no evidence
that the old frequent nominal loss of support returned.

Sampled forward motor torque RMS falls from **0.751 Nm at +60M to 0.546 Nm
at final**. Left-command foot XY paths shrink from **0.529/0.810 m** to
**0.221/0.364 m**. Leg-reference RMSE improves slightly despite worse movement;
TB's raw reference RMSE is higher than WR2's. Soft imitation error is therefore
insufficient for judging gait quality or changing the action range.

![Nominal tracking and standing support](../../results/wr2_walking/wr2_paper_20261017_080025_seed1/review_20261017/nominal_comparison.png)

These are CPU diagnostics with Brax **0.14.1**, JAX **0.6.2**, MJX **3.14.0**,
Python **3.11.14**, Flax **0.11.2** and Orbax **0.12.6**. Repeated fixed-reset
WR2 episodes repeat one trajectory. TB's previously captured nominal evaluation
uses its frozen Brax fork/MJX **3.2.4**, native random heading and unconditional
motor-home offsets **±0.02 rad**. Raw torque averages **21 WR2 / 30 TB motors**
with different gearing; cross-robot torque/RMSE ratios do not diagnose physics.

## 5. PPO updates are large; a smaller step has direct evidence

Confirmed against the actual **Brax 0.14.1** distribution/loss/optimizer and
[WR2 trainer](../../wr2/locomotion/train.py): `training/kl_mean` is analytic
**KL(rollout policy || updated policy)**, summed across the **12 raw Gaussian
action coordinates** and averaged across states and optimizer updates. It is
computed before each minibatch gradient against that rollout's original policy.
Epoch logs average many batches; these are not checkpoint-to-checkpoint KLs.

| Added transitions | Native epoch mean joint KL | Mean latent standard deviation | Native forward return |
| --- | ---: | ---: | ---: |
| +20M | 0.0211 | 0.0799 | 372.49 |
| +60M | 0.0520 | 0.0768 | 391.94 |
| +100M | 1.2522 | 0.0872 | 388.94 |
| +120M | **5.3177** | 0.0800 | 382.13 |
| +220M | 0.9651 | 0.0880 | 395.78 |
| +240M | 1.5430 | 0.1691 | 349.47 |
| +280M | 2.1057 | 0.0862 | 410.93 |
| +300M | 0.6116 | 0.0815 | 404.43 |

The fixed schedule maps to **`LRSchedule.NONE`**: YAML `desired_kl: 0.01`
does **not** constrain these updates. PPO's 0.2 surrogate ratio clipping is not
a KL bound. TB also uses Adam without a KL stop; its released cosine LR is
effectively 1e−4 over this budget. This is a measured transfer problem, with
no new confirmed adapter/loss bug. TB's fork does not log KL, so its late
survival regression cannot be assigned the same cause from these records.

The mean latent scale starts and ends near **0.08**, with a temporary spike at
+240M; there is no measured average-scale collapse during this fine-tune.
Positive entropy loss alone is insufficient evidence: successful TB 200M
also logs **0.01244**, versus WR2 final **0.01067**.

Native forward return rises **8.6% from +20M to final** while benchmark tracking
worsens. From restored +0 to final, angular XY stability reward rises
**2.57→18.73**, while energy penalty becomes **−1.46→−0.58**. This is compatible
with reduced movement; reward telemetry does not establish the cause or justify
changing the native paper weights.

### Actual rollout, identical-batch learning-rate comparison

Loaded **+60M** and collected one real noisy/randomized MJX rollout:
**32 environments × 20 steps after 200 warmup ticks**, seed **17017**. Reused
identical data, initial parameters, permutations and update RNG for both rates,
with fresh Adam and native **4 passes × 4 minibatches**. Checked the native KL
against an independent Gaussian formula. Batch and updated parameters are finite.

| Fixed LR | Mean pre-gradient joint KL | Post-batch joint KL | Ratios outside [0.8, 1.2] | Deterministic action change RMS |
| --- | ---: | ---: | ---: | ---: |
| Current 1e−4 | 0.24047 | **0.28847** | **81.41%** | 0.009256 |
| Proposed 1e−5 | 0.00978 | **0.01684** | **34.06%** | 0.002308 |

The lower LR reduces post-batch KL **17.1×** and action change **4.0×**; latent
scale stays approximately **0.072** for both. Small action changes can still
compound into substantial joint KL when exploration is narrow.
The small CPU batch and fresh Adam differ from the archived GPU batch/state;
long-run tracking and survival effects remain untested. This directly supports
a smaller-update experiment from this checkpoint.

## 6. TB source comparison and physical checks

The successful control is local TB `toddlerbot_walk_ppo_20261015_194347`, frozen
locomotion source **1982ef3**, Brax fork
**14147806f6c25724ba830ba36764df3243889b0d** and its actual episode exports.
The original TB source was dirty; prior verification matched the captured
11 locomotion files, 12 model/reference files and lock, not a complete historic
tree. The actively developed TB checkout uses a different recipe.

Rechecked WR2 `walking_env.py`, `paper_rewards.py`, `paper_evaluation.py`,
`ppo.py`, `train.py` and servo fitting against frozen TB `mjx_env.py`,
`walk_env.py`, `paper.gin`, `train_mjx.py`, `walk_zmp_ref.py` and its Brax loss:

- Moving leg-reference imitation **weight 5**, static `q_home`, delayed actions,
  observation/reward clocks and critic reference information remain aligned.
  The actor learns imitation through rewards/advantages.
- Swish, state-dependent NormalTanh, **1024 × 20** rollout, **256 × 4 × 4**
  updates, GAE/advantage normalization, native entropy/log probability, all
  **25 paper reward weights**, and actual value loss **0.25 × MSE** are retained.
- Landing normalization and summed-foot-force support fixes remain active.
  Model hash, Newton **10**, line search **4**, **5 ms × 4** physics steps and
  randomized-physics refresh match the parent. The physics configuration is
  unchanged through this fine-tune.
- Checked actual recorded servo damping against
  [`training_parameters`](../../wr2/tools/servo_sysid/fit.py): passive joint
  damping **0.17362248** plus explicit controller `kv=0.5` equals the fitted
  effective damping **0.67362248**. It is **not double counted**. Identified
  armature, torque-speed behavior and geometry remain physical differences;
  their loaded tracking limits deserve measurement if optimizer stabilization
  leaves the nominal gap.

## 7. Recommended next trial

1. **Retain +20M and +60M**; use +60M's lower-fall policy as the next start.
   +220M has equal falls and worse tracking; final has worse survival/tracking.
2. Run **another 300M at fixed LR 1e−5, seed 2**, restoring +60M. The LR is an
   explicit departure from TB's 1e−4, supported by the 560M training budget,
   sustained large KL and the identical-batch comparison. Keep the existing
   paper recipe, full variation and evaluation protocol.
3. Retain every **~20M** checkpoint. Check actual epoch KL and all command
   directions; success requires lower update size with improved tracking and
   survival. Keep **≤1% falls** eligibility and held-out seed **22102** for
   qualifying candidates. TB 200M's **0.0370 m/s equivalent error / 0.417%
   falls**, directional response and transitions remain the parity reference.
4. If KL becomes controlled but nominal/lateral tracking remains poor,
   measure loaded actuator target tracking and reference feasibility. The
   current evidence does not justify changing damping, hardware limits or
   paper reward weights.

Use the regular trainer on the GPU host:

```bash
uv run --extra training python -m wr2.locomotion.train \
  --config wr2/locomotion/configs/ppo_walking_paper.yaml \
  --restore-checkpoint results/wr2_walking/wr2_paper_20261017_080025_seed1/checkpoints/000060026880 \
  --learning-rate 1e-5 \
  --seed 2
```

The original +60M checkpoint directory must be present under ignored `results/`
on the GPU host. This proposal needs no additional training script or YAML.

## Evidence and references

All new JSONs, analysis scripts and PNG/PDF figures remain local under ignored
[`review_20261017`](../../results/wr2_walking/wr2_paper_20261017_080025_seed1/review_20261017).
The report and training guide are text documentation.

- [Independent metric audit](../../results/wr2_walking/wr2_paper_20261017_080025_seed1/review_20261017/metrics_audit.json)
  and [same-batch PPO update probe](../../results/wr2_walking/wr2_paper_20261017_080025_seed1/review_20261017/ppo_update_probe.json).
- [Parent result and push/reference diagnostics](wr2_paper_result_20261017.md)
  and [frozen TB control audit](toddlerbot_transfer_review_20261016.md).
- [ToddlerBot paper, Tables VI/VII](https://arxiv.org/html/2502.00893v2): recipe
  and original budget; implementation comparison uses local frozen source.
- [Schulman et al., PPO](https://arxiv.org/abs/1707.06347) and
  [OpenAI Spinning Up PPO documentation](https://spinningup.openai.com/en/latest/algorithms/ppo.html):
  ratio clipping does not guarantee a small policy change; explicit KL controls
  are a standard stabilization option. The proposed trial changes only fixed LR.
