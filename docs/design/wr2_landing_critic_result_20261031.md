# Landing-state critic: 400M cumulative result — 2026-10-31

## Decision

**Keep the accepted fresh 200M policy as the overall baseline.** Adding the
landing reward state to the critic produces useful intermediate candidates,
but does not improve aggregate walking tracking or establish a slower gait.
The final checkpoint is not the selected checkpoint and is not an overall
walking upgrade.

The missing direct reward information is now exposed and used. The result
**weakens missing critic memory as the leading explanation for fast cadence**:
value errors around landing payments do not consistently improve, and cadence
stays near 2.06 TB Hz. Reference-contact timing is the next reward priority.
This is a proposal; no new training configuration or reward change is made
in this review.

## Verified run and metrics

Run: `results/wr2_walking/wr2_paper_landing_critic_20261030_211823_seed3`.

- Clean source **571da9a**, v0.36.0; RTX 5070, Python 3.10.19, JAX 0.6.2,
  Brax 0.14.1, MuJoCo/MJX 3.14.0. These match the unchanged continuation's
  runtime. Time through final training evaluation: **6,573.7 s / 1.83 h**.
- Correct parent: fresh size-aligned `000200048640`. Saved `initial_params`
  equal the parent with exactly six zero-weight critic inputs appended.
  Actor parameters, old critic weights and normalizer fields are preserved.
- Actor remains 1005 inputs; critic grows 1515→1521. The GPU restore report
  records zero actor prediction error and zero float64 critic mapping error.
  Native GPU critic rounding difference is 0.000495; it is not a mapping change.
- Saved environment, rewards, network, PPO and randomization match the parent
  and completed control, apart from `critic_landing_state: true` and run labels.
  Recorded physics/model hash and actuator descriptions match exactly.
- **200,048,640 added transitions / 195,360 added Adam updates**; cumulative
  **400,097,280 transitions / 390,720 updates**. Native parameter restoration
  starts fresh Adam/RNG/environment state, as in the control.
- All 15 checkpoints, 16 training records and 99 rollout records are finite.
  Recomputed all 36 evaluation reports: **540 conditions / 69,120 attempts**.
  Episode flags, command vectors, sample partitions and selected summaries
  agree. Maximum group mean/std rounding differences: 2.77e-6 / 5.46e-7.
  Native evaluation reward components reconstruct return within 0.000582.

The selected checkpoint is **+40,009,728**, cumulative **240,058,368**.
`best_params` matches it exactly. Final `params` matches the continuation's
`000200048640`, cumulative 400,097,280. All 15 learned checkpoints have worse
selection-seed tracking than the initial policy. `beats_initial_tracking` is
correctly false: the selector ranks learned candidates separately from the
saved initial baseline.

## Standard randomized GPU result

Use held-out seed 22102: 128 attempts per condition, ten fixed nonzero commands
through 0.10 TB m/s, standing and four transitions. Both policies use the same
824-tick horizon, model/reset/actuator variation and observation noise, with
no pushes. Parent numbers below are its re-evaluation inside this run, avoiding
small recompilation differences from older reports.

| Held-out metric | Parent 200M | Selected +40M | Unchanged control selected +53M |
|---|---:|---:|---:|
| Desired-frame walking XY error, TB m/s | **0.03965** | 0.04018 | 0.04050 |
| Body-frame walking XY error, TB m/s | **0.03975** | 0.04017 | 0.04031 |
| Forward 0.10 desired-direction speed, TB m/s | **0.09439** | 0.09275 | 0.09084 |
| Forward-transition walking speed, TB m/s | **0.09312** | 0.08375 | 0.08544 |
| Fixed-walking leg-reference RMSE, rad | 0.16310 | 0.16362 | **0.16210** |
| Fixed-walking contact-phase agreement | **52.73%** | 52.26% | 52.37% |
| Fixed-walking absolute heading error | 4.79° | **3.91°** | 4.88° |
| Fixed-walking foot self-contact | 4.13% | **1.07%** | 2.03% |
| Standing desired-frame XY error, TB m/s | 0.02089 | **0.00663** | 0.00857 |
| Standing double support | 58.76% | **94.40%** | 92.12% |
| Falls | 0/1,920 | 0/1,920 | 1/1,920 |

Selected walking error worsens **1.34%** relative to its parent. Episode-index
clustered bootstrap change is +0.000530 TB m/s, 95% [0.000159, 0.000906].
Compared with the control's selected policy, the improvement is only 0.000322
TB m/s, with 95% [-0.000656, +0.000012]; this does not establish a tracking gain
over the control. These intervals describe evaluation variation for these
trained policies, not training-seed uncertainty.

Selected no-push falls are **0/3,840** across both seeds; nonfinite count is
zero. Optional held-out push stress gives 1,147/1,920 falls (59.74%), versus
the re-evaluated parent's 1,074 (55.94%). Pushes were absent from training;
this remains outside the current nominal-walking priority.

Final has selection-seed reports, not a separate held-out confirmation:
desired-frame walking error **0.04100**, +3.20% versus its same-seed parent;
leg RMSE 0.17697 versus 0.16332, and one fall in 1,920 attempts. Final forward
0.10 mean speed is 0.10203, but instantaneous XY error increases. Matching or
overshooting mean speed does not establish accurate velocity tracking.

## Original-policy Mac gait comparison

Generated 12 nominal native episodes at +13M/+40M/+107M/+160M/+200M, commands
0.10/0.20, plus 0.15 for selected/final. **9,888 transitions are finite**, with
no falls and exact target reconstruction. All actions come from the trained
policies; no reference angles or prescribed joint actions replace them.

These use the same exact production reference rows, CPU/high precision,
seed 22102, 20 TB-second duration and 1.5 TB-second warmup as the earlier
reviews. Speed divides physical velocity by `s=.8240547565448791`; distances
divide physical length by `s²`. Contact uses the same 0.612930 N upward-force
threshold and ten-sample landing qualification. Placement uses cached collider
centers along the initial commanded heading; body motion uses integrated root
poses and actual shoulder-anchor geometry.

| Nominal metric | Parent 200M | Selected +40M | Final +200M |
|---|---:|---:|---:|
| 0.10 desired-direction speed, TB m/s | 0.09937 | 0.09883 | 0.11327 |
| 0.10 median landing lead L/R, TB cm | 3.08/1.34 | **2.33/2.51** | 4.54/0.53 |
| 0.20 desired-direction speed, TB m/s | 0.19129 | **0.19277** | 0.17840 |
| 0.20 median landing lead L/R, TB cm | 5.98/2.77 | 5.99/3.05 | 5.28/2.81 |
| 0.20 same-foot cadence, TB Hz | 2.060 | 2.060 | 2.060 |
| 0.20 executed hip span L/R | 10.82°/12.61° | 11.04°/12.79° | 10.77°/10.84° |
| 0.20 absolute net turn | **1.57°** | 2.10° | 6.36° |
| 0.20 torso roll 5–95% span | 4.99° | 5.22° | **3.48°** |
| 0.20 root Z span, TB mm | 7.14 | 6.81 | **6.61** |
| 0.20 shoulder Z span L/R, TB mm | 14.53/12.16 | 14.68/12.24 | **10.69/8.47** |

Selected improves low-command placement and nominal forward pitch: at 0.10,
mean pitch drops 4.36°→2.18°. Its higher-command placement remains uneven.
Final improves rocking but loses forward progression and heading accuracy;
at 0.10 its right lead becomes only 0.53 cm. It is not a combined fix.

The **+160M checkpoint** is another candidate worth retaining: at 0.20 its
speed is 0.19550, leads 3.99/4.73 cm, and net turn 5.50°. At 0.10 its leads are
2.99/1.70 cm and speed 0.10597, so even placement is not consistent across
commands. Its aggregate GPU error is also worse than the parent. Neither a
single good command nor a single nominal episode justifies promotion.

Cadence stays at a median 20 ticks for selected/final, **2.060 TB Hz**, versus
the 30-tick reference, **1.373 TB Hz**. Selected 0.20 contact intervals are
10–11 samples; final intervals are 10–12, close to the ten-sample reward
qualification. Reference-contact agreement does not improve. At a fixed speed,
2.06 rather than 1.37 same-foot cycles per second implies roughly one-third
shorter same-foot stride. Alternating landing events alone do not imply equal
lead or reference-timed walking.

## Did the critic information help value learning?

Source and saved parameters confirm correct delivery and use:

- `walking_env.py::_critic_landing_features` exposes current supported air
  duration, queued credit and capped qualification progress for both feet.
  Updated next-state memory is appended after the old critic history and
  cleared with environment memory on reset/fall/timeout.
- `ppo.py::expand_landing_state_params` preserves old predictions at restore.
  New input-row norms grow from zero; selected pending-credit rows have L2
  norms 3.64/3.37, final 9.26/8.44. Actor inputs and reward formulas stay intact.
- Mean sampled training value loss is **0.06462**, versus control **0.06380**;
  last-five means are 0.06466/0.06445. These do not show a consistent loss gain.
  Losses are on changing on-policy batches, so they are not a paired accuracy
  test. KL is mostly 0.024–0.030, with one interval-average spike to 0.0832 at
  +120M; all saved parameters/metrics remain finite. No causal attribution to
  that spike is established here.

Ran an additional native **stochastic** snapshot: eight randomized/noisy
episodes per command, fixed 0.10/0.20, 320 ticks, matching seeds and physics,
for parent/control-selected/new-selected/new-final. All **20,480 transitions**
are finite with no falls. Old critics are zero-extended into the identical new
environment, preserving their predictions. Evaluate
`r + gamma*(1-done)*V(next) - V(before)` after the same warmup.

| One-step TD RMSE at actual payouts | Parent | Control selected | New selected | New final |
|---|---:|---:|---:|---:|
| 0.10 command | **0.184** | 0.207 | 0.337 | 0.256 |
| 0.20 command | 0.273 | 0.299 | **0.266** | 0.294 |

The original critic already predicts usual landing payments reasonably well
despite lacking the explicit counters. New selected improves slightly at 0.20
and worsens at 0.10. All-tick TD RMSE also does not consistently improve.
Zeroing the new memory changes selected payout-state values by about 0.11–0.12
on average, confirming use, not demonstrating improved accuracy. Ablation can
be out of distribution and is not a gait test.

These are frozen-policy snapshots on each policy's own trajectories, not
identical training batches or an estimate over all training commands/seeds.
They do not prove the original omission harmless. They **do not support it as
the primary explanation** for the persistent cadence. In these current
snapshots, air time contributes about 35–39% of return; older high-contribution
figures from different checkpoints should not be reused as current evidence.

## TB comparison and next priority

Re-read local **`paper-walking-baseline`** at `a4f693d`:

- `mjx_env.py::_get_obs`: phase, joint state/error, velocity, actual/reference
  contact and pushes reach the privileged critic; raw air-time counter does
  not. TB pays on first filtered landing, without WR2's pending-credit stage.
- `walk_env.py::_reward_feet_contact`: the same two-foot contact-reference
  equality count used by WR2. Its existing reward requests reference support
  timing; it does not directly request equal forward landing distances.
- `paper.gin`: **XY/leg/contact weights 5/5/1**. Current WR2 uses **15/15/1**.
  Thus contact/XY and contact/leg coefficients are one-third of TB's ratios.
  This is an intentional accumulated recipe difference, not a new size law.
- TB's saved cycle is 0.72 s; WR2's normalized/discretized reference is about
  0.728 TB s. TB Paper nominal 0.10 same-foot cadence was about **1.43 Hz**,
  substantially slower than WR2's 2.06. TB itself is not perfectly symmetric
  in the previously measured nominal sample.
- TB at 200M has body-frame error **0.03702**; new selected WR2 is **0.04017**,
  about 8.5% higher. Robot/randomization differences prevent a controlled
  robustness ranking. TB 300M uses 234,480 Adam updates, versus this WR2
  policy's cumulative 390,720; this is not a simple shortage of transitions.

**Recommended next experiment: `feet_contact` 1→3, with the current critic
state and all other rewards/physics/settings preserved, for 200M from the
accepted parent.** The completed landing-critic run supplies the matched
contact-weight-1 comparison from that same parent and budget. Keep all
intermediate checkpoints.

Weight 3 restores TB's contact/XY and contact/leg coefficient ratios while
retaining WR2's useful XY/leg weights. It is a focused relative-emphasis test,
not complete TB parity: contact/air ratio becomes three times TB's. Retain the
user-requested qualified landing rule and air weight 500. Do not add a stride
reward, change actor features, modify the ZMP reference or alter actuators in
the same experiment.

Success requires better reference-contact agreement and cadence approaching
the reference while preserving commanded speed/heading, improving positive
bilateral placement at 0.10/0.15/0.20, and keeping rocking/contact/falls within
the accepted baseline. A larger contact coefficient does not guarantee these
outcomes; this run justifies testing the existing TB timing objective before
inventing a different one.

References: local TB branch sources above;
[TB paper v2, Tables VI–VII](https://arxiv.org/html/2502.00893v2);
[Asymmetric Actor Critic for Image-Based Robot Learning](https://arxiv.org/abs/1710.06542);
[experiment setup](wr2_landing_critic_experiment_20261030.md);
[unchanged continuation review](wr2_size_continuation_result_20261030.md).

## Artifacts and visual review

Ignored `review_20261031/` contains audited `analysis.json`, `gait_analysis.json`,
contact-run counts, 12 nominal traces with provenance, stochastic value traces
and `landing_values/report.json`, reproducible analysis/rollout scripts and
comparison PNG/PDF. No result binaries or TB source are added to Git.

View the selected +40M candidate on the Mac:

```bash
uv run --no-sync mjpython -m wr2.tools.view_walking \
  --run-dir results/wr2_walking/wr2_paper_landing_critic_20261030_211823_seed3 \
  --command .1 0 0
```

The default `best_params` is the verified +40M selected checkpoint.
Repeat at `.2 0 0`. Retain the accepted fresh 200M policy as the overall
comparison. This review does not update completed-training entries in
`CHANGELOG.md` before user confirmation.
