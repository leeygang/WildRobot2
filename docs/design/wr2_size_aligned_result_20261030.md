# Fresh size-aligned 200M result — 2026-10-30

## Decision

**The fresh run improves walking and removes the almost fixed right-hip
strategy. It also substantially reduces nominal shoulder rocking.** The final
checkpoint is the best aggregate tracking candidate, while 120–160M have more
even nominal forward placement. Validate those retained policies on the GPU
before choosing a walking policy or another training change.

This was one fresh run with the integrated size-aligned recipe. The fresh
current-recipe control was intentionally skipped. The evidence establishes
practical improvement and shows that the existing physical model can execute
bilateral swing; it does not separate the effects of fresh initialization,
learning horizons, optimizer exposure and contact threshold.

## Verified run and metrics

Run: `results/wr2_walking/wr2_paper_size_aligned_20261030_085111_seed3`.

- **200,048,640 transitions /195,360 Adam minibatch updates**, starting fresh.
  `restore_checkpoint` and its hash are null. Initial zero-input mean logits
  are zero, rather than inherited right-hip endpoint means.
- Clean source **72eb805**, RTX5070, Python3.10.19, JAX.6.2, Brax.14.1,
  MuJoCo/MJX3.14.0; GPU matrix precision retains its native default.
- Saved typed config equals the prepared v0.34.0 config. Versus the preceding
  leg15 continuation, the only recipe differences are episode824, rollout16,
  gamma.9637121762, lambda.9396526368 and support threshold.612930N.
  Physics/model hash, actuator model, grounded reset/observation home and
  model-home hip target centers match exactly.
- All **15 retained checkpoints**, initial, selected and final parameters are
  finite. `best_params`, final `params` and checkpoint **000200048640** match
  exactly. Recomputed eligibility/selection reproduces the chosen final policy.
- Audited **36 reports /540 conditions /69,120 attempts**, including initial,
  no-push, push and held-out reports. Episode flags, sample partitions,
  commands and summaries are consistent. Maximum group mean/std rounding
  differences are 2.85e-6/3.97e-7.
- All16 training-metric and99 rollout records are finite. Native reward
  components reconstruct evaluation return within .000515. Use
  `episode/sum_reward` or `reward_per_step`, rather than the redundant rollout
  `episode/reward` field, which is zero.
- Time through final training evaluation: **7,990s /2.22h**. Prior leg15 took
  1.98h; TB Paper200M took2.14h. This includes compilation/evaluation, and is
  not a pure optimizer throughput benchmark. Shorter rollouts produce about
  25% more updates per transition; TB200M had156,320 updates.

The source uses moving executed-leg reference MSE, fixed action centers and
the requested sustained-landing rule. There are no policy-action overrides in
the training or nominal review. Landing credit still requires ten contact
samples, with gamma-derived compensation1.39468. Reference remains1000 frames,
and hardware timing remains20ms /5ms×4 physics.

KL averages .0174–.0281, final.0271. With fixed LR1e-4, `desired_kl=.01` is
a monitor; it is not an adaptive stopping gate. The finite learning curves
and improved retained policies do not identify a new PPO implementation bug.

## Standard randomized GPU comparison

All values below use held-out seed22102. The suite has ten fixed nonzero
commands through .10 TB-equivalent m/s, standing and four transition schedules.
There is model/reset/actuator variation and observation noise, with zero pushes.
The same normalized evaluator duration is824 physical ticks for all WR2 policies.

Parent is the model-home **+173,588,480** policy. Prior leg15 is its selected
**+186,941,440** continuation. Fresh aligned is its own **200,048,640** policy.

| Metric | Parent | Prior leg15 selected | Fresh aligned200M |
|---|---:|---:|---:|
| Desired-frame XY error, TB m/s | .04295 | .04258 | **.03966** |
| Body-frame XY error, TB m/s | .04339 | .04259 | **.03976** |
| Forward .10 desired-direction speed, TB m/s | .09241 | .09057 | **.09436** |
| Forward .10 desired-frame error, TB m/s | .04348 | .04240 | **.03647** |
| Forward transition walking speed, TB m/s | .0913 | .0872 | **.09305** |
| Fixed-walking leg-reference RMSE, rad | .21058 | .21707 | **.16309** |
| Fixed-walking heading error, degrees | 5.73 | **4.49** | 4.75 |
| Fixed-walking foot self-contact | 3.11% | **1.58%** | 4.13% |
| Fixed-walking contact-phase agreement | 52.6% | 43.5% | 52.7% |
| Standing XY error, TB m/s | .03058 | .03138 | **.02106** |
| Standing double support | 5.6% | 5.8% | **58.7%** |
| Falls, held-out seed | 0/1,920 | 1/1,920 | **0/1,920** |

Desired-frame error improves **7.65% versus parent**, and6.84% versus prior
leg15. Paired episode-index cluster bootstrap across ten fixed conditions:
parent change −.003286 TB m/s,95% [−.003762,−.002815]; prior leg15 change
−.002914,95% [−.003374,−.002451]. These intervals quantify evaluation variation
for these policies, not uncertainty across training seeds.

The selected policy has **0/3,840 falls** across selection and held-out seeds.
Native forward return increases276.38→571.15 within this run, but should not
be ranked directly against older runs with different episode/gamma-derived
payment settings. Air time contributes205.55/571.15 =36.0% of final native
forward return; this share alone is not evidence of its PPO gradient influence.

The self-contact increase occurs across multiple commands: forward.10 is2.28%,
rightward.05 is6.49%, and backward.05 is5.22%. Collision physics and its.1N
force cutoff are unchanged, so this metric is not redefined by the lower
floor-support threshold. Other contact/phase metrics use different native
support thresholds; common-threshold reconstruction is used for nominal
placement below before comparing strategies.

Held-out push stress **worsens**: 1,059/1,920 falls(55.2%), versus parent
994(51.8%) and prior leg15 855(44.5%). Push amplitudes/intervals are the same
TB-normalized bounds. Pushes remain absent from training. Better no-push
walking does not imply better perturbation recovery.

## Native Mac gait evidence

Generated14 original-policy rollouts at26.7/80/120/160/186.7/200M: forward.10
and.20 at each, plus.15/.30 at final. All **11,536 native transitions are
finite**, with zero falls and exact delivered-target reconstruction including
the timeout tick. These are nominal CPU MJX/high-precision screens, not GPU
randomized qualification. Native reference rows match production selection.

All geometric distances below are in TB-equivalent units. Actual physical
velocity is `v_TB*s`, time is `t_TB*s`, and length is `length_TB*s²`,
with `s=.8240547565`. Body motion uses integrated root poses; planting uses
cached final-substep collider centers/forces. Warmup is1.5 TB seconds.
Both old/new WR2 traces use common.612930N support classification and require
ten consecutive supported samples. Landing lead is the advancing foot's
distance ahead of the opposite foot, projected onto the initial commanded
heading; it is distinct from the same foot's distance between landings.

### Right hip now swings; placement improves but is not uniformly even

| Nominal .20 metric | Parent | Prior leg15 | Fresh final200M |
|---|---:|---:|---:|
| Desired-direction speed, TB m/s | .16225 | .16654 | **.19129** |
| Hip target5–95% span, left/right | 27.12°/1.68° | 24.71°/1.68° | **16.31°/16.08°** |
| Executed hip span, left/right | 26.61°/1.44° | 24.21°/1.15° | **10.82°/12.61°** |
| Right action below−.95 | 80.6% | 78.5% | **0%** |
| Median landing lead, left/right | 8.01/2.05cm | 9.08/.25cm | **5.98/2.77cm** |
| Net turn over episode | 18.5° | 7.4° | **1.6°** |

At .20 TB command(.16481 physical m/s), final nominal forward speed is
.19129 TB/.15764 physical m/s. Alternation of qualified left/right landing
events is100%; both feet swing and pass, although the left lead is still
larger. Same-foot advances are9.36/9.24cm, a separate stride measure.

Physics is identical across these policies. The executed right hip now moves
12.61°, so a physical range/torque prohibition cannot explain its old1.15°
excursion. Loaded response is still imperfect; that does not establish which
of the combined training changes enabled the improved coordination.

### Intermediate policies have more even placement

| Checkpoint | .10 lead L/R, cm | .20 lead L/R, cm | .20 speed, TB m/s | .20 net turn |
|---|---:|---:|---:|---:|
| 120,029,184 | 3.46/1.40 | **5.04/4.43** | .19566 | 5.7° |
| 160,038,912 | **2.50/2.32** | **4.12/4.97** | **.19950** | 10.8° |
| 186,712,064 | 3.51/1.21 | 6.66/2.33 | .19086 | 3.7° |
| 200,048,640 | 3.08/1.34 | 5.98/2.77 | .19129 | **1.6°** |

The selector optimizes aggregate XY tracking, not placement symmetry. The
final selection is correct for its metric, but should not imply that every
gait measure peaks at final. Nominal .20 self-contact is0% for160M/final;
120M has1.31%. Repeat under randomized GPU conditions before selecting160M
or120M as better walking policies.

Cadence remains faster than TB: final same-foot landing frequency is
**2.06 TB Hz**, versus native reference1.37Hz and TB Paper nominal.10 about
1.43Hz. At a given speed, faster cadence means shorter placement/strides.
This is an observed remaining gait difference, not proof that one particular
reward coefficient causes it. TB's measured .10 landing leads are4.74/1.68cm;
TB itself is not perfectly symmetric in that nominal sample, which also turns
26.9°. TB foot centers average the two cached collider centers on each same
foot; the two left/right collider pairs are kept separate before that average.

### Shoulder rocking substantially decreases

| Nominal .20 body measure | Parent | Prior leg15 | Fresh final | TB Paper200M |
|---|---:|---:|---:|---:|
| Roll5–95% span | 12.25° | 9.37° | **4.99°** | 18.26° |
| Root Z span, TB mm | 11.14 | 11.25 | **7.14** | 6.32 |
| Shoulder Z span L/R, TB mm | 25.34/32.11 | 20.62/24.91 | **14.53/12.16** | 18.36/24.05 |

Shoulders are measured at rigid torso-attached shoulder joint anchors, using
actual WR2/TB geometry and integrated root orientation/translation. This
isolates body rocking from arm motion. Fresh root bobbing is still about13%
above TB at.20; visible shoulder displacement is smaller in these native
samples because torso roll is much smaller. The final policy stays within
TB's ±.1rad roll band throughout the post-warmup .10/.15/.20 samples.

This improvement occurred with roll weight.5 and the original angular band.
The narrower shoulder-width band is no longer the immediate priority.
Native TB body-motion comparison uses the saved200M Paper policies and verified
Paper model; runtimes/models differ, and these are single nominal episodes.

### .30 is outside the trained range

Training maximum is.20 TB m/s and the native lookup saturates at that forward
command. At requested.30, final remains upright but turns **−100.1°**, has
only94.0% qualified alternation and larger shoulder motion. This is not a
qualified high-speed gait. Large heading changes also prevent treating its
initial-heading landing projection as a simple anatomical step-length measure.
Assess the trained range first; higher-speed training needs an explicit
command/reference-range decision.

## TB Paper progress and implementation comparison

Local `paper-walking-baseline` confirms Paper episode1000, rollout20,
gamma.97, fixed-home residual actions, moving leg-reference MSE and XY/leg
weights5/5. WR2 keeps hardware/model geometry and adds its established XY15,
leg15, desired-heading feedback/reference-frame reward and qualified landing.
Those are explicit departures; size alignment does not make every setting
identical to TB.

| Raw transitions | Fresh WR2 body XY error, TB m/s | TB Paper body XY error |
|---|---:|---:|
| About40M | .05092 | .04771 |
| About80M | .04216 | .03965 |
| About120M | .04047 | .03952 |
| About160M | .03992 | .03824 |
| About200M | .03987 | .03702 |

Use selection seed12102 and the same body-frame statistic here. At200M WR2's
remaining aggregate error gap is **7.7%**, compared with about15% for the prior
selected leg15 policy. TB200M had8/1,920 falls; differing randomization/model
physics prevent interpreting that as a controlled WR2/TB stability ranking.
Both seed/budget and normalized simulation time/optimizer exposure matter:
WR2 has more normalized simulated seconds and more updates at equal raw steps.

TB's body-frame speed does not measure desired-direction travel when turning.
In native nominal.20, TB cached body speed is.17546 but initial-heading
pose-derived speed is.12252, with72.8° net turning. Fresh WR2 has.19129
desired-direction speed and1.6° turn in its nominal sample. This illustrates
the benefit of heading feedback and the importance of measuring both frames;
it is not a matched randomized proof of overall TB parity.

## Next step

After the user's Mac visual confirmation, final is accepted as the current
walking baseline. The [prepared200M continuation](wr2_size_continuation_20261030.md)
tests longer learning with this recipe preserved. The GPU comparisons below
remain optional robustness/checkpoint comparisons rather than a training prerequisite.

1. Keep this recipe and its checkpoints. Validate120M/160M/final under the same
   randomized no-push GPU forward conditions. Select using actual travel,
   heading, stability and gait measurements, rather than final-step status alone.
2. Visualize160M versus final at .10/.20, retaining the viewer logs. The current
   higher-command GPU heading evaluator records root pose/velocity but no
   collider planting traces; its result validates progression/heading/stability.
   Placement conclusions still require the nominal traces or viewer contact logs.
3. Prioritize remaining cadence/placement and randomized self-contact after
   that comparison. Defer the narrower roll band: body rocking already improves
   without it. Treat.30 range expansion and push recovery as separate objectives.

Direct GPU progression/heading comparison from the updated WR2 checkout:

```bash
uv run --locked --python 3.10 --extra training python -m wr2.tools.evaluate_heading \
  --run-dir results/wr2_walking/wr2_paper_size_aligned_20261030_085111_seed3 \
  --checkpoint-steps 120029184 160038912 200048640 \
  --num-envs 128 --seeds 12102 22102 \
  --output results/wr2_walking/wr2_paper_size_aligned_20261030_085111_seed3/heading_comparison_gpu
```

It evaluates .10/.15/.20 TB commands, one nominal episode and two randomized
seeds per checkpoint/command. It applies original policies with original
observations/rewards and exact native lookup-row selection.

Mac live evaluation of the final tracking candidate:

```bash
uv run --no-sync mjpython -m wr2.tools.view_walking \
  --run-dir results/wr2_walking/wr2_paper_size_aligned_20261030_085111_seed3 \
  --command .2 0 0 --playback-speed 1
```

For the more balanced nominal160M candidate, add:

```text
--checkpoint results/wr2_walking/wr2_paper_size_aligned_20261030_085111_seed3/checkpoints/000160038912
```

## Artifacts and references

Ignored local `review_20261030/` contains `analysis.json`, `gait_analysis.json`,
14 nominal traces with an index/provenance, reproducible analysis scripts and
`comparison.png`/`.pdf`. Numerical report auditing uses stored per-episode
values; gait reviewing uses actual saved policies and original WR2 physics.

References: local TB **`paper-walking-baseline`** at `a4f693d`, `paper.gin`,
`walk_env.py`, `mjx_env.py`, and copied Paper training/evaluation source;
[ToddlerBot paper v2, Tables VI–VII](https://arxiv.org/html/2502.00893v2);
[size-aligned recipe](wr2_size_alignment_experiment_20261030.md);
[preceding reward result](wr2_leg_imitation_result_20261029.md).
