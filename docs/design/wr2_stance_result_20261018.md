# WR2 GPU stance diagnostic result — 2026-10-18

**The matching TB GPU traces have now been reviewed.** The
[paired result and CAD double check](wr2_tb_stance_result_20261018.md) supersede
the pending TB measurement and conditional next plan below. Box contact occurs
while the imported CAD feet remain separated; validate the collider before
tuning spacing around those constraints. The inward target/coordination issue
remains, so geometry correction alone is not established as a walking fix.

## Conclusion

The copied **GPU diagnostic completed successfully**. It confirms a contribution
from the learned inward hip-roll stance: replacing those two actions eliminates
foot self-contact and greatly reduces delivered-to-actual hip-roll error.
However, six-direction XY command error **increases 44.2%**, with additional
oscillation, heading drift and worse support-phase matching. The diagnostic
does not validate the hip-roll override as a training correction.

The leading measured failure is a nearly stationary, inward stance with
persistent foot contact that still receives maximum spacing reward. Clearing
that contact exposes further coordination problems. The next measurement is
the matching **TB Paper 200M policy trace**, before selecting a spacing or
actuator correction for a fresh 200–300M run.

## 1. Data and metric verification

Input: `wr2_paper_20261018_121025_seed3/stance_diagnostic`, comprising
`summary.json`, `reference_geometry.json` and **21 raw NPZ traces**.
The selected `best_params` is the **260,116,480-step** policy; its local SHA-256
matches the diagnostic's recorded checkpoint hash.

- Runtime: clean **ac91a6a**, Python **3.10.19**, RTX **5070**, JAX **0.6.2**,
  Brax **0.14.1**, MuJoCo/MJX **3.14.0**, Flax **0.10.7**.
- Every episode supplies **824 finite ticks**, **16.48 physical seconds**,
  approximately **20 TB-equivalent seconds**. There are zero height failures,
  nonfinite ticks or missing/padded samples.
- Recomputed summaries match the copied summaries within **1.49e−8**. The small
  difference comes from floating-point reductions across NumPy versions.
- Native MuJoCo signed distances recomputed from the traced geometry frames
  match within **9.20e−17 m**. Integrated root-position differences reconstruct
  the recorded path deltas. Reconstructed delivered targets have **zero error**.

Metric: mean per-tick `‖v_xy − command_xy‖ / s`, with **s=.824055**.
Net speed uses integrated displacement in the commanded heading frame,
divided by physical duration and by `s`. Distance divides by `s²`.
All controls use identical nominal resets, deterministic actions, seven fixed
commands, nominal plant parameters, and no observation noise or physical
pushes. These are one-reset interventions, without a multi-seed uncertainty
estimate. The actor responds to each altered trajectory through feedback.

Support uses the production summed floor/foot vertical force **>1 N**.
Self-contact uses active, penetrating sole-to-sole contact with force magnitude
**>.1 N**. Support fractions and conditional double-support matching have
different denominators; neither should be used to infer flight by subtraction.

Independent audit: [stance_gpu_audit.json](../../results/wr2_walking/wr2_paper_20261018_121025_seed3/review_20261018/stance_gpu_audit.json).
Raw measurements: [GPU summary](../../results/wr2_walking/wr2_paper_20261018_121025_seed3/stance_diagnostic/summary.json).

## 2. Paired control results

Six moving commands, excluding standing; speeds/errors in TB-equivalent m/s.
The forward/backward commands here are ±.10.

| Control | Six-direction XY error | Foot self-contact | Forward net speed | Forward net cross drift | Backward net speed |
| --- | ---: | ---: | ---: | ---: | ---: |
| Learned policy | **.06877** | **91.22%** | +.00564 | +.00011 | −.00219 |
| Reference hip rolls | .09918 | 0% | +.02503 | −.01998 | −.03054 |
| Full leg reference | .21501 | 0% | +.11589 | +.00856 | −.06037 |

The hip-roll intervention increases forward propulsion about **4.4×**, but
its forward error increases **.09910 → .10796**. Six-direction error increases
**.06877 → .09918**, or **44.2%**. After excluding the first two gait cycles,
the corresponding errors are **.06930 / .09913 / .21661**; startup alone does
not explain the outcome. The analytic zero-velocity score for these six
commands is **.06667**.

| Condition | Learned XY error | Reference hips | Full leg reference |
| --- | ---: | ---: | ---: |
| Forward .10 | .09910 | .10796 | .20302 |
| Backward .10 | .10077 | .12867 | .22094 |
| Backward .05 | .05363 | .07832 | .22153 |
| Forward .05 | .05319 | .08006 | .20480 |
| Right .05 | .05293 | .08110 | .21846 |
| Left .05 | .05297 | .11897 | .22129 |
| Standing | .02229 | .07362 | **.00018** |

### What changes physically in the forward episode

| Measurement | Learned | Reference hips | Full leg reference |
| --- | ---: | ---: | ---: |
| Left/right hip-roll target clipping | 29.2% /99.9% | 0% /0% | 0% /0% |
| Left/right actual–delivered RMSE, rad | .1123 /.0755 | .0167 /.0179 | .0114 /.0112 |
| Minimum signed sole gap, mm | −.693 | +9.772 | +17.289 |
| Mean body-Y spacing, mm | 62.443 | 75.410 | 77.093 |
| Torso velocity SD, forward/lateral | .0272 /.0259 | .0662 /.0600 | .0330 /.2338 |
| Final integrated root yaw, degrees | −9.1 | **−53.7** | +2.1 |
| Actual double support during 275 requested DS ticks | 5 | 3 | 61 |
| Actual double support during 549 requested SS ticks | 0 | **198** | 11 |
| Exact contact-state match over all ticks | 67.2% | 42.8% | 72.5% |

The hip correction clears the soles and reduces joint execution error, but
introduces frequent double support during the requested single-support phase.
The original 100% exact single-support match becomes **63.8%**. Its body also
turns roughly 54° despite the straight command, which helps explain the gap
between torso-frame mean velocity and net commanded-frame translation.
There is **no no-support tick** in either prescribed forward control.
Full leg-reference execution produces particularly large lateral oscillation;
its greater mean propulsion does not establish better walking.

The learned forward controller delivers mean hip-roll targets approximately
**+6.20° /−9.98°**, while the measured angles average **+.90° /−5.67°**.
The reference means are approximately **−.05° /−.005°**. Reference substitution
uses the existing home-centered .25-rad action mapping, bounds, one-tick delay,
quantization, slew and servo model. The comparison therefore demonstrates
that the large hip errors depend on the requested stance/load; it does not
isolate foot contact from all other changes to balance and actuator loading.

### How this revises the hypothesis

1. **Stance contributes.** Self-contact disappears and hip tracking improves
   substantially when the hip requests change. The learned policy's right hip
   clips on **99.9%** of ticks for every command, including standing.
2. **Clearance alone is insufficient.** Both prescribed controls clear the
   soles and survive, yet have worse command tracking. Whole-body coordination,
   heading and support timing still need comparison with the learned TB policy.
3. **The spacing reward has a concrete blind spot.** Every learned case receives
   maximum spacing reward on every tick despite frequent self-contact. Body
   spacing averages about **62–65 mm**, above the configured **55.728 mm**
   minimum. The code measures body-origin Y separation rather than surface gap.
4. **No new trainer or target-pipeline bug was demonstrated.** Finite execution
   and exact target reconstruction pass this diagnostic's checks. These checks
   do not establish that every PPO or plant parameter is correct.

CPU and GPU agree on these conclusions. Their six-direction learned/hip/leg
errors are respectively **.06895 /.10006 /.21962** on CPU and
**.06877 /.09918 /.21501** on GPU. Some prescribed trajectories are sensitive
to runtime arithmetic: full-leg backward net speed is **−.01152 CPU** versus
**−.06037 GPU**. Use the GPU results for the quantitative decision.

## 3. TB Paper comparison checked against source

Source inspected with `git show paper-walking-baseline:…`, branch **a4f693d**.
The completed Paper run records **1982ef3** and native Brax fork **1414780**.
Its archived `mjx_env.py`, `walk_env.py` and `evaluate_xy.py` match the requested
branch byte for byte. All **12** frozen model/reference hashes match the saved
run provenance. The active TB checkout was used for read-only checks; the
geometry/reference comparison uses the frozen Paper assets.

TB and WR2 both use fixed-home **.25-rad actions**, a **one-tick delay**, moving
leg-reference MSE at weight **5**, foot-contact matching at **1**, spacing at
**1**, collision at **.1**, and air-time at **500**. TB uses **20 ms /.72 s**
control/cycle; WR2 uses **20 ms /.6 s**, or **24.27 ms /.728 s** in TB time units.
Their reference is a soft imitation objective rather than a prescribed action.

Existing full benchmarks cover ten fixed moving conditions plus standing and
transitions, **128 episodes per condition**. They include each recipe's native
variation and must remain distinct from the seven nominal intervention cases.

| Full benchmark | Fixed walking XY error | Height failures /1920 |
| --- | ---: | ---: |
| WR2 initialization | .07162 | 0 |
| WR2 selected 260M | **.07684** | 0 |
| TB Paper 180M | .03838 | 3 |
| TB Paper 200M | **.03702** | 8 |
| TB Paper 300M | .03800 | 134 |

The established tracking gap remains **2.08×**. TB 200M's full-benchmark mean
forward velocity is **+.09053**, versus WR2's near-zero response. TB's own
forward leg-reference RMSE is **.2340 rad**, larger than WR2's nominal **.1474**;
different robot/reference/evaluation populations prevent interpreting those
values as a servo-quality ranking.

The full WR2 reference audit finds **32/227 rows** with overlap, up to
**2.352 mm**, all at ±.10 lateral TB speed; all physical joint limits are valid.
**8.30%** of uniformly counted table joint samples exceed the action range.
The prior frozen TB cache audit also finds reference overlap at ±.10 lateral
commands, up to **6.70 mm** for its main pair and **15.06 mm** for the second
pair. TB forward hip-pitch/knee reference samples also exceed its .25-rad
action range. These properties are shared with a successful TB training run.
They do not independently justify changing the action range or teacher design.

The GPU and CPU kinematic audits reproduce the same overlap/reachability
fractions; their maximum signed-distance difference is below **4.45e−16 m**.
Uniform lookup coverage is not the training command distribution.

## 4. Next plan

**Obtain the TB 200M policy's corresponding nominal traces next.** Use the
existing [matching TB task](wr2_stance_diagnostic_20261018.md#following-decision-and-matching-tb-task):
same seven commands, 1,000 ticks, frozen Paper source/native Brax, and recorded
requested/delivered/actual joints, both ankle collision pairs, support timing,
torso velocity/error, net displacement and heading. Retain 180M as the survival
reference. Matching TB policy stance traces are not in the copied local results.

- If successful TB avoids inward saturation/self-contact, compare the actual
  sole dimensions and joint limits, then select a **soft spacing correction**
  that preserves TB's reward form. Validate its penalties across the full WR2
  lookup and the learned traces. A global 70 mm floor penalizes valid rightward
  references; body-origin distance cannot guarantee clearance under yaw/tilt.
- If TB also walks with that stance, or WR2 has much greater normalized joint
  lag after contact is cleared, prioritize the **loaded actuator response and
  reference phase**. Revisit the nonunique hardware fit using measured target
  execution; copying TB constants would lack physical validation.
- Compare how TB executes its soft reference before changing contact-phase
  rewards. A prescribed trajectory's poor tracking is not equivalent to the
  learned TB policy's execution quality.

Following a justified correction, use a **fresh 200–300M run**, retain/evaluate
every approximately 20M checkpoint, and compare seeds **12102 and 22102**
against initialization, the completed WR2 control and TB 180M/200M. Require
improved per-tick XY error, correct commanded-axis translation, controlled
drift and the height-failure gate. Greater separation or survival alone is
insufficient. This result leaves the air-time ablation deferred.

## References

- [ToddlerBot paper, Tables VI/VII](https://arxiv.org/html/2502.00893v2), checked
  directly for PPO and reward settings.
- Frozen local TB `mjx_env.py`, `walk_env.py`, `evaluate_xy.py`, `mjx_config.py`,
  `paper.gin`, `motion_ref.py`, `walk_zmp_ref.py` and cached model/reference assets.
- WR2 [diagnostic](../../wr2/tools/diagnose_walking.py),
  [environment](../../wr2/locomotion/walking_env.py),
  [rewards](../../wr2/locomotion/paper_rewards.py),
  [reference](../../wr2/reference/paper_walk.py),
  [setup audit](wr2_tb_paper_setup_audit_20261018.md) and
  [actuator review](wr2_actuator_gap_review_20261018.md).
- [MuJoCo signed geometry distance](https://mujoco.readthedocs.io/en/stable/APIreference/APIfunctions.html#mj-geomdistance)
  and [collision detection](https://mujoco.readthedocs.io/en/stable/computation/index.html#collision-detection).
