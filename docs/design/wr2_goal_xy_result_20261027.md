# Goal-frame XY training result — 2026-10-27

## Result and next priority

**Update:** the [completed GPU comparison and shoulder-motion review](wr2_goal_xy_gpu_review_20261027.md)
confirms final+200M's higher-command forward gain and right-hip saturation;
it supersedes the pending-GPU step below. The
[completed final standard confirmation](wr2_goal_xy_final_confirmation_20261027.md)
also passes both seeds, closing the qualification gate for the forward priority.

Run: `results/wr2_walking/wr2_paper_goal_xy_20261026_193545_seed3`.
The 200M continuation completed correctly from the qualified heading+120M
checkpoint, with200,294,400 additional steps. The selected checkpoint is
**+173,588,480**; the final checkpoint is a separate **+200,294,400** candidate.

The result is mixed. Selected-policy held-out heading and foot self-contact
improve, while aggregate reference-frame XY error improves only1.9%.
Forward .10 tracking worsens, and push robustness deteriorates. A small Mac
screen finds that the **final** policy performs better at .20 than the
automatically selected policy.

The landing replay confirms real alternating steps, but placement remains
uneven. The strongest new constraint to investigate is **hip-pitch action
coverage and saturated physical exploration**. The final policy keeps its
right hip command near the posterior edge of the allowed action window.
This window excludes a substantial part of the .20 reference stance targets.
The joint follows the command closely, and measured torque stays below its cap.

Complete the larger GPU retained-policy comparison with the added per-joint
action logging before promoting a candidate or changing the training setup.
Use that comparison to audit action/posture compatibility before another
reward iteration.

## Implementation and metric checks

The saved config matches the prepared0.31.0 config. Compared with the source
run, only `reference_frame_xy_tracking`, version identity and output prefix
change. Training uses clean commit `b0ccc8e`, Python3.10.19/JAX0.6.2/
Brax0.14.1/MJX3.14.0 on the RTX5070. The goal frame is initialized from reset
heading and integrates commanded yaw. Actor/critic remain1005/1515 inputs.

Saved `initial_params` exactly equal source checkpoint `000120176640`.
`best_params` and final `params` exactly match their recorded checkpoints.
All15 saved checkpoints are finite. The model hash matches the trained model;
physics, actuator parameters, action origin/scale, ZMP, noise/randomization,
qualified landing rule and PPO settings retain the source setup.

All36 primary, push and confirmation reports were checked: summary means
recompute exactly, and group statistics agree within float32 rounding
(maximum3.54e-6). Sample counts, masks and fall flags are consistent.
There are no nonfinite transitions. One primary fall occurs at+133.53M;
selected and final primary reports have zero falls. The restored initial
held-out policy has one fall, and the selected held-out policy has zero.

The27 new Mac heading traces reproduce their summaries exactly. **The new
XY reward was reconstructed in the goal frame**, not the old body frame;
XY/yaw/orientation differences are below3.83e-7. The existing heading
measurement definitions and TB velocity/time/distance normalization apply.

## Standard suite

Errors use **TB-equivalent m/s**. Mean absolute heading and gait fractions
average the ten fixed nonzero walking conditions; fall counts cover all15
conditions×128 episodes. These heading means differ from net turn over the
post-warmup window in the next section.

| Evaluation | Policy | Body XY error | Reference XY error | Mean absolute heading error | Falls |
|---|---|---:|---:|---:|---:|
| Selection12102 | Restored source | .04338 | .04488 | 5.59° | 0/1,920 |
| Selection12102 | Goal+120M | .04396 | .04422 | 3.98° | 0/1,920 |
| Selection12102 | **Goal+173M selected** | .04373 | **.04420** | **3.80°** | **0/1,920** |
| Selection12102 | Goal+200M final | .04476 | .04519 | 5.23° | 0/1,920 |
| Held-out22102 | Restored source | .04399 | .04542 | 5.68° | 1/1,920 |
| Held-out22102 | **Goal+173M selected** | .04417 | **.04455** | **4.07°** | **0/1,920** |

Held-out reference error improves1.93%, with a paired physical change
−.000722m/s,95% interval[−.001252,−.000190]. Mean heading error improves
28.3%. Body-frame error changes+.000143m/s; its interval includes zero.
Resampling uses128 episode-index clusters shared across the ten conditions.

The aggregate hides a directional tradeoff:

- Forward .10 requested-direction speed changes **.08344→.08116 TB m/s**,
  and reference XY error **.04502→.04868**. Mean lateral velocity changes
  **+.00170→−.01416 TB m/s**, despite smaller heading error.
- Backward .10 error improves **.05088→.04351** and standing error
  **.03239→.02908**. These gains help select+173M over candidates with
  better higher-command forward walking.

The standard suite reaches .10, so its selected checkpoint should not be
assumed best for .15/.20 operation.

| Held-out fixed-walking metric | Restored | Selected+173M |
|---|---:|---:|
| Modeled foot self-contact | 4.02% | **2.83%** |
| Agreement with reference contact phase | **64.74%** | 58.75% |
| Torque RMS | .504Nm | .531Nm |
| Leg-reference RMSE | .1796rad | .1797rad |
| Actual double support | 5.11% | 5.25% |
| Actual single support | 94.89% | 94.75% |

Self-contact falls29.5%, while reference-phase agreement decreases6.0
percentage points and torque RMS increases5.2%. Air-time credit and support
fractions remain nearly unchanged. Reference-phase agreement alone does not
establish whether actual alternating stepping improved or worsened.

The independent push stress profile worsens from **829/1,920 to995/1,920
falls**(43.2%→51.8%). The paired increase is8.65 percentage points,
95% cluster-bootstrap interval[4.48,12.92]. This profile was not used for
training or selection. Lower mean heading error on surviving prefixes is not
evidence of better push recovery when more episodes terminate early.

Reported KL peaks at.219 around+120M and.075 around+147M; the final value
is.032. All metrics remain finite. These observations support retaining
intermediates, but do not prove that a particular update caused a gait change.
The training/evaluation log covers about1h53m.

## Higher-command Mac screen

Original policies use matching CPU runtime,16 randomized attempts per
checkpoint/command plus nominal episodes, no pushes, and20 TB-second
horizons. All144 new randomized attempts and9 nominal episodes complete.
Numbers below are **requested-direction speed / absolute net turn**, after
the1.5-TB-second warmup. That window lasts15.24 physical WR2 seconds.

| Policy | .10 command | .15 command | .20 command |
|---|---:|---:|---:|
| Source heading+120M | .0853 /5.0° | .1296 /4.8° | .1587 /11.4° |
| Goal+120M | .0884 /4.9° | .1265 /4.6° | .1599 /5.4° |
| Goal+173M selected | .0817 /4.8° | .1171 /7.9° | .1454 /9.4° |
| **Goal+200M final** | .0841 /4.7° | .1294 /6.3° | **.1668 /7.4°** |

At .20, final versus restored source improves physical forward travel
**1.993→2.094m**, lateral magnitude **.243→.118m**, and forward lean
**7.16°→6.47°**. Commanded forward distance is2.512m, so83.4% is achieved.
The paired16-draw speed gain is+.00808 TB m/s,95% interval[+.00172,+.01461].
The net-turn change interval includes zero, while lateral displacement
decreases by.125m, interval[−.244,−.012].

The selected+173M policy is slower than the source at .20 by.01324 TB m/s,
interval[−.02028,−.00643]. Final+200M beats selected+173M by.02132 TB m/s,
interval[+.01830,+.02449]. Thus+200M is the higher-command candidate to
qualify; its full held-out standard suite and larger GPU comparison are pending.

## Actual stepping and remaining asymmetry

The existing `LivePolicy` viewer was run headlessly at .20, with original
trained actions, no interventions, nominal physics and no noise/pushes.
The samples use the native controller and cached geometry. Landing events
require new supported contact lasting10 samples, preceding supported air,
and more than1mm box-sole clearance during that air interval. Measurements
exclude the1.5-TB-second warmup. This is one matched nominal replay per policy.

Each replay has25 geometric landings per leg,100% alternation between distinct
landing events, and median gait cadence **1.373 TB Hz**, close to TB's1.389.
The lower reference-phase score is largely timing: shifting the reference
by6–7 control ticks(.12–.14 physical seconds) restores agreement to85–86%.
This is a measured offset, not a modification of the policy or reward.

| Nominal .20 policy | Left landing lead | Right landing lead | Left/right same-foot stride |
|---|---:|---:|---:|
| Source heading+120M | 7.69cm | 3.89cm | 12.52/12.60cm |
| Goal+120M | 12.33cm | −.47cm | 12.46/12.39cm |
| Goal+173M | 9.30cm | .82cm | 10.85/10.81cm |
| Goal+200M | 9.87cm | 2.50cm | 13.14/13.13cm |

Lengths are **TB-equivalent**. Landing lead is the newly loaded foot's box
center ahead of the other foot in the desired-heading frame. Same-foot
stride is displacement between successive landings of that foot. Both feet
advance; the smaller right landing lead should not be interpreted as a
stationary right foot. Nevertheless, fore-aft placement is more uneven than
the source replay, and goal+120M's right foot can land behind the left.

## Hip action-space finding

In the final nominal .20 replay, the right hip-pitch request satisfies
`action<=−.99` for **92.8%** of post-warmup samples. Its delivered target spans
only **2.40°–2.88°**, while actual angle spans **.36°–2.76°**. Tracking RMS
is.70° and peak torque1.10Nm versus the4Nm cap. Left hip targets span
−29.04° to−2.40° and carry most hip swing. The current small right-hip
excursion is primarily present in the command itself.

The fixed action mapping is `q_target=q_home+.25*tanh(raw_action)`.
WR2's hip origin is16.63° in forward-angle coordinates, so its target
window is **+2.31° to+30.96°**. For comparison, TB's model home8.35° gives
**−5.98° to+22.67°**. The WR2 implementation substitutes the grounded
reference standing pose for both action and observation origin; TB uses
model default joints. WR2's original model hip home is6.88°, while the
grounded reference pose is16.63°.

At .10, WR2 reference stance targets fit the window, but posterior margin
is only about.23°. At .20, about30% of right-hip reference stance samples
fall below the policy target window; reference stance reaches−8.02°.
The physical joint range is−40° to+90°, so this is **command-space coverage**,
rather than contact with the mechanical stop. The robot can use body lean and
other joints as compensations, so this finding does not prove that the range
alone causes unequal placement. The forward part of the range also remains
available, yet the learned strategy requests the posterior edge.

TB's verified cached references also exceed its action window in some phases,
especially swing. Therefore total out-of-range reference fraction is not
sufficient evidence of a WR2-only bug. The matched .20 cache has about8.4%
posterior stance targets outside TB's range, versus about30% for WR2.
Raw motor signs differ; these coverage checks use each robot's native mapping.

The same replay was probed with4,096 native stochastic actions at six nominal
states. **86–100%** of right-hip samples quantize to the same2.40° target;
at one state all4,096 do. Left-hip probes yield11–69 different targets.
Some right-hip unsquashed standard deviations are as large as.61–1.07,
but tanh saturation collapses most physical variation below the.24° command
quantum. Global policy standard deviation therefore cannot establish useful
exploration at this joint. These fixed-input probes do not include training
observation noise or domain variation, which the GPU action trace will assess.

**Priority:** audit the hip action origin/range and physical exploration before
another reward modification. A revised center or range must preserve grounded
initialization and validate stance coverage. Changing the current home also
changes position-observation baselines and physical policy outputs, so a
continuation would need an explicit, tested conversion or a fresh controlled
run. Wholesale replacement by raw model home would also change WR2 knee home
from40.2° to13.8° and requires separate standing validation.

## TB comparison and interpretation

The five saved locomotion files match local `paper-walking-baseline`, revision
`a4f693d7cc4544f011dad23d6c353e752a6819ab`. The cached lookup used for the
coverage audit was read from that Git revision and its hash matches the
completed Paper run. The active TB development checkout was not used.

TB200M standard body-frame XY error is.03702 TB m/s versus selected goal
WR2 .04373, an18.1% WR2 gap, with8/1,920 versus0 sampled primary falls.
This is unpaired context across different robots/randomization/training seeds.
Its smaller native CPU .20 reference gives.1407 TB m/s and37.9° net turn over
15 completers from16 attempts. The new WR2 CPU forward candidate compares
favorably, but this is not a matched TB parity qualification.

The goal frame is implemented correctly and improves some directional metrics.
The run also includes200M additional training and a new Adam state; there is
no matched unchanged-reward continuation from the same source, so its gains
cannot all be attributed to the frame change. Retain the current goal reward
while investigating the newly measured action-space issue.

Sources: local TB Paper `mjx_env.py` action mapping and
`walk_zmp_ref.py` lookup; [ToddlerBot Table VII](https://arxiv.org/html/2502.00893v2);
[goal-frame experiment](wr2_goal_frame_xy_experiment_20261026.md);
[WR2 action implementation](../../wr2/locomotion/walking_env.py).

## Next GPU comparison

The existing evaluator now records per-step bounded `actor_action`, with
joint names, policy/model home positions, joint limits, action scale and
command resolution in WR2 metadata. Schema2 adds these fields; measurements
and policies retain their existing semantics. Seven heading-evaluation tests
and lint pass. The constructor comment was clarified; training behavior was
not modified for this review.

After syncing this review's commit to the GPU checkout:

```bash
uv run --locked --python 3.10 --extra training python -m wr2.tools.evaluate_heading \
  --run-dir results/wr2_walking/wr2_paper_goal_xy_20261026_193545_seed3 \
  --checkpoint-steps 120176640 173588480 200294400 \
  --num-envs 128 \
  --output results/wr2_walking/wr2_paper_goal_xy_20261026_193545_seed3/heading_evaluation_gpu
```

This supplies256 randomized attempts per checkpoint/command, plus nominal
episodes, to confirm forward ranking and the frequency of hip action saturation.
Qualify final+200M on the full held-out suite before any promotion if its
forward advantage survives this comparison.

## Artifacts

`review_20261027/analysis.json` under the goal run contains validated means,
paired intervals, input/source hashes and the comparisons. `headless_gait/`
contains original-policy traces, landing events, phase-offset measurements,
reference/action coverage and stochastic target-diversity probes. The
standalone figure is `review_20261027/goal_xy_result.png` and `.pdf`.
