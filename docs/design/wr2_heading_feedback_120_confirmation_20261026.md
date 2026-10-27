# +120M full walking confirmation — 2026-10-26

## Decision

**+120,176,640 qualifies as the next baseline for the current forward-walking
priority.** Both evaluation seeds pass the existing gate: complete15-condition
suite, finite results and≤1% falls. There is **1 fall across3,840 episodes**
(0.026%); held-out seed22102 has **0/1,920 falls**.

Together with the completed256-draw GPU heading comparison, this supports
using+120M for straighter walking at .15/.20 TB-equivalent commands. Keep
+173M as the broader walking control: it has better aggregate tracking,
backward walking and standing/stopping, and slightly better forward speed.

The recommended baseline checkpoint is:

```
results/wr2_walking/wr2_paper_heading_feedback_20261025_201358_seed3/checkpoints/000120176640
```

The run's `best_params` still identifies+173M, selected by the standard suite.
Commands for the+120M baseline should specify the checkpoint above.

## Data and verification

Input: `paper_120_confirmation.json` in the heading-feedback training folder.
It uses the saved0.30.0 config, native GPU runtime, observation/model/reset/
actuator randomization, deterministic actions, seeds12102/22102 and no pushes.
Each seed has15 conditions×128 episodes, with824 ticks at20ms.

Evaluation provenance is clean `d30f33a`; production WR2 code is unchanged
between that revision and the current review. Python3.10.19/JAX0.6.2/
Brax0.14.1/MJX3.14.0 and default matrix precision match training. The copied
+120M checkpoint hash matches the prior GPU heading evaluation.

Recomputed summary means agree exactly. Per-episode lengths, fall flags,
group sample counts and the standing/walking partition agree with the saved
reports. Group means/stds agree within float32 rounding, maximum2.12e-6.
All recorded measurements are finite. Finite terminal samples are included
in the per-episode means, according to the shared evaluator's contract.

The one fall is **backward_010**, seed12102, episode97, after444 samples
(8.88 physical seconds). It is a height-based terminal flag, without a
nonfinite transition. The report does not store the terminal pose or joint
trace, so its physical cause is unconfirmed.

The original training-time+120M evaluation at seed12102 recorded zero falls;
this repeat records one. Aggregate reference-frame error is essentially
unchanged(.044905→.044903 TB m/s), but individual trajectories differ. The
cause of the repeat discrepancy is not captured by these summary-only artifacts.

## Comparison with +173M

XY errors below are **TB-equivalent m/s**. Mean heading error and gait
fractions average the ten fixed nonzero walking conditions; fall counts
cover all1,920 episodes including standing and transitions.

| Seed | Checkpoint | Body XY error | Reference XY error | Mean absolute heading error | Falls |
|---|---|---:|---:|---:|---:|
| 12102 | +120M | .04341 | .04490 | 5.57° | 1/1,920 |
| 12102 | +173M | .04245 | .04369 | 5.47° | 0/1,920 |
| **22102 held out** | **+120M** | **.04402** | **.04547** | **5.72°** | **0/1,920** |
| 22102 held out | +173M | .04322 | .04438 | 5.79° | 0/1,920 |

+120M's held-out reference-frame error is **2.47% higher** than+173M.
The paired physical error difference favors+173M by.000902m/s,95% interval
[.000537,.001272]. Standard-suite mean heading error is similar; its paired
interval includes zero. One fall does not establish a reliable difference
in stability between these candidates.

| Held-out fixed walking metric | +120M | +173M |
|---|---:|---:|
| Modeled foot self-contact | **4.04%** | 5.92% |
| Contact-phase agreement | **64.75%** | 61.60% |
| Torque RMS | .504Nm | .511Nm |
| Leg-reference RMSE | .1796rad | .1760rad |
| Target clipping | 1.10% | 1.03% |
| Target slew limiting | .97% | .93% |

Self-contact is32% lower for+120M. The paired absolute reduction is
1.88 percentage points,95% interval[1.56,2.21]. Contact-phase agreement
improves3.14 percentage points, interval[2.55,3.73]. These gains also appear
on seed12102. Foot self-contact measures the modeled foot boxes; these
reports do not supply landing geometry or prove that alternating placement
is correct in every episode.

### Where the tradeoff comes from

Held-out reference-frame errors:

| Condition/window | +120M | +173M |
|---|---:|---:|
| Forward .10 | .04503 | .04593 |
| Backward .10 | .05130 | .04457 |
| Standing | .03241 | .02953 |
| Forward transitions, walking | .04574 | .04639 |
| Backward transitions, walking | .04989 | .04546 |

+120M is competitive in forward walking, while backward tracking and
standing/stopping favor+173M. This explains the slightly worse all-command
mean. The current forward priority favors+120M; broader walking performance
favors+173M.

The previously validated GPU heading comparison remains relevant at higher
commands. At .20, +120M/+173M achieve **.1543/.1624 TB m/s** requested-direction
speed and **13.5°/28.1°** absolute net turn, with zero falls in256 attempts
each. Their numbers of episodes turning over45° are **2/256 versus28/256**.
The full confirmation closes the missing standing/lateral/backward/transition
check; it does not change that speed-versus-direction tradeoff.

## TB comparison and next training direction

The five archived TB locomotion files match local **`paper-walking-baseline`**,
revision `a4f693d7cc4544f011dad23d6c353e752a6819ab`, byte-for-byte.
The comparison uses its saved200.09M checkpoint, the best eligible aggregate
tracking checkpoint from the300M Paper run.

On seed12102, TB body-frame XY error is.03702 TB m/s, versus WR2+120M
.04341 and+173M .04245. WR2+120M remains **17.3% above TB** in that metric.
TB has8/1,920 sampled falls; WR2+120M has1/1,920 here. Robot/randomization
profiles and training seeds differ, so this is unpaired context. The existing
native TB higher-command reference is smaller and uses a CPU runtime;
its .20 requested-direction speed/net turn are.1407 TB m/s/37.9°, averaged
over15 completers with one fall among16 attempts.

Code confirms that TB and current WR2 use body-local XY velocity tracking,
moving leg-reference imitation and a fixed home action origin. WR2's explicit
heading feedback is the tested observation extension; physics and reward
formulas were held fixed in this continuation. The new confirmation supports
retaining that feedback and using the qualified+120M policy as the next
forward-training baseline.

**Recommended next experiment:** change only XY reward tracking to the
desired-heading frame, with the existing coefficient15. Retain heading
feedback, physics, action mapping, ZMP, command/randomization distributions,
qualified landing reward and TB PPO settings. Continue for200M from+120M,
retain intermediates, and compare standard walking plus .10/.15/.20 forward
progress, turns, falls and foot contact before selecting a final policy.

Rationale: the [GPU reward review](wr2_heading_feedback_gpu_review_20261026.md)
found nearly equal native returns for+120M/+173M despite their direction
tradeoff. Replacing only the XY frame in a frozen-policy calculation favors
the straighter policy. That is evidence for objective alignment; the reset-
discounted comparison did not clearly separate them, so a PPO improvement
remains unproven. The experiment's goal would be better commanded progression
without larger turning. Record landing geometry to verify any benefit for
alternating foot placement.

Sources: local TB Paper `mjx_env.py` and `evaluate_xy.py`;
[ToddlerBot Table VII](https://arxiv.org/html/2502.00893v2);
[WR2 evaluator](../../wr2/locomotion/paper_evaluation.py);
[confirmed GPU comparison](wr2_heading_feedback_gpu_review_20261026.md).

## Artifacts

Validated summaries, per-condition comparisons, paired intervals, checkpoint/
source hashes and the failure record are in
`results/wr2_walking/wr2_paper_heading_feedback_20261025_201358_seed3/review_120_confirmation_20261026/analysis.json`.
The [goal-frame XY continuation](wr2_goal_frame_xy_experiment_20261026.md)
is implemented and ready with the standard trainer.
