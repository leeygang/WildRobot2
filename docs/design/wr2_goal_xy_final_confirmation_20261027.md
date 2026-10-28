# Goal+200M full XY confirmation — 2026-10-27

**Update:** the [Mac hip action-origin validation](wr2_hip_action_validation_20261027.md)
confirms physical feasibility and improved reference coverage, with changed
neutral posture and frozen-policy speed. It prepares the randomized GPU check
of those startup/feedback effects.

## Decision

**Final +200,294,400 passes the existing two-seed standard XY qualification.**
All 3,840 episodes complete 824 control ticks, with **zero height terminations
and zero nonfinite states**. Combined with the preceding .15/.20 GPU comparison,
it qualifies as the next baseline for the current **forward-walking priority**.
The existing gate requires the complete 15-condition suite, finite results
and at most 1% height terminations. Tracking and gait quality are compared below.

Keep goal +173,588,480 as the control for broader XY tracking: it retains lower
aggregate tracking error, less modeled foot self-contact and better reference
contact-phase agreement. Final is a forward improvement with a gait/tracking
tradeoff, not a general replacement for every operating condition or TB parity.

Use the explicit final checkpoint:

```
results/wr2_walking/wr2_paper_goal_xy_20261026_193545_seed3/checkpoints/000200294400
```

The run's automatic `best_params` remains +173M. No checkpoint files or policy
parameters are modified by this review.

**Scope correction:** the standard suite has ten fixed XY commands, standing,
and four standing/walking XY transition schedules. **Every condition commands
zero yaw.** The earlier conversational description included commanded turning
incorrectly. This result qualifies sampled XY walking/standing/transitions;
commanded turning and push recovery are outside this confirmation. Its fixed
commands reach .10 TB m/s; the previous separate GPU comparison covers .15/.20.

## Result verification

Input: `paper_final_confirmation.json` in the goal-XY run. It records clean
commit `fad653b`, saved config 0.31.0, seeds 12102/22102, 128 environments per
condition, deterministic policies, observation noise, model/reset/actuator
randomization and no physical pushes. The RTX5070 runtime uses
Python 3.10.19/JAX 0.6.2/Brax 0.14.1/MJX 3.14.0 and default matrix precision,
matching the earlier standard confirmations. The .15/.20 heading comparison
uses explicit high matrix precision and is a separate evaluation profile.

All 30 condition identities, physical commands and transition flags match the
source evaluator. Per-episode fall/nonfinite flags, lengths, sample counts and
standing/walking partitions agree. All reported measurements are finite.
Recomputed summary means agree exactly; group means/stds agree within float32
rounding, maximum 1.67e-6. Each episode lasts 16.48 physical seconds/20 TB seconds.

The report names the correct final checkpoint and saved config path. The local
checkpoint hash matches the preceding GPU heading evaluation. WR2 runtime
behavior has not changed since training: intervening source edits add action
logging/metadata and clarify the fixed home-origin comment.

Training's seed 12102 final reference error was .045187 TB m/s; this repeat is
.045233, a .10% change. Both have zero falls. These records do not imply bitwise
reproducibility of individual simulated trajectories.

The audited measurements and paired intervals are saved at
`results/wr2_walking/wr2_paper_goal_xy_20261026_193545_seed3/review_final_confirmation_20261027/analysis.json`.
The artifact includes result/checkpoint/config/source hashes and per-condition
comparisons; it is a local result and is not committed to Git.

## Standard-suite comparison

Errors are **TB-equivalent m/s**. Heading means and gait metrics average
per-episode values across the ten fixed nonzero XY conditions. They are not
net-turn angles over the higher-command post-warmup window. Fall counts include
all 15 conditions×128 episodes.

| Seed | Goal checkpoint | Body XY error | Reference XY error | Mean absolute heading error | Falls |
|---|---|---:|---:|---:|---:|
| 12102 | +173M selected | .04373 | **.04420** | **3.80°** | 0/1,920 |
| 12102 | +200M final | .04480 | .04523 | 5.21° | 0/1,920 |
| 22102 held out | +173M selected | .04417 | **.04455** | **4.07°** | 0/1,920 |
| **22102 held out** | **+200M final** | .04488 | .04527 | 5.37° | **0/1,920** |

Final held-out reference error is 1.61% higher than selected. The paired change
is +.000719 TB m/s, 95% bootstrap [+.000128,+.001324]. Heading error rises 1.30°,
interval [+.95,+1.66]. These differences also favor +173M on seed 12102.
Bootstrap sampling uses 128 episode-index clusters shared across the ten
conditions, separately for each seed.

Against the qualified body-frame +120M source, final held-out reference error
changes **.04547→.04527**. Its paired change interval includes zero. Body-frame
error changes **.04402→.04488**, about 2.0% higher, with a positive paired
interval. Thus final's higher-command forward gain should not be described as
a clear improvement of aggregate standard-suite tracking.

### Directional tradeoff

Held-out comparison with goal +173M:

| Condition/window | Selected reference XY error | Final reference XY error |
|---|---:|---:|
| Forward .10 | .04868 | **.04392** |
| Backward .10 | **.04351** | .04944 |
| Standing | **.02908** | .02989 |
| Forward transitions, walking | .04720 | **.04519** |
| Backward transitions, walking | **.04365** | .04961 |

At forward .10, requested-direction mean speed improves **.08116→.08417
TB m/s**, and lateral velocity changes **−.01416→+.00110**, bringing it nearer
zero. Backward .10 speed magnitude declines **.09613→.08494**. This explains
why the standard selector favors +173M even though final is the better forward
candidate. The same directional distinction was measured at .15/.20 in the
[completed GPU comparison](wr2_goal_xy_gpu_review_20261027.md).

### Gait tradeoff

| Held-out fixed walking metric | Goal+173M | Final+200M |
|---|---:|---:|
| Modeled foot self-contact | **2.83%** | 3.82% |
| Reference contact-phase agreement | **58.75%** | 48.76% |
| Leg-reference RMSE | **.1797rad** | .1875rad |
| Torque RMS | .531Nm | **.517Nm** |
| Mechanical target clipping | 1.26% | **1.00%** |
| Actual single support | 94.75% | 94.44% |
| Air-time reward credit | 11.986s | 11.930s |

Self-contact increases .99 percentage points, 95% interval [+.85,+1.13]. Contact
phase agreement falls 9.99 points, interval [−10.82,−9.17]. Leg-reference RMSE
increases 4.3%, while torque RMS falls 2.5%. Forward .10's modeled self-contact
also rises **2.67%→5.01%**. Zero falls does not establish that remaining contact
and placement quality is satisfactory.

Phase agreement measures synchronization with the teacher. The earlier .20
nominal replay showed real alternating landings despite a 6-tick phase offset.
This summary-only confirmation cannot determine whether the whole standard
suite's lower agreement has the same explanation. It also contains no torso
pose trace, so it does not establish a change in shoulder rocking or vertical
body motion. The preceding GPU/viewer measurements remain the evidence for
those gaps.

The 1.34% all-joint bounded-action fraction here averages 12 joints and ten
commands reaching .10. It is not the 92.5% **right-hip-only** near-bound fraction
at .20 in the GPU heading trace. Both use the .99 action threshold; their
joint/command aggregation differs.

## TB Paper comparison and next priority

The five archived TB locomotion sources match local **`paper-walking-baseline`**,
revision `a4f693d7cc4544f011dad23d6c353e752a6819ab`, byte-for-byte. TB's
`evaluate_xy.py` confirms the same ten XY/standing/four-transition suite and
zero commanded yaw, with per-episode averaging. WR2 scales speed by .82405 and
time by the same factor. The active TB development branch is not used.

TB Paper's best eligible 200.09M standard body-frame XY error is **.03702
TB m/s**, with 8/1,920 sampled terminations. Final WR2 is **.04480/.04488** on
the two seeds, about 21% higher. This is unpaired context across robots,
training seeds, runtime versions and randomization. TB's standard artifact
does not provide the equivalent reference-frame XY or torso-motion metrics.
WR2's better requested-direction .20 result and smaller sampled fall count
therefore do not establish complete TB parity.

**The final confirmation gate is closed.** Next, validate the TB-aligned hip
action-origin/coverage correction and its supported reset/checkpoint behavior
before preparing the 200M controlled experiment. The verified difference remains
WR2's grounded-reference fixed origin versus TB's model-home fixed origin.
Mechanical joint stops/torque do not explain the nearly fixed right-hip
request; the command window and useful physical exploration remain the leading
audit. A whole-home replacement would also change knees/ankles and observation
baselines, so those effects must be resolved explicitly.

Acceptance should include requested-direction progress at .10/.15/.20,
bilateral landing lead/stride, per-joint near-bound actions, modeled foot
self-contact, torso roll, normalized root-height variation and falls. Retain
intermediate checkpoints and the qualified final/control policies. The existing
TB torso rewards are present; this confirmation does not identify a missing
reward or prove that hip coverage alone causes the remaining body motion.

The audit can proceed using the qualified policy. This review changes
documentation only. `CHANGELOG.md` awaits the user's confirmation of
the analysis, per repository instructions.

References: [ToddlerBot paper v2, Table VII](https://arxiv.org/html/2502.00893v2),
local Paper `mjx_env.py`, `walk_env.py`, `evaluate_xy.py`, WR2
`wr2/locomotion/paper_evaluation.py`, and the
[training/action-space review](wr2_goal_xy_result_20261027.md).
