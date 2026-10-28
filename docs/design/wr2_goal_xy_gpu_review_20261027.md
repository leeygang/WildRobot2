# Goal-frame XY GPU comparison and shoulder rocking — 2026-10-27

## Conclusion

The GPU comparison confirms **goal+200M as the stronger .15/.20 forward
candidate**, despite the standard suite selecting goal+173M. At .20, final
requested-direction speed improves9.7% over the qualified body-frame+120M
source, while net turning falls53.9%. There are **2/256 height terminations**
at .20, versus zero for the source and goal+120M. Final qualification across
the full command suite is still pending.

The user's up/down shoulder observation is confirmed. The shoulders follow
torso roll; actual shoulder joint motion is below.02°. At .20, the new policy's
roll is only slightly larger than the source's and smaller than the available
native TB Paper reference. WR2's relatively wide shoulder mounts amplify the
visible motion. **Vertical body motion remains larger than TB after size
normalization**, so the naturalness concern is meaningful even though roll
angle alone is not unusually large relative to this TB policy.

The new GPU action traces also confirm the previously identified **right-hip
command saturation** under model variation and observation noise. This remains
the first implementation/design audit before another training change. Its
causal contribution to body rocking has not been isolated.

## Data and verification

- New run: `wr2_paper_goal_xy_20261026_193545_seed3/heading_evaluation_gpu`.
  Clean evaluator commit `8264e08`, schema2,27 traces,2,304 randomized attempts
  and9 nominal attempts. Checkpoints+120,176,640/+173,588,480/+200,294,400.
- Control: `wr2_paper_heading_feedback_20261025_201358_seed3/heading_evaluation_gpu`,
  qualified+120,176,640. GPU runtime, precision, seed/episode reset draws,
  horizons and model/actuator randomization match the new comparison.
- Two seeds12102/22102,128 episodes each, deterministic original policies,
  no pushes. Each full episode lasts16.48 physical seconds/20 TB seconds.
  Post-warmup measurements cover15.24 physical seconds/18.494 TB seconds.
- All27 new trace summaries recompute within2.67e-14. Valid masks are
  contiguous prefixes; finite terminal samples are retained, then excluded
  padding remains outside measurements. No nonfinite transitions occur.
- Saved checkpoint hashes, config, model hash and evaluator source hashes
  match. Goal-frame XY, yaw and orientation reward reconstruction differences
  are below5.09e-7. The XY calculation uses the native cached world velocity
  in the desired-heading frame. Root travel and visible motion use integrated
  free-root poses; these distinct samples are not mixed.
- Between training commit `b0ccc8e` and evaluation commit `8264e08`, environment
  behavior is unchanged: source changes add action/metadata logging and clarify
  the home-origin comment. The previous GPU control used the same measurement
  definitions before action logging was added.
- TB source is read from local **`paper-walking-baseline`**, verified runtime
  `a4f693d7cc4544f011dad23d6c353e752a6819ab`. Its five archived locomotion source
  files match that branch byte-for-byte. The active TB main checkout is not
  modified or used as the Paper source.

The machine-readable audit is
`results/wr2_walking/wr2_paper_goal_xy_20261026_193545_seed3/heading_evaluation_gpu/review/analysis.json`.
It records raw hashes, per-episode measurements, paired intervals, failure
identities, viewer sessions and shoulder geometry. These artifacts are local
results, not Git additions.

## GPU forward progress

Each cell is **requested-direction TB-equivalent m/s / absolute net turn**
after warmup. Means condition on completed episodes; counts include all
attempts. Each checkpoint has256 randomized attempts per command.

| Policy | .10 command | .15 command | .20 command | Falls .10/.15/.20 |
|---|---:|---:|---:|---:|
| Source body-frame+120M | .0845 /7.08° | .1283 /8.07° | .1543 /13.49° | 0/0/0 |
| Goal+120M | **.0885** /5.74° | .1271 /6.89° | .1608 /6.80° | 0/0/0 |
| Goal+173M selected | .0824 /5.68° | .1202 /7.46° | .1486 /6.85° | 0/0/1 |
| **Goal+200M final** | .0850 /6.62° | **.1331 /6.22°** | **.1693 /6.22°** | 0/0/2 |

At .20, final achieves84.7% of commanded forward speed. It travels **2.126m**
in the post-warmup window, versus about1.938m for the source; commanded travel
is2.512m. Mean absolute lateral displacement falls **.356→.146m**, and mean
forward lean falls **7.77°→7.14°**. Keeping all attempted finite prefixes and
assuming zero progress after a fall gives final speed **.1690**, so excluding
the two failures does not explain the forward gain.

Paired254-common-completer intervals for final versus source:

- Forward speed: **+.01498 TB m/s**,95% bootstrap[+.01348,+.01643].
- Absolute net turn: **−7.17°**, interval[−8.47,−5.93].
- Absolute lateral displacement: **−.208m**, interval[−.242,−.174].

Final versus goal+173M gains+.02080 TB m/s at .20, interval
[+.02001,+.02154], using253 common completers. Final versus goal+120M gains
.00850, interval[+.00756,+.00942]. Their net-turn difference interval includes
zero. Goal+120M remains the better sampled .10 candidate and completes all
768 randomized attempts. Three new sampled terminations are all at .20:
selected seed22102/episode83 at7.56 physical seconds; final seed12102/episode102
at15.44 seconds and seed22102/episode21 at8.90 seconds.

Native TB Paper+200M CPU reference has requested-direction speeds
.0870/.1227/.1407 and net turns13.54°/21.15°/37.91° at .10/.15/.20.
At .20 its body-forward speed is.1705, versus final WR2's.1676; WR2 retains
more of that speed in the requested direction. This is a **small unpaired
reference**,16 attempts per command with1/16 TB .20 terminations, and different
native MuJoCo/Brax versions and robot randomization. It cannot establish
matched-runtime robustness or complete TB parity.

## Shoulder rocking: measured motion and its visible cause

The user identified up/down rocking at **both .20 and .30**. The viewer logs
confirm final checkpoint `000200294400` and the trained config. Timestamps
are UTC2026-10-28, corresponding to Pacific2026-10-27.

| Actual visual session | .20 command | .30 command |
|---|---:|---:|
| Torso roll span | 15.39° | 22.12° |
| Root height span, physical | 7.54mm | 11.33mm |
| Left shoulder height span, physical | 20.57mm | 26.09mm |
| Right shoulder height span, physical | 28.19mm | 41.59mm |
| Actual shoulder-joint angle spans | <.014° | <.018° |
| Right-hip negative-bound fraction | 92.84% | 97.02% |

Spans are **p95−p5 after warmup**, not absolute maximum-minus-minimum. The .20
session has885 steps and no reset. For .30, the first full episode supplies
the table; seven full nominal episodes repeat exactly after20-physical-second
timeouts, followed by one partial episode. They are not independent robustness
trials. Viewer wall time is not used as the gait clock.

The policy controls12 leg joints. The shoulder targets remain fixed; measured
arm/waist joint excursions are tiny. Left/right shoulder mount heights are
computed from the integrated root pose and their fixed offsets. A MuJoCo
forward-kinematics check agrees within3.36e-9m. Therefore the visible shoulder
motion is explained by torso rotation plus vertical translation.

The source nominal .20 replay already has **15.29° roll span**, versus
current15.37° on the matching nominal replay. This rocking substantially
predates the goal-frame reward change. Across the larger GPU suite the
increase is small but measurable: source16.36°→final17.09°,
paired change+.723°, interval[+.602,+.841].

| Completed .20 episodes | Source WR2 GPU | Final WR2 GPU | TB Paper CPU reference |
|---|---:|---:|---:|
| Roll span | 16.36° | 17.09° | 19.32° |
| Root height span, TB-equivalent | 10.99mm | **12.37mm** | 7.45mm |
| Left-minus-right shoulder height span, TB-equivalent | 69.26mm | **72.38mm** | 44.74mm |

The shoulder differential removes common vertical translation. WR2 shoulder
mount spacing is **167.2mm**, versus TB's134.0mm, even though WR2's CoM-based
length ratio is.6791. Thus its normalized shoulder spacing is246.2mm,
**1.84× TB's**. For a nearly horizontal shoulder line,
`left_Z-right_Z ≈ shoulder_width*cos(pitch)*sin(roll)`.
Similar roll angles can therefore look larger on WR2. This is a direct
kinematic explanation of visible shoulder movement, not evidence that shoulder
geometry causes the learned leg asymmetry.

Root height variation is separately **66% larger than TB in normalized units**.
It rises12.6% over the WR2 source. This is a real remaining body-motion gap,
although the TB reference is small and is not a human natural-gait standard.
The trajectory figure is
`results/wr2_walking/wr2_paper_goal_xy_20261026_193545_seed3/heading_evaluation_gpu/review/shoulder_rocking.png`
(also saved asPDF). It plots torso roll, shoulder heights and root height in
shared TB time/length units, including the source and native TB nominal replay.

**.30 is outside the trained forward command range.** The logged actor command
is.30 TB m/s, but nearest-reference lookup remains at the maximum .20 row.
Observed body-forward speed is about.208 TB m/s. This is extrapolation with
an actor/reference command mismatch, and does not demonstrate stable trained
.30 performance. Its larger rocking should be assessed separately from .20.

## Reward/code implications and priority

WR2 retains TB Paper's relevant torso rewards: roll-range weight.5,
orientation weight1, angular-XY weight2 and vertical-velocity weight1.
The roll-range term is flat within **±.1rad/±5.73°**, then loses reward
smoothly; it is not a hard tilt limit or an objective of zero visible shoulder
motion. Orientation and angular-velocity rewards also penalize rocking. TB
uses the same roll band and formulas and its measured .20 policy also rolls
beyond that band. There is no missing shoulder-action reward to restore.

The remaining right-hip finding is stronger than nominal-only evidence:
final `.20` has **92.5%** negative-bound actions across254 completed randomized
episodes; at .10/.15 the fractions are46.6%/70.3%. This evaluation uses
deterministic policies with observation noise and model/reset/actuator
variation. It confirms saturated *mean-policy commands* in varied conditions;
it does not independently measure stochastic PPO exploration during training.
The earlier native stochastic fixed-input probes remain the evidence for
quantization-compressed physical exploration.

As [the action-space audit](wr2_goal_xy_result_20261027.md#hip-action-space-finding)
shows, WR2's reference-derived fixed hip origin gives a forward-angle target
window **+2.31°…+30.96°**, versus TB model-home **−5.98°…+22.67°**.
At .20 about30% of WR2 posterior stance reference targets fall outside its
window, versus about8.4% for TB. Mechanical joint stops and measured torque
do not explain the nearly stationary requested right hip. The forward part
of the window remains available, so coverage alone is not a demonstrated
sole cause of the learned asymmetric strategy or vertical bobbing.

### Next steps

1. **Retain final+200M as the higher-command candidate**, keep goal+120M as
   the stable .10/control policy, and finish final's full two-seed standard
   confirmation before promoting it across commands. The GPU comparison has
   qualified forward improvement, not the full standard suite.
2. **Audit hip action/posture compatibility before the next training config.**
   Resolve the verified TB-versus-WR2 origin difference while preserving the
   grounded reset, knee/ankle posture and checkpoint observation semantics.
   A blind whole-home replacement changes more than hip coverage. Define the
   supported target range and migration/reset behavior before implementation.
3. Use requested-direction progress, left/right landing lead and stride,
   action-bound fractions, torso roll and normalized root-height span as
   explicit acceptance measurements. Keep physics, qualified landing credit,
   moving leg-reference imitation and goal-frame XY feedback for this audit.
   More training cannot expand a fixed command window. Larger shoulder/roll
   reward changes would introduce another variable before its effect is clear.
4. Treat .30 as extrapolation. If .30 becomes an operating requirement, extend
   training/reference commands together after the .20 baseline is qualified.

Existing GPU command for step1, from the WR2 checkout:

```bash
uv run --locked --python 3.10 --extra training python -m wr2.locomotion.evaluate \
  --paper-benchmark --no-pushes \
  --config results/wr2_walking/wr2_paper_goal_xy_20261026_193545_seed3/training_config.yaml \
  --checkpoint results/wr2_walking/wr2_paper_goal_xy_20261026_193545_seed3/checkpoints/000200294400 \
  --seeds 12102,22102 --num-envs 128 \
  --output results/wr2_walking/wr2_paper_goal_xy_20261026_193545_seed3/paper_final_confirmation.json
```

No training or physics changes are made by this review. `CHANGELOG.md` awaits
user confirmation of the result, as required by repository instructions.

## References

- [ToddlerBot paper v2, Table VII](https://arxiv.org/html/2502.00893v2): torso,
  velocity, leg-reference and gait reward definitions.
- Local TB `paper-walking-baseline`: `toddlerbot/locomotion/walk_env.py`,
  `mjx_env.py`, frozen run config, and `descriptions/toddlerbot/toddlerbot.xml`.
- WR2 `wr2/locomotion/walking_env.py`, `paper_rewards.py`,
  `heading_evaluation.py`, `wr2/tools/evaluate_heading.py`, `view_walking.py`,
  and `wr2/descriptions/wr2/wr2_mjx.xml`.
- [Previous training and action-space review](wr2_goal_xy_result_20261027.md),
  [experiment definition](wr2_goal_frame_xy_experiment_20261026.md), and
  [TB Paper setup audit](wr2_tb_paper_setup_audit_20261018.md).
