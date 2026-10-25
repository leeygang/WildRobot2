# WR2 speed, foot contact and forward lean — 2026-10-25

## Findings

**The ZMP reference does not request foot overlap at the inspected forward
commands.** At .10 TB m/s, the current PPO gait produces occasional box
contacts while the imported CAD foot parts remain separated. The learned
stance is closer than the reference, and the rectangular colliders include
space outside the CAD convex hulls.

**Higher commands restore passing on both sides.** The selected 300M policy
has 0/25 right landings ahead at .10, but 25/25 at .15 and .20. Unequal left/right
placement remains. This weakens the case for adding a universal passing gate
based only on the low-speed recording.

**Forward lean grows with command:** about 5.0°, 7.6°, 10.0° for WR2 .10/.15/.20,
versus native TB Paper approximately 3.5°, 3.4°, 3.2°. The reference torso is
upright; the lean is an executed dynamic posture under learned leg commands.
It is not a commanded ZMP torso-pitch trajectory. This comparison does not
isolate whether reward tradeoffs or actuator/load response are the main cause
of the extra lean.

Keep XY15 and the requested sustained-landing treatment. The current evidence
does not select another reward, reference or physical-model change.

## 1. Sources and scope

Run: `wr2_paper_progress_20261025_085931_seed3`, selected 300M, version .29.0.
Replayed original deterministic policies at **40M, 120M and 300M**, each at
**.10/.15/.20 TB m/s**: nine WR2 nominal episodes, 824 control ticks each.
All survive without nonfinite events. No policy action is overridden.

The matching TB Paper selected **200M** policy completes three native nominal
episodes of 1000 ticks. Local frozen `paper-walking-baseline` source at
**`a4f693d`** is verified against the saved run and native Brax **`1414780`**.
The active TB checkout is read only. The frozen diagnostic is reused with only
its condition selector changed in memory; the driver hash and override are
recorded, and native environment/reward/controller code is unchanged.

WR2 uses Mac CPU JAX .6.2, Brax .14.2, MuJoCo/MJX 3.14.0, Flax .11.2;
TB uses its saved Python 3.10/native Brax .10.5/MuJoCo-MJX 3.2.4 runtime on
CPU. Both use high matrix precision, no noise/randomization or pushes, and a
zero-yaw reset. These are nominal CPU checks, not a randomized GPU benchmark
or the user's exact GPU visual session. The new visual logs capture that session.

Time/linear velocity scale by **`s=.8240547565`**, distance by **`s²=.67906624`**.
824 WR2 ticks and 1000 TB ticks are both approximately 20 TB seconds. Comparisons
exclude at least the first 1.5 TB seconds. Sustained landing requires ten WR2
or twelve TB contact samples, approximately matched elapsed contact spans.
Terminal/reset samples are excluded from placement/velocity summaries.

Read both robots' actual reference, action and reward code. Both controlled
leg targets use fixed home plus `.25*tanh(action)`; the moving joint reference
is used for imitation and critic information. Thus PPO can learn a stance
different from the ZMP reference. TB Table VII uses XY5 and soft torso rewards;
WR2's completed experiment uses XY15, with the other declared departures retained.

Reference: [ToddlerBot Tables VI–VII](https://arxiv.org/html/2502.00893v2).

## 2. What the .10 contact represents

Selected 300M, after warmup, physical WR2 millimetres:

| Pose | Minimum box gap | Minimum CAD convex-hull gap | CAD overlap |
| --- | ---: | ---: | ---: |
| Moving ZMP reference | +22.31 | +23.63 | 0 |
| Reference bounded to actor/joint limits | +22.31 | +22.96 | 0 |
| Ideal delivered-joint target pose | −1.41 | +5.58 | 0 |
| Executed cached foot frames | **−.192** | **+4.78** | **0** |

The executed recording has seven box-force contact ticks out of 762 post-warmup
ticks (**.92%**), with maximum recorded box self-force about 1.14 N. At those
ticks the minimum CAD gap is **+5.05 mm**. The negative box-gap fraction with a
10-micrometre overlap tolerance is a distinct .79%; geometric overlap and the
native force-based contact label are not substituted for each other.

The CAD check uses both imported parts of each foot and all four cross-foot
pairs at the recorded final-substep frames. Positive separation is independently
confirmed by projecting **every mesh vertex** onto the separation direction.
The minimum positive separator agrees with the +4.78 mm gap. This establishes
separation of the imported parts in these poses, including their convex hulls.

The reference geometry also clears both boxes at the selected .14 and .20
lookup rows, with minimum gaps **22.29 /22.26 mm** over all 1000 reference poses.
The .15 requested command selects WR2's .14 lookup row; TB's native selector
selects .16. Both rewards track the actual continuous command. That lookup
quantization remains explicit in the report and visual-evaluation parameters.

**Interpretation:** learning brings the feet closer than the reference. The
conservative box envelope then generates some modeled contacts outside the
CAD surfaces. This is evidence against reference-induced geometric overlap;
it is not evidence that every GPU/randomized pose has CAD clearance. Check the
logged `box_gap_m` and `CAD_convex_gap_m` for the actual visual session.

At .15/.20, executed CAD minima remain **+4.36 /+5.47 mm**; all actual CAD
frames remain clear in this audit. At .20, ideal delivered-target poses have
CAD overlap on **3.67%** of samples, minimum **−5.80 mm**, despite executed
clearance. Ideal position-target geometry is a kinematic counterfactual, not
observed physical collision or proof that a target pose is dynamically reachable.
Commanded and executed geometry need separate measurements.

## 3. Speed and alternating placement

Mean body-frame forward velocity at the native reward point, TB m/s:

| Command | WR2 300M | TB Paper 200M | WR2 right landings ahead |
| --- | ---: | ---: | ---: |
| .10 | .09121 | .09187 | **0/25** |
| .15 | .14708 | .13749 | **25/25** |
| .20 | .19596 | .17493 | **25/25** |

WR2 left landings are ahead on 25/25 events at all three speeds. Mean left/right
landing leads are **43.18/−1.05**, **57.57/12.06**, and **70.64/22.70 physical mm**.
The passing failure therefore depends on the command, but higher speeds still
show unequal leading distances.

At .20 the heading-aligned same-foot advances are approximately **95.37/100.61
physical mm**, consistent with the commanded .16481 physical m/s and .60-second
cycle: about **98.9 mm** per same-foot cycle. The fixed world-X advances of
34.35/40.60 mm are much smaller because the robot turns. They must not be called
short forward strides without accounting for the heading change.

The 40M and 120M policies also pass on both sides at higher commands. At .20,
120M achieves .17211 TB m/s with **49.98/35.65 mm** mean leading placements,
smaller asymmetry than final 300M but worse forward response. Intermediate
checkpoints remain useful alternatives, not just training history.

## 4. Lean and heading

| Command | WR2 mean pitch | TB mean pitch | WR2 heading variation | TB heading variation |
| --- | ---: | ---: | ---: | ---: |
| .10 | +5.03° | +3.46° | 32.8° | 23.6° |
| .15 | +7.57° | +3.39° | 79.9° | 26.1° |
| .20 | +10.03° | +3.19° | 106.5° | 54.1° |

Heading variation is the span of unwrapped yaw over the post-warmup window;
WR2 net heading change in these episodes equals that span. The yaw command is
zero. Stable locomotion and accurate body-frame forward speed do not establish
straight world-frame motion. The logger records both coordinate systems and
torso quaternion/Euler values to make that distinction measurable.

An additional check through the exact single-policy live-viewer path at .20,
on CPU with its default matrix precision, records forward speed **.19658 TB
m/s**, displayed mean pitch **10.04°**, net heading change **103.3°**, and minimum
CAD gap **+5.62 mm**. This confirms the qualitative lean/heading result through
the viewer path as well as the batched probe. It remains separate from the GPU
session; its `params.json` identifies the headless 824-step analysis.

The reference's torso pitch is zero for these straight commands. The body
is floating and controlled indirectly by leg torques and ground forces. Forward
lean therefore appears through the executed policy/dynamics interaction. Its
mean at .20 is within TB's configured soft pitch band of **±.20 rad /±11.46°**,
but is appreciably larger than TB's executed pitch. The coefficient change,
different native actuators and different robots are not independently isolated
by this cross-robot table.

No stronger upright penalty is selected from the visual observation alone.
Use the session's measured pitch, heading, tracking, contacts and weighted
reward contributions to understand the tradeoff first.

## 5. Next direction and logging

1. **Record the actual GPU live evaluations at .10/.15/.20**, using the current
   command and automatic `results/visual_eval/<session>/` logging. Compare the
   exact parameters and sampled body/world metrics, not playback appearance.
2. **Check collider versus CAD contact in those logs.** Reference poses are
   clear here; a CAD-positive box contact is a modeling/learned-clearance
   interaction. Validate the contact surface before designing another contact
   or placement incentive from box counts alone.
3. **Assess low-command asymmetry, unequal placement and heading together.**
   The higher-speed passing result changes the earlier proposal: a universal
   passing gate is not selected. Keep XY15 and sustained landing; compare the
   retained policies and TB before preparing another reward experiment.

Live logs include **`params.json`**, **`metrics.jsonl`** and **`summary.json`**.
See the [logging schema and copy command](wr2_stance_diagnostic_20261018.md#live-evaluation-logs).
`scp_from_remote.sh --latest visual_eval` copies the newest session to the same
Mac result path. A viewer episode-reset issue found during logging verification
is corrected: physical state restarts along with counters, while the final
pre-reset physics sample remains in the log.

## Artifacts

The audit is under
`results/wr2_walking/wr2_paper_progress_20261025_085931_seed3/review_20261025/speed_contact_audit`:
original-policy traces, source/provenance, `reference_geometry.json`,
`analysis.json`, native TB traces and the figure.

[Speed comparison](../../results/wr2_walking/wr2_paper_progress_20261025_085931_seed3/review_20261025/speed_contact_audit/speed_comparison.png)
and [numerical audit](../../results/wr2_walking/wr2_paper_progress_20261025_085931_seed3/review_20261025/speed_contact_audit/analysis.json).
