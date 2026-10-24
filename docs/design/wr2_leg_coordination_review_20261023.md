# Matched leg-coordination result — 2026-10-23

Follow-up to the [hip-pitch incentive analysis](wr2_hip_pitch_incentive_review_20261023.md).
Completed the requested hip/knee/ankle comparison at **80M and selected 120M**,
then checked full foot orientation to investigate the remaining yaw cost.

## Conclusion

**Joint coordination explains most of the destabilization and return loss in
the earlier hip-only test.** The exact same hip request, with knee/ankle
compensation, produces more forward movement and much less fore-aft oscillation.
At 120M, the return deficit shrinks **−1.604 →−.051**; full orientation
compensation reduces it further to **−.038**.

**Air credit is essentially restored**, while XY tracking earns more reward.
The remaining deficit is principally yaw tracking. These results weaken the
case for changing air weight to address this particular gap. They support
keeping the TB tracking terms while investigating the remaining coordination.

The success criterion of a consistent reward advantage over the original
policy is **not yet met**. The left foot still trails the right; these are
improved short controls, not a new trained walking policy or TB parity.

## 1. What was tested

Same WR2 model, controller, reference, reward weights and gamma **.97**. Walking
command is **+.10 TB m/s**, **+.0824055 physical WR2 m/s**. Size normalization
is `s=.8240547565`: distances divide by `s²`, time by `s`, linear velocity by `s`.

At ticks **180/190/200** of each checkpoint's nominal forward rollout, clone the
same complete environment state and compare:

| Control | Changed targets | Constraint |
| --- | --- | --- |
| Policy | None | Original deterministic action sequence |
| Hip only | Left hip pitch | Mean 3° forward plus native reference-shaped swing |
| Knee/ankle compensation | Same hip plus left knee/ankle pitch | Preserve commanded foot height and fore-aft sole angle |
| Full orientation | Same hip plus left hip yaw, knee, ankle pitch/roll | Preserve commanded height and full foot orientation |
| Duplicate policy | None | Verify the paired numerical baseline |

The hip waveform has maximum added deviation **5.00° at 80M /about 4.62° at
120M**, limited to the existing actor's reachable range. Its exact request and
delivered target are identical between all interventions. Other joint commands
are fixed to the original policy's recorded sequence; full orientation also
holds hip roll fixed. The actor does not react through those held commands.

Treatment lasts two gait cycles with six-tick smooth ramps. It is fully removed
after tick 66, and the original sequence continues to tick 99: **1.98 physical
/2.40 TB seconds** per control. These are three deterministic start phases,
not independent training seeds or a randomized performance benchmark.

The first experiment has **24 verified controls**. The final orientation check
has **30**, including the same policy/hip/knee-ankle comparisons. **All 54
complete 99 ticks without falls or nonfinite values.** Tables below use the
final 30-control dataset.

## 2. Kinematic constraints and verification

IK fixes hip pitch and solves the remaining selected targets with the existing
MuJoCo geometry, inside both joint limits and `q_home ±.25 rad` actor bounds.
The target baseline is the policy's intended pose **after joint-limit clipping**.
This matches the controller's bounded command; delay, encoder quantization and
slew remain active when the resulting targets are executed.

The knee/ankle solve has two constraints: foot-center height and fore-aft sole
angle. It does not constrain roll/yaw. Its maximum remaining full normal change
is **.0269 rad /1.54°**. The additional check uses four unknowns to preserve
height and all three orientation coordinates. Foot horizontal position is free
and measured; the maximum added lateral displacement is **1.64 mm**.

The initially attempted lateral-position constraint was unnecessary for this
test and overconstrained the solve near the 80M hip-roll limit. That policy
occasionally requests about **10.24°**, while the joint limit is **10°** and
the native controller clips it. The final comparison uses the bounded baseline
and preserves hip roll. No robot limit is changed.

All **390 target solves** in each variant pass. Maximum requested height/pitch
errors in the two-joint solve are **8.1e−11 m /3.3e−11 rad**. The full-orientation
check has maximum height error **6.1e−11 m**, rotation error **3.8e−11 rad**.
These are commanded-pose constraints, not guarantees of actual joint tracking.

Checks verify identical requested/delivered hip trajectories, exact unchanged
action entries, actuator reachability, controller-target reconstruction, reward
component sums and duplicate controls. **Seven physical IK tests and Ruff pass.**

## 3. Selected 120M result

Means across the three start phases. Velocity/placement measurements use the
treatment/ramp window; discounted return includes the continuation.

| Measure | Policy | Hip only | Knee/ankle compensation | Full orientation |
| --- | ---: | ---: | ---: | ---: |
| Net forward speed, TB m/s | .0522 | .0603 | **.0645** | **.0645** |
| Forward velocity std, TB m/s | .0320 | .0849 | **.0426** | **.0420** |
| Backward-velocity samples | 1.5% | 19.7% | **4.5%** | **5.1%** |
| Mean left advance between landings, actual WR2 mm | 26.2 | 26.4 | **34.8** | **35.0** |
| Loaded left sole-pitch range | .83° | 5.15° | **.95°** | **1.10°** |
| Discounted return change | 0 | −1.604 | **−.051** | **−.038** |

Compensation increases actual left landing advance about **33%** over the
policy and halves the hip-only control's fore-aft velocity variation. It recovers
**96.8%** of that control's return deficit; full orientation recovers **97.6%**.
The improvement follows changing knee/ankle coordination with an identical
hip request, rather than increasing the hip target again.

The original policy still has lower forward-velocity variation. Full orientation
has one slightly positive phase (**+.010**) and two negative phases; its overall
range is **−.072 to +.010**, mean −.038. This is not a consistent reward advantage.
Endpoint critic bootstrapping preserves the near-zero negative mean (**−.037**).
Those value estimates are supplementary: the critic was trained with stochastic
actions and noisy/randomized states, whereas these controls are nominal tapes.

### Where the return difference went

Mean discounted component changes from the policy:

| Component | Hip only | Knee/ankle compensation | Full orientation |
| --- | ---: | ---: | ---: |
| XY velocity tracking | **−.391** | **+.082** | **+.069** |
| Air time | **−.562** | approximately −.002 | approximately −.002 |
| Yaw tracking | −.487 | **−.160** | **−.125** |
| Angular XY tracking | −.027 | +.031 | +.026 |
| Leg-reference imitation | +.0025 | +.0054 | +.0052 |

The hip-only control's rejected physical credit disappears in both compensation
controls. Qualified physical credit is **1.887 s**, versus **1.893 s** in the
policy. Thus the previous large air loss is substantially a consequence of the
uncoordinated motion/support sequence, not evidence that any larger forward step
must lose that reward.

The existing XY tracking term now rewards the smoother movement. This supports
its present interpretation. Yaw remains the principal residual cost, even after
the commanded foot orientation is preserved.

## 4. 80M and TB comparison

At 80M, knee/ankle compensation raises net forward speed **.0630 →.0811 TB m/s**;
full orientation gives **.0816**. Mean left advance is **31.9 →45.3 physical mm**
with full orientation. The return deficit is **−.183 /−.106**, versus −1.160
for hip only. Every tested 80M full-orientation phase remains slightly negative.

The matched-duration native TB Paper recording, at corresponding reference-cycle
phases, gives approximately **.0954 TB m/s**, forward velocity std **.0137**, and
left advance **69.8 TB mm** (approximately **47.4 physical WR2 mm**). WR2 80M
therefore approaches that placement size under intervention, while smoothness
and mean speed remain worse. At 120M, full-orientation left advance is only
**51.5 TB mm**.

This is a normalized native comparison. TB uses its original model/controller,
MuJoCo 3.2.4/Brax fork and recorded observation behavior; WR2 runs nominally on
Mac CPU with MuJoCo 3.14.0/Brax .14.2. No claim of identical robot physics or
matched stochastic training conditions is made. The TB window is approximately
1.60 seconds, at ticks 216/228/240; the WR2 window is 66 physical ticks.

The [local `paper-walking-baseline` source](wr2_hip_pitch_incentive_review_20261023.md)
at `a4f693d` confirms a coordinated six-joint reference, converted to motor
targets through leg IK. WR2's reference likewise solves complete leg poses
from desired foot position/orientation. Fixing the hip for this diagnostic
requires constrained IK on the selected remaining joints. Reward coefficients
and the **.25-rad** action family remain the TB-aligned settings discussed in
[Tables VI–VII](https://arxiv.org/html/2502.00893v2).

## 5. What remains unresolved

**The left foot still trails the right.** At 120M, mean landing lead changes
from **−48.7 mm** to about **−36.0 mm**; at 80M it remains approximately −21.4 mm.
Larger same-foot advance does not establish normal alternating leading feet.
The right leg's learned posture and target sequence remain fixed in this test.

Full orientation improves the residual return slightly but does not eliminate
yaw cost. Its commanded orientation is exact; actual orientation and body
motion still depend on joint execution and the changed body's motion. These
data do not isolate the remaining yaw-force mechanism.

The most specific next investigation is **bilateral propulsion/yaw balance and
the learned posture bias**, using floor-force moments and the phase-dependent
right-leg response. A left-only correction acts against the existing right-leg
pattern. This is a concrete remaining hypothesis, not a proved explanation of
the historical PPO updates.

The result does **not select a new training recipe**. Keep the TB reward and
controller comparison intact while establishing a consistently better coordinated
control. The evidence from this test does not support disabling air time or
increasing XY weight to compensate for uncoordinated hip-only motion.

## Artifacts and visual evaluation

[Comparison figure](../../results/wr2_walking/wr2_paper_step_20261022_143550_seed3/review_20261023/leg_coordination_full_pose_diagnostic/coordination_comparison.png),
[verified rollout summary](../../results/wr2_walking/wr2_paper_step_20261022_143550_seed3/review_20261023/leg_coordination_full_pose_diagnostic/summary.json),
[placement/motion/component analysis](../../results/wr2_walking/wr2_paper_step_20261022_143550_seed3/review_20261023/leg_coordination_full_pose_diagnostic/analysis.json).
Data/figures remain in the ignored result directory. The source is
[diagnose_leg_coordination.py](../../wr2/tools/diagnose_leg_coordination.py),
with [IK tests](../../tests/test_leg_coordination.py).

The recorded request field is **`current_request_rad`**, including interventions.
It is not a newly learned actor output. The original recordings' field was
renamed from `actor_request_rad` without changing numerical arrays; metadata
retains the actual simulation source and records that schema correction.

On the Mac, view the short selected-policy baseline:

```bash
uv run --no-sync mjpython -m wr2.tools.view_walking \
  --trace results/wr2_walking/wr2_paper_step_20261022_143550_seed3/review_20261023/leg_coordination_full_pose_diagnostic/120M_start180_baseline.npz \
  --playback-speed 0.25
```

Use `120M_start180_hip_only.npz`, `120M_start180_coordinated.npz`, or
`120M_start180_full_pose.npz` in the same directory for the interventions.
The recordings begin at an intermediate episode state and last about 2.4
TB seconds. They are diagnostic controls, not newly trained checkpoints.

The diagnostic runs through the normal Python module entrypoint:

```bash
uv run --no-sync python -m wr2.tools.diagnose_leg_coordination \
  --run-dir results/wr2_walking/wr2_paper_step_20261022_143550_seed3 \
  --output results/wr2_walking/wr2_paper_step_20261022_143550_seed3/review_20261023/leg_coordination_repeated
```

It requires the saved configuration, 80M checkpoint and selected `best_params`;
it constructs its own native baselines and needs no ToddlerBot Python import.
GPU confirmation of these new WR2 controls is still absent.
