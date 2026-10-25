# Matched leg-coordination result — 2026-10-23

**2026-10-24 follow-up:** the [completed bilateral diagnostic](wr2_bilateral_diagnostic_20261024.md)
confirms larger executed steps, but its bilateral reference-wave control rocks
fore and aft. Use its corrected v2 result with lateral-position preservation.

Follow-up to the [hip-pitch incentive analysis](wr2_hip_pitch_incentive_review_20261023.md).
Completed the requested hip/knee/ankle comparison at **80M and selected 120M**,
then checked full foot orientation to investigate the remaining yaw cost.

**Diagnostic centering correction:** the initial reference-wave normalization
used an earlier cycle that still contained a startup transient. The recorded
comparisons below remain valid for their actual commands, but the waveform was
not strictly mean-preserving. The corrected repeat is documented in
[Section 6](#6-reference-wave-centering-correction); it confirms the coordination
benefit and the remaining short-stride/posture gap.

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
| Hip only | Left hip pitch | Constant 3° forward-offset component plus reference-shaped swing |
| Knee/ankle compensation | Same hip plus left knee/ankle pitch | Preserve commanded foot height and fore-aft sole angle |
| Full orientation | Same hip plus left hip yaw, knee, ankle pitch/roll | Preserve commanded height and full foot orientation |
| Duplicate policy | None | Verify the paired numerical baseline |

The initial hip waveform used scale coefficients **5.00° at 80M /about 4.62° at
120M**, limited to the existing actor's reachable range. Those coefficients
were not guaranteed peak deviations at the later reference times. Its exact request and
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
right-leg response. The tests changed the left targets while holding the
right policy command sequence fixed. Increased yaw is measured, but a right-leg
force opposing the correction has not been established. Bilateral imbalance
is a remaining hypothesis, not a proved explanation of the historical PPO updates.

The result does **not select a new training recipe**. Keep the TB reward and
controller comparison intact while establishing a consistently better coordinated
control. The evidence from this test does not support disabling air time or
increasing XY weight to compensate for uncoordinated hip-only motion.

## 6. Reference-wave centering correction

The WR2 reference retains a full episode trajectory, including settling of the
preview/IK solution. Its mean left hip reference at ticks 30–59 differs from
the later walking cycles. Normalizing the intervention with that early cycle
introduced an unintended mean into the added waveform.

In the originally viewed 120M start-180 recording, the constant offset component
was 3° forward, but the added reference term and ramps made the actual mean
change **2.01° forward over 66 ticks**, with a net target-change range of
**7.19° forward to 2.00° backward**. This is a diagnostic-generation issue;
the training reference and saved policies are not modified by the correction.

The generator now centers the waveform over the **actual applied window**,
including ramp weights, and normalizes its maximum applied deviation. The wave
has zero applied mean and **5° peak** in all six cases. The constant 3° offset
remains at full strength; with ramps its 66-tick average is **2.727° forward**.
An offset-invariance test prevents startup/reference shifts entering the wave.

The corrected **30-control repeat completes without falls or nonfinite values**;
all actuator/IK constraints and identical hip-request checks pass. **Eight tests
pass.** Corrected 120M means:

| Measure | Policy | Hip only | Knee/ankle compensation | Full orientation |
| --- | ---: | ---: | ---: | ---: |
| Net forward speed, TB m/s | .0522 | .0553 | .0640 | .0649 |
| Forward velocity std, TB m/s | .0320 | .0830 | .0444 | .0428 |
| Left same-foot advance, actual WR2 mm | 26.2 | 17.6 | 34.2 | 35.3 |
| Discounted return change | 0 | −2.307 | −.127 | −.071 |

The conclusion survives: coordinated targets remove most hip-only instability.
Air return is restored, while yaw remains a residual cost. Full-orientation
controls remain below the policy in all three phases, so the correction does
not establish a new training preference. Left landing position remains about
**34 mm behind** the right.

At the commanded **.0824055 physical m/s** and **.60-second gait cycle**, matching
speed requires about **49.4 physical mm per same-foot stride**. The current
34–35 mm advance remains short. The diagnostic changes the left leg; the right
leg retains its forward-biased target pattern. Bilateral posture/propulsion
coordination is still needed for natural alternating placements.

[Corrected recordings and summary](../../results/wr2_walking/wr2_paper_step_20261022_143550_seed3/review_20261023/leg_coordination_centered_reference_diagnostic/summary.json)
and [corrected comparison figure](../../results/wr2_walking/wr2_paper_step_20261022_143550_seed3/review_20261023/leg_coordination_centered_reference_diagnostic/coordination_comparison.png)
are in a separate folder; the originally viewed recordings are retained.

## 7. Physical limits versus reward hypotheses — 2026-10-24

Read the saved 0.28.8 actuator manifest, current controller and robot limits,
native nominal trace, and TB `paper-walking-baseline` controller/model/config.
Current WR2 actuator values match the saved run. Larger placement is feasible
in the tested nominal controls, but these tests do not certify a balanced
bilateral gait at the commanded speed under training randomization.

The selected 120M trace after 1.5 TB seconds shows:

| Left hip-pitch measure | Value |
| --- | ---: |
| Mechanical range, WR2 joint sign | -90° to +40° |
| Fixed-home policy target interval | -30.96° to -2.31° |
| Actual joint interval | -4.32° to -4.01° |
| Delivered target peak-to-peak range | 1.68° |
| Target clipping / slew fractions | 0 / 0 |
| Recorded peak motor torque | .459 Nm |
| Recorded peak joint speed | .103 rad/s |

The modeled motoring ceiling starts at 4 Nm and tapers to zero at 5.818 rad/s;
braking is capped at 4 Nm. Recorded torques are last-substep forces at control
ticks, while speeds are integrated end-of-tick values. They do not resolve
all four substeps or prove absence of loaded saturation in every leg. The
left hip's small requested motion is already upstream of any mechanical stop.
Pitch/knee/ankle target clipping and slew are zero in this nominal trace;
hip-roll limits have previously bound some 80M requests.

Physical response remains relevant. Modeled WR2 joint friction is .4655 Nm,
with Kp=24.157 Nm/rad. The isolated Coulomb-only error threshold `friction/Kp`
is about **1.10°**, versus **.88°** for TB hip pitch (.252/16.40625). The WR2
target wave is only about .84° amplitude. Gravity, contact, damping and solver
behavior affect breakaway, so this calculation is a clue rather than a measured
loaded deadband. Observed left target/actual error peaks at **1.084°**, close to
the isolated 1.104° threshold. The same servo model executes right-hip target
variation **10.32°** as **8.07°** actual motion. The amplitude dependence is
consistent with friction effects, with differing leg loads still unresolved.
Mean drift/tanh compression and small-motion attenuation can
compound each other without a hard range or torque ceiling preventing larger
motion.

Use the corrected intervention results when deciding whether reward should change:
at 120M, full-orientation compensation increases left same-foot advance
**26.22 ->35.25 physical mm** and speed **.05217 ->.06486 TB m/s**. Its discounted
XY contribution changes **+.0454**, yaw **-.1343**, air **+.0009**, and total
**-.0714**, averaged over three starting phases. Left landing lead remains
**-33.96 mm**. At 80M, corrected full-orientation advance is **43.74 mm**, versus
31.86 mm in the policy, with speed .07896 TB m/s. These are diagnostic controls,
not learned training improvements or complete normal alternating placements.

The next investigation should establish **bilateral** posture/propulsion and
yaw balance, audit per-substep loaded torque/speed and actual tracking, and
compare the complete native reward for a feasible smoother passing gait.
Higher native return with weak PPO acquisition points toward learning,
exploration or action-origin conditioning; lower return for genuinely improved
command following and stability would justify investigating the objective.
The current one-leg interventions do not establish the latter condition.

Keep the requested supported-swing/sustained-landing reward. TB's successful
recipe has no explicit forward-passing reward; its absence alone does not select
a new WR2 reward. Home-origin selection and delayed-credit learning remain
candidates, with the known mean compression and bilateral coordination gap
the most direct observed mechanisms.

PPO observes actual motor positions/velocities, and rewards actual motor-reference
error and body motion. The PD controller and friction model stay fixed; there
is no direct policy-target tracking-error reward. PPO can learn larger or
phase-advanced targets and coordinated other-joint changes if exploration finds
them and the full expected return improves. It does not automatically invert
the actuator response. The observed tracking gap and "loaded actuator response"
are one physical-response hypothesis rather than two independent causes.

No reward explicitly prefers the left foot behind the right. Air credit can be
paid for a trailing landing, while yaw and instantaneous velocity terms can
oppose an unbalanced correction. In the corrected 120M full-orientation test,
leg-imitation gain is only +.0065 and XY gain +.0454, versus yaw loss -.1343.
That explains this local comparison; it does not identify the historical reward
pressure that selected the original asymmetric posture. A higher-return nominal
bilateral control would motivate checking acquisition under the training
distribution, rather than proving a learning/exploration failure by itself.

The WR2 result's ignored `review_20261024/physical_feasibility_audit.json`
retains all twelve joint intervals, clipping, sampled torque/speed maxima,
corrected intervention comparisons, TB friction/Kp and the scope limitations.

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
For the corrected repeat, replace the directory with
`leg_coordination_centered_reference_diagnostic`.
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
