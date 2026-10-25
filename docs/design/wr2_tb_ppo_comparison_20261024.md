# Native TB Paper / WR2 PPO comparison — 2026-10-24

Results:

- TB: `results/toddlerbot_walk_ppo_20261015_194347/ppo_learning_diagnostic_v5`.
- WR2: `results/wr2_walking/wr2_paper_step_20261022_143550_seed3/ppo_learning_diagnostic_v5`.
- Prior context: [WR2 transition review](wr2_ppo_transition_review_20261024.md).

## Conclusion

**The large fresh-Adam command-switch update is not specific to WR2.** Probing
TB's already useful 80M policy produces final-policy KL **7.79**, larger than
WR2's **1.00** at 120M. Both have concentrated motion/standing critic residuals
and the same one-decision command-observation delay. This weakens our earlier
PPO-update hypothesis as an explanation of WR2's gait. The probes do not
reconstruct historical Adam updates or evaluate the resulting probe policy.

The robust gait difference is **WR2's nearly fixed, backward-biased left hip
target**: phase-conditioned swing is **1.65°**, versus **10–12°** in TB. WR2's
normalized same-foot stride is approximately half TB's, and its left foot stays
behind the right at sustained landings.

One verified source difference is **action-home selection**. TB uses its model's
`home` keyframe. WR2 initially reads its
configured model home, then its Paper branch overwrites that home using the
ZMP reference's standing pose. That moves the hip action origin from **6.88° to
16.63°** and shifts the permitted target interval. Both homes passed a preliminary
nominal MuJoCo hold check. This makes model-home selection a candidate comparison,
but does not establish that the current origin causes the short steps.

**The home-origin difference is verified; its causal effect on long-run gait
learning is not.** A controlled fresh training run is needed to test that effect.
The confirmed proximate mechanism remains mean bias plus tanh saturation reducing
cyclic target motion. These data do not prove why the biased solution was learned.

**A fixed-home action baseline is shared by TB and WR2.** The earlier WR2
walking-reward mismatch was penalizing departure from home; the current Paper
recipe instead rewards imitation of the moving leg reference. Changing controlled
leg targets to `q_ref(t) + residual` would be a new action design. TB Paper starts
full motor targets from the reference, then overwrites the twelve controlled leg
entries with its fixed-home policy targets.

## 1. Integrity and fair comparison

TB completed **24 batches**, three checkpoints ×8 ×1024 environments ×20 ticks
= **491,520 transitions**. WR2 completed 16 batches, **327,680 transitions**.
Both use the same diagnostic/math source hashes. TB ran clean
`paper-walking-baseline` **`a4f693d`**, with clean Brax **`1414780`** and its saved
Python/package versions; WR2 used its saved native stack. Both ran on GPU.

| Maximum audited reconstruction error | TB | WR2 |
| --- | ---: | ---: |
| Float32-reference actor parameter gradient | 5.426e−7 | 7.178e−7 |
| Native output gradient | 7.601e−8 | 7.568e−8 |
| Advantage reconstruction | 1.776e−15 | 1.776e−15 |
| Scalar reward reconstruction | 9.537e−7 | 1.431e−6 |

Exact first-minibatch indices and entropy keys passed. Recorded predictions and
GAE targets reproduce native `.25 MSE` within **1.36e−6 TB /1.12e−6 WR2**.
Native matrix precision is TB `high`, WR2 default; verification uses separate
`highest`/Float32 calculations. Native cross-graph kernel gaps remain recorded.

TB warmup **408** and WR2 **330** place the main command switch at batch tick 2.
Both use native 20-tick fragments. Their normalized durations are `.4000/.4854`
TB seconds, and their native gait phases/model distributions differ. This
comparison aligns the switch position while retaining those physical differences.

All reported updates start **fresh Adam** at `1e−4`, apply four epochs ×four
minibatches, and discard weights. Original frozen policies collect all batches;
their trajectories are shared across batches. A GAE target bootstraps the same
critic and is not a complete realized-return ground truth.

## 2. Command-switch residuals and KL also occur in TB

| Policy | Switch-batch value loss | Ordinary-batch median | Switch final KL | Ordinary final KL median |
| --- | ---: | ---: | ---: | ---: |
| WR2 80M | .41922 | .02216 | .08366 | .03416 |
| WR2 120M | .51315 | .01926 | **1.00085** | .03505 |
| TB 80M | .21321 | .01373 | **7.79019** | .08796 |
| TB 120M | .20252 | .02153 | .16703 | .09241 |
| TB 200M | .11262 | .01367 | .09877 | .09962 |

KL means `KL(original Gaussian || policy after all 16 steps)`, summed over
actions on the full original observation batch. It is separate from native
last-epoch minibatch KL. TB's 80M spike remains **7.804** after excluding
environments that fall in the batch; falls do not explain it. Its largest
deterministic target change is **3.49°**, versus **3.26°** for WR2 120M. Gaussian
KL also depends on the policy's state-specific std, so its magnitude alone is
not a physical walking-quality measure.

| Switch-batch audit | Residual squared mass in first 3 ticks | Residual squared mass from motion/zero status changes |
| --- | ---: | ---: |
| WR2 120M | 96.76% | 96.31% |
| TB 80M | 93.69% | 93.83% |
| TB 120M | 89.77% | 86.11% |
| TB 200M | 92.21% | 90.01% |

Squared normalized actor advantage is similarly concentrated. Motion includes
translation or turning, with normalized command norm above `.05`. Native code
and recordings confirm that observations are built before the next command is
assigned; one subsequent decision still sees the old command. This shared timing
is not a WR2-specific implementation failure.

WR2's later residual is larger, but that alone does not establish a faulty critic
or the cause of its mean bias. Future randomized commands can create irreducible
target variance. TB's successful checkpoint also has a substantial transition
residual. We should not redesign PPO from the earlier WR2-only spike.

## 3. The actual learned-motion difference

Forward samples have nearly identical mean commands, approximately **.108 TB
m/s**. WR2 velocities divide by `s=.8240547565`; distances divide by
`s²=.6790662418`. Angles below are nominal requested motor angles in physical
forward hip directions; both hip-pitch motors are directly transmitted.

| Forward-command measure | WR2 120M | TB 120M | TB 200M |
| --- | ---: | ---: | ---: |
| Achieved forward speed, TB m/s | **.04041** | .08241 | **.09520** |
| Left phase-conditioned hip target range | **1.65°** | 12.29° | 10.20° |
| Right phase-conditioned hip target range | 8.95° | 9.55° | 9.40° |
| Mean left forward hip target | **3.87°** | 7.52° | 8.10° |
| Mean right forward hip target | **28.17°** | 14.25° | 13.89° |
| Mean left tanh sensitivity | **.2036** | .8915 | .9146 |
| Raw left Gaussian std | .2193 | .0476 | .0359 |
| Sampled left target RMS deviation about mode | .720° | .807° | .547° |

Phase-conditioned range is max minus min over twenty phase-bin means, not a
single nominal video's full range. The right hip retains substantial cyclic
motion; the collapse is strongly asymmetric. WR2 still explores around its biased target;
the measured physical request variance has not vanished. Increasing raw noise
is not selected by this comparison.

The raw cyclic policy signal remains present. Re-reading all eight batches per
checkpoint with the same forward-command mask and twenty phase bins gives:

| Left-hip measure | WR2 80M | WR2 120M | TB 200M |
| --- | ---: | ---: | ---: |
| Mean raw Gaussian `mu` | .938 | **1.476** | -.014 |
| Phase-bin range of raw `mu` | .273 | **.564** | .776 |
| Phase-bin range of bounded target | 1.88° | **1.65°** | 10.20° |

WR2's raw phase variation grows while its bounded angle variation shrinks. The
increasing mean pushes that signal into tanh's compressed region. This confirms
the immediate loss of useful target motion; it does not identify which training
tradeoff selected the biased mean. The fixed home adds an offset to the bounded
target and does not itself remove the raw cyclic signal.

### Stride and landing placement

Same-foot stride below is **unsigned horizontal displacement** between sustained
landings at an unchanged forward command, excluding terminal intervals. Root
heading is not recorded, so signed forward stride cannot be recovered exactly.
Contact filters have approximately matched duration: WR2 ten samples span `.2184`
TB seconds, TB twelve samples span `.2200` seconds.

| Placement measure | WR2 120M | TB 120M | TB 200M |
| --- | ---: | ---: | ---: |
| Mean left horizontal stride, TB mm | **35.66** | 69.13 | **71.65** |
| Mean right horizontal stride, TB mm | **32.63** | 69.01 | **72.08** |
| Sustained left landings ahead of right | **0/862** | 471/697 | **731/784** |
| Sustained right landings ahead of left | 790/790 | 679/679 | 686/686 |

WR2's mean strides correspond to approximately **24.2/22.2 physical mm**. The
small steps are a real original-policy behavior, not merely an imitation metric
or a hypothetical target change. TB's two feet normally exchange the lead;
WR2 retains the right-leading posture.

## 4. Reward accounting does not select a weight change

| Native reward rate per TB-equivalent second | WR2 120M | TB 120M | TB 200M |
| --- | ---: | ---: | ---: |
| XY tracking | 1.842 | 3.715 | 3.900 |
| Yaw tracking | 3.422 | 3.856 | 3.877 |
| Torso stability | 2.334 | 2.459 | 2.364 |
| Leg imitation | −.203 | −.293 | −.279 |
| Air time | **10.655** | 8.129 | 7.889 |
| Other rewards | 2.016 | 1.530 | 1.733 |

Both native implementations use **negative mean squared leg-motor reference
error, weight 5**. WR2's left hip actual/reference RMSE is `.1647 rad`, versus
TB `.3246/.3020 rad`. TB walks better while deviating more from the hip reference.
Its reference's phase-bin hip range is about **37°**, versus its learned **10–12°**;
WR2's reference still varies about **21°**, versus learned **1.65°**. The reference
is not missing or static, and literal matching of its hip waveform is not the
successful TB policy's behavior.

Reward-component SGD projections remain an accounting convention with zero
component critics and an explicit value/bootstrap residual. TB also has large
negative air projections cancelling positive residuals, and SGD projection signs
can differ from actual Adam target changes. Those signs do not isolate causal
effects of reward coefficients. No missing symmetry augmentation was found in
the native Paper PPO path.

WR2 additionally qualifies supported air time after ten contact samples and pays
with **`.97⁻⁹ = 1.3154`** compensation. This explains part of its larger paid-air
rate, but the compensation is intentional discounted-return bookkeeping. With
an isolated reward and zero component critic, delaying its reward by nine ticks
attenuates earlier swing-action GAE contribution by **`.95⁹ = .6302`** after gamma
compensation. That is not a measured reduction of the total native actor gradient:
the learned critic, bootstrap, clipping and finite fragments intervene.
Short/asymmetric steps were already present in the preceding collision run with
no stance qualification, so this added rule cannot be the initiating cause.

## 5. A concrete TB contract difference: the action origin

TB Paper `mjx_env.py` reads `default_qpos` from the **model `home` keyframe** and
uses its motor positions for `default_action` and motor-home observations.
Reference-based reset initialization is computed separately.

WR2 initially sets `_home_ctrl = robot.home_position_rad`, its configured
**model `walk_home`**. The Paper branch then sets `_default_qpos` to
`PaperWalkReference.standing_qpos` and overwrites `_home_ctrl` with those reference
positions. Its comment attributes this behavior to TB, but TB does not overwrite
the model action origin that way. Reference root-height/standing-IK settings now
affect WR2's actor target origin and home observation offset.

| Nominal origin, absolute angles | Hip pitch | Knee | Ankle pitch |
| --- | ---: | ---: | ---: |
| TB model home used for actions | 8.35° | 30.64° | 21.74° |
| TB zero-command reference reset | 7.04° | 29.98° | 22.94° |
| WR2 configured model walk-home | **6.88°** | **13.75°** | **6.88°** |
| WR2 Paper origin from reference | **16.63°** | **40.19°** | **23.55°** |

TB's reset/model difference is at most **1.31°** in this neutral reference check.
WR2's model/used difference reaches **26.44°** at the knee. These are different
robot poses; restoring the home-selection rule uses WR2's own model positions,
not a copy of TB's joint angles.

Both use fixed home plus `.25*tanh(mu)`. Their hip control intervals therefore are:

| Nominal hip forward-angle interval | Degrees |
| --- | --- |
| TB model origin | −5.98 to +22.67 |
| WR2 origin actually used | **+2.31 to +30.96** |
| WR2 configured model origin | **−7.45 to +21.20** |

WR2's left mean **3.87°** is near the lower endpoint; its right mean **28.17°** is
near the upper endpoint. The same left physical mean expressed around WR2's
model home needs bounded residual **+.210**, versus current **+.891**. At that
mean, tanh sensitivity would be `.956` versus `.206`. This is a hypothetical
coordinate calculation, not a learned new policy. Existing right targets also
extend beyond the proposed model-origin interval, so old weights cannot simply
be reinterpreted as a corrected policy.

### Preliminary physical check

A native MuJoCo check used the saved WR2 **training scene**, including floor and
contact pairs, nominal PD/torque-speed equations and quantized constant targets.
Both targets started from the same current reference reset pose:

| Constant home target | Hold duration | Height/nonfinite failure | Final torso height | Maximum root rotation |
| --- | ---: | --- | ---: | ---: |
| Current reference home | 10 s | None | .2776 m | .86° |
| Configured model home | 10 s | None | .2868 m | 7.06° |

This establishes nominal feasibility for investigating model-home selection.
The candidate reaches a larger root rotation during this check. It is not a
randomized MJX walking test, a new PPO result, or proof that the change will cure
the learned bias.

## 6. Next analysis before a training change

The immediate question is why learning selects the biased left-hip mean despite
a surviving cyclic signal. The model-home difference alone is insufficient to
prioritize a long training run as a demonstrated fix.

Use the existing WR2/TB diagnostics to distinguish phase-averaged mean changes
from changes in the cyclic output. Compare the native full Adam probe and its
reward/value contributions, keeping the fresh-optimizer and fixed-observation
limitations explicit. The prior
[hip-pitch incentive review](wr2_hip_pitch_incentive_review_20261023.md) shows
that larger uncoordinated hip motion can incur tracking and stability costs;
the coordinated target intervention is evidence about feasible joint motion,
not a learned PPO improvement.

Model-home selection remains a candidate TB-aligned comparison. If selected,
verify the native MJX startup and new target bounds, preserve old checkpoint
origins, and use a fresh 200–300M run with retained intermediate checkpoints.
Its causal benefit must be measured rather than assumed from the source difference.

## Sources and retained artifacts

- `review_20261024/raw_phase_signal_comparison.json` in the WR2 v5 diagnostic
  retains the forward sample counts and all twenty raw/bounded phase-bin means.
- TB source: local `paper-walking-baseline`, `a4f693d`;
  `mjx_env.py:236–270, 677–723, 1173–1192`,
  `walk_env.py::_sample_command/_reward_feet_air_time`,
  `reference/walk_zmp_ref.py` and native Brax `1414780` PPO/distribution code.
- WR2: [home override](../../wr2/locomotion/walking_env.py),
  `RobotDescription.home_position_rad`, `paper_rewards.py`, `ppo.py`,
  `ppo_diagnostics.py`, saved configuration/model provenance.
- [ToddlerBot paper, Tables VI–VII](https://arxiv.org/html/2502.00893v2),
  [PPO](https://arxiv.org/abs/1707.06347),
  [GAE](https://arxiv.org/abs/1506.02438).

Under TB's ignored diagnostic `review_20261024` folder:

- `comparison.json`: verification, gait, stride, reward, update and home metrics.
- `home_reference_check.json`: neutral TB model/reference initialization check.
- `tb_wr2_comparison.png`: standalone four-panel comparison.

WR2's ignored `review_20261024/home_origin_nominal_check.json` retains the nominal
hold check. Review data and figures stay in the ignored result folders.
