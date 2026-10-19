# Why WR2 keeps its right leg inward — 2026-10-19

## Conclusion

**The current policy has learned to depend on its inward right-leg stance for
lateral balance.** Returning only that hip to the reference, or requesting a
small outward angle, causes a sideways fall. Changing only the left hip does
not. These right-only failures also occur during standing, with zero air-time
reward. During forward motion, air time additionally makes the inward stance
more valuable to the current policy.

The spacing reward supplies no additional benefit for moving outward from the
observed stance, and the achieved right-hip reference error costs very little.
Together, these measurements explain why the existing controller keeps this
stance. They do not reconstruct the training event that originally selected
the right side, or establish that a retrained policy must use it.

The [paired GPU review](wr2_tb_stance_result_20261018.md) remains the full TB
comparison. This review adds right-only controls and all native reward terms.

## 1. The inward request comes from the policy

Copied nominal GPU forward `.10 m/s` commands, in TB-equivalent units:

| Right hip measurement | WR2 | TB Paper 200M |
| --- | ---: | ---: |
| Policy's requested angle | −13.87° | −5.19° |
| Delivered target | −9.98° | −5.19° |
| Achieved angle | −5.67° | −4.37° |
| Mean walking reference | −.005° | +.093° |
| Physical direction of the mean delivered target | **Inward** | **Outward** |
| Joint-target clipping | 99.88% of ticks | 0% |
| Model foot self-contact | 95.87% of ticks | 0% |

**The right hip has opposite physical sign conventions in the two robots.**
At each native home pose, increasing WR2's right hip by `1e−5 rad` moves its
right foot farther toward negative world Y: `dy/dq = −.210491 m/rad`.
The same perturbation in TB gives `+.221829 m/rad`, toward the centerline.
Their left hips both move inward with increasing angle. Comparing the raw
negative right-hip numbers without this check would obscure the stance delta.

WR2 maps actions through fixed `q_home + .25 * action`, then the native delay,
quantization, limits and slew limiter. The right hip's inward joint limit is
−10°. The policy continually requests beyond that limit in every copied
command, including standing. Its walking reference stays approximately
neutral; its reference reward pulls the achieved right hip toward neutral.
The reference does not request the measured large inward deviation.

The new nominal CPU forward baseline applies mean right-hip actuator torque
**−1.815 Nm**, with mean absolute mechanical power **.096 W**. It continually
requests inward motion while the achieved joint remains about **4.32° RMS**
away from the delivered target. This supports the stance/execution diagnosis;
it does not establish a uniform actuator deficiency.

Sources: [`_policy_target`](../../wr2/locomotion/walking_env.py),
[`target_trace`](../../wr2/tools/diagnose_walking.py), the two compiled MJCF
models, and the copied GPU `actual_rad`, `actor_request_rad`, `delivered_rad`
and `reference_rad` arrays.

## 2. Right-only interventions isolate the balance dependence

The selected WR2 **260,116,480-step** policy and identical nominal reset/key
are reused. At every tick the policy still evaluates the changed state and
controls all unintervened joints. Only the specified hip action coordinates
are replaced. Native one-tick delay, limits, quantization, slew and physical
controller remain active. An outward/inward `3°` condition means the current
right-hip reference plus/minus `3°`, mapped through the regular action path.

Forward command `.10 m/s`; durations and speeds are TB-equivalent:

| Control | Observed duration | Outcome | XY error, m/s | Model foot contact |
| --- | ---: | --- | ---: | ---: |
| Learned policy | 20.00 s | Survives | .09910 | 95.75% |
| Right hip → reference | **2.50 s** | **Sideways fall** | .25011 | 37.86% |
| Right hip → 3° outward | **1.36 s** | **Sideways fall** | .39175 | 46.43% |
| Right hip → 3° inward | 20.00 s | Survives | .10570 | 49.39% |
| Left hip → reference | 20.00 s | Survives | .09851 | 85.07% |
| Both hips → reference | 20.00 s | Survives | .10784 | 0% |

The two right-only falling cases move rapidly toward the left side, with
terminal torso roll **−60.2° /−63.6°**. Their net cross velocities are
`+.165 /+.296 m/s`. Their errors and contact fractions cover their short
lifetimes, including the fall; they are not 20-second walking scores.

Standing provides the important air-time control:

| Standing control | Observed TB duration | Outcome | Air-time return |
| --- | ---: | --- | ---: |
| Learned policy | 20.00 s | Survives | 0 |
| Right hip → reference | **3.23 s** | **Sideways fall** | 0 |
| Right hip → 3° outward | **1.33 s** | **Sideways fall** | 0 |

This establishes that changing the right hip independently disrupts the
learned coordination even without air-time credit. The surviving left-only
control instead drives the achieved right hip farther inward, averaging
**−7.99°**. The surviving both-hip control shows that simultaneous stance
changes can remain upright, although it still tracks worse than the learned
policy and drifts off the commanded line.

These are forced controls of a frozen policy from a nominal reset. They
establish dependence of this controller on the stance; they do not measure
what a newly trained coordinated outward gait could achieve. They also do
not isolate foot collision forces from the hip-position change itself.

## 3. Which rewards maintain the stance

The new CPU replay records **all 25 native weighted, dt-scaled components**.
For the learned forward baseline over the full horizon:

| Component | Native episode contribution |
| --- | ---: |
| Air time | **+160.80** |
| Yaw-rate tracking | +81.10 |
| XY velocity tracking | +32.06 |
| Roll/pitch angular-rate tracking | +26.40 |
| Reference foot contacts | +27.64 |
| Feet spacing | **+16.48, the maximum** |
| Torso orientation | +13.44 |
| All leg-reference imitation | −1.791 |
| **Right hip's share of imitation** | **−.0677** |
| Foot collision | **−3.156** |
| Total, including every other term | **384.943** |

For the full-duration milder inward control, achieved right hip averages
−2.11° and foot contact drops to 49.39%. Its total return falls **71.302**:

- Air-time loss: **−41.00**.
- Orientation, roll/pitch-rate and yaw-rate losses combined: **−25.489**.
- Reference foot-contact loss: **−4.48**; XY reward loss: **−1.843**.
- Collision improvement: **+1.528**.
- Feet-spacing change: **0**, because both stances remain in its maximum band.
- Right-hip imitation improves about **+.058**; the other joints' errors
  offset that gain, leaving total leg imitation slightly worse.

The corresponding both-hip control recovers **+3.156** collision return but
loses **−36.40** air time and **−33.887** from orientation and angular stability.
Total return decreases **77.080**. Removing foot contact therefore supplies a
small gain compared with what the current coordination gives up.

For falling controls, the audit also compares **equal-length prefixes**.
Standing right-to-reference loses **15.213** relative to the learned policy
over the same first 133 ticks: **−7.322** from zero-velocity tracking and
**−5.804** from roll/pitch-rate and yaw-rate rewards, while collision improves
**+.244**. Air-time change is exactly zero. Thus the dependence is measurable
in the ordinary balance/velocity objectives as well as the moving air-time
objective; it is not inferred from shorter episode totals.

The copied GPU baseline independently gives **161.80** air-time credit, of
which **158.20** occurs on model foot-contact ticks. This arithmetic measures
floor-unloading credit during contact; it does not establish that all credited
foot motion is collision-supported or useful forward walking.

### Why the spacing term supplies no outward incentive here

Both WR2 and frozen TB measure the lateral distance between **foot-body
origins** in the torso frame. Their range reward is flat at 1 inside its
allowed band. It does not directly measure the gap between the foot surfaces.

WR2's forward body-origin separation is **61.87–81.14 mm**, averaging
**62.44 mm**, inside its **55.73–144.89 mm** band on every copied tick. Thus it
gets full spacing credit while the boxes touch.

The model proportions differ: WR2's home projected foot-box widths are
**59.38 /59.08 mm**, while TB's are **42 mm**. TB's minimum body spacing is
**50 mm**. WR2 scales that minimum by its measured stance-width ratio
**1.11456**, which does not preserve TB's spacing relative to its wider foot
colliders. These body-origin bounds are not universal surface-clearance
guarantees, but the widths and recorded reward demonstrate the lost incentive.

The reference-imitation term measures achieved angles. Its `−5/12 * error²`
contribution makes a sustained right-hip deviation near 6° quite inexpensive.
The Paper recipe also has no direct cost on a constant clipped hip request;
action-rate terms measure changes, and torque/energy terms measure physical
effort. This can make recovery from a saturating stance difficult, but PPO
does not differentiate through the physical clipping operation: these traces
do not establish a zero policy gradient or prove the training optimum.

Sources: [`paper_reward_terms`](../../wr2/locomotion/paper_rewards.py),
[`_paper_step`](../../wr2/locomotion/walking_env.py), and frozen TB
`walk_env.py::_reward_feet_distance/_reward_feet_air_time`,
`mjx_env.py::_reward_leg_motor_pos/_reward_collision`, and `paper.gin`.
TB retains the same weights: air time 500, leg imitation 5, spacing 1 and
collision .1. Reference: [ToddlerBot, Tables VI–VII](https://arxiv.org/html/2502.00893v2).

## 4. What is established about geometry and right-side asymmetry

The user's visual evaluation correctly identifies the inward tendency and
model foot contact. The existing GPU CAD audit covered only the two CAD parts
attached to each foot body. The new CPU traces also cache all leg-mesh frames
at the final physics substep and check **26 right-leg/left-foot CAD pairs**.

In the learned forward and standing CPU replays, all those convex hulls remain
separated: minimum non-foot right-leg/left-foot gaps are **13.26 /13.88 mm**.
Both foot-body CAD parts also remain separated, with minimum gaps
**1.81 /1.81 mm**, while the model boxes frequently contact. Independent vertex
projection separators confirm the closest distances. These findings apply to
the checked traces; a different visual episode requires its own frame audit.

The measured dependence therefore includes a contact approximation outside
the imported foot CAD. Its exact force/load-transfer contribution remains to
be isolated. [MuJoCo mesh collisions use convex hulls](https://mujoco.readthedocs.io/en/stable/computation/index.html#collision-detection);
positive hull separation also certifies separation of the enclosed CAD surface.

There is no explicit right-only inward bonus in either reward implementation.
WR2's modeled left/right leg subtree masses are identical, **.57921429 kg**,
and both use the same controller coefficients. Both recipes start their
walking reference with left support/right swing after initial double support.
The recorded WR2 recipe has no mirror loss. Those facts leave possible seed,
initial-phase and small geometric-asymmetry effects unresolved; they are not
evidence for selecting one as the cause of handedness.

## 5. Focused next step

1. **Validate the foot contact approximation against CAD**, including floor
   support and foot self-contact. Compare the same frozen checkpoint before
   and after the correction, with both hips' achieved angles, delivered
   targets, contacts, balance and signed displacement recorded. This tests
   how much the current inward coordination depends on the existing box
   constraint. Preserve a physically valid self-collision model.
2. **Confirm the new right-only results on the GPU** as part of that physics
   comparison. The CPU baseline reproduces copied GPU XY error within
   **7.60e−6 m/s** and contact fraction within one of 824 ticks, but the new
   interventions have not yet been measured in the training runtime.
3. **Train coordinated movement with the validated physics using the TB
   Paper recipe and a fresh 200–300M budget**, retaining approximately 20M
   checkpoints. Keep air-time 500 and leg imitation 5 for the initial physics
   comparison. Evaluate commanded net translation, XY tracking, both hips'
   clipping/execution, floor support and CAD/model clearance. These define
   progress toward the successful TB gait more accurately than total return.

If valid contact physics still produces the inward, stationary stance, the
next focused recipe check is **foot-spacing normalization against the actual
WR2 foot geometry**, with TB's formula as the starting point. The tests so far
support coordinated relearning; prescribing a right-only outward action fails
in the current controller, and tighter imitation is not supported by the
successful TB policy's larger overall reference deviation.

## Reproducibility and artifacts

The new probe uses seed **22102**, deterministic inference, no pushes,
randomization or observation noise, and the unchanged selected checkpoint
SHA256 `3d0fef64a99924474cd46ff5fac2c8e19fad7a115bf64390ba11c2871f51bf52`.
Control dt is `.02 s`, speed normalization `s=.8240547565`, and horizon 824
ticks = 16.48 physical seconds = 19.99867 TB seconds. Native finite terminal
ticks are included; following padding is invalid, with no autoreset samples.

Local CPU runtime is JAX/JAXlib .6.2, Brax .14.2, MuJoCo/MJX 3.14.0,
Flax .11.2 and NumPy 2.4.6, with high matmul precision. The copied GPU runtime
uses Brax .14.1, Flax .10.7 and NumPy 2.2.6. The model/controller/reward source
matches GPU diagnostic commit `ac91a6a`; current source is `baf032a`.
Frozen TB source is independently matched to `paper-walking-baseline`,
`a4f693d7cc4544f011dad23d6c353e752a6819ab`, in the local ToddlerBot repository.

All nine probes pass independent finite/termination, action-intervention,
reference-timing, native air-time/contact/collision/XY/imitation and component
sum checks. Maximum full-reward sum discrepancy is **9.54e−7 per tick**.

Generated local artifacts:

- [Full probe summary and all reward terms](../../results/wr2_walking/wr2_paper_20261018_121025_seed3/review_20261019/right_hip_incentives/summary.json).
- [Independent audit, joint signs and CAD separation](../../results/wr2_walking/wr2_paper_20261018_121025_seed3/review_20261019/right_hip_incentives/audit.json).
- [Probe source](../../results/wr2_walking/wr2_paper_20261018_121025_seed3/review_20261019/right_hip_incentives/probe.py)
  and [audit source](../../results/wr2_walking/wr2_paper_20261018_121025_seed3/review_20261019/right_hip_incentives/audit.py).
- [Comparison figure](../../results/wr2_walking/wr2_paper_20261018_121025_seed3/review_20261019/right_hip_incentives/right_hip_incentives.png)
  and [exportable PDF](../../results/wr2_walking/wr2_paper_20261018_121025_seed3/review_20261019/right_hip_incentives/right_hip_incentives.pdf).

The generated traces, analysis scripts and figures remain under ignored
`results/`; the review is the versioned record of this investigation.
