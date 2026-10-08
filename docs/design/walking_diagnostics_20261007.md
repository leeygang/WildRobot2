# v0.21.0 pre-training diagnostics

This is the matched diagnostic review of
`wr2_ppo_20261007_152924_seed0`, comparing its retained `best_params.pt`
(the migrated starting policy) with `params.pt` (the final policy). The final
policy is not promoted: nominal endpoint evaluation and an independent
randomized check both showed worse speed tracking despite improved heading.

## Reproduce

```bash
uv run --extra training python -m wr2.tools.diagnose_walking \
  --run-dir results/wr2_walking/wr2_ppo_20261007_152924_seed0 \
  --output results/wr2_walking_diagnostics/heading_phase_clipping \
  --seeds 101,202,303,404
```

The command runs on Mac CPU or Linux GPU. It writes one time-series NPZ for
each policy/scenario and `summary.json`. Existing completed runs are not
overwritten; `--summarize-only` recomputes summaries from those traces.

This audit uses deterministic actors, noiseless observations, nominal model
dynamics, and the configured reset pose/servo variation. Seeds are paired
between policies and disturbances. It is a causal diagnostic, not a
population-level deployment qualification or a replacement for randomized
held-out evaluation. Four seeds x five heading offsets x two policies plus
four transition seeds x two policies completed **48 twenty-second episodes**
without falls or non-finite states.

## 1. Heading recovery

At three seconds of 0.10 m/s walking, rotate the complete floating-base robot
by 0, -10, -5, +5 or +10 degrees about the vertical axis. Rotate world-frame
translation velocity too, but retain the body-local free-joint angular velocity,
joint state, reference heading, servo target, IMU memory and observation history.
Refresh only the newest observation frame. Reinitialize physics for the zero
control as well so reinitialization is not confused with a disturbance effect.
This state rotation isolates orientation tracking on flat ground; it is not a
physical torque impulse or a validation of push recovery.

Average absolute heading error during the post-disturbance steady window:

| Added heading | Starting policy | Final policy |
|---|---:|---:|
| 0 degrees | 4.07 degrees | 2.49 degrees |
| -10 degrees | 13.22 degrees | 4.34 degrees |
| -5 degrees | 9.12 degrees | 3.25 degrees |
| +5 degrees | 3.56 degrees | 2.62 degrees |
| +10 degrees | 6.22 degrees | 4.74 degrees |

Natural heading drift can make a one-sided perturbation appear to recover even
without feedback. Therefore compare each seed against its zero-offset control.
During the final three seconds, the residual disturbance-induced heading
difference fell from 8.79 to 1.52 degrees for -10 degrees, 5.26 to 0.53 degrees
for -5 degrees, 4.66 to 0.88 degrees for +5 degrees, and 8.56 to 1.30 degrees
for +10 degrees (mean absolute paired differences).

The final policy learned real heading sensitivity. Absolute recovery is not yet
reliable: only 2/4 negative-offset episodes stayed within 2 degrees for a full
0.72-second gait, and successful -10-degree recoveries took 5.0 and 11.5 seconds.
Even the final zero-offset policy did not maintain the heading tolerance in all
seeds. Meanwhile zero-offset forward-speed MAE worsened from 0.0211 to
0.0373 m/s. Do not remove the heading observation or increase its reward weight
on the assumption that the feature is broken.

## 2. Stance propulsion and slip

Sample foot-floor contact forces with MuJoCo's contact solver, transform into
the intended path frame, and retain their sign **on the foot**. MuJoCo reports
force on geom2; a foot in geom1 requires sign reversal. Ignore unrelated and
inactive contact pairs. Positive forward force is propulsion; negative forward
force is braking. Samples are taken at the last physics substep of each 20 ms
control period, so they are not full-period integrated impulses and cannot
alone establish the cause of a velocity change.

Slip uses contact-point velocity, not foot-origin velocity:

```text
v_contact = v_foot_origin + omega_foot x (contact_point - foot_origin)
```

Only upward-force support points above 1 N contribute to slip. This excludes
swing motion and permits a rolling/pivoting foot origin to move without calling
a stationary contact point a slip.

Zero-offset mean torso-forward speed by gait quarter:

| Quarter | Foot swinging | Starting policy | Final policy |
|---|---|---:|---:|
| 0 | Left, rising | 0.0859 m/s | 0.0654 m/s |
| 1 | Left, descending | 0.1099 m/s | 0.1261 m/s |
| 2 | Right, rising | 0.0760 m/s | 0.0689 m/s |
| 3 | Right, descending | 0.0842 m/s | 0.0581 m/s |

Both established stance feet carry about 17--21 N. There is no evidence that
the left foot cannot support the robot. Established-stance mean contact slip
changed from 6.12 to 5.66 mm/s on the left and 4.01 to 4.53 mm/s on the right;
these means do not exclude brief slip peaks or touchdown transients.

The slower final quarter has stronger sampled left-stance braking
(-0.97 -> -1.35 N mean forward force) and less sampled right touchdown load
(5.05 -> 2.91 N, averaged over that quarter). Quarter 1 has the opposite
touchdown imbalance: left-foot vertical load increases 3.84 -> 7.05 N.
This locates an uneven load-transfer/stride pattern. It does not prove a failed
servo or justify raising torque, increasing friction, or penalizing swing-foot
velocity. Net lateral travel increases about 0.071 -> 0.249 m over the
post-injection path even with the zero heading disturbance.

## 3. Clipping by joint and transition

Use the same 3s stand -> 6s walk -> 3s stand -> 5s walk -> 3s stand schedule
as checkpoint evaluation. Trace the **delayed** request, servo bias, physical
joint bounds, command quantization, slew limit, applied target and actual joint
position. Category membership uses the command applied on that step, not the
following observation's command.

All active-leg clipping is from `left_knee_pitch` at its 0-degree extension
boundary. Its limits remain the correct CAD range, approximately -90 to
0 degrees; no joint-range expansion is justified.

| Category | Starting knee clipping | Final knee clipping |
|---|---:|---:|
| Standing | 27.22% | 86.44% |
| Walking | 0.36% | 0.91% |
| First 0.5s after starting | 4.00% | 10.00% |
| First 0.5s after stopping | 0.00% | 12.50% |

Percentages above are per knee, not averaged over ten joints. Averaged over
all ten leg actions and the full episode, clipping is 1.245% -> 3.940%,
consistent with the training evaluator's 1.19% -> 3.93% over its larger seed
batch. Final standing requests go as far as **+8.91 degrees**, which are clipped
to 0 degrees. The first final-policy event is at 0.32 seconds, requesting
+0.133 degrees with an actual knee angle of -2.06 degrees. The actual joint
does not violate its limit; the commanded target is unreachable.

The final knee clips in all three standing segments, not just during stopping:
88.5%, 85.5% and 85.3% of samples respectively. Walking steady state rarely
clips, so standing clipping cannot by itself explain the steady-walking speed
regression. Inactive shoulder-roll bias also clips at its neutral CAD endpoints;
these are listed separately and excluded from the ten-action saturation metric.

## ToddlerBot/WR1 comparison and next decision

Source comparisons:

- ToddlerBot `toddlerbot/locomotion/mjx_env.py::_reward_torso_quat` and
  `toddlerbot/locomotion/walk.gin` use the same full-orientation kernel/weight.
  Its actor observes noisy quaternion orientation; WR2 observes gravity plus
  relative heading. The disturbance results verify WR2's learned heading path.
- ToddlerBot `mjx_env.py::_solve_contact` sums world-frame foot contact force
  and applies a 1 N support threshold. WR2's diagnostic adds explicit force-on-
  foot signs and point-velocity slip so braking and pivoting are not mislabeled.
- ToddlerBot `mjx_env.py::step` clips `default_action + 0.25 * delayed_action`
  to physical motor bounds. WR2 uses that residual strategy; the standing knee
  problem is a learned unreachable request, not a missing CAD limit.
- ToddlerBot has an extra hip-yaw axis per leg. WR2 cannot assume identical
  heading/propulsion decoupling; the measurements show a real gait trade-off,
  but do not quantify how much this missing axis causes it.
- WR1 `training/eval/eval_policy.py` records state traces and its environment
  separates velocity/yaw tracking from tilt. This audit follows its raw-trace
  review discipline without borrowing WR1's morphology-specific rewards.

Public project reference: [ToddlerBot](https://toddlerbot.github.io/) and its
[paper](https://arxiv.org/abs/2502.00893). The local code above, not presumed
paper settings, is the numerical source of truth.

Keep the starting checkpoint. Retain full orientation, its observable heading,
the physical joint ranges, and the fitted servo model. Before a new training
change, target the **standing knee request** and **uneven stride/load transfer**
as separate problems. A steady-walking clipping penalty or larger heading
weight does not address the measured speed regression. A physical-target
overrun cost would be a WR2-specific change requiring an explicit A/B test;
it should not be introduced as if ToddlerBot already uses it. A deeper
`walk_home` is likewise not supported by these diagnostics alone.

The tests cover force sign, active foot-floor pairing, angular contact velocity,
conditional denominators, terminating-run exclusion, state rotation, observation
history/reference preservation, and requested-versus-applied clipping traces.
The complete suite passes 126 tests and 12 subtests; focused diagnostics have
five regression tests and pass lint checks.
