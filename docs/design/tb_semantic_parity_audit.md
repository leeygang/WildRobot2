# ToddlerBot walking semantic parity audit

Audit date: 2026-10-01

Reference implementations:

- ToddlerBot `main` at `f81679b`, especially `walk.gin`, `walk_env.py`,
  `mjx_env.py`, `mjx_config.py`, `ppo_config.py`, and `train_mjx.py`.
- WR2 before this audit at `a05d262`, especially `ppo_walking.yaml`,
  `walking_env.py`, `ppo.py`, and `train.py`.

This audit compares equations and state transitions, not only similarly named
configuration fields. A difference is classified as aligned, a WR2 adaptation,
an intentional curriculum difference, a corrected defect, or an open measured
calibration gap.

## Audit result

Four accidental semantic mismatches were corrected in WR2 v0.9.2:

1. Action rate now uses ToddlerBot's dimensionless
   `sum((action_t - action_t-1)^2)`. The previous conversion through each
   joint's physical half-range made the same normalized action change cost
   roughly 6--21 times more, depending on the joint.
2. The asymmetric critic now scales torso-frame linear velocity by `2.0` and
   actuator force by `0.1`, matching ToddlerBot. Observation normalization is
   disabled, so these scales are behaviorally significant.
3. Foot support contact now requires upward world-frame force above 1 N rather
   than total force magnitude above 1 N. This prevents tangential force from
   being counted as vertical support.
4. Foot lateral separation now uses the torso yaw frame, not the full tilted
   torso frame. This matches ToddlerBot and prevents swing height or torso roll
   from contaminating foot-width measurements.

The regression suite evaluates the corrected equations directly. The
autonomous loop fingerprints the action-rate, support-contact, foot-width,
observation, and action-to-physics implementations so an experiment cannot
silently redefine them.

## Physics and control timing

| Contract | ToddlerBot | WR2 | Classification |
|---|---|---|---|
| Policy period | 20 ms | 20 ms | Aligned |
| Physics integration | 5 ms x 4 | 2 ms x 10 | WR2 adaptation for its stiff servo/contact model |
| Target delay | one policy step | one policy step | Aligned |
| Episode horizon | 1,000 policy steps | 1,000 policy steps | Aligned |
| Reward integration | weighted reward rate x 0.02 s | weighted reward rate x 0.02 s | Aligned |
| Flat-ground acquisition | flat tile | flat scene | Aligned |

ToddlerBot stores its delayed action in `last_act` after observation and reward
calculation, so the active source can expose a command older than the queued
command. WR2 intentionally exposes the most recently issued command, which is
the state needed to make its one-step-delay process Markov. The action-rate
cost is therefore defined against consecutive issued policy actions, matching
the stated ToddlerBot reward intent without reproducing that bookkeeping lag.

## Action and actuator path

| Contract | ToddlerBot | WR2 | Classification |
|---|---|---|---|
| Controlled legs | 12, including hip yaw | 10; WR2 has no hip-yaw DOF | Morphology adaptation |
| Policy action | unbounded Normal residual | bounded tanh-Normal absolute normalized position | WR2 hardware adaptation |
| Position mapping | `default + 0.25 * action` | safe midpoint + safe half-range x action | Deliberate WR2 full-range contract |
| Initial mean | residual zero/home | exact normalized `walk_home` | Equivalent physical initialization |
| Initial standard deviation | 0.5 normalized residual | 0.13 bounded action | WR2 physical-exploration adaptation |
| Joint-limit handling | clip motor target | distribution bound plus safe target range | WR2 avoids clipped-action aliasing |
| Servo command | floating target | 0.24-degree quantization and measured slew cap | WR2 hardware requirement |
| Controller | ToddlerBot motor envelope | WR2 fitted PD, torque-speed, and braking envelope | Robot-specific calibration |
| Action-rate input | normalized policy commands | normalized policy commands | Corrected/aligned |

Absolute WR2 actions are not expected to numerically match ToddlerBot residual
actions. Rate regularization is nevertheless applied in policy coordinates,
where the PPO probability model operates. Quantization, slew, torque, and
speed constraints remain in the action-to-physics path and in diagnostics.

## Actor and asymmetric critic observations

Both policies place the newest frame first and retain 15 frames. Each actor
frame contains phase sine/cosine, commanded velocity, all motor positions and
velocities, the controlled previous action, local angular velocity, and an
orientation signal.

| Feature | ToddlerBot | WR2 | Classification |
|---|---|---|---|
| Position scale | 1.0 | 1.0 | Aligned |
| Velocity scale | 0.05 | 0.05 | Aligned |
| Angular-velocity scale | 1.0 | 1.0 | Aligned |
| Orientation | full quaternion | projected gravity | Intentional yaw-invariant deployment contract |
| Actor linear velocity | absent | absent | Aligned |
| Critic position error | motor minus ZMP reference | motor minus WR2 ZMP reference | Aligned architecture; morphology-specific trajectory |
| Critic linear velocity | local, x2.0 | local, x2.0 | Corrected/aligned |
| Critic actuator force | x0.1 | x0.1 | Corrected/aligned |
| Critic contact/reference contact | two plus two | two plus two | Aligned analytic meaning |

Projected gravity intentionally avoids magnetometer heading and quaternion yaw
drift. The actor still receives local yaw rate and the commanded yaw rate.
WR2's current acquisition command has zero lateral and yaw components.

## Commands, phase, and foot guidance

The 0.72 s gait period, sine/cosine phase representation, alternating half-cycle
swing assignment, cubic smoothstep foot-height profile, 0.04 m swing height,
and 0.0007 square-metre foot-height kernel are aligned.

ToddlerBot samples forward/backward, lateral, turning, and standing commands.
WR2 gait acquisition intentionally samples 0.08--0.18 m/s forward commands and
10% standing commands; robust training expands the forward range and uses 20%
standing commands. Both resample every three seconds. WR2 starts a new gait at
phase zero when leaving stand, which is reproducible on hardware; ToddlerBot's
episode clock continues through command changes. This is an explicit WR2
deployment choice, not numerical parity.

ToddlerBot foot height is measured from a site whose nominal world height is
zero. WR2 subtracts each foot site's `walk_home` world height. These are the
same ground-relative quantity for their respective models.

## Active reward equations

All terms below are integrated over the 20 ms control period.

| Term | ToddlerBot active equation | WR2 equation | Status |
|---|---|---|---|
| XY velocity | `2 exp(-1000 ||v-c||^2)` | `2 exp(-||v-c||^2 / 0.15^2)` | Intentional acquisition kernel; final score remains TB-strict |
| Yaw rate | `1.5 exp(-4 (w-c)^2)` | same | Aligned |
| Roll/pitch rate | `-sum(w_xy^2)` | same | Aligned |
| Torso orientation | `2.5 exp(-20 angle_to_ref^2)` | `2.5 exp(-20 tilt^2)` | Yaw-invariant WR2 adaptation |
| Alive | `1.0` | `1.0` | Aligned |
| Action rate | `-2 sum((a_t-a_t-1)^2)` | same | Corrected/aligned |
| Weighted pose | `-0.5 sum(weight * q_error^2)` | same, mapped to ten WR2 leg DOFs | Aligned by morphology |
| Close feet | `-10` below 0.06 m yaw-frame width | same | Corrected/aligned |
| Foot phase | `7.5` times the same phase/height kernel | same | Aligned |
| Foot tilt | `-5` times summed sine tilt | same relative to WR2 home sole axes | Model-frame adaptation |
| Torque/power | disabled in active TB config | small WR2 torque-squared and power costs | WR2 servo safety requirement |

The wider 0.15 m/s acquisition velocity kernel is the largest intentional
reward divergence. It gives a cold policy gradient while standing. Success
scoring still uses ToddlerBot's strict `exp(-1000 error^2)` tolerance. It should
be tightened only as a separately measured experiment, not combined with this
semantic correction.

## ZMP-reference boundary

ToddlerBot's active reward set disables motor-position imitation, and active
leg targets are policy residuals around the static first-frame pose. It is
therefore direct PPO rather than ZMP-policy pretraining. However, the source
still constructs `WalkZMPReference`: phase and stance come from it, torso and
velocity rewards use its path reference, and the privileged critic receives
motor error relative to its ZMP joint trajectory.

The v0.9.2 comparison run remained stable but converged to a one-sided local
optimum: the left foot stayed in support while only the right foot swung, and
forward speed remained near zero. WR2 v0.10 therefore closes this evidenced
structural gap. A periodic Linear Inverted Pendulum Model sets lateral CoM
motion whose implied ZMP alternates at the WR2 foot centers. Forward footsteps
and swing height are conditioned on command and phase, and damped numerical IK
generates a lookup for WR2's five-DOF legs. The critic's 17-value error slot is
now measured against that trajectory, matching ToddlerBot's learning
architecture without exposing simulator-only reference positions to the actor
or enabling a joint-imitation reward.

This is architecture parity, not reuse of ToddlerBot's numerical table. Its
six-DOF leg IK and dimensions are incompatible with WR2. Every generated WR2
lookup is checked against a configured maximum foot-position residual; the
active 0.08--0.18 m/s table has a 2.1 mm worst residual under a 6 mm gate.

## Reset, termination, and contact

WR2 resets delayed action, target slew state, command, gait phase, observation
history, IMU state, torque exposure, and its episode counter with the physical
state. The multi-episode wrapper regression protects this lifecycle.
ToddlerBot terminates primarily by torso height. WR2 also terminates on severe
tilt and non-finite state; these are intentional safety checks. WR2 reset joint
and velocity perturbations are small acquisition randomization not present in
ToddlerBot's same form.

Both use a 1 N foot-support threshold. WR2 now applies it to upward world force
and explicitly requires the named foot geom/floor pair. Contact phase is a P0
metric and privileged critic feature, not an active dense reward.

## PPO and optimization

The policy/value widths (512/256/128), ELU activation, 20-step unroll, four
updates, learning rate `3e-5`, entropy cost `5e-4`, discount `0.97`, GAE
`0.95`, PPO clip `0.2`, gradient norm `1.0`, disabled observation
normalization, and normalized advantages are aligned.

WR2 uses 2,048 environments with batch 256 x 8 minibatches to fit its target
GPU; ToddlerBot's checked-in defaults use a larger environment/batch layout.
WR2 deterministic evaluations and checkpoint gates are operational additions.
The tanh-Normal distribution and home-centered output head are intentional
consequences of WR2's bounded absolute action contract.

## Randomization and remaining measured gaps

Nominal gait acquisition keeps dynamics randomization disabled. The later WR2
robust stage covers the same classes as ToddlerBot plus explicit target bias,
speed, braking, and WR2 servo uncertainty. Its numerical ranges are not claimed
to be ToddlerBot parity because the robots and actuators differ.

The remaining unresolved values require WR2 hardware measurements:

- bidirectional torque versus speed and braking torque;
- backlash and hysteresis under installed loads;
- continuous thermal envelope beyond the bounded tests;
- BNO085 mounting transform residual, noise, drift, and latency; and
- ground friction and installed mass/COM uncertainty.

## Required review method for future changes

Any action, observation, reward, reset, or command-contract change must record:

1. the exact equation on both robots;
2. input units, coordinate frame, and normalization;
3. sum versus mean reduction and control-period integration;
4. whether the value is sampled, delayed, clipped, quantized, or filtered;
5. actor versus privileged-critic visibility;
6. reset and command-transition timing; and
7. a synthetic regression with a known numeric result.

Matching a field name or weight is not sufficient evidence of semantic parity.
