# WR2 training and runtime interface

The policy must see the same semantic signals in MJX and on the physical robot.
Backend-specific raw values are converted into this contract before policy
preprocessing.

## Action contract

The action vector has 17 values in `actuator_order.txt` order. Each value is
clipped to `[-1, 1]`, scaled by 0.25 rad, added to the configured home position,
and clipped to the MJCF joint limits. Training initially controls the ten leg
joints; waist and arm action channels are retained for WR1-compatible ordering
but masked to zero. A single global joint-limit margin remains zero because
some upper-body neutral positions are CAD range endpoints.

This is a position-residual interface:

```text
joint_target = clip(walk_home_position + 0.25 * policy_action, joint_limits)
```

## Actor observation v1

`wr2_proprio_v1` contains 60 float32 values:

| Field | Size | Notes |
|---|---:|---|
| command velocity | 3 | torso-frame vx, vy, yaw rate |
| joint position error | 17 | measured position minus home |
| joint velocity | 17 | scaled by 0.05 |
| previous action | 17 | normalized policy action |
| torso angular velocity | 3 | torso frame |
| projected gravity | 3 | unit gravity vector in torso frame |

Foot contacts, actuator force, base linear velocity, and exact simulator state
may be used for rewards and the privileged critic, but are not actor inputs.
The deployed robot cannot measure them equivalently without extra hardware or
state estimation.

## IMU sim-to-real bridge

The BNO085 is mounted with a fixed sensor-to-torso transform `q_TS` stored in
`robot.yml`. If its hardware adapter produces `q_WS` (sensor frame to world),
the canonical torso orientation is:

```text
q_WT = q_WS * inverse(q_TS)
omega_T = rotate(q_TS, omega_S)
```

MJX starts from exact torso orientation and angular velocity. Before these
signals reach the policy, training must randomize the effects seen in hardware:

- residual mounting/calibration rotation;
- gyro white noise, scale error, fixed bias, and bias random walk;
- orientation/filter error;
- sample-and-hold delay, timestamp jitter, and occasional stale samples; and
- policy/control latency.

The actor uses projected gravity instead of magnetometer heading. This avoids
dependence on indoor magnetic-field disturbances and quaternion yaw drift.

Noise ranges must be fitted from WR2 logs rather than copied blindly from TB.
Collect at least stationary logs in several orientations and slow/fast manual
motion logs, then compare the BNO085 stream with MuJoCo-replayed kinematics.

## Initial walking environment

`wr2.locomotion.walking_env.WR2WalkingEnv` now provides the first flat-ground
MJX environment. It starts from `walk_home`, runs at 50 Hz over a 500 Hz
physics model, applies one control-step action delay, exposes only the 60-value
deployable observation, and uses simulator-only state for rewards and
termination. Its first curriculum is forward standing/walking with leg-only
control; lateral/yaw fields remain in the interface but their initial command
ranges are zero. The environment currently adds provisional joint, gyro, and
projected-gravity white noise. Episode bias, mounting-error, and IMU-delay
models remain gated on WR2 sensor measurements.

Actuator targets are quantized to HTD-45H command units (0.24 degrees) and
slew-limited to 27 units per 20 ms control step, or 5.655 rad/s. This is below
the vendor's 5.818 rad/s no-load speed at 11.1 V. The MJCF force cap remains
4.0 N m, below the 4.413 N m vendor stall value, and is randomized downward;
neither value is treated as a continuous-torque rating.

Run the JIT environment gate with:

```bash
uv run --extra training python -m wr2.locomotion.train --smoke
```

Start PPO with:

```bash
uv run --extra training python -m wr2.locomotion.train
```

Before hardware deployment, complete the remaining gates:

1. characterize WR2 servos under representative leg loads and fit the
   randomization distributions;
2. characterize the installed BNO085 noise, mounting error, and latency;
3. add policy export plus a runtime adapter that reproduces the v1 action and
   observation contract;
4. train standing and forward walking, then evaluate multiple seeds with
   perturbations and held-out dynamics; and
5. run restrained hardware standing before any untethered walking test.
