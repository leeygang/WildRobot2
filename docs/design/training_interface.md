# WR2 training and runtime interface

The policy must see the same semantic signals in MJX and on the physical robot.
Backend-specific raw values are converted into this contract before policy
preprocessing.

## Action contract

The action vector has 17 values in `actuator_order.txt` order. Each value is
clipped to `[-1, 1]`, scaled by 0.25 rad, added to the configured home position,
and clipped to the MJCF joint limits. The initial limit margin is zero because
the straight-leg home places both knees on a CAD range endpoint; this should be
revisited with the future bent-knee locomotion home.

This is a position-residual interface:

```text
joint_target = clip(home_position + 0.25 * policy_action, joint_limits)
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

## Next implementation steps

1. Implement a MuJoCo backend that emits `RobotObservation`.
2. Implement a BNO085 adapter that emits the same torso-frame fields.
3. Record synchronized servo and IMU logs and fit randomization parameters.
4. Generate the torque-actuated MJX model.
5. Build a standing environment using the v1 action and observation contract.
