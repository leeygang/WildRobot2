# WildRobot2 architecture

WildRobot2 follows ToddlerBot's principal separation between robot description,
simulation/hardware backends, policies, motion references, and training.

## Source-of-truth ownership

- MJCF owns geometry, mass, inertia, joint axes and limits, collision geometry,
  sites, and mechanical constraints.
- `descriptions/default.yml` owns defaults shared by future WR2 variants.
- `descriptions/wr2/robot.yml` owns versioned WR2 hardware choices and robot
  constants.
- `descriptions/wr2/motors.yml` owns calibration for one physical build and is
  not committed.
- Training configuration owns rewards, observations, randomization, and timing.
- A deployable policy bundle must identify its exact model and configuration.

## Initial implementation order

1. Add and validate one canonical WR2 MJCF model.
2. Implement the shared observation and motor-target contract.
3. Implement MuJoCo and real-hardware backends against that contract.
4. Add calibration and actuator system-identification tools.
5. Add a standing policy and regression tests.
6. Add motion references, MJX training, and learned locomotion.

Task-specific manipulation, depth, and skill-classification packages remain out
of scope until the core robot loop is verified.

The versioned action, observation, and IMU-frame contract is documented in
[`training_interface.md`](training_interface.md).

## ToddlerBot comparison baseline

The local `toddlerbot_2xm` model used during initial planning has a summed MJCF
body mass of 3.768872 kg, configured COM height of 0.2874 m, hip-to-knee length
of 0.1015 m, knee-to-ankle length of 0.1100 m, and a 20 ms policy period.

Before reusing gains, speeds, or motion timing, record the corresponding WR2
measurements and compare dimensionless quantities:

```text
length ratio = L_WR2 / L_TB
mass ratio   = M_WR2 / M_TB
torque scale = (M_WR2 * L_WR2) / (M_TB * L_TB)
time scale   = sqrt(L_WR2 / L_TB)
```
