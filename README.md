# WildRobot2

WildRobot2 is a clean, ToddlerBot-aligned software stack for the second
WildRobot hardware generation.

The repository currently contains only the initial architecture scaffold. Robot
models, hardware drivers, policies, and training environments will be added only
after their interfaces and measurements are defined.

## Architecture

```text
robot description + configuration
              |
              v
            Robot
              |
       +------+------+
       |             |
   MuJoCoSim      RealWorld
       +------ observation/command API ------+
                                                |
                                                v
                                             Policy

description + motion reference -> MJX environment -> training -> policy
```

## Repository layout

- `wildrobot2/descriptions`: MJCF assets and robot configuration.
- `wildrobot2/sim`: common simulation/hardware interface and MuJoCo backend.
- `wildrobot2/actuation`: physical actuator drivers.
- `wildrobot2/sensing`: physical sensor drivers.
- `wildrobot2/reference`: nominal and recorded motion references.
- `wildrobot2/locomotion`: MJX environments and training entry points.
- `wildrobot2/policies`: deployable policies and the policy runner.
- `wildrobot2/tools`: calibration, SysID, and sim-to-real utilities.
- `motion`: version-controlled motion data.
- `examples`: small component and workflow examples.
- `tests`: model, transmission, and interface regression tests.
- `results`: generated runs and checkpoints; ignored by Git.

See [`docs/design/architecture.md`](docs/design/architecture.md) for ownership
rules and the implementation sequence.

## Development

```bash
python -m unittest discover -s tests
```

## References

- [ToddlerBot project](https://toddlerbot.github.io/)
- [ToddlerBot paper](https://arxiv.org/abs/2502.00893)
- [ToddlerBot source](https://github.com/hshi74/toddlerbot)
