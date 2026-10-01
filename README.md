# WildRobot2

WildRobot2 is a clean, ToddlerBot-aligned software stack for the second
WildRobot hardware generation.

The repository contains the canonical WR2 MuJoCo model, a versioned
action/observation contract, and the first flat-ground Brax/MJX walking
environment. A bench-only HTD-45H qualification driver is included; full
robot hardware drivers and policy deployment remain future phases.

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

- `wr2/descriptions`: MJCF assets and robot configuration.
- `wr2/sim`: common simulation/hardware interface.
- `wr2/actuation`: physical actuator drivers.
- `wr2/sensing`: physical sensor adapters and frame conversion.
- `wr2/reference`: nominal and recorded motion references.
- `wr2/locomotion`: MJX environment and PPO training entry point.
- `wr2/policies`: deployable policies and the policy runner.
- `wr2/tools`: model post-processing, calibration, and sim-to-real utilities.
- `motion`: version-controlled motion data.
- `tests`: model, transmission, and interface regression tests.
- `results`: generated runs and checkpoints; ignored by Git.

See [`docs/design/architecture.md`](docs/design/architecture.md) for ownership
rules and the implementation sequence.

## Development

```bash
uv sync --extra training --extra dev
uv run python -m unittest discover -s tests
uv run python -m wr2.tools.post_process
uv run python -m wr2.locomotion.train --smoke
```

On Linux, the training extra installs CUDA 12 JAX. Confirm that JAX sees the
GPU before starting a long run:

```bash
uv run --extra training python -c 'import jax; print(jax.devices())'
```

See [`docs/design/training_interface.md`](docs/design/training_interface.md)
for the 15-frame, 825-value phase-aware actor observation, ten-value leg action
contract, and remaining hardware gates. The tracked training and hardware work
is in [`docs/TODO.md`](docs/TODO.md). See
[`docs/design/servo_model.md`](docs/design/servo_model.md) for the evidence and
limitations behind the initial HTD-45H model.
Training values are defined in
[`wr2/locomotion/configs/ppo_walking.yaml`](wr2/locomotion/configs/ppo_walking.yaml);
the trainer accepts explicit CLI overrides and snapshots the effective YAML in
every run directory.
The bounded train/evaluate/promote campaign and Mac-to-GPU command are described
in [`docs/design/walking_training_agent.md`](docs/design/walking_training_agent.md).
That guide also covers the autonomous Mac supervisor, which analyzes each GPU
cycle, asks Codex for one high-confidence ToddlerBot-aligned improvement,
validates and pushes its commit, and then fast-forwards the GPU before
continuing.
The staged high-load qualification procedure is in
[`docs/hardware/htd45h_deployment_test.md`](docs/hardware/htd45h_deployment_test.md).

## Copy results from a remote machine

The transfer script copies artifacts into the matching local `results/`
directory. JSON/NPZ servo captures with the same stem are treated as one run.

```bash
./scripts/scp_from_remote.sh --latest servo_sysid
./scripts/scp_from_remote.sh --latest servo_sysid 3
./scripts/scp_from_remote.sh --latest training 2
./scripts/scp_from_remote.sh --list servo_sysid

# GPU/training host through its public address
./scripts/scp_from_remote.sh --linux-pc --latest training

# Deployment host
./scripts/scp_from_remote.sh --wrdev --latest servo_sysid
```

It defaults to `leeygang@linux-pc.local:/home/leeygang/projects/wildrobot2`.
`--linux-pc` selects `LINUX_PUBLIC_IP` and optional `LINUX_PUBLIC_PORT`;
`--wrdev` selects
`leeygang@wrdev.local:/home/leeygang/projects/WildRobot2`. Use `--host`,
`--user`, `--port`, or `--remote-base` for other overrides. Any directory
directly below remote `results/` can be used as a result group; run with
`--dry-run` to inspect the selected transfers first.

## References

- [ToddlerBot project](https://toddlerbot.github.io/)
- [ToddlerBot paper](https://arxiv.org/abs/2502.00893)
- [ToddlerBot source](https://github.com/hshi74/toddlerbot)
