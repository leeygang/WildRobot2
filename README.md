# WildRobot2

WildRobot2 is a clean, ToddlerBot-aligned software stack for the second
WildRobot hardware generation.

The repository contains the canonical WR2 MuJoCo model, a versioned
action/observation contract, and the first flat-ground MJX walking environment
with ToddlerBot-aligned RSL-RL and Brax PPO backends. A bench-only HTD-45H
qualification driver is included; full
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
for the 15-frame, 1005-value heading/phase-aware actor observation, twelve-value leg action
contract, and remaining hardware gates. The tracked training and hardware work
is in [`docs/TODO.md`](docs/TODO.md). See
[`the heading, stance/slip and clipping diagnostic review`](docs/design/walking_diagnostics_20261007.md)
for the current pre-training evidence and reproducible Mac/GPU checks. See
[`the model/reference, actor-symmetry and solver-convergence verification`](docs/design/walking_symmetry_verification_20261007.md)
and [`the latest regression audit and controlled solver A/B plan`](docs/design/walking_regression_ab_20261007.md)
for the full-substep impulse checks and physics-first next experiment. See
[`docs/design/servo_model.md`](docs/design/servo_model.md) for the evidence and
limitations behind the initial HTD-45H model.
Training values are defined in
[`wr2/locomotion/configs/ppo_walking.yaml`](wr2/locomotion/configs/ppo_walking.yaml);
the trainer accepts explicit CLI overrides and snapshots the effective YAML in
every run directory.
For signed forward/backward and lateral walking, use
[`ppo_walking_xy.yaml`](wr2/locomotion/configs/ppo_walking_xy.yaml).
It follows TB's XY command sampler with turning excluded; see
[the scope, reference validation and GPU command](docs/design/xy_walking_training.md).
The [corrected-physics XY continuation](docs/design/xy_physics_continuation.md)
records the 1B review and subsequent 200M experiment. That continuation completed
at 600M cumulative transitions; lateral tracking and backward launch stability
remain incomplete. The
[ToddlerBot paper control](docs/design/toddlerbot_transfer_review_20261016.md)
completed and supplies the recipe for the
[WR2 paper training experiments](docs/design/wr2_paper_training.md).
The 300M continuation at fixed LR 1e−5 completed: reported PPO update size is
controlled, but best normalized tracking error remains 2.82× TB Paper 200M.
See the [WR2/Paper/Main progress comparison and next investigation](docs/design/wr2_tb_progress_review_20261017.md).
The **fresh 300M Paper control, seed 3 / LR 1e−4**, also completed. In the
recorded native profiles, its best error is **3.53× TB Paper 200M**; nominal
replays confirm hopping.
See the [latest result, code comparison and improvement proposal](docs/design/wr2_fresh_paper_result_20261018.md).
The [deeper actuator review](docs/design/wr2_actuator_gap_review_20261018.md)
identifies the modeled response mechanisms and confirms TB Paper's configured
pushes are physically absent. The **0.28.5 fresh 300M acquisition without
pushes** has now completed. Selected 260M has **zero height failures on both
evaluation seeds**, but barely translates: error remains **2.08× TB Paper
200M** and exceeds the stationary initial policy. See the
[completed no-push result and revised next plan](docs/design/wr2_no_push_result_20261018.md)
and [experiment setup](docs/design/wr2_no_push_acquisition_20261018.md).
The next step is a **paired stance diagnostic** of the selected 260M policy:
learned actions, reference hip-roll targets, and the full leg reference.
It checks foot clearance, inward hip-roll clipping, support timing and actual
translation through the existing actuator pipeline. See the
[setup and decision criteria](docs/design/wr2_stance_diagnostic_20261018.md).
Run from the WR2 checkout on the GPU host:

```bash
uv run --python 3.10 --extra training python -m wr2.tools.diagnose_walking \
  --paper-stance \
  --run-dir results/wr2_walking/wr2_paper_20261018_121025_seed3 \
  --output results/wr2_walking/wr2_paper_20261018_121025_seed3/stance_diagnostic \
  --seeds 22102
```

The canonical config retains the completed TB-aligned weight-500 baseline.
The previously prepared air-time ablation is deferred while stance/geometry
is investigated. The guide specifies the matching TB measurements and the
evidence needed to choose a training change.
The [WR2/TB Paper setup audit](docs/design/wr2_tb_paper_setup_audit_20261018.md)
compares actual reward formulas, clocks, effective variation, resets and metrics.
Actor/critic inputs are 975/1485 for this recipe. WR2 training is self-contained.
The [six-DOF model-import guide](docs/design/hip_yaw_model_import.md) documents
both new hip yaws, 21 actuators, model-derived home/CoM geometry and validation.
Neck/head stay outside walking actions; the updated export now includes their
CAD mass properties. Historical ten-leg-action
checkpoints and their continuation/migration commands do not work on this model.
Historical experiment guides remain as evidence for the previous hardware model.
The bounded train/evaluate/promote campaign and Mac-to-GPU command are described
in [`docs/design/walking_training_agent.md`](docs/design/walking_training_agent.md).
That guide also covers the autonomous Mac supervisor, which analyzes each GPU
cycle, asks Codex for one high-confidence ToddlerBot-aligned improvement,
validates and pushes its commit, and then fast-forwards the GPU before
continuing.
The staged high-load qualification procedure is in
[`docs/hardware/htd45h_deployment_test.md`](docs/hardware/htd45h_deployment_test.md).
Use `uv run python -m wr2.tools.servo_sysid list-plans` to inspect the versioned
servo campaigns; `run`, `capture`, `fit`, `hysteresis`, and `report` share that
single module entry point. The canonical servo protocol remains
[`wr2/actuation/htd45h.py`](wr2/actuation/htd45h.py).

## Copy results from a remote machine

The transfer script copies artifacts into the matching local `results/`
directory. JSON/NPZ servo captures with the same stem are treated as one run.

```bash
./scripts/scp_from_remote.sh --latest servo_sysid
./scripts/scp_from_remote.sh --latest servo_sysid 3
./scripts/scp_from_remote.sh --latest training 2
./scripts/scp_from_remote.sh --list servo_sysid

# Copy a whole custom results folder, including all nested runs
./scripts/scp_from_remote.sh --results wr2_solver_ab
# Or copy just one subtree
./scripts/scp_from_remote.sh --results wr2_solver_ab/newton10_seed0

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
- [RSL-RL project](https://github.com/leggedrobotics/rsl_rl)
