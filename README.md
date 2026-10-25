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

For MuJoCo's macOS viewer with uv's standalone Python, complete the
[one-time library setup](docs/design/wr2_stance_diagnostic_20261018.md#one-time-macos-library-setup-for-uvs-standalone-python).
The viewer then uses the regular `uv run --no-sync mjpython` command.

### View the ZMP reference walks

```bash
uv sync --extra training --extra dev --extra zmp-viewer
uv run --no-sync mjpython -m wr2.tools.view_zmp_reference --robot wr2
uv run --no-sync mjpython -m wr2.tools.view_zmp_reference --robot tb
```

Run each viewer separately. Both default to forward 0.10 m/s in TB units
(0.0824 m/s for WR2), quarter-speed playback, COM/ZMP markers and support
outlines. Space pauses; `[` / `]` steps; `1` / `2` / `3` / `4` selects
front / side / top / perspective. TB uses the local frozen Paper checkout;
`--tb-root` selects its location. See the
[commands, reference sources and overlay guide](docs/design/zmp_reference_viewer.md).

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
The **GPU stance diagnostic completed**: reference hip rolls eliminate foot
self-contact and improve joint execution, but worsen six-direction command
tracking **44.2%**, with greater drift and poorer support timing. See the
[result and revised next plan](docs/design/wr2_stance_result_20261018.md).
The **matching TB Paper 200M GPU diagnostic completed**. TB walks with no foot
self-contact; nominal tracking error is **2.69× lower** than WR2. An independent
CAD check finds that WR2's boxes contact while the imported foot parts remain
separated. The policy also requests an inward stance that would overlap CAD
convex hulls; geometry's contribution to learning this stance remains
unisolated. See the
[paired result and focused next plan](docs/design/wr2_tb_stance_result_20261018.md).
The [right-hip investigation](docs/design/wr2_right_hip_incentives_20261019.md)
finds that the current policy falls sideways when only the right hip is
returned to neutral or moved outward, including during zero-air-time standing.
The [phase and retained-checkpoint review](docs/design/wr2_phase_learning_review_20261020.md)
then finds that WR2 partially learns outward movement at 20M, loses it by 40M
as angular-stability return improves, and survives phase-specific outward
movement with the existing geometry. The current priority is understanding
that early loss of coordination with matched TB Paper checkpoints.
The [TB setup](docs/design/tb_stance_diagnostic_20261018.md) and
[WR2 diagnostic guide](docs/design/wr2_stance_diagnostic_20261018.md) record the
completed commands and measurements.
Use the [recorded-policy viewer](docs/design/wr2_stance_diagnostic_20261018.md#replay-the-recorded-policy-visually)
to inspect the copied GPU rollout on the Mac with slow motion, frame stepping
and foot collision outlines.

The **collision-weight experiment completed at 300M**: increasing collision
**0.1→1.0** almost eliminates foot self-contact and changes the right hip
outward. Selected tracking improves **10.1% /10.5%** on the two evaluation
seeds, but error remains **1.87× TB Paper** and rightward response is incomplete.
See the [result and proposed next step](docs/design/wr2_collision_result_20261020.md)
and [experiment setup](docs/design/wr2_collision_experiment_20261020.md).
The [GPU confirmation and policy-feedback review](docs/design/wr2_policy_feedback_review_20261021.md)
confirms rapid, tiny foot taps. Past-action feedback sustains the 6.1 Hz
command loop even without physics integration; TB Paper's tested feedback
remains stable. The [acquisition audit](docs/design/wr2_tb_acquisition_root_review_20261021.md)
shows the fast strategy already appears at 20M. Recipe changes are deferred
until its initiating cause is identified; the late feedback mechanism alone
does not explain why training diverges from TB.
The [reward review](docs/design/wr2_reward_review_20261022.md) identifies weak
step validity and competing incentives: brief taps collect air credit, while
a more propulsive reference control loses reward through lateral sway and
angular motion. That accounting motivated stance qualification.
The [qualified-step experiment completed](docs/design/wr2_step_reward_result_20261023.md)
at 300M: nominal learned replays show sustained alternating steps instead of
rapid taps, but selected **120M** tracking is **11.0% /9.9% worse** than the
control and **2.07× TB Paper**. Forward/rightward response improves; short
strides, lateral velocity variation and standing remain gaps. See the
[experiment setup](docs/design/wr2_step_reward_experiment_20261022.md) for the
supported-swing/sustained-landing treatment, which remains enabled in
`ppo_walking_step.yaml` at the user's request; `ppo_walking_paper.yaml` is its
original-air control. The
[reward and speed audit](docs/design/wr2_tb_reward_audit_20261024.md)
explains both landing-credit definitions and distinguishes physical speed,
normalized speed and foot advance.
The [hip-pitch diagnostic](docs/design/wr2_hip_pitch_incentive_review_20261023.md)
identifies backward left-target mean drift and tanh compression. Larger isolated
left motion can gain translation while losing reward through fore-aft/yaw
oscillation and reduced air credit; joint phase coordination remains a priority.
The [completed bilateral diagnostic](docs/design/wr2_bilateral_diagnostic_20261024.md)
shows that larger left strides execute within the existing limits. Its tested
bilateral pattern lets feet exchange the lead but increases fore-aft rocking;
executed timing and weight transfer remain the next priority.
The [matched coordination test](docs/design/wr2_leg_coordination_review_20261023.md)
confirms that compensating knee/ankle targets removes most of that instability
and restores air credit. Larger placements still retain a right-leading posture
and a small yaw-driven return deficit; a new training recipe is not selected.
The canonical config retains collision 1.0 and air-time 500; the earlier
air-time ablation remains deferred.
The [progression reward audit and next experiment](docs/design/wr2_progress_reward_experiment_20261025.md)
prepares a fresh **300M seed-3** run in `ppo_walking_progress.yaml`: increase
the existing TB XY tracking coefficient **5→15**, retaining qualified landings
and all other training settings. Recorded full-orientation controls gain mean
return at 80M/120M under this coefficient; unstable controls remain penalized.
The new PPO acquisition and alternating-placement benefit remain to be tested.
The [PPO learning-direction diagnostic](docs/design/wr2_ppo_learning_diagnostic_20261023.md)
is prepared to measure normal stochastic learning at WR2 80M/120M and native
TB Paper 80M/200M. It verifies reward/advantage/actor-gradient reconstruction
and measures fresh-Adam PPO target changes. Probe weights are discarded;
the coordinated target overrides are not a learned training improvement.
The [first WR2 PPO review](docs/design/wr2_ppo_learning_result_20261024.md)
identified an asymmetric posture and a large command-resampling update.
The [v5 transition review](docs/design/wr2_ppo_transition_review_20261024.md)
localizes most critic training residual to motion/zero command switches before
the network receives the new command. The
[completed native TB comparison](docs/design/wr2_tb_ppo_comparison_20261024.md)
shows that TB also has those residuals and can have larger fresh-Adam probe KL.
WR2 retains much shorter strides: its left-hip mean moves toward the tanh limit,
compressing a raw cyclic signal that remains present. Both robots use fixed-home
leg targets. WR2's deeper reference-derived action home is a verified difference;
its causal role remains unproven, so the next analysis examines the mean drift.
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
./scripts/scp_from_remote.sh --latest training

# Deployment host
./scripts/scp_from_remote.sh --wrdev --latest servo_sysid
```

It defaults to the Linux PC's `LINUX_PUBLIC_IP` and optional
`LINUX_PUBLIC_PORT`, with remote checkout `/home/leeygang/projects/WildRobot2`.
`WR2_PUBLIC_IP` and `WR2_PUBLIC_PORT` are fallback environment variables;
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
