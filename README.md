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
Use `wr2.tools.view_walking --run-dir <training-run>` to execute a trained WR2
Paper policy directly; see the [Linux/Mac commands and live controls](docs/design/wr2_stance_diagnostic_20261018.md#run-a-trained-wr2-policy-directly).
Live evaluation parameters, timestamped metrics/events and summaries are saved
under `results/visual_eval/`; copy sessions with
`./scripts/scp_from_remote.sh --latest visual_eval`.

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
The **XY tracking-priority run completed at 300M**, seed 3, with the existing
TB coefficient **5→15** in `ppo_walking_progress.yaml`. Selected tracking improves
**40.9% /39.4%** on selection/held-out seeds, with zero primary falls. Forward
response and actual advances approach TB, but the right hip loses swing and
foot self-contact increases. See the
[result, retained-checkpoint comparison and next priorities](docs/design/wr2_progress_reward_result_20261025.md)
and [experiment setup](docs/design/wr2_progress_reward_experiment_20261025.md).
The [speed/contact audit](docs/design/wr2_speed_contact_review_20261025.md)
finds that both feet pass at .15/.20 TB m/s. At .10, modeled box contacts occur
with clear CAD foot parts; higher commands increase forward lean and heading
drift. Live evaluation logs capture the actual session and effective parameters
under `results/visual_eval/` for the next comparison.
The [retained-checkpoint heading evaluation](docs/design/wr2_heading_evaluation_20261025.md)
compares 120M/200M/300M with native TB Paper across randomized .10/.15/.20
commands. At .20, 300M improves body speed but loses requested-direction progress;
the report includes the prepared larger GPU confirmation command.
The [completed GPU review](docs/design/wr2_heading_gpu_review_20261025.md)
confirms that direction loss over 256 episodes per command. At .20, 120M/200M
achieve similar requested-direction progress; 200M has two sampled falls.
The [heading-feedback continuation completed](docs/design/wr2_heading_feedback_result_20261026.md)
at +200M from 300M. Selected +173M cuts held-out mean heading error
11.8°→5.8°, with zero primary falls. The
[completed GPU comparison](docs/design/wr2_heading_feedback_gpu_review_20261026.md)
confirms .20 requested-direction speed .103→.162 TB m/s and net turn
80°→28°, with zero falls across2,304 randomized attempts. +120M is straighter
(13.5°) but slower(.154 TB m/s), with better sampled contact-phase agreement.
The [full +120M confirmation](docs/design/wr2_heading_feedback_120_confirmation_20261026.md)
passes both seeds with1/3,840 falls(held-out0/1,920). Lower foot self-contact
and better phase agreement support it as the next forward-walking baseline;
aggregate tracking/standing still favor+173M. The
[goal-frame XY continuation](docs/design/wr2_goal_frame_xy_experiment_20261026.md)
completed200M from+120M. The
[result and action-space audit](docs/design/wr2_goal_xy_result_20261027.md)
finds a1.9% held-out reference-error gain and better heading/self-contact,
with worse push recovery. A small Mac screen favors final+200M for .20 forward
walking over the standard selector's+173M. Actual steps alternate, but landing
placement remains uneven and the right hip command stays near its action bound;
the [completed GPU comparison and shoulder review](docs/design/wr2_goal_xy_gpu_review_20261027.md)
confirm final .20 speed .169 TB m/s, net turn6.2° and2/256 falls. Right-hip
commands stay near their bound92.5% of the time. Shoulder rocking follows torso
roll; normalized body-height variation remains larger than TB. The
[final-policy confirmation](docs/design/wr2_goal_xy_final_confirmation_20261027.md)
passes both seeds with 0/3,840 falls, qualifying +200M for the forward priority;
+173M retains better aggregate XY/contact metrics. The hip action/posture audit
has [completed on the Mac](docs/design/wr2_hip_action_validation_20261027.md):
posterior stance coverage improves with model-home hip centers, but zero-action
stance leans backward and frozen-policy speed falls17–24%. Supported reset is
preservable. The [randomized GPU check](docs/design/wr2_hip_action_gpu_review_20261028.md)
covers 3,584 attempts and confirms coverage/initialization, but the frozen
candidate loses 16–28% forward speed and retains short right-foot placement.
The [next comparison is prepared](docs/design/wr2_hip_action_training_20261028.md):
two 200M continuations with current versus model-home hip action centers.
The [completed training review](docs/design/wr2_hip_action_training_review_20261029.md)
finds better low-command progression/tracking at candidate +173M, but nominal
right-hip suppression and uneven placement persist; the final candidate regresses.
The paired GPU higher-command comparison is ready before policy promotion.
The [next training-setup test](docs/design/wr2_leg_imitation_experiment_20261029.md)
compares leg-reference weights 5 and 15 from the same +173M policy, preserving
XY15 and the sustained-landing rule. It tests actual right swing/placement
over 200M per arm with reference, physics and PPO settings held fixed.
The [completed reward comparison](docs/design/wr2_leg_imitation_result_20261029.md)
finds better tracking/contact metrics with weight15, but no right-swing or
placement fix. The retained right mean remains near its posterior endpoint.
Fresh initialization of this recipe is the next proposed training-setting test.
The [latest shoulder-motion review](docs/design/wr2_shoulder_rocking_review_20261029.md)
confirms torso roll plus root bobbing and proposes a separate roll-reward
comparison scaled for WR2's wider shoulders.
The [size-alignment audit](docs/design/wr2_training_size_audit_20261029.md)
verifies commands/noise and local geometry, and identifies unscaled learning
time horizons plus force/effort normalization differences. These are explicit
recipe candidates; existing experiment configurations are preserved.
The [fresh size-alignment comparison](docs/design/wr2_size_alignment_experiment_20261030.md)
is ready: two fresh 200M runs with current versus normalized episode/rollout,
discount/GAE and nominal support-force threshold. Hardware timing, history,
geometry and angular roll reward remain shared. Native timeout diagnostics now
retain the target delivered on the final tick.
The [completed fresh aligned run](docs/design/wr2_size_aligned_result_20261030.md)
improves held-out tracking and restores bilateral hip swing, with smaller nominal
shoulder rocking. Final has the best tracking;120–160M have more even nominal
placement. Randomized self-contact and push recovery remain concerns. The retained
higher-command GPU comparison is ready before selecting a walking policy.
This standard suite commands zero yaw throughout. The
[experiment](docs/design/wr2_heading_feedback_experiment_20261025.md) retains
the previous training reward formulas and weights.
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
