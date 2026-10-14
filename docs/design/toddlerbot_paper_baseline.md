# Next run: ToddlerBot paper walking baseline

## Decision and scope

Train **ToddlerBot first**, using the published walking recipe and the local
published-era release. This is the positive control needed before transferring
the recipe to WR2. The completed corrected-physics WR2 continuation did not
close the lateral tracking gap, and the available current-code TB archive
eventually contains nonfinite policies. Neither establishes a healthy baseline
for the current default recipe.

The numerical reference is the [ToddlerBot paper, Tables VI and VII](https://arxiv.org/html/2502.00893v2).
Implementation details come from local TB `v1.0.0`, commit
`59832cde2be3ab029486958f2e5ba92604f87307` (May 2025), and its Brax gitlink
`14147806f6c25724ba830ba36764df3243889b0d`. This release follows publication;
the paper does not identify every runtime/initialization detail. Call this a
**paper-aligned baseline**, not an exact replay of the authors' experiment.

Run the comparison directly in the ToddlerBot repository on branch
`paper-walking-baseline`. The prepared branch tip is `586d4dd`, with setup and
launch instructions in its `PAPER_WALKING.md`. It retains the release's model,
meshes, arm datasets and original ZMP cache. WR2 tracks this experiment guide
and the small review diff; the TB run uses the TB repository and its own
environment.

## Effective recipe

| Setting | Prepared TB control | Current WR2 XY continuation |
| --- | --- | --- |
| PPO | Released Brax PPO | RSL-RL PPO |
| Initialization | Fresh native Brax networks/Adam/counters | Continued 400M policy/Adam/LR |
| Networks | 512/256/128, Swish | 512/256/128, ELU |
| Action distribution | Native state-dependent Normal followed by tanh | Unbounded Normal, learned independent log std |
| Action targets | Home + 0.25 rad × bounded residual | Home + 0.25 rad × residual |
| Learning rate | 1e-4; released cosine schedule | Adaptive, restored 1e-5 at launch |
| Rollout/PPO | 1024 envs, 20 steps, 256 × 4 minibatches, 4 passes | 2048 envs, 20 steps, 256 × 8 minibatches, 4 passes |
| Leg motor imitation | `-mean((q_motor - q_ZMP)^2)`, weight 5 | Disabled |
| Velocity XY reward | Weight 5, released tracking sigma 100 | Weight 2, current-TB sigma 1000 equivalent |
| Survival reward | Terminal penalty, weight 10 | Positive alive reward, weight 1 |
| Actor reference information | Sin/cos phase; no reference joint vector | Sin/cos phase; no reference joint vector |
| Critic reference information | Motor position error and stance | Joint position error, phase/stance context |
| Actor/critic history | 83/126 × 15 = 1245/1890 | 67/116 × 15 = 1005/1740 |
| Control/gait | Native 20 ms / 0.72 s | 20 ms / 0.72 s |
| Physics | Released TB model, MuJoCo/MJX 3.2.4, 4 ms × 5, Newton 1 | WR2 model, MuJoCo/MJX 3.14, 5 ms × 4, Newton 10 |

`paper.gin` explicitly sets all paper reward weights, including feet clearance
**0.05**; released `walk.gin` has **1.0**. Other unspecified behavior stays with
the released implementation: original `toddlerbot` robot, ZMP lookup, servo
parameters, noise, randomization, action delay and command sampler. Native
commands are X [-0.10, +0.20], Y [-0.10, +0.10] m/s and yaw [-0.50, +0.50]
rad/s, with 20% standing, 30% turning and 50% XY walking. This sampler is
broader than WR2's 20% standing / 80% XY sampler.

Leg imitation affects the actor through PPO rewards/advantages. The joint
reference remains a critic input; adding an actor reference input is not part
of this experiment. The native policy distribution uses
`std = softplus(raw_std) + 0.001` before tanh, not current-code `init_noise_std=0.5`.
On zero observations, the released zero biases give initial pre-tanh std
approximately 0.694; real observations change its state-dependent output.

The cosine decay's 50M counter counts optimizer updates, not environment
transitions. At this budget it is almost constant at 1e-4. Preserve that released
schedule; do not substitute WR2's adaptive LR.

## Changes needed to run and measure this release

- Add a training-only dependency lock for Python 3.10, with JAX/JAXlib 0.4.28,
  MuJoCo/MJX 3.2.4, pinned local Brax 0.10.5, Orbax 0.5.23, Flax 0.8.4 and
  Optax 0.2.2. Linux installs the corresponding CUDA 12 plugin. TB and WR2 use
  separate environments.
- Adapt the released TB randomizer's two-value return to the pinned Brax
  wrapper's three-value interface, supplying an empty, unused embedding. The
  Brax source pin stays unchanged. Seed the host mass-grid permutations.
- Fix reference generation: NumPy target-foot updates now preserve the start
  pose for interpolation; JAX stride updates, segment clipping and coefficient
  indexing use array operations. The original cache is retained and fingerprinted.
  Corrected NumPy generation reproduces five complete cached trajectories
  (960 frames each), with identical stance masks and maximum joint difference
  below 8.1e-8 rad. The prior NumPy aliasing error produced a 0.595 rad discrepancy.
  JAX also regenerates the full forward trajectory within 9.5e-6 rad, with
  identical stance masks; its calculation uses float32.
- Retain every native checkpoint, audit policy/value parameters for finiteness,
  and stop after retaining a nonfinite checkpoint or nonfinite native metrics.
  Correct the top-level `policy` copy to the actual latest checkpoint.
- Record raw native metrics, source/runtime/model/reference provenance and a
  deterministic quantitative XY suite. The optional explicit reset command
  leaves ordinary training resets unchanged. Quantitative evaluation uses a
  separate environment object and does not change training rewards or sampling.
  `provenance.json` records the prepared TB branch commit and dirty status,
  the Brax commit and runtime/model fingerprints.
- Add `--skip-video` to the existing trainer. Its native intermediate rendering
  is disabled for this run; no launcher wrapper is required. Video evaluation
  also gets the correct stacked network input dimensions.

WR2's corrected randomized physics remains active in WR2: native `mj_setConst`
refreshes derived constants after mass/inertia/armature changes. The TB control
preserves released TB mass preparation and later armature randomization. It is
not a new claim that the older randomizer is a complete physics oracle.

## Budget and checkpoint measurements

Requested training budget: **300,000,000 transitions**, seed 0. Native Brax
rounding yields **300,134,400 transitions**, 14,655 rollout iterations and
234,480 minibatch optimizer updates. With `num_evals=16` and the pinned fork's
disabled initial evaluation, there are **15 trained checkpoints**, separated
by **20,008,960 transitions**. Retain all checkpoint directories.

Each checkpoint has its native random-command reward evaluation plus 15
deterministic conditions, 128 episodes each, up to 1000 control steps:

- X +0.10/-0.10/+0.05/-0.05 and Y +0.05/-0.05 m/s.
- Four signed diagonals: X ±0.0707107, Y ±0.0353553 m/s.
- Reset standing and four cardinal start/stop schedules: 3 s stand, 6 s walk,
  3 s stand, 5 s walk, 3 s stand.

Seed 12102, model draws and reset keys are held fixed across checkpoints and
conditions. Actor observation noise, reset/servo variation, model variation and
released pushes remain enabled. Only policy sampling is deterministic. Every
episode stops contributing after its first terminal/nonfinite transition; the
terminal finite sample is included. A nonfinite sample contributes to episode
length and the nonfinite count, and is excluded from measurement sums.

`xy_evaluation.jsonl` stores signed torso-frame X/Y velocity, instantaneous
`norm(v_xy - command_xy)` MAE, instantaneous XY speed, leg motor RMSE, falls,
nonfinite counts, lengths and per-episode values. Group means first normalize
within each episode, then average episodes. Standing/walking/start/stop each
have their own counts; an empty category is `null`. Full surviving transition
episodes have 450 standing, 550 walking, 50 start and 50 stop samples.

These command values and metric definitions match the WR2 review protocol;
the physical robot and disturbance distribution differ. The paper's 0.016 m/s
trajectory error is not the same instantaneous vector-MAE benchmark. Judge TB
control success from its measured signed response, tracking, standing/recovery
and survival, then set an explicit comparison gate. `best_policy` is still the
native reward-best policy; it does not certify XY parity. Evaluate retained
intermediates instead of assuming the final checkpoint is best.

Native checkpoints contain normalizer, policy and value parameters. They
**do not contain Adam, RNG, counters or environment state**; `--restore` is a
warm start, not exact continuation. Start this control fresh.

## Direct TB branch setup

Once the prepared branch has been transferred to your existing TB checkout,
run from its repository root on the GPU host (`~/projects/ToddlerBot`). The
branch contains the paper configuration, dependency lock, implementation fixes,
readiness checks and quantitative evaluator. The separate `.venv-paper`
environment preserves the existing TB `.venv`.

```bash
(
  set -eu
  cd ~/projects/ToddlerBot
  git switch paper-walking-baseline
  git submodule update --init --recursive
  uv venv --python 3.10 .venv-paper
  uv pip sync --python .venv-paper/bin/python requirements-paper.lock
)
```

The review diff records the prepared branch's changes relative to `v1.0.0`.
Those changes are already committed on the branch. It contains no model
binaries or upstream package copies. The Brax gitlink remains pinned and its
URL uses HTTPS, so initialization does not require a GitHub SSH key.

## Direct launch on the GPU host

After setup, check the GPU backend and launch TB's existing trainer directly.
A failed `cd` or GPU check stops before the trainer runs.

```bash
(
  set -eu
  cd ~/projects/ToddlerBot
  .venv-paper/bin/python -c 'import jax; print(jax.devices()); assert jax.default_backend() == "gpu"'
  WANDB_MODE=offline .venv-paper/bin/python -m toddlerbot.locomotion.train_mjx \
    --robot toddlerbot --env walk --gin-files paper --skip-video
)
```

Run output is `results/toddlerbot_walk_ppo_<timestamp>/` inside the TB repository.
Keep the complete directory, including `metrics.jsonl`, `xy_evaluation.jsonl`,
`provenance.json`, effective configs, dependency lock and all checkpoint folders.
No full training run has been launched during preparation.

## Verification and transfer

The preparation patch applies to a fresh checkout of the pinned TB release.
The native trainer imports from that checkout with the pinned Brax runtime.
The direct branch's separate environment installs all 106 locked packages and
passes eight lightweight recipe/reference/metric/checkpoint checks. Its setup
commands, original ZMP hash, Brax pin and ignored environment were also checked.
The locked dependency installation, complete walking environment (12 actions),
native reset/step integration and PPO smoke below were verified during the
original preparation.

The CPU PPO smoke completes **80 transitions**, two rollout iterations / eight
minibatch optimizer updates, and saves finite 40/80-step checkpoints. The policy
weights change by up to **0.0004067** between checkpoints; the top-level policy
is byte-identical to the 80-step checkpoint. Both checkpoints complete all 15
quantitative conditions: **60 finite episodes**, each reaching the reduced
20-step horizon. This horizon exercises checkpoint/evaluation execution; its
transition cases end before the first commanded start at 3 s.

Nine focused checks cover paper settings, native imitation/terminal rewards,
unequal episode lengths and empty categories, history-preserving command
updates, complete schedule counts, native reset/step agreement, both nonfinite
abort cases, NumPy/JAX foot interpolation and traced polynomial evaluation.

Reference preflight covers sixteen signed/standing/turning commands and 73
samples per command. All sampled leg motor references are finite and within
physical motor limits; the stance cycle is 36 frames / 0.72 s. Some native swing
motor references are **1.064 rad from home**, exceeding the native ±0.25 rad residual target
range (plus reset target bias). The paper uses a soft imitation reward; exact
reference position targets are not fully representable by this actor. Preserve
the native recipe for the control and measure its actual motor RMSE. Validate
target representability and actuator feasibility explicitly before WR2 transfer.

Focused configuration/reward/measurement checks:

```bash
(
  set -eu
  cd ~/projects/ToddlerBot
  WANDB_MODE=disabled .venv-paper/bin/python -m unittest discover \
    -s tests -p test_paper_walking.py -v
)
```

The readiness smoke uses the same trainer, paper networks, full released
randomization, checkpoint saving and all 15 conditions; only environment count,
batch/minibatches, horizon, budget and evaluation cadence are reduced:

```bash
(
  set -eu
  cd ~/projects/ToddlerBot
  WANDB_MODE=disabled .venv-paper/bin/python -m toddlerbot.locomotion.train_mjx \
    --gin-files paper --skip-video \
    --config-override 'PPOConfig.num_envs=2,PPOConfig.num_eval_envs=2,PPOConfig.num_timesteps=80,PPOConfig.num_evals=3,PPOConfig.batch_size=2,PPOConfig.num_minibatches=1,PPOConfig.episode_length=20'
)
```

CPU readiness checks establish execution and metric contracts, not learned
walking, CUDA memory fit or hardware qualification. GPU availability and the
long TB run remain the next execution steps.

After a healthy TB control is measured, transfer the complete recipe to WR2
with its measured geometry/servo limits and verified reference. Normalize
speed as `v/sqrt(g*h_COM)` and timing as `T/sqrt(h_COM/g)` using measured COM
height, rather than copying physical speeds/timing by robot name. If choosing
RSL-RL for WR2, first check the equivalent recipe on TB under RSL-RL so a backend
difference is measurable. Then train fresh WR2 under corrected physics in
affordable 200M stages, retaining the current 400M/600M policies as controls.

## References

- [ToddlerBot paper](https://arxiv.org/abs/2502.00893) and
  [public project](https://toddlerbot.github.io/).
- Released TB plus preparation changes (paths relative to the TB repository root):
  `toddlerbot/locomotion/{walk.gin,mjx_config.py,mjx_env.py,train_mjx.py}`
  and `toddlerbot/reference/walk_zmp_ref.py`; pinned Brax
  `brax/training/agents/ppo/{networks.py,train.py}` and `brax/training/distribution.py`.
- Tedrake et al., *A Closed-Form Solution for Real-Time ZMP Gait Generation and
  Feedback Stabilization* (2015), [DOI](https://doi.org/10.1109/HUMANOIDS.2015.7363473).
- [MuJoCo `mj_setConst`](https://mujoco.readthedocs.io/en/stable/APIreference/APIfunctions.html#mj-setconst).
