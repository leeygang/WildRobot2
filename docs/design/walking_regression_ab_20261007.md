# Heading-regression audit and solver/asymmetry experiment

## What the latest training changed

The regressed run is `wr2_ppo_20261007_152924_seed0`, v0.21.0, following
`wr2_ppo_20261007_084251_seed0`, v0.20.1. The relevant commit is `7aae26a`.
The saved effective configurations differ only in version labels and enabling
`environment.heading_observation`. That flag both adds relative heading to the
actor/critic and switches the tilt-only upright reward to full orientation.

Source and actual-checkpoint checks:

- Both runs have identical servo snapshots, action scale/home, ZMP preflight,
  command sampling, rewards weights, randomization and PPO settings.
- The v3-to-v4 migration inserts two zero-weight columns after the first 55
  inputs in **each of fifteen frames**, including the critic's corresponding
  Adam moment columns. Every resulting parameter and optimizer tensor matches
  the expected migration exactly. Counters, LR, policy std and RNG are retained.
  The latest run's retained step-zero model exactly matches the migrated source.
- Full orientation uses the shortest quaternion angle and `exp(-20 * angle^2)`
  with weight 2.5. This matches ToddlerBot's `_reward_torso_quat` and `walk.gin`;
  the reference is initial home orientation plus integrated commanded yaw.
- Noise uses `R_noise * R_true`, matching ToddlerBot. Tests verify projected
  gravity and heading describe this same measured orientation, and that all
  pre-existing actor/critic features are unchanged by adding heading.
- Replaying identical motion with v3/v4 environments gives identical physical
  states and non-orientation metrics. Only upright reward and total reward
  change. Transition evaluation and path diagnostics do not drive training.
- The preceding tilt-only continuation also regressed slightly: nominal
  0.10 m/s MAE 0.02468 -> 0.02699 m/s. The latest regression cannot be attributed
  entirely to adding heading merely because it followed that commit.

For the latest run, nominal full-episode 0.10 m/s MAE changed
0.02474 -> 0.04110 m/s and heading error 4.16 -> 3.41 degrees. No falls occurred.
The total return rose 2.77667 despite tracking regression. Reconstructing it
from logged, already-signed reward components (use **absolute** penalty weights
when decoding these logs) gives 2.77666, agreeing within floating-point error:

| Contribution to return change | Change |
|---|---:|
| Full orientation | +1.75459 |
| Foot phase | +0.61571 |
| Pose | +0.52380 |
| Gaussian XY velocity | +0.41650 |
| Action rate | -0.40656 |
| Yaw rate | -0.21383 |
| Remaining terms combined | +0.08646 |

This is evidence of an objective/behavior trade-off, not a proven reward bug.
The existing paired heading disturbances show learned heading recovery.
Gaussian reward averages need not improve with mean absolute velocity error;
their kernels and metrics measure different things. No heading-weight increase,
reward removal, new progress reward, or standing-knee penalty is justified here.

## Heading-input ablation on Mac

Hide heading in **every actor history frame** by replacing sin/cos with `[0, 1]`.
Keep the final policy weights, physics, reward, servo targets, reset seeds and
real heading state unchanged. This removes feedback available to an already
trained policy; it is **not** retraining without the heading objective.

```bash
uv run --extra training python -m wr2.tools.verify_walking_symmetry \
  --run-dir results/wr2_walking/wr2_ppo_20261007_152924_seed0 \
  --output results/wr2_walking_diagnostics/20261008_heading_ablation_solver10 \
  --solver-iterations 10 --ablate-heading --seeds 101,202,303,404
```

Results use the same world/path-forward velocity, settled 4--20 s window and
seeds as the previous solver-10 probe. Do not compare these MAEs directly to
the full-episode torso-frame training evaluator above.

| Command | Final, heading available | Final, heading hidden |
|---|---:|---:|
| 0.05 m/s | 0.02276 | 0.02412 |
| 0.075 m/s | 0.02694 | 0.02823 |
| 0.10 m/s | 0.03310 | 0.03553 |

At 0.10 m/s all four paired seeds worsen (+0.00317, +0.00172, +0.00409,
+0.00072 m/s). Actual half-cycle joint mismatch also does not recover:
4.91 -> 4.96 degrees. For the retained source policy, every seed's MAE is
**exactly unchanged** at all three speeds, validating its zero heading weights.
All 24 episodes survive 1000 steps with no non-finite states. This argues against
removing the heading input as an immediate fix; it does not prove the changed
training objective played no role in learning a worse gait.

## Pre-training checks under changed physics

The retained policy was checked with ten solver iterations, full configured
dynamics/reset randomization, held-out seeds 505/606 and sixteen environments
per seed. Scripted standing/walking transitions completed 32 episodes without
falls or non-finite states. All-action target clipping was 1.07%/1.24%;
mean torque RMS 0.409/0.400 N m, mean per-step peak 0.995/0.979 N m, and the
provisional exposure metric 0.372/0.356. These are per-seed evaluation means,
not a guarantee of every instantaneous torque or physical thermal safety.
Standing speed MAE was 0.0120/0.0116 m/s; walking-only MAE was
0.0325/0.0345 m/s, which the lower full-schedule MAE would otherwise hide.
This is a development preflight, not a passed final walking gate.

Fixed-command checks completed another 96 episodes with no falls, non-finite
states or target clipping. Mean values across the two equal-sized seed batches:

| Command | Torso-forward velocity | Full-episode speed MAE | Absolute heading error |
|---|---:|---:|---:|
| 0.05 m/s | 0.06031 m/s | 0.02981 m/s | 13.53 degrees |
| 0.075 m/s | 0.07653 m/s | 0.02796 m/s | 13.41 degrees |
| 0.10 m/s | 0.09016 m/s | 0.03065 m/s | 12.31 degrees |

Contact-phase match was 84.3--84.6%; seed-mean torque RMS 0.494--0.511 N m,
mean per-step peak 1.193--1.239 N m and exposure 0.499--0.536. The source policy
is not qualified for straight-path walking under changed physics. These are
small development batches, not the planned 128-env independent confirmation.

Paired nominal-model heading disturbances at the same two held-out seeds
also compared one and ten iterations. The zero-offset retained policy's mean
steady heading error increases **2.67 -> 16.58 degrees** under accurate physics,
while the final heading-aware policy changes **2.45 -> 5.05 degrees**. Neither
falls. After +/-10-degree state rotations, the retained ten-iteration policy
preserves about 8.8/9.0 degrees of the perturbation relative to its zero-offset
control; the final policy reduces those differences to about 1.14/1.20 degrees.
Thus changing physics is a real policy-domain shift, and heading feedback is
useful. Do not equate the retained policy's good torso-forward speed with
straight-path walking, or disable heading because the older policy ranked best
under one-iteration physics.

Artifacts: `20261008_solver10_transition_preflight.json`,
`20261008_recovery_solver1/summary.json` and
`20261008_recovery_solver10/summary.json`, all under
`results/wr2_walking_diagnostics/`. The artifact names identify this audit's
batch, not a new training-result date. Recovery checks add 48 twenty-second
diagnostic episodes (24 per solver), all surviving; their model is nominal
with configured reset/servo variation, unlike the fully randomized evaluator.
The full fixed-command report is `20261008_solver10_heldout_preflight.json`.

Validation: all **139 unit tests** pass; the checks above plus heading ablation
cover **200 twenty-second episodes**, all surviving. A separate restored RSL-RL
CPU smoke at ten solver iterations completes two PPO updates/80 transitions,
preserves the exact source weights at step zero, saves finite actor/critic
parameters and captures the effective solver/model/source hashes. Its 40-step
episodes and tiny batch verify plumbing only, not learning effectiveness.

## Hypothesis and controlled training test

Hypothesis: the one-iteration solver's measured contact-dynamics error biases
learning toward uneven propulsion, impairing speed tracking. This is currently
**unproven**. See [the numerical and symmetry evidence](walking_symmetry_verification_20261007.md).
Asymmetry itself is not a failure gate: both feet can work differently while
meeting tracking, contact, stability and servo criteria.

Run two 20M continuations from the **same retained v4 checkpoint**, not the
regressed final or an earlier arm's output. They differ only in Newton's
iteration ceiling. Preserve heading reward, observations, PPO, actuator model,
5 ms physics, four substeps, command sampling and observation noise.
Historical configs default to one iteration. The canonical MJCF remains
unchanged; the environment applies the budget before converting the model to MJX.
Run metadata records effective solver/line-search budgets, timestep, substeps,
model hash and source checkpoint hash. Each arm snapshots its effective YAML.

On the GPU machine, after syncing this commit:

```bash
# A: current physics control arm
uv run --extra training python -m wr2.locomotion.train \
  --config wr2/locomotion/configs/ppo_walking.yaml \
  --num-timesteps 20000000 --num-evals 6 --seed 0 \
  --solver-iterations 1 --output-root results/wr2_solver_ab/newton1_seed0 \
  --restore-checkpoint results/wr2_walking/wr2_ppo_20261007_152924_seed0/best_params.pt

# B: accurate-physics treatment arm; same starting checkpoint
uv run --extra training python -m wr2.locomotion.train \
  --config wr2/locomotion/configs/ppo_walking.yaml \
  --num-timesteps 20000000 --num-evals 6 --seed 0 \
  --solver-iterations 10 --output-root results/wr2_solver_ab/newton10_seed0 \
  --restore-checkpoint results/wr2_walking/wr2_ppo_20261007_152924_seed0/best_params.pt
```

Run IDs remain automatic. RSL-RL restores Torch RNG/Adam/LR from the same source;
`--seed` selects environment/reset/randomization streams, not a cold network.
Each run takes 489 updates, 20,029,440 transitions. Each initial evaluation
measures the same source policy under that arm's physics, so compare learning
improvement within arms as well as their endpoints. Six evaluations are not six
PPO updates. Do not deploy the inaccurate control arm's policy.

Copy both arm directories to Mac after completion (these are separate from
the usual `training` result-group alias):

```bash
./scripts/scp_from_remote.sh --copy wr2_solver_ab newton1_seed0
./scripts/scp_from_remote.sh --copy wr2_solver_ab newton10_seed0
```

For confirmation, repeat the pair with `--seed 1` and separate `seed1` output
roots. A single training seed is an exploratory result, not high-confidence
causal attribution. No training job is automatically launched by these tools.

### Common evaluation and decision rule

Evaluate both trained arms under **the same ten-iteration physics**, same
held-out seeds and command schedule; otherwise training and evaluation dynamics
are confounded. Include final and selected-best checkpoints, not only the
highest-return snapshot. Example (substitute an actual run's checkpoint/config):

```bash
uv run --extra training python -m wr2.locomotion.evaluate \
  --checkpoint PATH_TO_PARAMS.pt --config PATH_TO_TRAINING_CONFIG.yaml \
  --solver-iterations 10 --seeds 707,808,909 \
  --commands 0.05,0.075,0.10 --num-envs 128 --output HELDOUT.json

uv run --extra training python -m wr2.locomotion.evaluate \
  --checkpoint PATH_TO_PARAMS.pt --config PATH_TO_TRAINING_CONFIG.yaml \
  --solver-iterations 10 --seeds 707,808,909 --commands 0.10 \
  --num-envs 128 --transitions --output TRANSITIONS.json

uv run --extra training python -m wr2.tools.verify_walking_symmetry \
  --run-dir PATH_TO_RUN --output SYMMETRY_OUTPUT \
  --solver-iterations 10 --seeds 707,808,909
```

Mac evaluation adds `--allow-cpu`; symmetry verification supports Mac directly.
Use the paired heading diagnostic with `--solver-iterations 10` to verify
heading recovery separately. Check constant and transition clipping, bilateral
contact use, falls, finite states, torque RMS/peak/exposure and GPU throughput.
The provisional torque model is not a thermal/deployment qualification.

Predeclared interpretation:

- Lower actor/half-cycle mismatch **and** improved held-out tracking in the
  accurate arm across both training seeds, without new falls/servo violations,
  supports the hypothesized learning bias. It does not prove a sole root cause.
- Improved tracking with unchanged asymmetry supports accurate physics but
  **not** the proposed asymmetry-mediated explanation.
- Reduced asymmetry without better walking is not success.
- No reproducible advantage means do not claim the solver caused the gait
  regression. Investigate PPO/objective trade-offs next, with a separate change.

No symmetry loss or diagnostic action projection is enabled in training.
ToddlerBot also disables its optional mirror loss by default; WR1 provides
an optional loss but its morphology-specific coordinate signs cannot be copied.

References: [MuJoCo numerical solvers](https://mujoco.readthedocs.io/en/stable/computation/index.html#algorithms),
[ToddlerBot](https://toddlerbot.github.io/), and
[Yu, Turk and Liu (2018)](https://arxiv.org/abs/1801.08093).
The symmetry paper adds a loss plus assisted curriculum; it does not establish
that imposing symmetry alone solves WR2's regression.
