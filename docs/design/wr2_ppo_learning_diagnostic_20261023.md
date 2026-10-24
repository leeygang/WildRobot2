# PPO learning-direction diagnostic — 2026-10-23

Follow-up to the [coordination diagnostic](wr2_leg_coordination_review_20261023.md).
The improved intervention videos override joint targets. They do not show a
newly learned policy. This diagnostic investigates the **normal PPO learning
signal** behind the short, asymmetric learned steps.

## Question and current evidence

WR2's nominal left hip action mean moves toward the backward endpoint:
`.728 at 80M →.894 at 120M →.978 at 300M`. Its requested cyclic range shrinks
`2.50° →1.65° →.44°`. TB Paper's successful hip remains near the middle of the
same action family. The mechanism is verified; the initiating PPO pressure is
not. Coordinating knee/ankle targets restores air credit and improves placement,
but the corrected controls still have lower mean return, principally through
yaw costs. Those interventions do not select a training fix.

**No training reward, action mapping, physics, reference or checkpoint changes
are part of this diagnostic.** There are no joint-target overrides.

## Protocol

Run [diagnose_ppo.py](../../wr2/tools/diagnose_ppo.py) on the GPU that trained the
policies. It loads the saved configuration and checks runtime provenance.

| Setting | WR2 | TB Paper |
| --- | --- | --- |
| Default checkpoints | 80,035,840 and 120,053,760 | 80,035,840 and 200,089,600 |
| Actor | Original checkpoint's sampled policy | Original checkpoint's sampled policy |
| Commands, noise, model/controller randomization | Saved training recipe | Saved native Paper recipe |
| Parallel environments | 1024 | 1024 |
| Warmup | 330 physical ticks, approximately 8 TB seconds | 400 ticks, 8 seconds |
| Measured batches/checkpoint | 8 ×1024 ×20 transitions | 8 ×1024 ×20 transitions |
| Probe on each batch | Four epochs ×four minibatches, Adam at 1e−4 | Same native recipe |

Every batch is collected with the **original frozen checkpoint**, rather than
with the previous probe's weights. Each probe begins with fresh Adam state,
updates both actor and critic through the installed native Brax PPO loss, and
discards its weights. There are 327,680 measured transitions across the two
checkpoints, plus warmup. The batches share environment trajectories and are
not independent training seeds.

The checkpoint gradients are independent of optimizer state. The measured
Adam updates describe a parameter-only restart; they **do not reconstruct the
historical updates**, because neither saved checkpoint contains Adam moments,
training RNG or environment state. TB's exported `policy` pickle also omits
the critic: the tool restores its adjacent full Orbax checkpoint and checks
that its actor exactly matches the exported policy.

One first shuffled minibatch per measured batch is attributed, with exactly
the entropy key used by that probe. WR2's `.25 MSE` value coefficient maps to
Brax's `.5` coefficient; TB's native loss also uses `.5`. Both use gamma `.97`,
lambda `.95`, clipping `.2`, normalized advantages and entropy `.0005`.
Original precision and XLA settings are retained: WR2 saved default precision,
TB `high`. A CPU development check is explicitly labeled and records runtime
differences; GPU measurements require the completed run's core package versions.

## What is measured and verified

The actor's sampled loss gradient is split into:

1. XY tracking.
2. Yaw tracking.
3. Torso stability: orientation, vertical velocity, angular XY, roll and pitch.
4. Leg-reference imitation.
5. Air time.
6. Every remaining reward, including collision and termination.
7. Native value-baseline/bootstrapping residual.
8. Entropy.

The implementation in [ppo_diagnostics.py](../../wr2/locomotion/ppo_diagnostics.py)
uses the installed Brax GAE function. Reward groups use zero component values;
the seventh component retains the native critic and bootstrap. Every component
shares the total advantage's standard deviation and the **total surrogate's
clipping branch**. Independently clipping or normalizing each reward would
produce a different gradient.

Checks require reward reconstruction, advantage reconstruction and full actor
parameter-gradient reconstruction. Native first-minibatch loss metrics must
also match the actual first Adam step. This is an exact sampled-loss decomposition
up to float32 arithmetic. It is not a set of learned component critics, and
component sizes alone do not establish independent causal reward effects.

Two direction measurements are reported:

- **Local forward-mu pressure:** the loss gradient at the hip output, expressed
  in physical forward directions. It is grouped by command and four gait phases.
- **Projected forward-target SGD change:** each component's infinitesimal
  parameter-gradient effect on mean forward hip targets, on the same forward
  observation subset. Positive means a plain SGD step would move forward.

Actual post-probe target changes are measured separately on the same observations.
Adam is nonlinear; separately applying Adam to each component would not give an
additive explanation of its update. Post-probe fields contain **requests only**;
their physical motion is not simulated or claimed improved.

Recordings retain sampled actions, Gaussian means/scales, deterministic targets,
per-step rewards, hip actual/reference angles, command/phase, physical velocity,
support, landing lead and world foot centers. The centers allow same-foot stride
analysis across batch boundaries. Falls are separated from timeouts. WR2 linear
velocity divides by `s=.8240547565`, angular velocity multiplies by `s`, and foot
distances divide by `s²=.6790662418`. Support and termination fractions are
fractions of recorded **transitions**, not episode-level benchmark fall rates.
Joint signs are WR2 `left−/right+` and TB
`left+/right−` for forward hip pitch. Both use sole-collider centers. TB uses its
sole collider pair,
not its second ankle-link collider pair.

## GPU command: WR2

From the GPU's WR2 checkout, after pulling the committed diagnostic:

```bash
uv run --locked --python 3.10 --extra training python -m wr2.tools.diagnose_ppo \
  --run-dir results/wr2_walking/wr2_paper_step_20261022_143550_seed3
```

The output is:

```text
results/wr2_walking/wr2_paper_step_20261022_143550_seed3/ppo_learning_diagnostic
```

The full native reference grid is constructed before the first rollout; this
startup work runs on the host CPU. `--output` selects a new folder for a repeat.
The tool refuses to overwrite an existing diagnostic directory.

## GPU command: native TB Paper

Use a checkout/worktree containing **`paper-walking-baseline`** and its clean
native Brax submodule. The active TB development checkout is not switched or
modified by the diagnostic. `--tb-root` must point to the existing Paper copy;
its source/model/runtime checks reject a different training implementation.

If an isolated Paper copy does not already exist, prepare one using only the
local repositories. This leaves the active TB source/branch and its Brax
checkout in place. The separate local Brax clone avoids sharing its working
directory with active development:

```bash
git -C ~/projects/toddlerbot worktree add --detach \
  ~/projects/toddlerbot/.worktrees/paper-ppo paper-walking-baseline
git clone --no-hardlinks ~/projects/toddlerbot/toddlerbot/brax \
  ~/projects/toddlerbot/.worktrees/paper-ppo/toddlerbot/brax
git -C ~/projects/toddlerbot/.worktrees/paper-ppo/toddlerbot/brax \
  checkout --detach 14147806f6c25724ba830ba36764df3243889b0d
```

Use that Paper copy in the command below, or substitute an existing Paper
checkout's path. The saved `.venv-paper-blackwell` environment is reused from
the TB project:

```bash
uv run --no-project \
  ~/projects/toddlerbot/.venv-paper-blackwell/bin/python \
  ~/projects/WildRobot2/wr2/tools/diagnose_ppo.py \
  --tb-root ~/projects/toddlerbot/.worktrees/paper-ppo \
  --run-dir ~/projects/toddlerbot/results/toddlerbot_walk_ppo_20261015_194347 \
  --output ~/projects/WildRobot2/results/toddlerbot_walk_ppo_20261015_194347/ppo_learning_diagnostic
```

This directly uses TB's existing runtime/source; WR2 does not vendor TB or its
lookup table. Outputs are placed under WR2 results for the same copy workflow.

## Copy and review

From the Mac's WR2 checkout:

```bash
./scripts/scp_from_remote.sh --linux-pc --results \
  wr2_walking/wr2_paper_step_20261022_143550_seed3/ppo_learning_diagnostic

./scripts/scp_from_remote.sh --linux-pc --results \
  toddlerbot_walk_ppo_20261015_194347/ppo_learning_diagnostic
```

`summary.json` and `batches.jsonl` contain checks and grouped measurements.
Compressed NPZ files retain the per-environment data and the shuffled audit
environment indices. Source hashes, package versions, precision, settings and
the fresh-Adam protocol are recorded in `provenance.json`.

The decision is whether WR2's actual loss/updates consistently reinforce the
backward hip bias, and whether TB shows a different pressure under its native
recipe. Check agreement across batches, checkpoints and command phases,
including actual target changes after the probe. A reward claim needs more than
a large component contribution or one nominal trajectory.

Only after reviewing those GPU measurements should we choose one controlled
200–300M training change, with retained intermediate checkpoints and the existing
held-out walking benchmark. Success requires longer strides and alternating
leading feet alongside improved commanded tracking and acceptable falls.

## Source and references

- Local TB branch `paper-walking-baseline`, runtime source `a4f693d`, native Brax
  `14147806f6c25724ba830ba36764df3243889b0d`.
- TB `mjx_env.py`: moving leg-motor imitation, home-based `.25-rad` targets.
- TB native `brax/training/agents/ppo/losses.py`: GAE, clipped total advantage,
  separate privileged critic and value-loss coefficient.
- WR2 `paper_rewards.py`, `walking_env.py`, `ppo.py` and `train.py`: matching
  action/reward path and native Brax adapter. No adapter bug is asserted here.
- [ToddlerBot paper, Tables VI–VII](https://arxiv.org/html/2502.00893v2).
- [Generalized Advantage Estimation](https://arxiv.org/abs/1506.02438).
- [Proximal Policy Optimization](https://arxiv.org/abs/1707.06347).

Verification: targeted attribution tests exercise both installed WR2 Brax and
the frozen TB fork, including opposing components at clip boundaries, zero
advantages, falls/timeouts, bootstrap and full actor gradients. The WR2 driver
test verifies the actual first native Adam minibatch, with and without domain
randomization. Physical CPU smoke checks are development checks, not new GPU
training results or evidence of better learned walking.

Completed checks: **8 WR2 tests, 6 frozen-fork tests**, and full simulator/PPO
CPU probes at 80M for both robots (4 environments, 8 warmup ticks, one batch,
16 Adam steps). Actor-gradient relative errors were **9.54e−6 WR2 /3.9e−6 TB**.
TB used the saved Python/package versions; WR2's Mac Brax/Flax differ from the
GPU run and are recorded explicitly. These checks establish that the diagnostic
executes and reconstructs the sampled native loss, not a training conclusion.

Development results remain in ignored folders:

- `results/wr2_walking/wr2_paper_step_20261022_143550_seed3/review_20261023/ppo_learning_cpu_verified`
- `results/toddlerbot_walk_ppo_20261015_194347/review_20261023/ppo_learning_cpu_verified2`
