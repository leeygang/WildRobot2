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
uses the installed Brax GAE function. Total advantage and normalization use the
**original native scalar reward**, rather than a sum of regrouped terms. Reward
groups use zero component values and float64 bookkeeping. The seventh component
is the native-total residual, retaining the critic, bootstrap and native rounding.
Every component shares the native total advantage's standard deviation and
**total surrogate's clipping branch**. Independently clipping or normalizing
each reward would produce a different gradient.

Checks require reward reconstruction, advantage reconstruction and full actor
parameter-gradient reconstruction, plus an independent output-gradient check.
Native first-minibatch loss metrics must
also match the actual first Adam step. Large cancelling component gradients are
combined in float64 and cast to float32 once for native network backpropagation.
This is not a set of learned component critics, and
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

### Fix for the first GPU reconstruction failure — 2026-10-24

The first GPU attempt stopped at 80M with parameter-gradient relative error
`3.4111e−4`, despite reward/advantage reconstruction errors of only `9.54e−7`
and `1.81e−5`. The original diagnostic recomputed total advantage from regrouped
rewards and rounded each component gradient to float32 before adding them.
That changed PPO's normalization input and lost precision through cancellation.

Regression cases with an accurate constant critic and nearly steady reward
reproduced errors `5.63e−4` and `.0526` on CPU, from scalar-reward differences
of just `5.96e−8`. The corrected calculation uses the original scalar reward,
native float32 score/clip/entropy derivatives, and float64 deterministic
bookkeeping. Separate JIT stages keep JAX 0.6's dtype handling correct. Globally
enabling x64 would change Brax's default entropy random draws and is not used.

The parameter-gradient tolerance remains **5e−5**; an output-gradient check
uses the same limit. Native loss, matrix precision and fresh-Adam probes remain
the saved recipe. Tests verify native loss/entropy and caller x64 state.
If another numerical reconstruction check fails, the exact input/checkpoint/key are retained in
`failed_minibatch.pkl` under results for reproduction. No result binary is
committed.

After syncing the fix, rerun into a new folder to retain the failed attempt:

```bash
uv run --locked --python 3.10 --extra training python -m wr2.tools.diagnose_ppo \
  --run-dir results/wr2_walking/wr2_paper_step_20261022_143550_seed3 \
  --output results/wr2_walking/wr2_paper_step_20261022_143550_seed3/ppo_learning_diagnostic_v2
```

GPU verification of this fix is pending that rerun. The CPU reproduction
establishes the numerical failure mechanism, not the exact failing GPU batch.
Validation passed **10 WR2 tests and 8 tests against the pinned TB Brax source**.
A separate check used the actual WR2 80M actor/critic, full 975/1485 observation
dimensions and a 256×20 synthetic batch: parameter/output-gradient relative
errors were **2.54e−7 /6.81e−8**, with advantage closure **2.22e−16**. This is a
numerical verification, not a physical walking result.

### Replay of the remaining GPU parameter-gradient mismatch

The v2 attempt has output-gradient error **7.59e−8** and advantage closure
**8.88e−16**, but parameter-gradient error **2.51e−4**. Thus the component
bookkeeping closes; the remaining difference appears during network
backpropagation. The exact saved batch was copied to the Mac. It produces
parameter-gradient error **2.12e−7**, including with the GPU run's Python 3.10,
Brax .14.1, Flax .10.7, JAX .6.2 and other core package versions.

Backend matrix arithmetic is the leading explanation. JAX documents that
GPU `DEFAULT` and `HIGH` may use TensorFloat32, while `HIGHEST` uses Float32.
CPU results do not test GPU precision behavior. TB Paper explicitly sets
`high`; that alone does not establish that it disables TensorFloat32.
References: [JAX precision](https://docs.jax.dev/en/latest/_autosummary/jax.lax.Precision.html)
and [default matrix precision](https://docs.jax.dev/en/latest/_autosummary/jax.default_matmul_precision.html).

The new replay reads `failed_minibatch.pkl`, restores the saved actor/critic and
key, and compares saved/default, `high`, and `highest` precision. It constructs
no walking environment or reference table. After syncing the replay code, run
this on the **same GPU**:

```bash
uv run --locked --python 3.10 --extra training python -m wr2.tools.diagnose_ppo \
  --run-dir results/wr2_walking/wr2_paper_step_20261022_143550_seed3 \
  --replay-minibatch results/wr2_walking/wr2_paper_step_20261022_143550_seed3/ppo_learning_diagnostic_v2/failed_minibatch.pkl
```

It prints three comparison rows and saves
`ppo_learning_diagnostic_v2/replay/report.json`. It reports mismatches rather
than aborting on the first precision, so all three modes can be inspected.
`native_head_backprop_relative_error` also sends the native head gradient,
without component reconstruction, through the same cached network VJP. This
distinguishes sensitivity to tiny cotangent differences from a difference
between the standalone VJP and the native full-loss backpropagation.
The full diagnostic's tolerances and the training recipe have not changed.
GPU replay is required before selecting a further fix to the backpropagation
check; another full physics diagnostic is not needed for that decision.

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
