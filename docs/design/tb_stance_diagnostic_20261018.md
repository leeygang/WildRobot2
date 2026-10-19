# Matching TB Paper stance diagnostic — 2026-10-18

**The GPU measurement completed and its copied traces passed the independent
audit.** See the [paired result, CAD check and focused next plan](wr2_tb_stance_result_20261018.md).
The setup below records the completed measurement. The diagnostic loads TB Paper's completed
200M policy and uses the same learned, hip-roll-reference and full-leg-reference
controls as the [completed WR2 test](wr2_stance_result_20261018.md). It retains
the frozen local `paper-walking-baseline` source at `a4f693d`, native Brax
`1414780` and the run's actual Blackwell-compatible training environment.

Preparation used a separate local TB checkout at `.worktrees/tb-paper-stance`,
branch `paper-stance-diagnostic`. Git writes in `~/projects/toddlerbot/.git`
failed with `Operation not permitted`, including elevated execution. The active
`main-walking-compare` checkout remains clean. The portable
[text patch](../../patches/toddlerbot-paper-stance.patch) contains the new
diagnostic, four measurement tests and documentation. It contains no robot
assets, checkpoints, ZMP cache, dependency source or binaries.
Prepared TB commit: `b52001c2fa2881776990853bc17651a9160f1869`. The patch is
**45,281 bytes of text** and was verified to apply cleanly to `a4f693d`.

## Prepare the GPU checkout

First sync the WR2 commit containing `patches/toddlerbot-paper-stance.patch` to
the GPU's `~/projects/WildRobot2` checkout. Create a separate TB worktree and
apply the patch:

```bash
cd ~/projects/toddlerbot
git fetch origin paper-walking-baseline
git worktree add -b paper-stance-diagnostic \
  .worktrees/paper-stance-diagnostic \
  a4f693d7cc4544f011dad23d6c353e752a6819ab
git -C .worktrees/paper-stance-diagnostic am \
  ~/projects/WildRobot2/patches/toddlerbot-paper-stance.patch
cd .worktrees/paper-stance-diagnostic
git submodule update --init --recursive
```

This preserves the active TB development checkout and applies the diagnostic
as its own local commit. A Mac worktree is not transferred by pulling WR2.

## Run directly with uv

Reuse the environment that completed Paper training:

```bash
export UV_PROJECT_ENVIRONMENT="$HOME/projects/toddlerbot/.venv-paper-blackwell" UV_PYTHON=3.10
export PYTHONPATH="$PWD/toddlerbot/brax"
uv run --no-sync -m toddlerbot.locomotion.diagnose_stance \
  --run-dir ../../results/toddlerbot_walk_ppo_20261015_194347 \
  --output ../../results/toddlerbot_walk_ppo_20261015_194347/stance_diagnostic \
  --seeds 22102
```

`PYTHONPATH` selects this worktree's pinned Brax source when the shared
environment has an editable installation pointing at the original checkout.
The command uses `200089600/policy`, requires a GPU and refuses to overwrite
a completed `summary.json`. An optional `--checkpoint` selects the retained
`180080640/policy`; use a separate output directory.

The installed packages must match the saved provenance: Python 3.10,
JAX/JAXlib 0.6.2, MuJoCo/MJX 3.2.4, Brax 0.10.5, Flax 0.10.6, Optax 0.2.5,
Chex 0.1.90 and Orbax 0.11.13. The module checks package versions, archived
training/environment source, all saved model/reference hashes and the frozen
reference/controller/robot/utility implementation. `STANCE_DIAGNOSTIC.md` in
the prepared TB checkout contains the full measurement guide and references.
For a missing environment, follow its existing Blackwell installation guide.

## Measurements and scope

- **21 episodes:** three controls × seven commands, 1,000 ticks each at 20 ms.
  Forward/backward .10, forward/backward .05, left/right .05 and standing match
  WR2 in TB-equivalent units. WR2 uses its recorded size/clock normalization.
- **Nominal conditions:** no domain randomization, pushes or white observation
  noise. Reset uses TB's native grounded reference pose with zero yaw and zero
  action-origin bias. The deterministic backlash observation offset remains .02.
- **Native execution:** fixed-home .25-rad bounded actions, one-tick delay,
  joint limits and motor/collision physics. Requested, delayed, delivered,
  actual and reference leg positions are separate raw fields. Delivered-target
  reconstruction is checked at the controller call; `pipeline.ctrl` is torque.
- **Useful movement:** torso-frame XY velocity mean/std and per-tick vector
  error, plus net displacement/velocity in the initial commanded heading frame.
  Compare commanded-axis progress and cross-direction drift.
- **Stance and geometry:** native support at 1 N, conditional double-support
  matching, both active ankle pairs' signed gaps and penetrating constraint
  forces, floor force, sole clearance, spacing reward and landing air-time.
- **Accounting:** include a finite fall tick once; exclude nonfinite ticks and
  later padding. Record duration and reference lookup choice. Signed gaps use
  traced final-substep geometry and offline CCD distance queries, preserving
  TB's saved MJX simulation settings.

Outputs are `summary.json` and 21 compressed `.npz` traces in the TB results
folder, with checkpoint/configuration hashes, source/runtime provenance and
nominal overrides. Copy the complete diagnostic directory back to the Mac.

## Verification and following decision

Four focused tests passed: bounded reference interventions, native delay/limits,
separated/rotated/penetrating/touching-box distances and censored support metrics.
The saved 200M policy completed the **21-episode, 100-tick CPU execution check**
on the actual Paper dependency stack. All measured samples are finite; all
seven learned-policy episodes reached 100 ticks. Nine reference-control
episodes fell, exercising finite-terminal accounting. Delivered-target
reconstruction error is **exactly zero** throughout valid samples.

An independent raw-trace audit checked intervention coordinates, one-tick delay,
joint limits, constant commands/clocks, terminal padding and tracking,
displacement, support and contact summaries. Local ignored artifacts are at
`.worktrees/tb-paper-stance/results/tb_stance_cpu_verified/`, including
`summary.json`, `audit.json` and the raw traces. This short check covers about
2.8 native gait cycles; use the full GPU diagnostic for the walking comparison.

Compare learned TB/WR2 clearance, loaded joint tracking, support timing,
translation, oscillation and drift first. Then compare each paired intervention.
TB's short reference controls also fail, reinforcing the need to assess learned
coordination. Select a surgical correction from the measured delta, then use a
fresh 200–300M Paper-aligned run with intermediate checkpoints. This diagnostic
adds no training reward or actuator changes.

References: [ToddlerBot paper, Tables VI/VII](https://arxiv.org/html/2502.00893v2),
frozen local TB environment/rewards/reference/controller and native PPO;
[WR2 metric definitions](wr2_stance_diagnostic_20261018.md);
[MuJoCo 3.2.4 geometry-distance API](https://mujoco.readthedocs.io/en/3.2.4/APIreference/APIfunctions.html#mj-geomdistance)
and [collision detection](https://mujoco.readthedocs.io/en/3.2.4/computation/index.html#collision-detection).
