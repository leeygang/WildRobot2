# Pre-restart controller recovery diagnostic

## Question and controls

Can the older controller repair the newer controller's walking/standing state
before the second walking restart? The previous exact-restart handover still
fell in 5/384 episodes, versus 7/384 when the newer controller continued. That
does not isolate a posture problem or prove the newer controller cannot restart.

Use the same old/source and Newton-10 nominal-best actors, saved control config,
0.10 m/s walking command, ten Newton iterations and full noise/randomization
as the preceding matched-state tests. Keep the 20-second command schedule:
stand 0--3 s, walk 3--9 s, stand 9--12 s, walk 12--17 s, stand 17--20 s.

Swap actor weights at steps 450/550/585/600: the beginning of the preceding
stand, 1.0 s before restart, 0.3 s before restart, and exactly at restart.
Run both newer-to-older and older-to-newer directions, plus uninterrupted
old/old and new/new controls. Each case uses the same 128 model/reset keys
at each of seeds 707/808/909. These are paired episodes, not independent
draws for each intervention.

No simulator or history reconstruction: preserve pose, velocities, warm-start,
sensor state/RNG, all 15 observation frames, reference heading, gait clock and
delayed actions. The first receiver action still has the one-step command
delay. Check every recorded field against the donor control before the swap.
The seed-707 self-controls must also match the independent original replay.

The 0.3 s lead equals the actor's 15-frame history horizon, but it is not a
history-only intervention: physical motion and filters also evolve. Even a
longer lead does not isolate posture from velocity or history.

## Reproduction on Mac

Use new output directories; the tool refuses to replace earlier results.

```bash
uv run python -m wr2.tools.check_walking_stance \
  --checkpoints \
    results/wr2_walking/wr2_ppo_20261007_152924_seed0/best_params.pt \
    results/wr2_solver_ab/newton10_seed0/wr2_ppo_20261007_215905_seed0/best_params.pt \
  --config results/wr2_solver_ab/newton10_seed0/wr2_ppo_20261007_215905_seed0/training_config.yaml \
  --switch-steps 450,550,585,600 --seeds 707 --num-envs 128 \
  --reference-traces results/wr2_walking_diagnostics/investigation_20261008_full707_v2 \
  --output results/wr2_walking_diagnostics/recovery_standing_20261008_seed707 \
  --allow-cpu

uv run python -m wr2.tools.check_walking_stance \
  --checkpoints \
    results/wr2_walking/wr2_ppo_20261007_152924_seed0/best_params.pt \
    results/wr2_solver_ab/newton10_seed0/wr2_ppo_20261007_215905_seed0/best_params.pt \
  --config results/wr2_solver_ab/newton10_seed0/wr2_ppo_20261007_215905_seed0/training_config.yaml \
  --switch-steps 450,550,585,600 --seeds 808,909 --num-envs 128 \
  --output results/wr2_walking_diagnostics/recovery_standing_20261008_seeds808909 \
  --allow-cpu
```

On GPU omit `--allow-cpu`. No training or deployment behavior changes.
The existing independent 808/909 replay contains only the newer actor (indexed
as policy 0), so it cannot be passed as a two-actor `--reference-traces` folder.
Compare the newer self-controls to that replay and the older self-controls to
the preceding crossed-state controls separately when analyzing the NPZ files.

## Interpretation

Report falls before restart and all episodes still alive at restart separately
from falls in the following five-second walking window. Never label a case
safe merely because its difficult episodes already fell during standing.
Check whole-episode length, non-finite states, walking speed MAE and active
target clipping as well as restart falls. Compare the same failed episode
indices across controls and handovers, not only aggregate counts.

Pre-start foot geometry/load statistics are last-second means. A 0.3 s lead
therefore mixes 0.7 s of donor behavior with 0.3 s of receiver behavior; use
the NPZ trajectories for instantaneous comparisons. Contact-force samples
are the last physics substep, not control-period integrated impulses.

If earlier old-controller takeover removes failures without preceding falls,
the tested states are recoverable with preparation time. If takeover exactly
at restart still fails, this supports a transition-preparation/recovery issue,
not an inherently impossible pose. The reverse direction checks whether the
newer controller can create vulnerability during a short stand. A policy swap
can itself introduce a transient, and rare count differences are not enough
to rank controller robustness universally. This diagnostic does not establish
mirror loss as the root-cause fix or justify phase/pose resets in training.

## Observed results, 2026-10-08

All 30 batches completed on Mac: ten cases at each of three seeds, 128 episodes
per batch (3,840 paired rollouts). These represent the same 384 randomized
conditions across cases, not 3,840 independent samples. The canonical model
and actor hashes match the preceding investigation:

```text
model: cbbb5e0cfe0d7c3663a52acc18a502e2e2052e0ccc1c6d9e0d4c95f06b4abacd
older: 7b25653d5d5b305884d923d15664ca8bb25e41fc1865707212db0333f7e1bfe1
newer: 78092e0b092d024be253dc9524b589da4fcca1b53c20ccc1567be72551807b5f
```

Every crossed run matches every recorded donor-prefix field exactly. Both
seed-707 self-controls match the independent original replay; the newer
808/909 controls match their independent replay, and older 808/909 controls
match the preceding self-replays. There are no pre-restart falls or non-finite
states in any case. All 384 episodes therefore remain eligible at restart.
The uninterrupted controls reproduce older 0/384 and newer 7/384 falls.

| Lead before restart | Newer walks, older takes over and remains in control | Older walks, newer takes over and remains in control |
|---|---:|---:|
| 3.0 s, entire preceding stand | 0/384 falls | 3/384 falls |
| 1.0 s | 2/384 | 0/384 |
| 0.3 s | 5/384 | 0/384 |
| 0 s, at restart | 5/384 | 0/384 |

Newer-to-older failure indices by seed (707 / 808 / 909):

- 3.0 s: none / none / none.
- 1.0 s: 33 / none / 22.
- 0.3 s: 33,55,111 / none / 22,81.
- 0 s: 33,111 / 61 / 77,81.

Older-to-newer with the entire stand fails in seed 707 episodes 33,55 and
seed 808 episode 54; the latter did not fail under uninterrupted newer control.
The small count differences and newly affected episodes prevent a claim of
monotonic per-episode rescue or universally superior recovery by one actor.

Last-second pre-restart means, averaged over the equally sized seed batches:

| Case | Left sampled load share | Signed left-minus-right forward offset | Lateral foot separation |
|---|---:|---:|---:|
| Older uninterrupted | 42.1% | -23.71 mm | 87.63 mm |
| Newer uninterrupted / takeover exactly at restart | 24.2% | +3.87 mm | 83.79 mm |
| Newer to older, 3.0 s lead | 43.8% | -19.88 mm | 88.47 mm |
| Newer to older, 1.0 s lead | 27.7% | +3.20 mm | 85.28 mm |
| Newer to older, 0.3 s lead | 24.7% | +3.80 mm | 83.91 mm |
| Older to newer, 3.0 s lead | 24.2% | +0.81 mm | 82.54 mm |

At the instantaneous pre-restart boundary, newer-to-older 3.0 s lead also
changes mean pitch from -2.32 to -0.74 degrees and knees from -1.11/+10.81
to -1.65/+14.66 degrees, closer to the older actor's -0.80-degree pitch and
-1.99/+14.96-degree knees. These jointly changing quantities are descriptive,
not an isolated posture/load intervention or a reason to force 50/50 loading.

For the second walking interval only, newer-to-older 3.0 s lead has forward
speed 0.08531 m/s and MAE 0.03386 m/s, versus older uninterrupted 0.08588 /
0.03362 and newer uninterrupted 0.07769 / 0.03355. These are pooled valid
walking samples; neither standing nor post-fall resets dilute the denominator.
Whole-episode active target clipping is 1.325% for the 3.0 s repair case,
1.121% for older control and 1.942% for newer control. The repair case survives
all 1000 steps, but its MAE remains above the 0.03 m/s gate. This is not policy
qualification or evidence that a controller-switch deployment design is needed.

Conclusion: the newer actor's walking state is recoverable when the older
actor handles the full preceding stand. Conversely, the newer actor can
introduce restart vulnerability during standing even after an older-controller
walk. Clearing approximately one input-history horizon through ordinary
control is not sufficient to eliminate all failures. Physical posture,
velocities, delayed commands and persistent filters still evolve together;
the causal variable remains unisolated. This supports investigating learned
standing/transition preparation without changing the running actor mirror-loss
experiment. It does not prove mirror loss will fix the issue.

Artifacts: `results/wr2_walking_diagnostics/recovery_standing_20261008_seed707/`
and `recovery_standing_20261008_seeds808909/`, with `summary.json` and all
per-case NPZ trajectories. Regression validation: 166 tests and 30 subtests
pass. No training, servo or model configuration was changed.

## Code and strategy references

- WR2 `wr2/tools/check_walking_stance.py::switched_unroll`: actor-only switch,
  Brax `generate_unroll` key order, retained state and exact donor-prefix checks.
- Local ToddlerBot `toddlerbot/locomotion/mjx_env.py:1537`: delayed actions;
  `:1716` retains prior action and updates commands without a stand-to-walk
  pose/phase reset. This diagnostic deliberately leaves that strategy intact.
- Local WR1 `training/algos/ppo/ppo_core.py:538`: actor observation/action
  reflection is a separate loss mechanism, not state repair during deployment.
- [Previous matched-state investigation](walking_solver_followup_20261008.md).
- [Actor mirror-loss experiment](actor_mirror_loss_experiment.md), which remains
  unchanged while this read-only policy diagnostic runs.
