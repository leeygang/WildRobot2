# 200M continuation from seed1 best120 — 2026-10-10

## Confirmed result and decision

Continue `wr2_mirror_off_20261010_125338_seed1/checkpoints/000120012800.pt`,
not its slower final200 actor. Retain the preceding best80 policy as fallback.
The selected checkpoint matches `best_params.pt` exactly. Full-state restore,
all 11 checkpoints and logged numeric metrics passed the integrity checks.

Paired full-variation native Mac evaluation covered three policies, four
conditions, three seeds and 128 episodes per seed: 4,608 episodes. Every
episode completed 1000 steps without a fall or non-finite state.

| Held-out metric | Source best80 | Seed1 best120 | Seed1 final200 |
|---|---:|---:|---:|
| Speed at 0.10 command | 0.0819 m/s | 0.0829 m/s | 0.0796 m/s |
| Speed MAE at 0.10 | 0.0265 m/s | 0.0259 m/s | 0.0285 m/s |
| Heading MAE at 0.10 | 3.86 degrees | 2.99 degrees | 2.87 degrees |
| Scripted start-window MAE | 0.0939 m/s | 0.1017 m/s | 0.1019 m/s |
| Scripted stop-window MAE | 0.0487 m/s | 0.0373 m/s | 0.0395 m/s |

Best120 gives a small fixed-speed gain and better heading/stopping, but slower
starts. Scripted/random walking-only MAE is slightly worse, not improved
uniformly. Speed reaches 82.9% of the 0.10 command, below the 90% target;
MAE remains above 0.02 m/s. Scripted clipping is 8.90%, random clipping 3.79%.
These reused validation seeds and simulation torque aggregates are not final
walking or hardware qualification. No new restore/metric/solver bug was found.

## Controlled follow-up

`ppo_walking_mirror_off_seed2.yaml` changes only seed 1 to 2 and version labels.
Train another 200M **additional** transitions with the same TB-aligned recipe.
Historical configs remain unchanged. No reward, action, observation, network,
command-distribution, servo-envelope, robot-size or physics change is made.

Restore actor, critic, Adam, adaptive LR, learned exploration, counters and
saved Torch RNG. Seed 2 changes JAX reset, dynamics variation, sensor/command
and inline-evaluation streams, not the restored Torch sampling RNG. This is
a full-state continuation, not an independent cold-start PPO trial. Different
step-zero evaluation metrics are expected; model/Adam must match the source.

Actor mirror loss remains off, matching ToddlerBot's default
(`toddlerbot/locomotion/train_mjx.py:981`). Keep Normal residual actions,
strict Gaussian velocity tracking, standing plus forward 0.05–0.10 m/s,
20% standing, 150-step command resampling and 5 ms physics/four substeps.
Existing WR2 hardware, observation, parallelism and solver exceptions remain
unchanged. Reference: [ToddlerBot paper](https://arxiv.org/abs/2502.00893),
with the local ToddlerBot implementation as the active-code source of truth.

## GPU command

Push the preparation commit from Mac using `git push`, then run in
`/home/leeygang/projects/WildRobot2` on the GPU machine:

```bash
git pull --ff-only
uv run --extra training python -m wr2.locomotion.train \
  --config wr2/locomotion/configs/ppo_walking_mirror_off_seed2.yaml \
  --restore-checkpoint results/wr2_walking/wr2_mirror_off_20261010_125338_seed1/checkpoints/000120012800.pt
```

The existing GPU checkpoint is not Git-tracked. Source SHA256:
`5ef1bbebcbc9818a7e79185681865c8195830c61828b14b4fed043bc3ccae92e`.

Expected setup:

- Generated `wr2_mirror_off_<timestamp>_seed2` ID, 2,048 environments.
- 4,883 additional native PPO updates / 200,007,680 actual transitions.
- Eleven evaluations with retained checkpoints approximately every 20M.
- Source cumulative counters 1,120,092,160 transitions / 27,346 updates;
  final cumulative counters 1,320,099,840 / 32,229. Checkpoint filenames
  count this run's added transitions, not lifetime transitions.
- Restored LR 0.00001 and mean residual std about 0.02385, not fresh defaults.
- Approximately 85–90 minutes if throughput matches the completed run.

## Candidate confirmation

Keep intermediate, selected-best and final checkpoints; final is not
automatically best. Inline evaluation disables actor noise and omits vector
model variation, while retaining local reset/servo variation. Its fixed
endpoint selection score is not standing/walking qualification.

Before promotion, repeat the [fixed, scripted transition and random-command
checks](mirror_off_continuation.md#held-out-candidate-confirmation) with the
new run's saved `training_config.yaml` and compare against source best120.
Use identical keys and full configured variation. Reuse seeds 707/808/909
for paired validation, reserving additional unused seeds for qualification.
Prioritize starts as well as tracking, stopping, survival, contact timing,
target clipping and servo demand. Acceptance thresholds are unchanged.
No automatic deployment promotion or physical hardware motion is included.

## Preparation verification

- All 203 full-suite tests and five focused continuation regressions pass,
  including seed-only parity, CLI source selection, distinct/repeatable JAX
  streams and native full-state restoration. Ruff and diff checks pass.
- An 80-transition, four-environment Mac MJX/Torch smoke restores the actual
  best120 model/Adam/LR/counters/CPU Torch RNG exactly at step zero. It completes
  a finite native PPO update, saves seed 2 without mirror loss and advances
  cumulative counters to 1,120,092,240 / 27,347.
- The smoke's 20-step episodes verify startup, not walking quality. No GPU
  training or physical hardware motion was started during preparation.
