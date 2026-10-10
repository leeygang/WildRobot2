# 200M mirror-off continuation — 2026-10-10

## Controlled comparison

Continue the original 920M checkpoint using
`wr2/locomotion/configs/ppo_walking_mirror_off.yaml`. The only training-behavior
change versus the completed mirror-on 200M continuation is
`ppo.mirror_loss_coeff: 1.0 -> 0.0`. Budget and cadence match that control:
200M additional transitions, 11 evaluations including step zero.

Keep the same rewards, residual actions, observations, command distribution,
network, adaptive-KL PPO settings, randomization, servo envelope, and Newton-10
physics. Restore actor, critic, Adam, learned exploration, adaptive learning
rate, cumulative counters, and saved Torch RNG; do not reset exploration or
continue the regressed +200M final policy. Historical configs are unchanged.

ToddlerBot removes `symmetry_cfg` unless `--symmetry` is explicitly requested
(`toddlerbot/locomotion/train_mjx.py:981`, flag default at `:1264`). Mirror loss
is supported but optional, not its default PPO strategy. This comparison
tests that difference; it does **not** establish that mirror loss caused the
tracking plateau or random-command falls.

Reference: [ToddlerBot paper](https://arxiv.org/abs/2502.00893), with the local
project implementation in `~/projects/toddlerbot` as the source of truth.
Robot size/timing, reward kernels and hardware limits do not change here.

## GPU training command

Push the preparation commit from Mac, then run in the GPU repository:

```bash
git pull --ff-only
uv run --extra training python -m wr2.locomotion.train \
  --config wr2/locomotion/configs/ppo_walking_mirror_off.yaml \
  --restore-checkpoint results/wr2_walking/wr2_fresh_20261009_104213_seed0/checkpoints/000920043520.pt
```

The restore file must have SHA256
`b1aa97b73a6b4f0fb5d61f795e800ab87645f66534f579daed79c6b754566992`.
It is an existing GPU result, not a Git-tracked asset.

Expected setup:

- Generated `wr2_mirror_off_<timestamp>_seed0` run ID; no manually supplied ID.
- Native RSL-RL PPO, 2,048 environments, 20-step rollouts, 4,883 updates;
  actual added transitions 200,007,680. The budget is **additional**, not a
  cumulative 200M target.
- Eleven evaluations and retained checkpoints, about every 20M transitions.
  Checkpoint filenames count run-local transitions; their stored cumulative
  counters start at 920,043,520 and end at 1,120,051,200.
- Restored adaptive learning rate 0.00001 and action std approximately 0.0292,
  not the fresh-config initialization values. Step-zero weights/Adam match
  the source. `mirror_loss_coeff` in newly saved checkpoints is 0.0.
- Standing plus forward 0.05–0.10 m/s; 20% standing, resampling every 150 steps.
- Approximately 90 minutes if throughput matches the completed control.
  Preserve intermediate, selected-best and final policies; final is not
  automatically best.

Training checkpoint evaluations remain unchanged for a controlled comparison:
actor noise is off and vectorized model variation is omitted, but configured
reset/local actuator variation is still on. These are not fully nominal or
deployment qualification.

## Held-out candidate confirmation

Before promotion, compare source and candidate using identical seeds and full
configured observation noise, model variation and reset/local actuator
variation. Keep fixed endpoints and the scripted two-restart test; add random
commands sampled from the **unaltered training distribution**.

With 150-step command resampling and a 36-step gait cycle, command changes
occur at six gait-clock phases. Random commands can exercise restarts around
180 degrees, which the standard scripted starts around 60/240 degrees miss.
This is an evaluation-coverage correction, not a change to training commands.

Set `WALK_RUN` to the generated run directory. For each candidate checkpoint
(start with `best_params.pt`, also check `params.pt` and promising intermediate
checkpoints), run:

```bash
WALK_RUN=results/wr2_walking/wr2_mirror_off_REPLACE_WITH_RUN_ID_seed0
WALK_CHECKPOINT="$WALK_RUN/best_params.pt"

uv run --extra training python -m wr2.locomotion.evaluate \
  --config "$WALK_RUN/training_config.yaml" \
  --checkpoint "$WALK_CHECKPOINT" --seeds 707,808,909 --num-envs 128 \
  --commands 0.05,0.10 --output "$WALK_RUN/held_out_fixed.json"

uv run --extra training python -m wr2.locomotion.evaluate \
  --config "$WALK_RUN/training_config.yaml" \
  --checkpoint "$WALK_CHECKPOINT" --seeds 707,808,909 --num-envs 128 \
  --transitions --commands 0.10 --output "$WALK_RUN/held_out_transitions.json"

uv run --extra training python -m wr2.locomotion.evaluate \
  --config "$WALK_RUN/training_config.yaml" \
  --checkpoint "$WALK_CHECKPOINT" --seeds 707,808,909 --num-envs 128 \
  --random-commands --output "$WALK_RUN/held_out_random.json"
```

On Mac add `--allow-cpu`; these commands require no physical hardware.
Repeat the same commands with the original `000920043520.pt` and separate
output filenames for paired comparison. Random-command keys match the prior
review's seed convention; fixed 0.10 uses seed + 100000 as the second endpoint.

Reports identify the actual schedule and all active variation sources. Use
`conditional_tracking` for walking/start/stop/standing errors; episode-wide
errors are diluted by standing. For pooling seeds, sum category error sums
and sample counts before dividing, rather than averaging conditional means.
Scripted/random reports have `walking_score: null`: standing-containing
contact/double-support aggregates cannot be used as a fixed walking score.
Raw contact, clipping, finite-state and servo metrics remain available.

## Confirmed control and decision

The completed mirror-on run `wr2_fresh_20261009_205016_seed0` restored the
source exactly and finished 200,007,680 transitions; all 11 checkpoints were
finite. Its selected best stayed at step zero.

Paired held-out source versus final control (384 episodes per mode):

| Metric | Original 920M | Mirror-on +200M |
|---|---:|---:|
| Fixed 0.10 speed | 0.0798 m/s | 0.0770 m/s |
| Fixed 0.10 speed MAE | 0.0289 m/s | 0.0300 m/s |
| Scripted start-window MAE | 0.0976 m/s | 0.0890 m/s |
| Scripted stop-window MAE | 0.0405 m/s | 0.0340 m/s |
| Scripted all-step target clipping | 8.87% | 8.79% |
| Random-command falls | 0/384 | 2/384 |

Fixed/scripted survival remained 1000 steps with no falls. Two random-command
falls are a warning, not proof of broad collapse. A physically replayed
180-degree restart failure had no target clipping during deterioration; no
PPO, clipping or solver root cause was established. Slightly higher return
included less lateral oscillation and lower movement costs, not better
forward tracking. Aggregate torque gates are not hardware thermal approval.

Retain 920M as champion until independently confirmed improvement. Judge
tracking, survival, contact timing, starts/stops and servo demand together.
Final walking targets still include at least 90% of commanded speed, at most
0.02 m/s MAE, at least 995 steps, at most 1% falls, at least 80% contact match,
and at most 2% clipping. A failed mirror-off comparison is not permission to
declare mirror loss the cause or add WR2-specific rewards/actions.

## Preparation verification

- Typed config equals the completed mirror-on control except mirror coefficient
  and version/run labels; regression coverage also checks budget and cadence.
- 197 full-suite tests and six focused evaluation regressions pass, including
  schedule conflicts, unchanged random-command sampling, paired seed keys,
  six command-clock phases, and non-standing-diluted conditional metrics.
- Ruff and `git diff --check` pass.
- An 80-transition, four-environment Mac MJX/Torch smoke restores the **actual
  920M source** exactly at step zero (model, Adam, LR, counters and CPU RNG),
  completes one native PPO update without mirror loss, and saves finite model
  and Adam tensors. It uses 20-step episodes solely for startup validation.
- A two-environment 1000-step random-command CLI smoke confirms full configured
  variation, starts/stops, valid conditional reports and no mixed-mode walking
  score. Neither smoke establishes walking quality or deployment safety.
- No GPU training or physical servo motion was started during preparation.
