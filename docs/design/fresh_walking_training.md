# 1B fresh-start walking run

Use `wr2/locomotion/configs/ppo_walking_fresh.yaml` for a long direct-PPO run
without inheriting policy, critic, Adam moments or learned exploration from
the earlier curriculum. This is a controlled initialization experiment, not a
guarantee that a fresh policy will outperform continuation.

The config preserves the verified `ppo_walking_mirror.yaml` environment,
rewards, observations, randomization, servo limits, network and PPO settings.
Only budget, evaluation cadence and run labels differ. Mirror loss stays at
1.0 with no data augmentation, so initialization is the experimental factor;
ToddlerBot supports this optional loss but disables it by default.

## GPU command

After pushing the preparation commit from Mac, run on the GPU machine:

```bash
git pull --ff-only
uv run --extra training python -m wr2.locomotion.train \
  --config wr2/locomotion/configs/ppo_walking_fresh.yaml
```

Do not add `--restore-checkpoint`. Without it, the native trainer creates new
actor/critic weights and an empty Adam state, starts counters at zero, uses
the configured learning rate of 0.00003, and initializes Normal residual
noise std at 0.5 (about 7.16 degrees of target-angle noise before clipping).
Existing result/checkpoint directories are never implicitly restored.

Expected setup:

- 1,000,000,000 requested transitions, 2,048 environments, native RSL-RL PPO.
- 51 evaluations including step zero; checkpoints approximately every 20M
  transitions, plus the selected best actor and final `params.pt`.
- 24,415 full PPO rollout iterations produce 1,000,038,400 actual transitions;
  the small budget overshoot is normal rollout rounding.
- Generated `wr2_fresh_<timestamp>_seed0` run under `results/wr2_walking/`.
- No `Restore:` line; saved `run_config.json` arguments have
  `restore_checkpoint: null`. Step-zero checkpoint counters are zero and
  its optimizer state is empty.
- Standing plus 0.05--0.10 m/s forward commands; nominal fixed-endpoint and
  standing/walking evaluations. Higher speeds are not part of this trial.
- Ten Newton iterations, 5 ms physics with four substeps per 20 ms control
  period, and unchanged hardware-bounded servo envelope.

Allow roughly 7--8 hours if throughput resembles the recent 200M run; fresh
training and system load can change that estimate. Preserve intermediate and
best policies. Judge tracking, contact timing, falls, transitions and servo
demand together rather than promoting the final actor or highest return
automatically. Nominal evaluations are not deployment qualification; confirm
the selected actor independently with full configured noise/randomization.

## ToddlerBot references

- Local `~/projects/toddlerbot/toddlerbot/locomotion/walk.gin:2` requests 1B
  transitions; `ppo_config.py:28` initializes exploration std at 0.5.
- `rsl_rl_config.yml:40` defines optional actor mirror loss coefficient 1.0;
  `train_mjx.py:981` removes it unless symmetry is explicitly enabled.
- [ToddlerBot paper](https://arxiv.org/abs/2502.00893).

The historical 20M mirror-treatment config remains unchanged for reproduction
and controlled continuation. No action/reward redesign or physics/servo-model
change is introduced by this fresh-start config.

Validation: 169 tests pass. An 80-transition Mac MJX/Torch smoke exercised the
fresh training entry point, verified step-zero counters/empty Adam and std 0.5,
and completed one finite native PPO update with a saved final checkpoint.
Its four environments and 20-step episodes validate startup, not gait quality
or deployment readiness.
