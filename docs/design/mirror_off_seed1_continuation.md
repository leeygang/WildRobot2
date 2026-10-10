# 200M continuation from mirror-off best80 — 2026-10-10

## Confirmed result and decision

The mirror-off run `wr2_mirror_off_20261010_083343_seed0` completed
200,007,680 additional transitions. Source model/Adam/LR/counters/Torch RNG
restoration was exact; all 11 checkpoints and logged metrics were finite.
Its selected 80M checkpoint improves tracking modestly, while its final 200M
checkpoint regresses. Continue **best80**, not final200; keep the original
920M policy as the comparison reference, not an overwritten artifact.

Paired native Mac MJX evaluations enabled configured observation noise,
model variation and reset/local servo variation: four policies, four
conditions, three seeds and 128 episodes per seed, totaling 6,144 episodes.
Conditional walking/start/stop metrics use active sample counts, not
episode-wide means diluted by standing.

| Held-out metric | Original 920M | Mirror-off best80 | Mirror-off final200 |
|---|---:|---:|---:|
| Fixed 0.10 forward speed | 0.0798 m/s | 0.0819 m/s | 0.0743 m/s |
| Fixed 0.10 speed MAE | 0.0289 m/s | 0.0265 m/s | 0.0314 m/s |
| Scripted walking speed MAE | 0.0355 m/s | 0.0329 m/s | 0.0417 m/s |
| Scripted stop-window MAE | 0.0405 m/s | 0.0487 m/s | 0.0395 m/s |
| Scripted falls | 0/384 | 0/384 | 27/384 |
| Random-command falls | 0/384 | 0/384 | 5/384 |

Best80 survived all 1,536 candidate episodes without falls/non-finite state.
Its tracking improvement repeated across all three seeds. However, stopping
and heading worsened: fixed-0.10 heading MAE increased 2.88 to 3.86 degrees.
Speed remains below the 0.09 m/s acceptance target and MAE above 0.02 m/s.
Scripted target clipping remains 8.87%; continuous-walking clipping is zero.
These reused validation seeds are not an untouched final test set, and
simulation torque aggregates do not qualify hardware thermal safety.

All 27 final scripted falls occurred shortly after the **first** walking
start, at steps 204–252. No new PPO/physics/initial-stance root cause was
established. This single training seed does not prove that removing mirror
loss always improves walking or that mirror loss caused the original plateau.

## Controlled follow-up

Use `wr2/locomotion/configs/ppo_walking_mirror_off_seed1.yaml`: another
200M additional transitions, changing only the environment/evaluation seed
from 0 to 1 and version labels. Historical configs remain unchanged.

Restore actor, critic, Adam, learned exploration, adaptive learning rate,
counters and saved Torch RNG. Seed 1 changes the JAX reset, dynamics variation,
sensor/command and inline-evaluation streams. It does **not** reset the
restored Torch action-sampling RNG or start a new independent PPO policy.
Different step-zero evaluation results are expected with different JAX keys;
step-zero model and optimizer tensors must still match the source exactly.

No reward, action, observation, network, servo-envelope, command distribution
or physics change is introduced. Continue Normal residual actions, strict
Gaussian velocity tracking, standing plus forward 0.05–0.10 m/s, 20% standing,
150-step command resampling, and 5 ms physics with four substeps per control
step. Actor mirror loss stays disabled, matching ToddlerBot's default
(`toddlerbot/locomotion/train_mjx.py:981`, `--symmetry` default off at `:1264`).
Robot dimensions, speed/timing calibration and hardware limits are unchanged.
Reference: [ToddlerBot paper](https://arxiv.org/abs/2502.00893); the local
ToddlerBot implementation remains the code source of truth.

## GPU command

Push the preparation commit from Mac with `git push`, then run in
`/home/leeygang/projects/WildRobot2` on the GPU machine:

```bash
git pull --ff-only
uv run --extra training python -m wr2.locomotion.train \
  --config wr2/locomotion/configs/ppo_walking_mirror_off_seed1.yaml \
  --restore-checkpoint results/wr2_walking/wr2_mirror_off_20261010_083343_seed0/checkpoints/000080035840.pt
```

Source SHA256:
`e8e461305b242d09ec1c71264e8239912d993d21ca79238fc10d93b36bdaedd7`.
The checkpoint is an existing GPU result, not a Git-tracked file.

Expected setup:

- Generated `wr2_mirror_off_<timestamp>_seed1` ID; no manual run ID needed.
- 2,048 environments, 20-step rollouts, 4,883 additional native RSL-RL updates:
  200,007,680 actual additional transitions and 11 evaluations/checkpoints.
- Source counters 1,000,079,360 transitions / 24,416 updates; final cumulative
  counters 1,200,087,040 transitions / 29,299 updates. New checkpoint filenames
  count **run-local added transitions**, not cumulative lifetime transitions.
- Restored LR 0.00001 and mean action std about 0.02607, not fresh defaults.
- Approximately 85–90 minutes if throughput matches the completed run.

## Candidate confirmation

Keep intermediate, selected-best and final checkpoints. Inline evaluations
disable actor noise and omit vector model variation, while retaining local
reset/servo variation; their selection score is not transition qualification.

Before promotion, repeat the [fixed endpoint, scripted transition and random
training-command checks](mirror_off_continuation.md#held-out-candidate-confirmation)
using the generated **seed1 run's saved `training_config.yaml`**. Compare
against best80 with identical keys and full configured noise/model/local
variation. Reuse seeds 707/808/909 for paired validation; reserve additional
unused seeds for final qualification. Check stopping and heading as well as
speed, contact timing, survival, target clipping and servo demand. Do not
assign a fixed walking score to standing-containing schedules.

The existing acceptance thresholds remain unchanged; no automatic deployment
promotion or hardware motion is part of this training preparation.

## Preparation verification

- All 201 full-suite tests and three focused continuation tests pass. They
  cover seed-only config parity, repeatable distinct JAX streams, CLI source
  selection, and exact native model/Adam/LR/counter/Torch-RNG restoration.
- Ruff and `git diff --check` pass.
- An 80-transition, four-environment Mac MJX/Torch smoke restores the actual
  best80 checkpoint exactly at step zero, including populated Adam and CPU
  Torch RNG. It completes one finite native PPO update, advances cumulative
  counters to 1,000,079,440 / 24,417, saves seed 1 and contains no mirror loss.
  Its 20-step episodes verify startup only, not walking quality.
- No GPU training or physical servo motion was launched during preparation.
