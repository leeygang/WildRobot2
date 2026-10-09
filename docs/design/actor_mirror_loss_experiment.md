# Actor mirror-loss experiment

## Question and controls

Does ToddlerBot's optional actor symmetry loss prevent WR2's learned
walking/standing behavior from producing fragile restart states, without losing
forward tracking? This is a hypothesis test, not a proven root-cause fix.

Use `wr2/locomotion/configs/ppo_walking_mirror.yaml`, a complete frozen copy of
the saved 20M Newton-10 control configuration. The only training intervention is
`ppo.mirror_loss_coeff: 1.0`. Version/output labels identify the treatment.
Default `ppo_walking.yaml` and historical snapshots have coefficient zero.

Keep Normal residual actions, walk_home, rewards, observations/noise, domain
randomization, PPO layout and servo limits unchanged. Physics remains 5 ms with
four substeps and ten Newton iterations. Commands remain standing and forward
0.05--0.10 m/s; this experiment does not introduce lateral/backward walking.

Restore the **shared source**, not the regressed 16M/final policy:

```text
results/wr2_walking/wr2_ppo_20261007_152924_seed0/best_params.pt
SHA256: 7b25653d5d5b305884d923d15664ca8bb25e41fc1865707212db0333f7e1bfe1
```

Actor/critic parameters, Adam moments, adaptive learning rate, counters and
Torch RNG are restored without migration. No parameters or input channels are
added. As in the existing solver A/B, MJX environments start fresh from seed 0;
the checkpoint does not contain live simulator/filter state.

The existing control is
`results/wr2_solver_ab/newton10_seed0/wr2_ppo_20261007_215905_seed0`.
Compare its intermediate checkpoints as well as nominal best/final; return alone
does not establish a safer policy.

## Implementation and parity

Use native RSL-RL 2.3.3 `PPO.update` with `use_mirror_loss=true`, coefficient
1.0, and `use_data_augmentation=false`. Its callback temporarily constructs
original-plus-reflected actor inputs to compute the loss; it does **not**
augment PPO transitions, advantages, returns or critic observations.

The native term is:

```text
mean_squared_error(actor(mirror(observation)),
                   stop_gradient(mirror(actor(observation))))
```

Reflect all fifteen history frames, including phase shifted by 180 degrees,
commands, measured positions/velocities, prior actions, gyro, projected gravity
and IMU-relative heading. Joint signs come from WR2's neutral model axes, not
ToddlerBot/WR1 hardcoded signs. Training and existing JAX diagnostics share the
definitions in `wr2/locomotion/symmetry.py`; Torch/JAX parity and physical
reflection regression tests guard the implementation.

Raw, explicitly scaled observations are required. Empirical normalization is
already unsupported by WR2's RSL adapter and is rejected rather than reflecting
normalized channels incorrectly. The adapter also rejects non-leg policies or
non-mirrored safe limits/home. The loss is soft: it does not require equal
instantaneous foot loading or make both legs execute the same action.

Logs expose raw `training/mirror_loss`, its weighted contribution, and a total
loss including that contribution. Checkpoints record the active coefficient;
loading a control checkpoint does not disable the treatment coefficient.

## GPU command

Push the local commit from Mac, then on the GPU machine:

```bash
git pull --ff-only
uv run --extra training python -m wr2.locomotion.train \
  --config wr2/locomotion/configs/ppo_walking_mirror.yaml \
  --restore-checkpoint results/wr2_walking/wr2_ppo_20261007_152924_seed0/best_params.pt
```

The YAML requests 20M additional transitions, six evaluations, 2,048
environments, and generated `wr2_mirror_*` run IDs under `results/wr2_walking/`.
Confirm the banner says actor-only coefficient 1 and Newton iterations 10,
and the recorded restore hash matches above. Do not cold-start the treatment.

## Evaluation and decision

Training-time endpoint/transition evaluations retain the control's nominal
evaluation path. They are not randomized deployment qualification. After
copying results, independently evaluate **every saved checkpoint**, including
step zero, with full noise/dynamics variation at seeds 707/808/909 and 128
environments per seed. Run fixed-command and scripted-transition evaluations
separately. Use identical command lists/key conventions for source/control/
treatment; multi-command evaluation offsets RNG keys by command index.

For a selected candidate, GPU held-out commands are:

```bash
uv run --extra training python -m wr2.locomotion.evaluate \
  --config wr2/locomotion/configs/ppo_walking_mirror.yaml \
  --checkpoint <candidate.pt> --commands 0.05,0.10 \
  --seeds 707,808,909 --num-envs 128 --output <fixed_report.json>

uv run --extra training python -m wr2.locomotion.evaluate \
  --config wr2/locomotion/configs/ppo_walking_mirror.yaml \
  --checkpoint <candidate.pt> --commands 0.10 --transitions \
  --seeds 707,808,909 --num-envs 128 --output <transition_report.json>
```

On Mac, append `--allow-cpu`. Keep transition walking-only speed-error
denominators; standing samples must not dilute tracking error.

Required outcome: randomized transitions reach episode length >=995 and fall
rate <=1%, action/physical target clipping <=2%, and no non-finite state.
Fixed-command speed MAE must meet the existing 0.03 m/s gate and not regress
against the shared source, especially at 0.10 m/s. Contact match must retain
the existing 80% gate. Existing torque/exposure safety gates remain mandatory;
this experiment makes no new servo deployment claim.

Reject a balance-only improvement that reduces speed. A lower mirror MSE, more
equal standing load, or a larger reward is not itself success. If transition
fragility persists, do not continue this treatment unchanged or assume that
symmetry was the root cause.

## References

- Local ToddlerBot `locomotion/rsl_rl_config.yml:40`: actor mirror coefficient
  1.0 and no transition augmentation. `train_mjx.py:1264` defaults `--symmetry`
  off: this is an **optional** TB capability, not its default training strategy.
- Installed/public [RSL-RL PPO](https://github.com/leggedrobotics/rsl_rl/blob/v2.3.3/rsl_rl/algorithms/ppo.py):
  mean-action mirror MSE with detached target, added to optimizer loss.
- [Yu et al., Learning Symmetric and Low-energy Locomotion (2018)](https://arxiv.org/abs/1801.08093):
  symmetry in the loss, not reward. That paper also uses assisted curriculum;
  its results do not prove mirror loss alone will solve WR2 walking.
- WR2's preceding diagnostics:
  [solver/transition investigation](walking_solver_followup_20261008.md).

Development smoke/regression checks validate implementation and restore
semantics only. The GPU treatment has not been run or qualified yet.

Validation: 165 tests and 24 subtests pass, including native detached-target
gradients, all-frame Torch/JAX and clipped-target parity, legacy configuration
loading, and unchanged action/Adam/LR/RNG restore. A four-environment Mac smoke
restored the real shared source and completed 80 transitions plus a native PPO
mirror-loss update. Its step-zero checkpoint matches every source model/Adam
tensor, parameter group, LR, counter and CPU RNG exactly; the final model is
finite. Its 20-step episodes and small-batch KL are not walking-quality or
full-batch optimizer calibration evidence.
