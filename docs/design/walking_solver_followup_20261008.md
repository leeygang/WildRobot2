# Solver A/B follow-up investigation

This is an investigation record, not deployment qualification or approval to
promote a checkpoint. PPO, rewards, actuator limits, action semantics and the
canonical one-iteration configuration are unchanged.

## Controls and measurements

Both 20M solver arms start from the same RSL-RL checkpoint, including optimizer,
learning rate and RNG state. Source checkpoint SHA-256 is
`7b25653d5d5b305884d923d15664ca8bb25e41fc1865707212db0333f7e1bfe1`.
The ten-iteration arm's nominal best is step 16,056,320, SHA-256
`78092e0b092d024be253dc9524b589da4fcca1b53c20ccc1567be72551807b5f`.

Held-out comparisons use a common ten-iteration solver, 5 ms physics and four
substeps, seeds 707/808/909 and 128 environments per seed. Fixed-command and
transition results must not be mixed: the latter contain standing samples.
The canonical model SHA-256 is
`cbbb5e0cfe0d7c3663a52acc18a502e2e2052e0ccc1c6d9e0d4c95f06b4abacd`.
Velocity is torso-frame forward velocity; heading is absolute error to the
integrated commanded path. Signed lateral displacement can cancel across
episodes and is not an absolute-drift statistic.

The new replay uses the exact model-key and first Evaluator reset-key streams.
It records inside the unwrapped environment because Brax's randomized vmap
bypasses ordinary wrappers. Physical state is retained **before** AutoReset
restores the initial pose. The first terminal sample is included; all later
autoreset episodes are excluded. Reward-component sums are checked against
the actual reward, and episode lengths/falls against Brax EvalMetrics.
Contact forces are last-substep samples, not integrated impulses.

## 1. Reproduced transition failures

The source survives the original 384 transition episodes. The ten-iteration
nominal best fails 7/384 (1.82%). Instrumented replay reproduces all seven and
the original per-seed average episode lengths exactly:

| Seed | Episode indices | First fall times | Average episode length |
|---|---|---|---:|
| 707 | 33, 55, 63, 111 | 13.20, 13.30, 13.68, 13.38 s | 989.671875 |
| 808 | 30, 61 | 13.24, 13.92 s | 994.984375 |
| 909 | 77 | 14.00 s | 997.656250 |

All occur 1.20--2.00 s after the second walking restart at 12 s. There are no
non-finite states in these failures. In seed 707, none of the active joint
requests clips in the final 0.5 s before each fall. This contradicts a claim
that immediate joint-limit clipping is the direct cause of these falls.
Earlier standing knee clipping is real but is not established as their cause.

The first start arrives near 60 degrees of gait phase; the second near
240 degrees. In the four failing seed-707 episodes, the newer policy initially
keeps the right foot loaded instead of transferring support to the left for
the right-foot swing. At 0.2 s after restart its left foot carries no sampled
vertical force in all four; the right carries 18.2--20.6 N. The source performs
the weight transfer in those same episodes. Over the last second of standing,
all 128 newer-policy episodes have mean sampled left load share about 24.5%,
versus 42.6% for the source. The policies also differ in standing posture:
median knees approximately -1.25/+11.49 degrees versus -1.91/+15.22 degrees.
These are observed differences, not proof that forcing symmetric standing or
changing walk-home is the right training intervention.

## 2. Speed regression and reward trade-off

Independent fixed 0.10 m/s evaluation (three equally sized seed batches):

| Checkpoint | Mean forward speed | Speed MAE | Mean absolute heading error |
|---|---:|---:|---:|
| Source | 0.09083 m/s | 0.03019 m/s | 10.52 degrees |
| One-iteration final | 0.07060 m/s | 0.04150 m/s | 3.69 degrees |
| Ten-iteration best, 16M | 0.08232 m/s | 0.03009 m/s | 4.06 degrees |
| Ten-iteration final | 0.07743 m/s | 0.03429 m/s | 3.38 degrees |

Both arms improve heading and lose mean speed. Ten iterations improve
numerical contact consistency but do not, by themselves, solve the learning
regression or prove solver error caused asymmetry.

The ten-iteration arm's nominal 0.10 m/s return rises by 16.06534. Logged signed
components reconstruct 16.06533: full orientation contributes +10.99835,
Gaussian velocity +2.56202, angular-velocity cost +0.99234, and foot orientation
+0.98624. There is no sign/weight decoding defect. A nonlinear Gaussian
reward average and mean speed/MAE are different objectives.

Intermediate nominal checkpoints matter: the 8M/12M 0.10 m/s MAEs are
0.02201/0.02202, versus 0.02306 at the selected 16M checkpoint. Selection is
the worst score over both 0.05 and 0.10 m/s, not just 0.10: the 12M worst score
is 0.53485 and 16M is 0.53619. Selection did not malfunction, but a tiny
nominal-score advantage is not robustness evidence.

RSL-RL KL, optimizer losses and parameters remain finite. Standard deviation
changes 0.04319 -> 0.04030, equivalent to about 0.62 -> 0.58 degrees of target
noise. Final KL is 0.00912 against desired 0.01. These facts do not establish
an optimizer bug or justify changing entropy, learning rate or reward weights.

## 3. Paired diagnostic ablations

`full` includes configured observation noise and dynamics/reset/servo variation.
`dynamics` removes observation noise only. `noise` makes the model, initial
roll/pitch and servos nominal but retains observation noise. `nominal` removes
both. All cases retain the standard joint/velocity reset jitter and identical
episode indices/RNG streams; these are not four unrelated seed batches.

Seed 707, same 128 episode indices for both policies:

| Case | Source falls | 16M best falls | Source walking MAE | 16M walking MAE |
|---|---:|---:|---:|---:|
| Full | 0 | 4 | 0.03342 m/s | 0.03354 m/s |
| Dynamics only | 0 | 0 | 0.03114 m/s | 0.02930 m/s |
| Noise only | 0 | 0 | 0.02630 m/s | 0.02573 m/s |
| Nominal | 0 | 0 | 0.02298 m/s | 0.02076 m/s |

This supports an observation-noise/dynamics interaction in the learned
transition. It does not identify one sensor channel, one actuator or one
randomization parameter as the cause, and is not evidence to disable either
noise or dynamics randomization in training. Isolated modes retain a slower
newer-policy mean speed even when their MAE improves.

Moving the second restart from step 600 to 618 shifts its phase by 180 degrees
while preserving the 250-step walking interval. It also extends the preceding
standing interval by 0.36 s, so it is explicitly a **timing sensitivity probe**,
not a phase-only causal experiment or a proposal to reset phase in training.
In seed 707 the delayed-restart probe survives all 128 episodes versus four
falls at the original restart. Walking MAE is 0.03276 m/s and active clipping
2.55%, versus 0.03354 m/s and 1.94% before. Timing eliminates these falls but
does not establish better walking quality or pass the 2% clipping gate.

An optional actor projection reuses WR2's model-derived reflection of all
15 history frames. It tests learned left/right asymmetry without rewriting
the checkpoint. It is not a deployment adapter or evidence that a symmetry
loss will necessarily train successfully.

In seed 707, projection removes all four falls and reduces active target
clipping from 1.94% to 0.00055%. Walking speed MAE changes 0.03354 -> 0.03456
m/s: it does **not** solve tracking. Mean standing left load share moves from
24.5% to 38.0%. This is evidence that the learned action pattern contributes
to transition fragility, not proof that a mirror loss alone solves locomotion.
Repeating projection at seeds 808/909 also produces no falls: **0/384** versus
7/384 without projection. Across the three batches, walking MAE is 0.03431 m/s
versus 0.03298 m/s unprojected, and mean speed remains about 0.0790 m/s. Active
clipping is approximately 0.00034%. This separates the balance benefit from
the still-unsolved speed objective.

Earlier ten-iteration checkpoints were also replayed with full randomization
at all three held-out seeds:

| Checkpoint | Falls / 384 | Avg episode length | Walking-only MAE | Active clipping |
|---|---:|---:|---:|---:|
| 8M | 5 | 995.80 | 0.03207 m/s | 1.50% |
| 12M | 3 | 997.57 | 0.03130 m/s | 1.20% |
| Selected 16M | 7 | 994.10 | 0.03298 m/s | 1.94% |

The table averages equally sized seed-batch summaries; trace MAEs weight valid
walking samples within each batch. The original aggregate evaluator uses
means of per-episode averages, so slight differences in terminated batches
are expected and must not be called a reproduction failure.
The 12M checkpoint is a better recovery starting candidate, but not qualified:
seed 909 has 2/128 falls (1.56%) and average length 994.984, failing the robust
stage's per-seed 1%/995 gates. All new intermediate failures also occur after
the second restart. Rolling back to an earlier checkpoint is not a full fix.

Artifacts are under `results/wr2_walking_diagnostics/investigation_20261008_*`.
The `summary.json` and per-batch NPZ files retain checkpoint/config hashes,
keys, first-failure indices/times, commands, physical states, delayed joint
requests/applied targets, active clipping, foot forces/slip and reward terms.

Example reproduction on Mac (use a new output directory):

```bash
uv run --extra training python -m wr2.tools.investigate_walking \
  --config results/wr2_solver_ab/newton10_seed0/wr2_ppo_20261007_215905_seed0/training_config.yaml \
  --checkpoints results/wr2_walking/wr2_ppo_20261007_152924_seed0/best_params.pt \
    results/wr2_solver_ab/newton10_seed0/wr2_ppo_20261007_215905_seed0/best_params.pt \
  --seeds 707 --cases full dynamics noise nominal --num-envs 128 \
  --output results/wr2_walking_diagnostics/my_paired_replay --allow-cpu
```

Projection uses `--project-actor`; timing sensitivity uses `--restart-step 618`.
For fixed-command replays, `--fixed --command 0.10` matches a single-command
evaluation. To match a command inside a multi-command report, supply its keyed
seeds (`original_seed + 100000 * command_index`), just as evaluate.py does.

Validation: **151 tests and 21 subtests pass**, including pre-autoreset trace
retention, delayed requests, conditional denominators, all-history actor
reflection, and local/remote transition-confirmation regression cases.
The new matrix contains **2560 matched twenty-second diagnostic episodes**.
These paired episodes are not 2560 independent draws and are not a training run.

## 4. Validation/promotion fix

Both local and Mac-owned remote supervisors now require independent randomized
transition confirmation as well as fixed-command confirmation. Transition
gates reuse the stage's episode length, fall, tracking, active clipping and
finite-state limits, plus existing hard servo limits. Tracking uses
`walking_velocity_error_sum / walking_sample_count`; standing samples cannot
dilute it. Fixed-command speed ratio, walking score and foot-use gates are not
misapplied to a mixed-command schedule. Missing transition results fail closed.

`best_params.pt` is still a nominal candidate, not a deployment-qualified
policy. This fix changes validation, not training dynamics or PPO.

## Next controlled training experiment (not implemented here)

Before training the proposed treatment, use `wr2.tools.check_walking_stance` to
cross old/new controllers at the first (step 150) and second (step 600) starts.
Only actor weights change. The complete ongoing simulator state, velocities,
physics warm-start, sensor RNG/filter state, 15-frame observation history,
phase and delayed commands are retained. The first new action still incurs the
configured one-step delay. Exact donor-prefix checks and independent original
self-replay controls must pass before interpreting counterfactual outcomes.
This is a controller-from-donor-state test, not a geometry-only or phase-only
intervention; policy handover can itself create a transient.

For foot geometry, use **signed left-minus-right robot-forward x**, not the
absolute stagger. Last-second standing means in seed 707 are:

| Checkpoint | Before first start | Before second start |
|---|---:|---:|
| Source | +5.74 mm | -24.07 mm |
| Ten-iteration 16M | +11.64 mm | +3.91 mm |

Positive means left ahead of right. The newer mean does not reverse sign;
the source does. Neither a sign change nor keeping the same sign proves a
coordinate bug: the robot stops at different phases, and standing does not
force its feet back to a symmetric reset pose. The newer second start actually
has less absolute stagger, so increased foot stagger is not an established
explanation of its failures.

The diagnostic follows Brax's `training/acting.py::generate_unroll` key stream
and uses WR2's existing TB-aligned continuous gait clock; it does not introduce
a new standing controller or reset strategy. Reproduction command:

```bash
uv run python -m wr2.tools.check_walking_stance \
  --checkpoints \
    results/wr2_walking/wr2_ppo_20261007_152924_seed0/best_params.pt \
    results/wr2_solver_ab/newton10_seed0/wr2_ppo_20261007_215905_seed0/best_params.pt \
  --config results/wr2_solver_ab/newton10_seed0/wr2_ppo_20261007_215905_seed0/training_config.yaml \
  --output results/wr2_walking_diagnostics/stance_cross_seed707 \
  --reference-traces results/wr2_walking_diagnostics/investigation_20261008_full707_v2 \
  --seeds 707 --allow-cpu
```

Review the crossed-state results before selecting a training intervention;
they do not isolate physical posture from sensor/action history.

The previously proposed next direction is RSL-RL's optional **actor mirror loss**, as
exposed by ToddlerBot, rather than changing physics again, resetting phase or
disabling noise/randomization. Keep the Gaussian velocity and orientation
rewards, residual action map, 5 ms/four-substep timing and ten solver iterations.
Start from the original shared source checkpoint with its optimizer/RNG intact;
the existing ten-iteration 20M arm supplies the control. Run one bounded 20M
treatment, not an unbounded continuation. Compare fixed-speed tracking and
randomized transitions at every checkpoint, not just nominal best/final return.

The symmetry adapter must use WR2's tested model-derived signs and all-history
reflection. Mirror raw actor features; if RSL-RL supplies normalized features,
undo normalization, reflect, then renormalize with the same current statistics.
Naively reflecting asymmetric per-channel normalized values would not reproduce
the validated actor projection. Use mirror loss only, not critic/transition data
augmentation, and verify involution, physical observation parity, gradients and
action-preserving checkpoint restore before GPU training.

This is a justified WR2 use of an **optional** TB capability, not default TB
training or a proven solution. Projection improves balance but not speed here;
the treatment must demonstrate improved transitions without losing tracking.
If speed remains deficient, investigate that separate objective trade-off next
instead of adding a progress reward, changing servo ranges and symmetry at once.

## ToddlerBot / WR1 comparison and references

- Local ToddlerBot `reference/walk_zmp_ref.py:70` computes continuous phase
  from elapsed time; `locomotion/mjx_env.py` advances it through command changes.
  WR2 does the same. No evidence supports replacing this with phase reset.
- ToddlerBot `locomotion/walk.gin` uses the same full orientation weight and
  Gaussian velocity kernel. The trade-off above is possible with that objective;
  identical weights are not a guarantee of identical learned behavior.
- ToddlerBot `locomotion/rsl_rl_config.yml:40` supports actor mirror loss, but
  `train_mjx.py:1264` defaults `--symmetry` off and removes its configuration
  without that option. It is an optional framework capability, not default TB
  training or a ready-made WR2 morphology transform.
- WR1 `training/eval/diagnose_policy_symmetry.py` separates reference, actor and
  closed-loop symmetry. WR2 reuses that diagnostic method with independently
  verified WR2 signs; WR1's joint signs must not be copied.
- No equivalent randomized TB transition campaign was run. This investigation
  does not claim TB is immune to this failure or that servo hardware caused it.

Public sources: [MuJoCo constraint-solver algorithms](https://mujoco.readthedocs.io/en/stable/computation/index.html#algorithms),
[ToddlerBot](https://toddlerbot.github.io/),
[Yu, Turk and Liu, Learning Symmetric and Low-energy Locomotion](https://arxiv.org/abs/1801.08093).
The symmetry paper introduces a loss plus assistance curriculum, not a claim
that equal instantaneous foot forces are necessary for walking.
