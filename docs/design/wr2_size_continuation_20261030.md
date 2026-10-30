# Size-aligned 200M continuation — 2026-10-30

## Purpose

Continue the accepted fresh size-aligned final policy for another200M transitions.
Test whether longer learning improves cadence and alternating placement while
preserving forward tracking, heading and the smaller shoulder motion.

The [completed run](wr2_size_aligned_result_20261030.md) restores bilateral hip
swing and improves held-out tracking. The user confirms substantial Mac visual
improvement and accepts nominal Mac evaluation for the current walking priority.
Further GPU robustness comparison remains optional.

Local TB **`paper-walking-baseline`** uses300M transitions in `paper.gin`.
WR2 has200M fresh transitions so far. This is an affordable longer-training
test with the established recipe. Different normalized simulation time and
optimizer exposure mean raw step counts are not exact TB learning equivalence.
Longer training may preserve or worsen the current strategy; improvement must
be measured on retained checkpoints.

## Parent and recipe

Restore exactly:

```text
results/wr2_walking/wr2_paper_size_aligned_20261030_085111_seed3/checkpoints/000200048640
```

This checkpoint equals the accepted parent's `best_params` and final `params`.
The selected parameter-file SHA256 is
`6c11c19aa60b169eee902bbb6933e4ef1ca9f6b065aa5039acd4261d46012a4d`.
The preflight also records the checkpoint-directory digest independently.

Complete v0.35.0 `ppo_walking_size_continue.yaml` differs from the parent's
saved training config **only in version, descriptive name and output prefix**.
It preserves seed3, reward weights, reference generation, command ranges,
model/actuation, action centers, grounded reset, observations/noise and
randomization. In particular:

- Episode824, rollout16, gamma.9637121762 and lambda.9396526368.
- Nominal support threshold.612930N, qualified landing after ten contact
  samples and gamma-derived compensation1.39468.
- XY/leg-reference coefficients15/15; roll weight.5 with the existing ±.1rad
  band. Leg reward remains executed joint MSE against the moving reference.
- LR1e-4 fixed, entropy.0005, clipping.2, four epochs/four minibatches,
  batch256 and1024 environments.
- SiLU512/256/128 actor/critic and state-dependent tanh Gaussian.
- Actor1005/critic1515,12 leg actions; model-home hip-pitch action centers.
- 20ms control,5ms×4 physics, actual hardware limits and.60s native gait.
- Forward range−.10…+.20 TB-equivalent m/s, zero training pushes.

Size conversions remain `v_physical=v_TB*s`, `t_physical=t_TB*s`,
`length_physical=length_TB*s²`, with `s=.8240547565448791`.

The native Brax restore preserves actor, critic and normalizer parameters;
Adam state, RNG/environment state and counters start anew. This is a parameter
continuation, not restoration of the original optimizer trajectory. No heading
input expansion is needed, and restored deterministic actions are verified.
The actor's learned means and scale-head parameters are restored exactly.

## Direct GPU command

From the updated GPU WR2 checkout:

```bash
uv run --locked --python 3.10 --extra training python -m wr2.locomotion.train \
  --config wr2/locomotion/configs/ppo_walking_size_continue.yaml \
  --restore-checkpoint results/wr2_walking/wr2_paper_size_aligned_20261030_085111_seed3/checkpoints/000200048640
```

The regular trainer applies its GPU/backend and full-reference preflight gates.
Output uses `results/wr2_walking/wr2_paper_size_continue_<timestamp>_seed3`.
The restore path in this command is required; the training config alone does
not choose a parent checkpoint.

Requested200M rounds to **200,048,640 added transitions /195,360 added Adam
updates** with the unchanged batching/evaluation schedule. Including the
parent, total learning becomes **400,097,280 transitions /390,720 updates**.
New-run metrics/checkpoint names count added steps from zero:

| New-run step | Cumulative training transitions |
|---|---:|
| 0 / `initial_params` | 200,048,640 |
| 200,048,640 / final | 400,097,280 |

Sixteen evaluations retain15 learned checkpoints roughly13.34M transitions
apart, plus initial parent parameters. Standard selection and held-out
confirmation remain enabled. The automatic best file ranks learned
checkpoints by desired-frame XY error; `beats_initial_tracking` indicates
whether it beats its own parent. It does not select on cadence or placement.

## Evaluate improvement

Use existing final and160M as benchmarks at .10/.15/.20 TB commands, with
the same nominal Mac protocol and1.5 TB-second warmup. Optional randomized
GPU batches can assess robustness. Do not interpret.30 as a trained condition.

Current final nominal.20 baselines are forward speed.19129 TB m/s,
landing leads5.98/2.77 TB cm, same-foot cadence2.06 TB Hz, roll5–95% span4.99°
and shoulder Z spans14.53/12.16 TB mm. The160M policy has more even4.12/4.97cm
placement but10.8° net turn, versus final1.6° in those nominal samples.

Look for lower cadence toward the native TB/reference approximately1.4Hz and
more balanced forward landings, while retaining speed/heading, stability,
executed bilateral swing and body motion. Check self-contact as well. Use
identical force classification and report actual placement, not reward alone.
The standard GPU suite already runs each checkpoint and measures tracking,
contact and stability; its commands stop at.10 TB speed, so higher-command
placement/body checks are a separate part of result review.

Retain the parent if no learned checkpoint improves the walking priorities.
Do not promote final merely because it has the largest step count. If the
additional budget leaves cadence/placement unchanged, review the measured
timing incentives before choosing another reward change.

## Verification

Preparation checks passed for the full typed config against the saved parent and a YAML
round trip, finite/exact parent tensors, unchanged input dimensions and
restored deterministic policy outputs (maximum action difference0). A small native CPU PPO continuation
uses the actual parent parameters with16-tick rollouts, four epochs/four
minibatches and64 real WR2 transitions, verifying exact initial parameters,
finite changed actor parameters and exact saved-checkpoint reload.

Validation uses nominal physics and exact reduced standing/forward reference
rows for CPU integration. The production trainer retains its full command-grid
gate. Validation weights are discarded. Evidence is saved locally under the
parent's ignored `continuation_preflight_20261030/` folder.

References: local TB **`paper-walking-baseline`** `paper.gin` and native
parameter-only Brax checkpoint restore; [TB paper v2, Tables VI–VII](https://arxiv.org/html/2502.00893v2);
[size-alignment design](wr2_size_alignment_experiment_20261030.md);
[completed policy review](wr2_size_aligned_result_20261030.md).
