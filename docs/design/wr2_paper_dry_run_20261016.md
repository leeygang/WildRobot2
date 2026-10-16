# WR2 paper GPU dry-run review — 2026-10-16

## Decision

Proceed with a **fresh 300M run** of the prepared paper recipe. The full GPU
batch completed, PPO updated both networks, measurements agree with the raw
episodes, and checkpoints reload. One standalone evaluation path bug was
reproduced and fixed. This dry run provides no evidence to change rewards,
actuator limits, reference targets or the training budget.

Reviewed run: [`wr2_paper_20261016_111519_seed0`](../../results/wr2_walking/wr2_paper_20261016_111519_seed0).
The copied artifacts contain configuration, native metrics, two complete
paper evaluations and initial/final/native checkpoint parameters. A terminal
stdout/stderr log was not included.

## Configuration and update checks

- Source **0a63ae9**, clean at launch; fresh seed 0, no restore. The only
  differences from the prepared configuration are **20,480 transitions** and
  **2 evaluations**, replacing 300M and 16.
- Actual batch: **1024 environments × 20 ticks**, 4 minibatches × 4 passes:
  one rollout and **16 optimizer updates**. The GPU guard remained enabled.
- Runtime: Python **3.10.19**, Brax **0.14.1**, JAX/JAXlib **0.6.2**,
  MuJoCo/MJX **3.14.0**. `uv.lock` selects Brax 0.14.1 for Python below 3.11;
  local Python 3.11 checks used 0.14.2. These are supported lock resolutions.
  Inspection of the 0.14.1 loss, optimizer, distribution and checkpoint code
  confirms the configured scalar-value, fixed-LR semantics. Keep this runtime.
- The physics model hash matches the prepared model. Newton 10, 5 ms × 4
  substeps, fixed crouched home, measured CoM scale **0.824054757**, all 227
  command-reference validation entries and their diagnostics match.
- All **263 initial / 272 final native metrics** are finite. All 25 weighted
  reward terms are present; their sum matches return within **0.0005**,
  consistent with float32 accumulation. Disabled terms remain zero.
- Final loss **0.810338** = policy **−0.011569** + value **0.825784** + entropy
  **−0.003878**. Mean KL **0.002634**, LR **0.0001**, mean pre-tanh standard
  deviation **0.693612**. No numerical instability appears in this update.

The native 1000-tick forward evaluation return changes **202.415 → 205.664**,
with falls **65/128 → 59/128** and zero nonfinite states. This single-condition
evaluation differs from the normalized paper suite below.

## Measurement audit and walking status

Both paper reports contain **15 conditions × 128 episodes**, seed 12102,
deterministic policy, noise and randomization enabled, and the normalized
**824-tick / 16.48 s** horizon. Commands, boundaries **124/371/494/700** and
21-tick recovery windows match the WR2 source's normalized TB protocol.

Recomputed group means/population standard deviations, selection summaries,
fall/nonfinite totals, episode lengths and group sample counts from all
**3,840 episode-condition records**. Maximum summary rounding difference is
**9.94e−8**. Finite terminal samples contribute; tracking is the per-episode
mean instantaneous XY vector error, then averaged across episodes. Short
failed episodes also contribute, so tracking must be read with survival.

| WR2 transitions | Falls / 1,920 | Nonfinite | Fixed walking XY error, physical m/s | TB-equivalent m/s |
| --- | ---: | ---: | ---: | ---: |
| 0 | 787 (40.99%) | 0 | 0.103805 | 0.125968 |
| 20,480 | 768 (40.00%) | 0 | 0.103874 | 0.126053 |

The trained dry-run policy still behaves similarly across commands: about
**−0.034 m/s forward velocity** and **92.5% double support**. Walking leg
reference RMSE is about **0.159–0.167 rad**, versus **0.0124 rad** when standing.
Actual target clipping and slew limitation are zero in every condition; mean
episode torque RMS is about **0.167 Nm**. There is no learned walking yet.
One rollout cannot establish a persistent reward or hardware limitation.

No `best_params` or held-out confirmation is expected: neither checkpoint
meets the configured **≤1% falls** eligibility gate. Initial parameters,
final parameters and the 20,480-step native checkpoint are present.

## ToddlerBot comparison

Checked the frozen successful TB run's
[`evaluate_xy.py`](../../../toddlerbot/results/toddlerbot_walk_ppo_20261015_194347/locomotion/evaluate_xy.py)
and raw `xy_evaluation.jsonl`, alongside WR2's environment, trainer adapters
and paper evaluator. The suite conditions and episode aggregation match;
WR2 additionally normalizes physical commands/time and records hardware loads.
TB source is **1982ef3**, with Brax **1414780**. Its paper-aligned moving leg
reference reward, fixed action home and native policy semantics are the
prepared WR2 targets; see the [recipe](wr2_paper_training.md).

| TB transitions | Falls / 1,920 | Fixed walking XY error, m/s |
| --- | ---: | ---: |
| 20,008,960 | 9 (0.469%) | 0.062016 |
| 180,080,640 | 3 (0.156%) | 0.038379 |
| 200,089,600 | 8 (0.417%) | 0.037025 |

TB's first saved comparison has about **1,000×** this dry run's training.
These results support completing the intended budget and retaining intermediate
checkpoints; they do not isolate a WR2 transfer defect after one update batch.
Compare the WR2 180M/200M checkpoints and selected candidate with TB 180M/200M,
including direction response, transitions, survival and foot use. Confirm the
selected candidate on seed 22102; final checkpoint quality is not assumed.

## Checkpoint issue and verification

The standalone paper evaluator passed relative directory paths directly to
Orbax, which raises **“Checkpoint path should be absolute.”** The documented
relative-path evaluation command therefore failed. Resolve the path before
loading; automatic training-time evaluation and training restore are unaffected.
The regression exercises real Brax checkpoint saving/loading through this entry
point and failed before the fix. **Nine targeted tests pass**, and the regression
also passes with the exact Brax **0.14.1** package in the local CPU environment.

Loaded the actual GPU native checkpoint locally with Brax 0.14.1 and an absolute
path: all **25 leaves / 1,599,600 values** are finite and exactly match final
`params`. Both actor and critic changed from initialization. Metadata-based
`load_policy` succeeds and produces the same deterministic actions as the
configured network factory on three nontrivial observation vectors.

Logged training throughput **241 transitions/s** includes the first JIT compile
in an **84.9 s** single-rollout training interval. It does not estimate warmed
300M throughput; use later training progress to measure that rate.

## Next command

From the updated WR2 checkout on the GPU host:

```bash
uv run --extra training python -m wr2.locomotion.train \
  --config wr2/locomotion/configs/ppo_walking_paper.yaml
```

This requests 300M, rounds to **300,134,400 transitions**, and retains initial
parameters plus 15 trained checkpoints approximately every 20M.

References: [ToddlerBot paper, Tables VI/VII](https://arxiv.org/html/2502.00893v2),
[ZMP preview control](https://doi.org/10.1109/HUMANOIDS.2015.7363473),
and the [TB control/WR2 source comparison](toddlerbot_transfer_review_20261016.md).
