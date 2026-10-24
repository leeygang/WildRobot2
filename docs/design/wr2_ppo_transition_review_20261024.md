# WR2 v5 PPO command-transition review — 2026-10-24

Follow-up: the [completed native TB comparison](wr2_tb_ppo_comparison_20261024.md)
shows that the command-switch residual and large restart-probe KL also occur in
TB. Its action-home source audit supplies the next proposed correction.

Result: `results/wr2_walking/wr2_paper_step_20261022_143550_seed3/ppo_learning_diagnostic_v5`.
Source: clean `8e3bc6d`; protocol: [diagnostic guide](wr2_ppo_learning_diagnostic_20261023.md).
The user confirmed that **only WR2 ran**; the matching TB Paper GPU measurement
is still needed. TB source and existing training results were reviewed locally.

## Conclusion and next priority

The large 120M update repeats at command resampling. The new recordings localize
the critic training residual to **switches between zero and nonzero motion
commands**, before the networks receive the newly sampled command. Motion here
includes translation or turning, using the reward's normalized `.05` deadzone.

Pending qualified-air state is **not supported as the leading explanation for
this update spike**. Most residual occurs without pending credit; prediction and
GAE target align after the new command becomes available even while credit is
pending. Air reward can still contribute to the value difference between motion
and standing. These are different claims and the latter has not been isolated.

The next priority is the same diagnostic on **TB Paper at 80M, 120M and 200M**.
Both source implementations have the observed command delay and command-gated
air reward. The recordings establish a WR2 transition condition worth comparing;
they do not establish why TB avoids WR2's short, asymmetric learned steps.

## 1. Verification and interpretation

All **16 batches** passed: 1024 environments ×20 ticks ×8 batches ×2 checkpoints
= **327,680 measured transitions**, after 330 warmup ticks per checkpoint.
Tool/math/config hashes match the committed and saved files. The GPU core
packages match training, with no runtime differences.

| Maximum over the audited minibatches | Error |
| --- | ---: |
| Float32-reference actor parameter gradient | 7.178e−7 |
| Float32-reference output gradient | 7.776e−8 |
| Native output gradient | 7.568e−8 |
| Advantage reconstruction | 1.776e−15 |
| Scalar reward reconstruction | 1.431e−6 |

Exact first-minibatch indices and entropy keys passed. Named comparisons show
the formerly failing scalar behavior: maximum policy/total-loss differences
are **3.982e−6 /4.061e−6**, below their cancellation-aware limits. Entropy and
value-loss differences are at most **1.77e−8 /5.66e−7**. The recorded native
matrix-kernel gradient gap is at most `4.662e−4`; the separate Float32 check
continues to close tightly.

Across the full recordings, paid air credit reconstructs exactly from contact,
pending/supported credit, stance samples and command eligibility. Pending/raw
counter recurrences also reconstruct exactly; supported-credit recurrence differs
by at most `1.07e−8` seconds. Weighted air reward reconstructs within `4.77e−7`.
Full reward-group sums differ from the native scalar by at most `1.907e−6`.
These checks include cross-batch continuity and native episode resets.

The saved value prediction and GAE target reproduce native `.25 MSE` within
`1.12e−6` absolute error across all 16 audited minibatches. This small separate
GPU graph rounding is retained in the review data. Normalized advantages have
zero mean and unit standard deviation within `2e−6`.

**A GAE target bootstraps the same critic.** Its residual is a native training
signal, not an error measured against a complete realized future return.
Agreement with the target does not establish that the critic's value estimate
is unbiased.

Each probe starts fresh Adam, applies four epochs ×four minibatches at `1e−4`,
and discards its weights. Batches share original-checkpoint trajectories. Target
changes are evaluated on those original observations; post-probe physical motion
is not simulated. Historical Adam moments/RNG/environment states are unavailable.

## 2. The command-switch response repeats

The main switch is in zero-based batch 2. The 80M/120M batch records command
changes in **980/985 environments**. New fields locate the timing precisely:

| Original episode tick | Recorded event |
| --- | --- |
| 370 | Reward and network inputs use the previous command. |
| 371 | The environment samples its next command after building the observation. |
| 372 | Reward uses the new command; actor and critic still receive the old one. |
| 373 onward | Network inputs contain the new command. |

Some environments have different counters following resets. In each switch
batch's randomly selected 256-environment audit, 244 show this command change
and one-tick input mismatch.

| Probe measure | 80M | 120M |
| --- | ---: | ---: |
| Native first-minibatch value loss, switch batch | .41922 | .51315 |
| Median value loss, seven other batches | .02216 | .01926 |
| Final-policy/full-batch KL, switch batch | .08366 | **1.00085** |
| Median final-policy KL, seven other batches | .03416 | **.03505** |
| Switch/ordinary KL ratio | 2.45× | **28.56×** |

This is `KL(original || policy after all 16 steps)`, summed over actions on the
full original observation batch. It differs from native last-epoch KL, which
averages metrics from four intermediate minibatch steps. The latter is `1.1831`
at 120M here, versus `1.6580` in v3. That difference is not a walking improvement:
both are probes of the same saved policy.

### A small set of transitions dominates the training signal

At 120M, the first three ticks hold **96.76% of squared value residual** and
**96.59% of squared normalized actor advantage** in the audited minibatch.

| Motion-state transition, 120M audit | Environments | Share of squared value residual |
| --- | ---: | ---: |
| Zero → nonzero motion | 40 | 37.85% |
| Nonzero motion → zero | 45 | 58.46% |
| Nonzero motion → nonzero motion | 159 | 3.32% |
| Zero → zero | 12 | .37% |

The 85 environments changing motion status account for **96.31%** of squared
value residual and **96.25%** of squared actor advantage across the batch.
Their first three ticks occupy only **4.98% of its 5120 samples**.

| 120M transition | Mean value, first three ticks | Mean GAE target | Mean residual | Residual RMSE after tick 372 |
| --- | ---: | ---: | ---: | ---: |
| Zero → motion | 8.899 | 14.566 | +5.667 | .331 |
| Motion → zero | 15.008 | 8.305 | −6.703 | .207 |

The first two ticks include a future randomized command in the target; at the
third tick that command also remains absent from network inputs. The data
support a large motion/standing value gap and a concentrated transition signal.
Stochastic future commands can create irreducible target variance. Source alone
does not establish that this should produce WR2's much larger later probe update.

### The largest target response is hip yaw during turning

At 120M, **left hip yaw contributes .5910, about 59% of final KL**. In the
largest target-change sample, its request changes **3.263°**, with raw Gaussian
mean shift `.22881` and old std `.01523`: **15.0 old standard deviations**.
This sample is before resampling and has yaw command `−.48464 TB rad/s`; it is
not a fixed-forward walking sample.

Gaussian mean scores are sensitive to small stds. This explains an update
amplification mechanism to compare with TB. It does not prove that WR2 alone has
smaller stds or that this discarded probe caused historical hip-pitch saturation.

The average absolute left-yaw mean shift is `.00634` at 80M versus `.02700` at
120M. Its full switch-batch std averages `.19577/.14563`; on the audited
motion-status changes' first three ticks, std instead averages `.03102/.03801`.
Thus simple std shrinkage does not explain the stage difference. Most of the
later yaw KL comes from the mean changing relative to its narrow distribution,
not from a large update of std itself.

## 3. What the pending-credit check tells us

At 120M, pending-credit samples contribute **4.37%** of squared value residual;
samples paying air credit contribute **.451%**. The payment peak is later in the
batch, around ticks 381–382, after the main residual has subsided. At 80M the
corresponding shares are **8.80% /5.96%**.

This lowers the priority of adding pending-credit counters to observations as
a remedy for this particular spike. It does not exclude air reward's effect on
future targets: switching to zero command disables future air credit, and
switching to motion enables it. The total value-gap calculation includes other
command-dependent rewards and future critic predictions as well.

The source audit remains valid: WR2's added supported/pending/stance counters
are absent from explicit observations, while TB also omits its raw air counter.
The new recordings provide no basis for selecting a counter-input change or
another air-weight change before the TB comparison.

## 4. The learned gait gap remains

Forward-command statistics below are time weighted, with nearly identical mean
commands `.10829/.10823 TB m/s`. Reward rates are native weighted return per
TB-equivalent second, with `s=.8240547565`; foot distances divide by `s²`.

| Measure | 80M | 120M |
| --- | ---: | ---: |
| Achieved forward speed, TB m/s | .05294 | **.04041** |
| Weighted return rate | 19.577 | **20.066** |
| XY reward contribution | 1.968 | **1.842** |
| Yaw reward contribution | 3.165 | 3.422 |
| Torso-stability contribution | 1.912 | 2.334 |
| Air reward contribution | 10.624 | 10.655 |
| Left/right forward hip target, degrees | 6.184 /26.941 | **3.871 /28.165** |
| Left hip tanh sensitivity | .4647 | **.2036** |
| Sustained left landings ahead of right | 1/863 | **0/862** |
| Sustained right landings ahead of left | 790/790 | **790/790** |

At 120M, mean sustained landing leads are **−50.6/+70.2 actual WR2 mm**.
Contacts must persist ten recorded samples; unresolved end events are excluded.
This is a contact-duration filter, not every air-eligibility condition, and
landing separation is not same-foot stride length.

The stable observation repeats: slower translation accompanies higher total
return and the right-leading posture. It does not isolate a causal reward weight.
There are 8/1 early-termination transitions and no timeouts in the short 80M/120M
recordings; these are not benchmark episode fall rates.

Small fresh-Adam hip directions are less consistent across recordings. In v5
the left target moves forward in **4/8 probes at 80M /3/8 at 120M**; the right
does so in **7/8 /5/8**. Mean changes per probe are left `+.00199°/−.00269°`,
right `+.00283°/+.01171°`. These use per-observation target deltas, averaged per
probe. The v3 counts were `6/8 /2/8` left and `7/8 /7/8` right. The same seed and
configuration do not make these different compilations independent experiments.
The small update directions are weaker evidence of persistent PPO bias than the
posture and transition-condition observations.

## 5. TB comparison: established result and missing measurement

Both runs trained 300M steps. The table uses **1280 per-episode walking errors
from ten fixed nonzero conditions**; WR2 speed errors divide by `s`. Falls cover
all 1920 native evaluation episodes. This reproduces the selection metric,
avoiding a different average over all conditions or shortened episode samples.

| Checkpoint | WR2 XY error, TB m/s | TB Paper XY error, m/s | WR2 falls | TB falls |
| --- | ---: | ---: | ---: | ---: |
| 80M | .07589 | .03965 | 20 | 17 |
| 120M | .07666 | .03952 | 0 | 39 |
| 200M | .08241 | **.03702** | 0 | **8** |
| 300M | .07820 | .03800 | 0 | 134 |

The existing <=1% fall gate selects WR2 120M and TB 200M. WR2's later low fall
counts do not close its translation gap. TB's final checkpoint is also worse
than its selected intermediate checkpoint, supporting retention and evaluation
of intermediates.

Raw native training value loss at 80M/120M is WR2 `.16975/.16393` versus TB
`.07445/.07174`. Both native losses use `.25 MSE`, but these aggregate logs sample
different native state/reward distributions and optimizer histories. They support
checking the value/transition difference; they cannot replace the matched probes.

Code was checked from local **`paper-walking-baseline` `a4f693d`**, its clean Brax
fork `1414780`, and WR2 source `8e3bc6d`. Both use home plus `.25*tanh` actions,
state-dependent Gaussian std, swish, leg-reference imitation, gamma `.97`, lambda
`.95`, 20-tick rollouts and four-by-four Adam at `1e−4`. Both build observations
before assigning the next resampled command and gate air credit on motion.
WR2 additionally qualifies supported-air credit after ten contacts and uses
collision weight 1 rather than TB's .1.

The matched TB GPU measurements of transition residual, raw std and final-policy
KL are **not available**. The four-environment TB CPU smoke is implementation
validation, not a statistical substitute.

## 6. Next measurement and decision

Use the [isolated Paper checkout setup](wr2_ppo_learning_diagnostic_20261023.md#gpu-command-native-tb-paper)
and saved TB runtime. Include 120M to compare the same training stage, and 200M
to compare the selected successful policy:

**Align the command change within the PPO fragment.** WR2 warmup 330 plus two
20-tick batches starts the switch batch at tick 370; the new command is used at
372, batch tick 2. TB's default warmup 400 starts that batch at 440; its new
command is used at 450, batch tick 10. Directly comparing the batch loss would
mix a different amount of pre-switch GAE/bootstrapping with policy differences.

Use **TB warmup 408**, so its switch batch starts at 448 and uses the new command
at batch tick 2. This is an existing diagnostic argument, not a training change.
The warmups become approximately 8.009 TB seconds for WR2 and 8.16 seconds for
TB. The native 20-tick fragments still span `.4854/.4000 TB seconds`; compare
event-relative time and retain this duration difference. Native gait phase,
model/randomization and reward differences also remain part of the comparison.

```bash
uv run --no-project \
  ~/projects/toddlerbot/.venv-paper-blackwell/bin/python \
  ~/projects/WildRobot2/wr2/tools/diagnose_ppo.py \
  --tb-root ~/projects/toddlerbot/.worktrees/paper-ppo \
  --run-dir ~/projects/toddlerbot/results/toddlerbot_walk_ppo_20261015_194347 \
  --checkpoint-steps 80035840 120053760 200089600 \
  --warmup-steps 408 \
  --output ~/projects/WildRobot2/results/toddlerbot_walk_ppo_20261015_194347/ppo_learning_diagnostic_v5
```

Compare the first three switch ticks, motion/zero transitions, per-joint std
and KL, and full probe target changes. If TB has a similar value residual but
smaller KL, focus on how its policy/optimizer handles the signal. If TB lacks
the residual, compare native command/reward/value timing and normalized reward
distributions. If both show comparable probe spikes, investigate whether the
fresh-optimizer protocol explains the spike before using it to explain WR2's gait.

Select one controlled 200–300M training change after that comparison, with
retained checkpoints and the existing tracking/fall/stride evaluation.

## Artifacts and references

Local review data and the standalone four-panel figure are retained under the
ignored result folder:

- `ppo_learning_diagnostic_v5/review_20261024/analysis.json`
- `ppo_learning_diagnostic_v5/review_20261024/ppo_transition_review.png`

Source checks: WR2 `walking_env.py::_paper_step/_paper_observation`,
`paper_rewards.py::qualified_air_time_step`, `ppo_diagnostics.py`, and native
TB Paper `mjx_env.py::step/_get_obs`, `walk_env.py::_reward_feet_air_time`,
plus both Brax `losses.py` and Gaussian distributions.

- [ToddlerBot paper, Tables VI–VII](https://arxiv.org/html/2502.00893v2).
- [Generalized Advantage Estimation](https://arxiv.org/abs/1506.02438).
- [Proximal Policy Optimization](https://arxiv.org/abs/1707.06347).
- [Prior v3 review](wr2_ppo_learning_result_20261024.md) and
  [hip incentive/coordination evidence](wr2_hip_pitch_incentive_review_20261023.md).
