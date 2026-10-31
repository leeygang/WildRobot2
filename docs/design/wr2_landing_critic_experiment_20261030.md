# Landing reward state in the critic — 2026-10-30

## Question

Does explicit access to delayed landing credit improve value estimation and
walking learning? Add the current reward-memory state to the privileged critic,
then run a matched 200M continuation from the accepted fresh 200M policy.

The [400M review](wr2_size_continuation_result_20261030.md) shows that another
unchanged 200M did not improve cadence/placement. It also confirms that current
critic inputs omit the internal state controlling qualified air-time payments.
A controlled refreshed-observation probe demonstrates that queued credit can
change the next reward without changing the old observation inputs. This
confirms missing direct information; its contribution to gait failure remains
a hypothesis, not an established result.

## TB comparison and rationale

Local **`paper-walking-baseline`** `_get_obs` includes phase, joint state/error,
true velocity, actual/reference contact and push information. TB does not feed
its raw air-time counter directly to the critic. Its air reward pays on first
filtered landing contact, without WR2's pending-credit/sustained-contact stage.

WR2 retains the user-requested qualifying landing rule. Ten consecutive contact
samples span nine intervals/.18 physical seconds. The extra waiting period can
move the rewarded swing's beginning outside the 15-frame observation history.
Explicit reward state is a WR2 extension justified by that changed reward,
following the asymmetric-critic principle of exposing available simulator
state to value learning while preserving deployable actor inputs.

## Exact implementation

Complete v0.36.0 `ppo_walking_landing_critic.yaml` sets
`environment.critic_landing_state: true`. All other recipe values equal the
completed v0.35.0 control; only version/name/output labels also change.
The new flag defaults false, preserving historical configurations/replay.
It requires the Paper recipe and a positive qualifying-stance duration.

Append six **current** values after the existing critic history:

| Flat critic index | Value | Units |
|---:|---|---|
| 1515 | Left supported air duration | TB-equivalent seconds |
| 1516 | Right supported air duration | TB-equivalent seconds |
| 1517 | Left pending landing credit | TB-equivalent seconds |
| 1518 | Right pending landing credit | TB-equivalent seconds |
| 1519 | Left stance qualification progress | `min(samples,N)/N` |
| 1520 | Right stance qualification progress | `min(samples,N)/N` |

Durations divide physical seconds by `s=.8240547565448791`. Progress is clipped
to [0,1]; counter values beyond qualification do not need a growing input.
Queued duration excludes the constant reward weight and discount compensation.

Actor input remains **1005**, critic becomes **1521**. The existing 15×101
critic history stays intact; the six current values are not duplicated across
every historical frame. The previous tail is excluded before rolling history.
Actor shape/history and the controller/deployment action contract are preserved.

Features describe the **updated next-state memory**, after accumulating,
queuing, paying or rejecting credit on the physical transition. Reset,
fall and timeout clear the tail consistently with environment memory.
Features are included in the existing observation-finiteness check.

Reward formulas/weights, sustained landing, reference generation, actuation,
noise/randomization, commands, network hidden sizes/distribution, optimizer
and learning horizons match the completed control. This changes critic
information and its input layer, not the walking objective.

## Checkpoint conversion

Restore the accepted parent:

```text
results/wr2_walking/wr2_paper_size_aligned_20261030_085111_seed3/checkpoints/000200048640
```

Original actor/critic/normalizer tensors are restored. Six zero-weight rows
extend the critic's first-layer kernel; its existing rows and all actor
parameters are preserved. Critic normalizer statistics append zero mean/
summed variance and unit standard deviation. The experiment keeps observation
normalization disabled, as in the parent.

Initial policy and value predictions are verified on CPU in float64 while
native arithmetic differences are recorded. This distinguishes GPU matrix-
width rounding from a wrong mapping. This experiment's source actor shape is
1005 and critic 1515; already-converted 1521 checkpoints restore without another
extension. Combined conversion from older unheaded actor layouts is not needed
for this parent.

`restore_report.json` records conversion kind, old/new dimensions, parameter
digest and prediction differences. This uses the regular Paper restoration
path. Native continuation starts new Adam/RNG/environment state and step
counters while preserving learned actor means/scales and critic weights.

## Direct GPU command

From the updated WR2 GPU checkout:

```bash
uv run --locked --python 3.10 --extra training python -m wr2.locomotion.train \
  --config wr2/locomotion/configs/ppo_walking_landing_critic.yaml \
  --restore-checkpoint results/wr2_walking/wr2_paper_size_aligned_20261030_085111_seed3/checkpoints/000200048640
```

The normal trainer performs its GPU and full-reference preflight checks.
Output prefix is `wr2_paper_landing_critic`. Requested 200M rounds to 200,048,640
added transitions/195,360 Adam updates, reaching 400,097,280 cumulative
transitions from this parent. Sixteen evaluations retain 15 learned checkpoints
plus `initial_params`; initial/selected policies receive held-out confirmation.

The comparison is already available:

```text
results/wr2_walking/wr2_paper_size_continue_20261030_153953_seed3
```

Both restore the same 200M actor/critic/normalizer and use the same 200M budget,
seed3, optimizer reset and recipe. Another unchanged control run is unnecessary.
The control's selected checkpoint is +53M, final is +200M; compare trajectories
and equal added-step policies as well as selected candidates.

## Measurement and limits

First verify finite learning and recorded 1515→1521 conversion. Compare value
loss and, during result review, actual on-policy value/TD residuals around
qualified payouts. Global value loss alone does not identify payout-specific
accuracy; stochastic rollouts must retain current reward state and predictions
for that analysis. The six fields are directly available in the new critic
input and the environment's existing `step_air_time` state.

Primary gait checks are cadence and more even positive landing leads at
.10/.15/.20 TB commands while preserving desired-direction speed, heading,
support/self-contact and body motion. Use the same 1.5 TB-second warmup and
force classification. Nominal Mac measurements remain acceptable for the
current walking priority; randomized GPU testing is optional for higher-command
robustness. The existing standard inline GPU suite runs each checkpoint through
.10 TB speed, standing and transitions.

Retain the accepted parent if no learned policy improves walking. The selector
still optimizes eligible desired-frame XY error among learned checkpoints;
`beats_initial_tracking` and actual gait checks determine whether to promote.
Do not assume the largest checkpoint is best. Preserve the existing 160M parent-
run placement benchmark. Do not change contact weight or air-time weight in
this experiment; critic information needs an interpretable comparison.

Explicit state may improve learning accuracy without changing the preferred
gait. The current objective may still prefer fast qualified steps. A contact-
reference weight experiment remains separate if cadence persists.

## Verification

All 18 focused tests pass on the Mac. Checks cover configuration isolation,
strict flag validation and round trip; matched actor observations, physics,
rewards and the old 1515 critic features;
updated credit/progress, once-only payment, stop cancellation and reset/fall/
timeout clearing; parameter conversion with and without observation
normalization; native-rounding versus mapping-error discrimination; native
PPO updates/new input gradients and checkpoint-policy reload.

The accepted real parent also receives a 64-transition native CPU PPO restore
test, retaining its actor parameters exactly and verifying finite updates and
exact reload. Initial actor and critic prediction differences are zero in that
CPU check. All changed Python files pass Ruff. These development checks use
exact reduced reference rows, while
the production trainer retains its full grid gate. Validation weights are
discarded and remain in the parent's ignored `landing_critic_preflight_20261030/`.

References: local TB **`paper-walking-baseline`** at `a4f693d`,
`mjx_env.py::_get_obs` and `walk_env.py::_reward_feet_air_time`;
[TB paper v2, Tables VI–VII](https://arxiv.org/html/2502.00893v2);
[Asymmetric Actor Critic for Image-Based Robot Learning](https://arxiv.org/abs/1710.06542);
[400M review and information probe](wr2_size_continuation_result_20261030.md).
