# Qualified step air reward experiment — 2026-10-22

**Prepared: 0.28.8, fresh seed 3, 300M transitions**, using the regular WR2
trainer and [experiment YAML](../../wr2/locomotion/configs/ppo_walking_step.yaml).
The completed 0.28.7 collision-1 run is the control. The existing
`ppo_walking_paper.yaml` remains that baseline; the experiment has its own YAML.

## Treatment and rationale

Replace only the Paper air-time credit definition:

1. Accumulate a foot's unloaded time only while the opposite foot is loaded.
2. At an eligible walking landing, save that supported swing credit.
3. Require sustained sampled contact before paying it once.
4. Cancel pending credit on a contact break, inactive command or episode reset.

Air coefficient **500**, collision **1**, all other rewards, physics, command
sampling, noise/randomization, action mapping, reference, actor/critic inputs,
network and PPO settings match the control. Raw TB air/clearance accumulators
continue normally; the other 24 reward terms retain their definitions.

The sole non-metadata configuration change is
`environment.air_time_min_stance_s`: **0 → .1648109513 physical seconds**,
which is **.20 TB seconds × s=.8240547565**. Zero selects the original TB rule;
a positive value selects this completed single-support step treatment.

The .10 TB-second accounting example rejected right-foot taps but retained
much left-foot short-stance credit. This experiment uses .20 TB seconds to
reject both feet's fast contacts. The copied TB policy's measured normal
landings still pass that threshold.

At the fixed 20 ms control period, qualification requires **10 consecutive
positive contact samples**. They span **9 intervals /180 physical ms**,
approximately **.218 TB seconds**, the first observable span above the requested
164.8 ms. A single force sample does not establish elapsed support. The native
summed floor-force threshold remains **1 N**; no collider or force rule changes.

The count is a sampled support criterion. It does not prove uninterrupted
contact between physics/control samples or independently qualify horizontal
foot placement. Actual commanded translation remains an evaluation requirement.

## Resolve the payout timing issue

Let supported swing credit `A` be recorded at landing tick `t`. Payment becomes
eligible `d=9` ticks later. Use the **actual PPO discount** `gamma=.97` and pay:

`A * gamma**(-d)`.

Then the contribution to discounted return at the landing time is preserved:

`gamma**(t+d) * A * gamma**(-d) = gamma**t * A`.

The compensation factor is **1.31538956**. It applies only to qualifying credit;
short contacts, flight-only airtime and cancelled pending credits receive none.
Training, smoke tests and evaluation pass the configured PPO discount to the
environment, including CLI overrides. This changes no PPO loss or optimizer.

This preserves the discounted value of eligible credit, not identical TD errors,
GAE estimates or policy updates. Delayed events can cross rollout boundaries;
existing histories and value learning still need to predict them. No additional
actor/critic features are introduced in this controlled trial.

### Production-function checks on existing recordings

The implementation was checked against the recorded contacts using the actual
JAX reward transition, not just the earlier accounting prototype:

| Existing WR2 control | Original total | New total | Original discounted return from reset | New discounted return from reset |
| --- | ---: | ---: | ---: | ---: |
| Learned tapper | 367.31 | 203.91 | 14.77 | **9.04** |
| All-leg reference | 308.13 | 352.64 | 9.86 | **9.45** |

Both raw and discounted reset ranking now prefer the reference control. The
reference remains a prescribed, swaying control rather than a balanced trained
policy, so this is reward discrimination evidence, not proof of training success.

On the copied successful TB Paper forward recording, physically credited time
remains **16.94 foot-seconds**, and discounted air return remains **4.506896**
within numerical tolerance. The original TB implementation has no stance gate;
this is an explicitly justified WR2 reward change targeting its measured taps.

Because payment is compensated, **raw weighted air return can increase for valid
steps** despite the unchanged coefficient. For the reference recording, physical
qualified credit is **14.62 s**, while its compensated air reward is **192.31**.
Use the physical-credit metrics to assess behavior; raw reward totals alone do
not establish useful walking or a matched cross-robot objective.

[Verification data and source](../../results/wr2_walking/wr2_paper_collision1_20261020_072851_seed3/review_20261022/prepared_step_experiment/verification.json)
include configuration, source hashes, event timing and retained/cancelled credit.

## Run on the GPU

From the GPU's WR2 checkout after syncing the prepared commit, first exercise
the actual GPU PPO update path with the normal training batch:

```bash
uv run --locked --python 3.10 --extra training python -m wr2.locomotion.train \
  --config wr2/locomotion/configs/ppo_walking_step.yaml \
  --num-timesteps 40960 --num-evals 2 --num-eval-envs 8 \
  --run-id dry_step_reward
```

This retains **1024 environments ×20 ticks**, four minibatches and four PPO
passes. Only the short budget and evaluation count/env count differ from the
full experiment. The saved config records those overrides. It verifies the
Linux Python-3.10/Brax-0.14.1 profile and logs qualified/rejected credit and
reward components. The GPU run has not been executed from this Mac session.

Start the full experiment using the same normal entrypoint:

```bash
uv run --locked --python 3.10 --extra training python -m wr2.locomotion.train \
  --config wr2/locomotion/configs/ppo_walking_step.yaml
```

It starts fresh: no `--restore-checkpoint`. Output is automatically named
`results/wr2_walking/wr2_paper_step_<timestamp>_seed3`.
Use the initial parameter file from the GPU dry run to check initialization
against the completed seed-3 control; reward-state additions do not change the
network dimensions or RNG splits. Numerical identity across runtimes is not
assumed before that GPU check.

## Evaluation and interpretation

Keep the existing **300M** budget, 16 evaluations and roughly 20M checkpoint
spacing. Retain every trained checkpoint and select using the original
TB-comparable `paper_xy_error` suite; the best need not be the final checkpoint.
The independent 15-condition suites and pushed diagnostic remain configured.

Require useful progress, not just disappearing air credit:

- Forward/backward/right/left commanded-axis speed and cross-drift improve.
- Right and left foot contact intervals sustain normal support, with meaningful
  placement advances rather than a slower tap or prolonged one-leg stance.
- Standing and command transitions remain controlled.
- Lateral sway, yaw/roll rates, falls, clipping and execution stay acceptable.
- Check qualified, rejected and raw landing credit alongside the unchanged
  primary tracking metrics and visual recordings.

After training, use the same diagnostic on the saved run:

```bash
uv run --locked --python 3.10 --extra training python -m wr2.tools.diagnose_walking \
  --paper-stance \
  --run-dir results/wr2_walking/wr2_paper_step_<timestamp>_seed3 \
  --output results/wr2_walking/wr2_paper_step_<timestamp>_seed3/stance_diagnostic
```

Schema 4 records all reward components plus physical candidate/qualified/rejected
credit and observed stance duration. `air_time_credit_s` is the **compensated
seconds-equivalent reward credit** in this mode; `qualified_air_time_s` is the
uncompensated physical time paid. The metadata records threshold, sample count,
delay, discount and compensation. Stance-duration samples are current ages;
use contact intervals for a duration distribution rather than summing ages.

## Local verification and references

**42 focused tests pass**, including the native Brax update check. Checks cover one-time payment, short contacts, separate contact runs, supported
versus flight time, stop/restart gating, exact discounted equivalence, real
environment timeout clearing, unchanged physics/observations/other rewards and
configuration-only treatment differences. A native Brax CPU update and the
regular entrypoint's full 227-command reference preflight/30-step smoke were
also exercised. The remaining runtime gate is the GPU dry update above.

Sources: [reward transition](../../wr2/locomotion/paper_rewards.py),
[environment](../../wr2/locomotion/walking_env.py),
[tests](../../tests/test_step_reward.py), [reward audit](wr2_reward_review_20261022.md).
Local TB code was checked on `paper-walking-baseline`, revision `a4f693d`.
Public references are [ToddlerBot, Tables VI–VII](https://arxiv.org/html/2502.00893v2)
and [Isaac Lab v2.2 biped reward definitions](https://github.com/isaac-sim/IsaacLab/blob/v2.2.0/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/mdp/rewards.py).
They provide the baseline and single-support reward intent; neither contains
this exact delayed-qualification implementation.
