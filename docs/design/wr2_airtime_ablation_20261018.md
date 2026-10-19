# WR2 air-time acquisition control — 2026-10-18

**Prepared: 0.28.6, fresh seed 3, 300M transitions.** The controlled objective
change is `feet_air_time: 500 → 0`. Use the regular WR2 trainer.

## Hypothesis and evidence

The leading working hypothesis is that the reward allows WR2 to acquire
near-stationary foot unloading instead of commanded translation. The
[completed 0.28.5 review](wr2_no_push_result_20261018.md) identifies a concrete
incentive competing with the reference support schedule. In the selected
260M policy's 824-tick nominal forward trace:

- All **549 planned single-support ticks** have the expected supporting foot.
- Only **7 of 275 planned double-support ticks** have actual double support.
- Foot-to-foot contact occurs on **789/824 ticks**; the unloaded sole is
  typically only millimeters above the floor.
- Recorded contacts earn **161.40** weighted air-time credit. Applying the
  same reward arithmetic to the reference contact schedule gives **108.00**,
  while contact-reference matching would gain only **5.38**.

The alternate contact-schedule calculation is undiscounted reward arithmetic,
not a simulated translating gait. It establishes a substantial competing
incentive, but does not prove that air time caused convergence or that the
stationary policy has greater total return than every feasible walker.

Frozen TB `paper-walking-baseline` (**a4f693d**) also uses weight **500**, pays
accumulated air time at landing and requires no actual progress or minimum
clearance. It succeeds: Paper 200M has normalized walking error **.037025**,
versus WR2 260M **.076838**. No corresponding TB policy support-phase trace
has been measured yet. This ablation tests whether that incentive contributes
to WR2's acquisition failure; it is an experimental departure from the Paper
objective, rather than an established correction to TB's recipe.

## Controlled training setup

Relative to the saved 0.28.5 configuration, effective settings differ only in
the air-time coefficient, version/name and output prefix. The new prefix is
`wr2_paper_airtime0`. Use fresh seed-3 actor/critic/Adam initialization, fixed
LR **1e−4**, **1024 ×20** rollouts, four PPO passes and **300M requested steps**.
Use the same actuator model, joint limits, fixed grounded home, moving
leg-reference imitation, other reward coefficients, reference, observations
and model/reset/noise variation for the comparison. Physical pushes remain
zero during acquisition; sliding friction remains **.7–1.3**.

Physical control ticks are **20 ms**, about **24.27 ms in TB time units**.
The WR2 gait cycle is **.6 physical seconds**, about **.728 TB-equivalent
seconds**. Training episodes are 1,000 ticks; comparison episodes are
824 ticks, approximately **20 TB-equivalent seconds**. Speed normalization
uses **s=.824055**, with WR2 velocity divided by s for the TB comparison.

## Start on the GPU host

From the WR2 checkout:

```bash
uv run --python 3.10 --extra training python -m wr2.locomotion.train \
  --config wr2/locomotion/configs/ppo_walking_paper.yaml
```

Python 3.10 selects the existing lock's **Brax 0.14.1 / Flax 0.10.7** profile
used by the completed GPU control. Python 3.11 selects a different profile.
The trainer checks for a GPU and records package versions, devices, actual
matmul precision and XLA flags. The command starts a fresh run.

For an optional compilation/update check with the actual training batch:

```bash
uv run --python 3.10 --extra training python -m wr2.locomotion.train \
  --config wr2/locomotion/configs/ppo_walking_paper.yaml \
  --num-timesteps 20480 --num-evals 2
```

Then start the full command fresh. The short run verifies execution and
reporting; the full 300M control tests acquisition.

## Collect the old policy's matching measurements

The updated evaluator can measure the completed 260M control with its saved
weight-500 configuration, on both comparison seeds:

```bash
uv run --python 3.10 --extra training python -m wr2.locomotion.evaluate \
  --paper-benchmark \
  --config results/wr2_walking/wr2_paper_20261018_121025_seed3/training_config.yaml \
  --checkpoint results/wr2_walking/wr2_paper_20261018_121025_seed3/checkpoints/000260116480 \
  --seeds 12102,22102 --num-envs 128 \
  --output results/wr2_walking/wr2_paper_20261018_121025_seed3/review_airtime0_baseline.json
```

The saved configuration and checkpoint are run artifacts that must already
be present on the GPU host. This evaluation retains native variation and
zero physical pushes. The nominal single-episode trace above is evidence
for the hypothesis; this full evaluation supplies the comparable baseline.

## Measurements and decision

Retain all approximately **20M** checkpoints, initial parameters and final
parameters. Each callback evaluates **15 conditions ×128 episodes**, selection
seed **12102**, with paired no-push and pushed profiles. At completion, initial
parameters and the selected trained candidate also receive paired held-out
**22102** evaluations. `best_checkpoint.json` records whether its tracking
actually beats the initial policy; the existing finite/≤1%-fall selector still
chooses a trained candidate for confirmation.

New schema-3 measurements are independent of the air-time coefficient:

| Measurement | Meaning |
| --- | --- |
| `reference_double_support_fraction` | Fraction of evaluated ticks requesting both supporting feet |
| `reference_double_support_match_fraction` | Actual double support during requested double support, divided by requested ticks within each episode; null when none are requested |
| `contact_phase_match_fraction` | Fraction of the two individual stance labels matching the reference |
| `foot_self_contact_fraction` | Fraction of ticks with penetrating foot-to-foot contact and contact-force magnitude >.1 N |
| `air_time_credit_s` | Sum of command-eligible landing air durations before reward weight and control-period multiplication |

Average conditional support matches over eligible episodes, with their count
reported explicitly. Air-time credit is an episode sum. Fall-censored
episodes retain their measured length, so air-time totals need that context.
Compare actual speed/error, rather than total returns across the two different
objectives. Existing support, clipping, slew, leg-reference error, forces and
start/stop metrics remain available.

The acquisition result needs **better commanded translation and lower walking
XY error than its own initial policy and the completed 0.28.5 control**, on
both evaluation seeds, with finite states and ≤1% height failures. Report the
six cardinal speeds and all fixed-direction errors; a favorable aggregate
alone can hide a lost direction. TB Paper 200M **.037025** and 300M **.037999**
remain tracking references, with TB 180M also retained for survival comparison.

If translation improves and unwanted support timing/foot contact decreases,
the result supports an air-time incentive contribution. Repeat a promising
recipe with another training seed. If double support increases while speed
stays near zero, the term affected contact strategy without solving propulsion.
If tracking does not improve or survival degrades, air-time removal is
insufficient; use matched TB/WR2 reference execution and requested/delivered/
actual joint traces to choose the next plant or geometry experiment.

## Verification and sources

**20 targeted tests passed**, including native MJX transitions, reward timing,
PPO/checkpoint contracts and fall/nonfinite evaluation accounting. The new
test verifies exact q/qd/observation equality between weights 0 and 500 for
the same transition, and a reward difference exactly equal to
`500 ×dt ×unweighted_credit`. Conditional support denominators and summed
air credit have separate checks. The effective config audit found only the
four intended differences. CPU checks used JAX 0.6.2 with read-only Brax
0.14.1 sources and the existing Mac Python 3.11 environment; the GPU batch
check remains a host verification step.

Sources: [ToddlerBot paper, Tables VI/VII](https://arxiv.org/html/2502.00893v2),
local `paper-walking-baseline:toddlerbot/locomotion/{walk_env,mjx_env}.py` and
`paper.gin`, WR2 [rewards](../../wr2/locomotion/paper_rewards.py),
[environment](../../wr2/locomotion/walking_env.py),
[evaluation](../../wr2/locomotion/paper_evaluation.py) and the
[full setup audit](wr2_tb_paper_setup_audit_20261018.md).
