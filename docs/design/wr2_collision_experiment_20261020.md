# WR2 collision-weight experiment — 2026-10-20

**Completed:** `wr2_paper_collision1_20261020_072851_seed3` reached 300M.
Selected tracking improves 10.1% /10.5% on the two evaluation seeds and foot
self-contact is almost eliminated, but rightward walking and phase-dependent
weight transfer remain incomplete. See the
[result and proposed continuation](wr2_collision_result_20261020.md).
The preparation and commands below preserve this experiment.

**Prepared: 0.28.7, fresh seed 3, 300M requested transitions.** Test whether
increasing the existing collision coefficient from **0.1 to 1.0** teaches
foot separation together with better commanded walking. Use the regular
training entrypoint and canonical Paper YAML.

## Controlled change and TB comparison

The control is completed
`results/wr2_walking/wr2_paper_20261018_121025_seed3/training_config.yaml`.
The effective configuration differs in exactly four fields:

| Field | Completed control | Prepared experiment |
| --- | --- | --- |
| `reward_weights.collision` | **0.1** | **1.0** |
| `version` | 0.28.5 | 0.28.7 |
| `version_name` | Paper acquisition without physical pushes | Paper collision weight 1.0 control |
| `output.run_prefix` | `wr2_paper` | `wr2_paper_collision1` |

Version 0.28.6 identifies the historical, deferred air-time ablation.

Native TB **`paper-walking-baseline`**, revision `a4f693d`, uses collision
weight **0.1** in `paper.gin`. Its `_reward_collision` counts both directions
of active non-floor collider pairs with force magnitude above **0.1 N**.
WR2 additionally requires penetration and deduplicates manifold points on
one pair. One active foot pair contributes an unweighted **−2**. With the
shared `.02 s` control period, the penalty changes from **−.004 to −.04**
per contact tick. No force-proportional term or hip-angle penalty is added.

This is an approved departure from TB's coefficient, motivated by WR2's
frequent model foot self-contact while successful TB has none. The selected
WR2 nominal forward replay pays only **−3.16** collision return while earning
**+160.80** air-time return. Rescoring that unchanged trajectory at weight 1
would give approximately **−31.56** collision return; that arithmetic does
not predict the newly trained behavior.

The current foot boxes define the conservative clearance being penalized.
The [CAD audit](wr2_tb_stance_result_20261018.md) found box contact while the
imported foot parts remained about 2 mm apart. This experiment measures
avoidance of those model contacts; every such contact is not necessarily
real foot-on-foot loading. The
[phase review](wr2_phase_learning_review_20261020.md) also shows that outward
movement is possible with this model. This run tests an objective change,
without establishing the original cause of inward convergence.

## Training setup

Fresh actor, critic and optimizer; seed **3**, **300M** requested steps,
fixed **LR 1e−4**, **1024 environments ×20 ticks**, four minibatches and four
PPO passes. Native rounding retains 15 trained checkpoints about every 20M,
ending at approximately **300,134,400** transitions.

Air-time weight **500**, all other rewards, action mapping, target pipeline,
actuator model, reference, network/distribution, noise, model/reset variation
and command sampler retain the completed control's values. Acquisition has
zero physical pushes. Friction variation remains **.7–1.3**.

Size normalization is still `s=.8240547565`: physical linear velocity divides
by `s` for TB comparison; angular velocity multiplies by `s`. The `.6 s`
gait cycle is `.728 TB s`; comparison horizons are 824 ticks, approximately
20 TB seconds. The collision coefficient multiplies a contact-event count
and is not additionally scaled by robot length or mass.

## Run on the GPU host

After syncing the prepared commit to the WR2 checkout:

```bash
uv run --locked --python 3.10 --extra training python -m wr2.locomotion.train \
  --config wr2/locomotion/configs/ppo_walking_paper.yaml
```

Python 3.10 selects the lock's **Brax 0.14.1 / Flax 0.10.7** profile used by
the completed GPU control. The trainer checks for a GPU and records the
effective YAML, package versions, devices and source provenance. The command
starts fresh and writes
`results/wr2_walking/wr2_paper_collision1_<timestamp>_seed3/`.

Optional GPU compilation/update check with the full training batch:

```bash
uv run --locked --python 3.10 --extra training python -m wr2.locomotion.train \
  --config wr2/locomotion/configs/ppo_walking_paper.yaml \
  --num-timesteps 20480 --num-evals 2
```

Start the full command fresh after the short check. The short check verifies
execution/reporting; it does not measure learned walking.

## Measurements and decision

The trainer retains initial parameters, approximately 20M checkpoints,
`best_params` and final `params`. Every checkpoint has **15 conditions ×128
episodes**, selection seed **12102**, with primary no-push reports in
`paper_evaluations/` and a paired robustness profile in
`paper_push_evaluations/`. Completion confirms initial and selected parameters
on held-out seed **22102**. The existing selector minimizes fixed-condition
XY error among finite, complete suites with **≤19 height failures /1920**.
It does not select on collision reward or total return.

Compare the full checkpoint curve, especially **20M and 40M**, with the
completed WR2 control and TB Paper's corresponding stages and selected 200M.
Use each old run's saved configuration for its evaluation.

| Measure | Interpretation |
| --- | --- |
| `foot_self_contact_fraction` | Fraction of valid ticks with penetrating foot-pair contact and force magnitude >.1 N; independent of reward weight |
| Walking XY error | Existing ten fixed nonzero-command conditions, normalized to TB units |
| Signed net velocity/displacement | Actual progress along the command, including lateral drift; inspect nominal trace summaries |
| Hip requested/delivered/actual angles | Whether the policy learns separation and reduces inward saturation; distinguish actual support from desired phase |
| Floor support, starts/stops and height failures | Whether separation preserves useful coordination and survival |

Run the existing stance diagnostic after training, replacing the timestamp:

```bash
WR2_COLLISION_RUN="results/wr2_walking/wr2_paper_collision1_<timestamp>_seed3"
uv run --locked --python 3.10 --extra training python -m wr2.tools.diagnose_walking \
  --paper-stance --run-dir "$WR2_COLLISION_RUN" \
  --output "$WR2_COLLISION_RUN/stance_diagnostic"
```

Its `learned` traces and `net_velocity_TB_equivalent_m_s` summarize actual
commanded displacement. The reference-action controls remain diagnostics.
For earlier stages, add `--checkpoint "$WR2_COLLISION_RUN/checkpoints/000020008960"`
or `000040017920`, and use a distinct output directory for each.

Success requires **lower foot self-contact together with improved commanded
translation and XY tracking**, on both evaluation seeds, relative to the
completed control and the new run's initial policy. Wider standing with no
tracking improvement is not a walking improvement. Report each direction;
an aggregate can hide a regression. Compare physical metrics across runs:
their total rewards have different collision coefficients.

## Preparation verification

- Effective typed-config audit: only the four fields listed above differ
  from completed 0.28.5.
- Native MuJoCo test: floor contact costs zero; a real solver foot-pair contact
  costs **−.004 /−.04**; total reward difference is exactly **−.036** within
  numerical tolerance. q, qd, actor/critic observations and all other reward
  components are identical between the two weights.
- **3 targeted tests passed**: native collision-weight transition, frozen TB
  numeric reward oracle and finite/survival checkpoint selection.
- `uv lock --check` passed. Local checks used CPU Python 3.11 / Brax 0.14.2;
  the GPU batch check and long training run are host execution steps.

References: [ToddlerBot, Tables VI–VII](https://arxiv.org/html/2502.00893v2),
local `paper-walking-baseline:toddlerbot/locomotion/{paper.gin,mjx_env.py}`,
WR2 [collision calculation](../../wr2/locomotion/walking_env.py),
[rewards](../../wr2/locomotion/paper_rewards.py) and
[independent evaluation](../../wr2/locomotion/paper_evaluation.py).
