# WR2 XY tracking-priority result — 2026-10-25

## Conclusion

**Increasing the existing XY reward coefficient 5 →15 substantially improves
learned command following and forward progression. It does not complete
balanced alternating placement.** This is the original PPO policy, with no
manual reference/joint overrides.

The fresh **300M** run selects **300,134,400** transitions. Independent XY error
is **.045281 / .045869 TB m/s** on selection/held-out seeds: **40.9% /39.4% lower**
than the qualified-step control. Both primary suites have **0/1920 falls** and
zero nonfinite episodes. Error remains **22.3% above TB Paper selected 200M**.

Mean forward response at +.10 TB m/s is **.08787**, versus control **.04467**
and TB **.09053**. Nominal actual same-foot advances reach approximately TB's
size-normalized values and forward rocking decreases substantially.

The remaining gait asymmetry changes sides: the left hip now swings well,
while the right hip approaches the backward policy bound and loses useful
cyclic motion. At 300M the right foot lands almost level with the left. Foot
self-contact also increases under randomized evaluation. Keep **XY 15** as
the stronger progression baseline and focus the next review on preserving
right-leg passing and reducing contact, rather than changing actuator physics
or increasing commanded speed.

Run: `results/wr2_walking/wr2_paper_progress_20261025_085931_seed3`, **0.29.0**.
[Experiment and its prior evidence](wr2_progress_reward_experiment_20261025.md).

## 1. Source and experiment integrity

Saved source is clean **`948d577`**. Compared against completed qualified-step
0.28.8 at `wr2_paper_step_20261022_143550_seed3`:

- The only training configuration change is **`lin_vel_xy: 5 →15`**; other
  differences are version, name and output prefix.
- `initial_params` are byte-identical, SHA256
  `8318472aee60c1dc2ac8898ab59e57fd3765794c1ee6d52e755def2ab38a170b`.
  Both runs start fresh, seed 3, without restoration.
- Physics/model hashes, complete actuator manifests, action scale, home,
  reference diagnostics and full-grid preflight results match exactly.
- Both GPU runs use RTX 5070, Python 3.10.19, JAX .6.2, Brax .14.1,
  Flax .10.7 and MuJoCo/MJX 3.14.0, with the same matrix precision and flags.
- Production environment, reward, controller and reference source are unchanged
  between the two training commits. Evaluator changes only describe qualified
  air-credit metadata; PPO diagnostic additions do not change the trainer.

Read current WR2 reward/environment/evaluator/trainer and local TB
**`paper-walking-baseline` at `a4f693d`**, including `paper.gin`, `walk.gin`,
`mjx_env.py`, `walk_env.py` and reference path integration. TB's effective XY
coefficient remains **5**, air **500**, leg imitation **5**. Both use
`exp(-100 * squared XY error)` against commanded body velocity, moving
actual-leg-reference imitation, and fixed-home controlled leg targets.

WR2 retains its requested supported-air/sustained-landing treatment and
collision coefficient **1**, versus TB's immediate landing credit and collision
**.1**. The new XY coefficient is an explicit additional departure. TB's
successful recipe remains the comparison, not its active development branch.

Reference: [ToddlerBot Tables VI–VII](https://arxiv.org/html/2502.00893v2).

### Metric checks

Recomputed every checkpoint's independent score from the ten fixed nonzero
conditions' per-episode errors; reports/history agree within `1e−8`.
All fifteen conditions count toward falls. The selector uses the existing
1% survival gate and independent XY error, not the changed training reward.
Selected `best_params` and final `params` are byte-identical.

WR2 velocities divide by **`s=.8240547565`**, times by `s`, distances by
**`s²=.67906624`**. A .10 TB m/s command is **.0824055 physical WR2 m/s**.
All primary tables below use TB-equivalent velocity units. Nominal foot
distances are physical WR2 mm; TB distances are multiplied by `s²` for comparison.

Native WR2 component returns already include `dt=.02`; TB native component
metrics omit it and are multiplied by .02. Component sums reconstruct native
returns within .002. For unequal episode lengths, rates use sum of returns
over sum of TB-equivalent durations, not an assumed full-length episode.
PPO losses and all primary checkpoint evaluations remain finite; late native
`training/kl_mean` is approximately **.0235**. No optimizer explosion is evident.

## 2. Independent evaluation and progress

| Policy | Selected transitions | Selection XY error, TB m/s | Held-out XY error, TB m/s | Primary falls: selection / held-out |
| --- | ---: | ---: | ---: | --- |
| Qualified-step control | 120M | .076661 | .075710 | 0 /5 |
| **XY15 progression** | **300M** | **.045281** | **.045869** | **0 /0** |
| TB Paper | 200M | .037025 | Not in this copied suite | 8 /— |

The new policy also beats its own initial .071623 tracking score. Initial
policies and independent metrics match the control before training.
These are one paired WR2 training seed and two evaluation seeds; they do not
establish generalization across independent training seeds.

Paired bootstraps preserve the 128 shared model/reset groups across all ten
commands. New-minus-control error:

- Selection: **−.031380 TB m/s**, 95% interval **[−.032431, −.030375]**.
- Held out: **−.029840**, interval **[−.030871, −.028852]**.

These intervals quantify evaluation variation of these particular policies,
not training-seed uncertainty.

| Transitions | New XY error | Control XY error | TB Paper XY error | New forward +.10 response |
| --- | ---: | ---: | ---: | ---: |
| 20M | .063263 | .096929 | .062016 | .07312 |
| 40M | .055216 | .085939 | .047711 | .07756 |
| 80M | .052354 | .075887 | .039653 | .07707 |
| 120M | .050811 | .076661 | .039524 | .07567 |
| 200M | .047329 | .082414 | .037025 | .08712 |
| 300M | .045281 | .078199 | .037999 | .08787 |

All sixteen evaluations/checkpoints remain available. 300M wins the selector,
although 280M is nearly tied at **.045290**; the last 20M bring little score
change. Earlier checkpoints remain essential for analyzing the developing
asymmetry. Training plus recorded evaluation takes about **2.72 hours**, versus
2.68 for the control; this pair does not establish an efficiency change.

### Directional response

Mean body commanded-axis velocity on selection seed 12102:

| Command, TB m/s | Control 120M | XY15 300M | TB Paper 200M |
| --- | ---: | ---: | ---: |
| Forward +.10 | +.04467 | **+.08787** | +.09053 |
| Backward −.10 | −.02713 | **−.08365** | −.07357 |
| Forward +.05 | +.02157 | **+.04109** | +.04401 |
| Backward −.05 | −.00713 | **−.02782** | −.04111 |
| Right −.05 | −.01601 | **−.03405** | −.03204 |
| Left +.05 | +.01354 | **+.02772** | +.03499 |

Mean forward response almost doubles and reaches 88% of command. Physical
mean body forward velocity is **.0724 m/s**. Forward instantaneous XY error
also decreases **.08158 →.04252**, versus TB **.03729**. Better mean speed
alone is not substituted for better tracking.

Every fixed nonzero condition has lower XY error than the control. Remaining
gaps are largest for small backward/left commands and backward diagonals:
backward-right error **.05284**, versus TB **.03768**; backward-left
**.05080**, versus **.03859**. Standing XY speed/error improves
**.05556 →.03231**, close to TB **.03196**. Held-out standing is .03269.

The pushed held-out companion suite improves error **.20118 →.14346**, with
falls **1154 →846 /1920**, **60.1% →44.1%**. This remains poor push robustness.
Both WR2 acquisition runs train without physical pushes, matching TB Paper's
effective implementation; this companion suite is separate from primary
checkpoint selection and does not establish a TB pushed-policy comparison.

## 3. Original PPO motion and the remaining asymmetry

Replayed thirteen nominal cases on the Mac: seven selected commands,
20M/40M/80M/120M/200M forward, and the control's selected forward policy.
All thirteen complete **824 ticks**, approximately 20 TB seconds, without
falls or nonfinite events. Actor outputs are used directly; there are no joint
or reference overrides. Every required reference frame matches the older
full-grid GPU recording within **9.1e−13 rad**. Reward sums and delayed-target
reconstruction pass. The control reproduces its prior gait closely.

Mac runtime is JAX .6.2, Brax .14.2, MuJoCo/MJX 3.14.0, Flax .11.2, high
matrix precision. These detailed gait numbers describe nominal CPU recordings,
alongside TB's existing native GPU recording. Native randomized GPU evaluation
remains the source for the primary scores and contact-frequency regression.

After at least 1.5 TB seconds of warmup:

| Actual gait measurement | Control 120M | XY15 300M | TB Paper 200M |
| --- | ---: | ---: | ---: |
| Net world forward, TB m/s | .05186 | **.08495** | .08841 |
| World-X interval velocity std, TB m/s | .03270 | **.01587** | .01565 |
| Lateral body velocity std, TB m/s | .04278 | **.02693** | .01965 |
| Left same-foot world-X advance, physical WR2 mm | 26.08 | **41.30** | 41.52 at WR2 size |
| Right same-foot advance, physical WR2 mm | 25.73 | **43.17** | 41.07 at WR2 size |
| Left landings ahead | 0/25 | **25/25** | 25/27 |
| Right landings ahead | 25/25 | **0/25** | 27/27 |
| Mean left landing lead, physical WR2 mm | −48.80 | **+43.18** | +18.74 at WR2 size |
| Mean right landing lead, physical WR2 mm | +72.36 | **−1.05** | +22.99 at WR2 size |
| Left hip actual range | .31° | **15.45°** | 14.65° |
| Right hip actual range | 8.07° | **1.07°** | 15.92° |

Same-foot advance measures one foot between its successive sustained landings.
Landing lead measures that foot relative to the other foot at its landing.
Their distinction matters: both WR2 feet now advance in the world, while they
still do not normally exchange the lead. Right landings are near level rather
than clearly forward. The right knee still moves about **17.75°**, and both
feet lift and land repeatedly; a nearly fixed hip does not mean a frozen leg.

### When the right swing diminishes

| New checkpoint | Right hip actual range | Right phase-binned requested range | Right mean raw mu | Right tanh sensitivity | Right landings ahead |
| --- | ---: | ---: | ---: | ---: | ---: |
| 20M | 7.57° | 8.48° | −.695 | .636 | 25/25 |
| 40M | 6.43° | 6.79° | −1.008 | .445 | 25/25 |
| 80M | 4.04° | 5.13° | −1.730 | .183 | 25/25 |
| 120M | 1.64° | 3.03° | −2.018 | .109 | 25/25 |
| 200M | 1.34° | 1.88° | −2.401 | .059 | 0/25 |
| 300M | 1.07° | 2.34° | −2.208 | .075 | 0/25 |

The right raw phase-bin range grows **.855 →1.539** from 20M to 300M, while
the bounded requested range shrinks. Its mean shifts to the backward endpoint
of WR2's fixed-home policy interval. Right physical-forward target mean declines
**8.46° →2.88°**, close to the **2.31°** endpoint. Both hip-pitch joints have
zero physical target clipping and slew limiting in these nominal windows.
The proximate mechanism is mean bias/tanh compression, followed by attenuation
of the small remaining target wave; the gait phase signal is still present.

The control had this mechanism at the left hip; the new policy develops it at
the right. Early new checkpoints execute passing on both sides using the same
model and action family. This weakens a one-sided mechanical impossibility as
the explanation, while not proving that physical response has no influence.

Nominal reward rate rises **26.828 at 20M →29.102 at 300M**, with XY
**9.753 →11.182** and yaw **3.352 →3.910**, while air is **10.413** in both
windows. Left imitation contribution improves **−.0615 →−.0452**, right
worsens slightly **−.0648 →−.0671**, and total imitation improves. The shared
mean-over-joints imitation term permits those changes to offset each other.
These are trajectory-level incentives, not isolated historical PPO gradients.
The current objective accepts the final asymmetric policy while its overall
progression and smoothness improve.

## 4. Contact and reward interpretation

Under native randomized fixed-walking evaluation, foot self-contact rises
**.359% →3.366%** of ticks; forward-only **.395% →2.814%**. This is a
regression despite zero primary falls. Mean fixed-walking torque RMS changes
only **.467 →.470 Nm**, so improved motion does not require a large increase
in this average load measure.

Nominal selected forward self-contact is **.92%** after warmup, versus zero
in the reproduced control and TB's nominal recording. The existing viewer's
cached-frame distance calculation finds maximum overlap about **.192 mm** and
maximum recorded foot self-force **1.14 N** over the whole selected recording.
These nominal magnitudes do not describe every randomized collision or certify
the collider's match to real foot geometry.

Current-contact double support is **4.07%** nominal, versus control .13%
and TB 13.95%; randomized fixed-walking averages **4.44%**, versus control
2.72%. These are sampled contact labels, not vertical load shares or proof that
a prescribed double-support fraction would improve walking.

The larger native total return **664.18 versus 498.28** cannot by itself prove
better behavior because the coefficient differs. Rescoring the new native
selected evaluation at XY5 gives approximately **501.32**, close to the old
selected 498.28. The unweighted native mixed-command XY reward mean improves
approximately **.546 →.814**, versus TB selected **.819** after correct
coefficient/timestep/episode-length accounting. This native mixed-command
statistic remains separate from the fixed-command benchmark, where the 22.3%
TB gap remains.

The independent scores, actual forward movement and learned step recordings
provide the evidence of improvement. The controlled configuration/initialization
comparison supports a reward-balance contribution to the previous progression
gap; it does not imply that every residual problem is solved by this coefficient
or that it works across all training seeds.

## 5. Next priorities

1. **Keep XY15 and selected 300M as the progression baseline.** Retain the
   qualified-landing rule and collision 1.0. The result supports the coefficient
   change; another generic increase in XY or removal of air is not selected.
2. **Focus the next reward investigation on preserving bilateral passing.**
   Use saved 20M/40M and 120M as comparators before the right-side collapse.
   Compare their per-leg reference/placement incentives and the later mean
   drift, while preserving the stronger command tracking. A body-speed reward
   can be satisfied by the observed asymmetric motion. A new placement or
   symmetry incentive requires a specific comparison with TB's reference and
   successful policy; its coefficient/formula is not selected in this review.
3. **Treat foot-contact regression as a parallel qualification metric.**
   Inspect the learned viewer recordings and confirm the detailed passing
   pattern with the existing native GPU stance diagnostic, particularly under
   randomization. Primary GPU contact data already confirm the regression;
   CPU placement details alone do not quantify its full exposure.
4. **Evaluate progression, small-command/diagonal tracking and contact together.**
   Keep intermediate checkpoints because better tracking and more balanced
   passing are not the same selector. Push robustness is a later acquisition
   objective, separate from the current no-push TB comparison.

## 6. Artifacts and visual evaluation

Recomputed data, input hashes and figure:
`results/wr2_walking/wr2_paper_progress_20261025_085931_seed3/review_20261025`.
It contains `analyze.py`, `analysis.json`, `analysis.log`,
`progress_comparison.png`, and thirteen original-policy recordings under
`nominal_cpu`. The viewer loader validates the selected 824-frame recording.

Run the selected trained policy directly on the Mac:

```bash
uv run --no-sync mjpython -m wr2.tools.view_walking \
  --run-dir results/wr2_walking/wr2_paper_progress_20261025_085931_seed3 \
  --playback-speed 0.25
```

For an earlier policy, add `--checkpoint` with its directory, for example
`results/wr2_walking/wr2_paper_progress_20261025_085931_seed3/checkpoints/000040017920`.
On Linux use the [direct viewer command](wr2_stance_diagnostic_20261018.md#run-a-trained-wr2-policy-directly).
The archived NPZ files remain available through `--trace` for exact recorded
frame inspection. All actions are from their saved PPO policies.

[Comparison figure](../../results/wr2_walking/wr2_paper_progress_20261025_085931_seed3/review_20261025/progress_comparison.png)
and [audited report](../../results/wr2_walking/wr2_paper_progress_20261025_085931_seed3/review_20261025/analysis.json).
