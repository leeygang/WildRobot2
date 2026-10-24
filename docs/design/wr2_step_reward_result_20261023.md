# WR2 qualified step reward result — 2026-10-23

## Conclusion

**The stance treatment produces sustained alternating steps in the nominal
replays, replacing the control's rapid foot taps. Overall command tracking
worsens.** The selected checkpoint is **120M**, with **.076661 / .075710 TB m/s**
selection/held-out error, **11.0% / 9.9% worse** than the collision-1 control.
Its selection error is **2.07× TB Paper 200M**.

This is useful evidence that the air-credit definition contributed to the
tapping strategy. It does not establish the complete cause of WR2's remaining
tracking gap. The [visual follow-up](wr2_step_placement_review_20261023.md)
confirms a large first right swing, then short swings with the left foot
consistently trailing the right. The current priorities are **missing left
forward hip swing, short strides and lateral velocity variation**, with
standing/stop behavior also worse. Another physics
change or a longer continuation is not justified by this result alone.

Run: `results/wr2_walking/wr2_paper_step_20261022_143550_seed3`, configuration
**0.28.8**, fresh seed 3, **300,134,400 transitions** completed.
[Prepared treatment](wr2_step_reward_experiment_20261022.md).

## 1. Source, configuration and metric audit

The saved configuration differs from the completed 0.28.7 collision-1 control
only in `air_time_min_stance_s` **0 → .1648109513 physical seconds**, version/name
and output prefix. Physics and actuator manifests match; `initial_params` are
byte-identical. Home/action mapping, reference, randomization, network, PPO and
reward coefficients match. Both runs start fresh, without restoration.

New training source is clean **`47d2d93`**, on an RTX 5070. The two GPU runs use
the same Python 3.10.19, Brax .14.1, JAX .6.2, Flax .10.7 and MuJoCo/MJX 3.14.0
profile. The run completes without nonfinite events in any primary checkpoint
evaluation. Training plus recorded evaluations take about **2.68 hours**, versus
2.75 for the control; this single timing pair cannot establish a speedup.

Checked current WR2 reward/environment/evaluator/trainer code and local TB
**`paper-walking-baseline` at `a4f693d`**. The frozen copies used for native TB
diagnostics exactly match branch objects for `walk_env.py`, `mjx_env.py`,
`paper.gin`, `mujoco_control.py` and `walk_zmp_ref.py`. The active TB branch is
not used as the baseline. TB retains air coefficient **500**, XY **5**, leg
imitation **5**, contact **1**, and its immediate landing-credit definition.
WR2's sustained-contact rule and collision coefficient **1** are declared
departures from TB. The published [Table VII](https://arxiv.org/html/2502.00893v2)
was also checked directly.

Metric accounting is verified as follows:

- Recomputed benchmark scores from the ten fixed nonzero conditions' episode
  arrays, preserving their averaging. All 15 conditions count toward survival.
- WR2 velocity divides by `s=.8240547565`; 824 physical ticks are approximately
  20 TB seconds. TB uses 1000 ticks. Foot advances divide by `s²`, times by `s`.
- Native WR2 `paper_reward/*` values already include `dt=.02`. TB native reward
  metrics include coefficients but omit `dt`; multiply by .02 before comparing
  component returns. Components reconstruct native returns within .001 WR2
  and .00005 TB, consistent with accumulated float precision.
- Native `*_per_step` evaluation metrics are episode averages. Multiply by
  1000 to obtain episode sums only when all native episodes reach 1000 ticks.
  The 80M/160M/200M native evaluations contain shorter episodes; means of
  rates times means of lengths are not substituted for exact sums.
- In this experiment, independent-evaluation `air_time_credit_s` is compensated
  seconds-equivalent credit. Divide by **1.31538956** for qualified physical
  foot-seconds. The copied report's landing-credit descriptor is stale; a
  metadata correction now describes this mode accurately, without changing
  numerical measurements, reward or training.

## 2. Training progress and comparison

Each primary WR2 suite has **15 conditions ×128 episodes**. Selection seed is
12102, held-out seed 22102. The selector requires no nonfinite episodes and
at most 1% height failures; it minimizes tracking error within that gate.

| Policy | Checkpoint | Selection error, TB m/s | Held-out error, TB m/s | Falls: selection / held-out |
| --- | ---: | ---: | ---: | --- |
| WR2 initial | 0 | .071623 | .071623 | 0 / 0 |
| WR2 collision-1 control | 300M | .069066 | .068908 | 0 / 0 |
| **WR2 qualified step** | **120M** | **.076661** | **.075710** | **0 / 5** |
| WR2 qualified step final | 300M | .078199 | Not separately copied | 0 / — |
| TB Paper | 200M | .037025 | Not in this suite | 8 / — |

The selected checkpoint also has **5.7% more held-out tracking error than its
own initial policy**. A nearly stationary initial policy can score well under
small commands; this comparison does not mean it walks better. It does mean
training has not yet produced uniformly accurate command following.

80M has the lowest unrestricted new score, **.075887**, but **20/1920 falls
(1.0417%)** narrowly exceed the configured 1% gate. Selecting 120M is correct.
`best_params` differs from final `params`; all intermediate checkpoints remain.

| Transitions | New WR2 error | Control WR2 error | TB Paper error |
| --- | ---: | ---: | ---: |
| 20M | .096929 | .181385 | .062016 |
| 40M | .085939 | .166579 | .047711 |
| 80M | .075887 | .130923 | .039653 |
| 120M | .076661 | .104939 | .039524 |
| 200M | .082414 | .078006 | .037025 |
| 300M | .078199 | .069066 | .037999 |

Early acquisition improves substantially; the control's strong early backward
motion under forward commands is absent. Forward progress peaks near 80M,
then declines. More transitions do not recover a consistent advantage over the
control. Late PPO KL is around **.02**, comparable to the control; the copied
losses do not show an optimizer explosion that explains the plateau.

Paired bootstraps resample the 128 shared model/reset groups across all ten
commands. New-minus-control error is **+.007595** on selection
(95% interval **+.00660 to +.00863**) and **+.006801** held out
(**+.00588 to +.00777**). These quantify evaluation variation for these policies,
not variation across independent training seeds.

This is a normalized native performance comparison: WR2 uses training seed 3,
TB seed 0, and each robot retains its native friction, observation/reset variation
and actuator implementation. It is not an experiment with identical robot physics.

### Directional response

Mean commanded-axis body velocity on selection seed 12102, in TB m/s:

| Command | Control 300M | New 120M | TB Paper 200M |
| --- | ---: | ---: | ---: |
| Forward +.10 | +.01696 | **+.04467** | +.09053 |
| Backward −.10 | −.03764 | **−.02713** | −.07357 |
| Forward +.05 | +.00525 | **+.02158** | +.04401 |
| Backward −.05 | −.01375 | **−.00713** | −.04111 |
| Right −.05 | +.00151 | **−.01601** | −.03204 |
| Left +.05 | +.01552 | **+.01354** | +.03499 |

Forward speed increases about 2.6× and rightward response has the correct sign.
Backward and leftward tracking worsen. Rightward instantaneous XY error also
worsens despite improved mean direction: displacement and smooth tracking are
different measurements. Held-out results confirm the directional pattern.

Standing XY speed/error rises **.0190 → .0556 TB m/s**; TB is .0320. All four
transition conditions have higher all-phase XY error than the control. These
are whole-condition measurements, not isolated stopping distances.

Pushed held-out evaluation remains poor: **1154/1920 falls (60.1%)**, versus
1218 (63.4%) for the control; error rises .19265 → .20118 TB m/s. This companion
suite is separate from acquisition and checkpoint selection. Both WR2 runs
train without physical pushes, matching TB Paper's effective implementation.

## 3. Actual steps and remaining coordination gap

Twelve nominal CPU replays use the unchanged native model/controller, learned
deterministic actions, no pushes/noise/variation, seed 22102. They cover seven
selected commands plus 20M/40M/80M/300M forward and the control forward policy.
**No actions are replaced with reference angles.** All twelve survive 824 ticks.
Every needed reference frame matches the original full-grid GPU reference
within `1e−7` rad; maximum measured difference is `1.82e−12`.

These replays use Mac Brax .14.2/Flax .11.2 and high matmul precision. The
control reproduces its copied GPU tapping pattern. New GPU evaluation directly
confirms balanced contact fractions, but its interval/placement distributions
have not yet been independently copied from GPU. Those detailed numbers below
describe nominal CPU episodes and the existing native TB GPU recording.

After a 1.5-TB-second warmup, forward +.10:

| Measure | WR2 control | New WR2 120M | TB Paper 200M |
| --- | ---: | ---: | ---: |
| Right contact onsets | 93 | **25** | 27 |
| Median right loaded interval, TB seconds | .049 | **.388** | .340 |
| Median same-foot onset interval, TB seconds | .194 | **.728** | .700 |
| Median right foot-center advance, TB mm | 2.95 | **38.04** | 63.53 |
| Right loaded fraction after warmup | 24.5% | **52.6%** | 49.8% |
| Lateral velocity std, TB m/s | .0369 | **.0446** | .0211 |

New left support also sustains **.340 TB seconds**, with **38.51 mm** median
advance. The forward request spectrum moves from **5.03 TB Hz** in the control
to **1.35 TB Hz**, close to TB's **1.4 Hz**. This is the full walking-cycle
frequency: successive landings of the same foot, approximately .74 TB seconds
apart. With alternating feet it corresponds to about 2.7 individual steps per
TB second. Frequencies satisfy `f_TB = s * f_physical`; 1.35 TB Hz is about
1.64 cycles per physical second for WR2. New forward net world velocity is
**.0530 TB m/s**, versus .0200 in the control. These are actual placements and
translation, beyond mere contact duration.

The short-stride gap remains: at 80M right advance is **44.22 mm**; at 300M
it falls to **29.26 mm**, while the .728-second period remains. Backward −.05
has only about **4 mm** advances per foot. Double support in nominal selected
forward is **.6%**, versus **13.9% TB**. GPU fixed-walking averages are
2.7% double support and 49.3%/53.4% left/right contact. These are support
labels, not body-weight fractions or proof of uninterrupted contact between
control samples.

The new selected nominal right ankle has **zero slew limiting** and execution
RMSE **.0304 rad**, versus 44.0% limiting and .1100 rad in the control after
warmup. Fixed-walking mean torque RMS falls .563 → .467 Nm, and target slew
fraction falls 3.81% → .25%. Thus the old rapid ankle-request/execution problem
is substantially reduced; persistent poor tracking cannot simply be attributed
to the same fast tapping loop.

Selected nominal foot self-contact is zero on all seven commands. Across
randomized GPU fixed walking, it increases from **.0091% → .359%** of ticks.
The improvement in stepping therefore comes with a small contact regression,
as well as worse standing and tracking. New leg-reference RMSE is **.216 rad**,
versus control .168 and successful TB .232; tighter imitation is not established
as the missing ingredient.

The [per-landing follow-up](wr2_step_placement_review_20261023.md) separates
startup from continued swings and qualifies the stepping interpretation:
the selected left foot keeps landing behind the right despite alternating
support, with nearly absent left hip-pitch motion.

## 4. What the reward experiment establishes

The same-seed, identical-initialization treatment changes the learned behavior
from taps to sustained alternating steps by 20M. This provides stronger evidence
for a reward contribution than frozen-action reward rescoring alone. It also
shows the current WR2 physical model can learn repeated foot loading at nearly
TB walking-cycle frequency. Robot physics can still influence acquisition and smoothness;
this experiment does not identify or exclude every physical contributor.

The new air rule qualifies support but does not require meaningful placement
or directly encourage double support. Forward nominal 120M realizes **16.10
physical foot-seconds**, with zero rejected candidates. Compensation turns that
into **21.18 seconds-equivalent credit /211.78 weighted return**. The same old
control under this rule earns zero qualified credit and rejects **16.32 s**.
The gate works as implemented; the learned policy can now satisfy it fully
while making shorter strides than TB.

At 20M, native air return is **56.9%** of total, versus control 67.2% and
TB Paper 39.8%. Removing compensation for accounting gives new qualified
physical credit **19.29 s**, versus control raw credit 22.44 s. This confirms
reduced early unloaded/flight exploitation; the higher new raw reward does not
indicate more physical air time. By 120M air is still 50.3% of native return.

There is direct evidence of reward competition on the new learned trajectories:
from 80M to 120M nominal forward net speed drops **.0631 → .0530**, but episode
return rises **413.85 →432.86**. Gains in orientation/angular/vertical stability
outweigh the small XY loss. Discounted return from reset also rises
**12.634 →13.114**. These compare complete recorded trajectories; they do not
reconstruct PPO advantages or prove which update caused the speed reduction.

An illustrative continuous three-term calculation on the same 16.48-second
timeline makes the remaining incentive explicit. Actual 120M air + XY + contact
return is **289.94**. Perfect XY and reference-like contacts with 2/3 single
support, 1/3 double support yield about **259.88**, including the new compensation.
This ignores boundary credit, other rewards and physical feasibility. It shows
why qualification alone does not ensure that accurate, longer walking has the
larger reward. TB also succeeds without exact reference contacts; blindly forcing
those contacts is not justified.

## 5. Next priorities

1. **Retain 80M and selected 120M for comparison.** Inspect the supplied learned
   replays and confirm their foot intervals/placements with the existing GPU
   stance diagnostic before treating the nominal measurements as runtime-wide.
2. **Focus the next reward review on left forward hip swing, progress versus
   stability and support duty.**
   Compare 80M/120M/300M and TB: commanded displacement, lateral velocity variation,
   double support, qualified physical credit and individual reward terms. The
   simple air landing gate has passed its behavioral test, but the objective
   still accepts short, swaying steps. Additional contact penalties, stronger
   reference imitation or a new PPO implementation do not follow from this data.
3. **Select any next training change only after that comparison.** Keep the
   sustained-step candidate for diagnosis and the collision-1 YAML as the
   control. The evidence supports reviewing air/translation/stability balance,
   not claiming that reducing air weight alone solves the remaining gap. No new
   reward coefficient, physics or training run is prepared in this review.

## Artifacts and visual evaluation

[Training and contact comparison figure](../../results/wr2_walking/wr2_paper_step_20261022_143550_seed3/review_20261023/training_comparison.png),
[GPU metric audit](../../results/wr2_walking/wr2_paper_step_20261022_143550_seed3/review_20261023/analysis.json),
[contact/placement summaries](../../results/wr2_walking/wr2_paper_step_20261022_143550_seed3/review_20261023/planting.json).
Audit/replay/figure sources are beside those ignored local artifacts.

On the Mac, view the selected learned forward episode:

```bash
uv run --no-sync mjpython -m wr2.tools.view_walking \
  --trace results/wr2_walking/wr2_paper_step_20261022_143550_seed3/review_20261023/nominal_cpu/selected_forward_010_learned.npz \
  --playback-speed 0.25
```

Use `80M_forward_010_learned.npz` or `300M_forward_010_learned.npz` in the same
folder to compare the retained checkpoints. These are recorded learned actions,
without action interventions. The existing TB Paper trace remains viewable with
`--robot toddlerbot`.

Sources: [WR2 reward](../../wr2/locomotion/paper_rewards.py),
[environment](../../wr2/locomotion/walking_env.py),
[evaluator](../../wr2/locomotion/paper_evaluation.py),
local `paper-walking-baseline` TB source and saved result provenance,
[ToddlerBot paper Tables VI–VII](https://arxiv.org/html/2502.00893v2),
[preceding reward audit](wr2_reward_review_20261022.md).
