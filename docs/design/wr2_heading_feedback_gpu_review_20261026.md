# Heading-feedback GPU comparison — 2026-10-26

## Conclusion

The larger GPU evaluation confirms the continuation's improvement. All
**2,304 randomized attempts** and **9 nominal episodes** complete without
falls or nonfinite states. At .20 TB-equivalent m/s, the selected +173M
policy advances **.1624 TB m/s** in the requested direction, compared with
**.1025** for the previous300M policy; absolute net turn falls **79.7°→28.1°**.

The retained checkpoints have a meaningful tradeoff:

- **+120M is the straighter higher-command candidate.** At .20 it averages
  13.5° net turn, and only2/256 episodes turn more than45°. Its speed is.1543.
- **+173M is faster and remains the selected standard walking candidate.**
  At .20 its speed is.1624, but28/256 episodes turn more than45°.
- **+200M final deteriorates.** It is slower and turns more than+173M.

For the current priority of forward progression with controlled direction,
qualify +120M on the full held-out walking suite before choosing the next
training baseline. The GPU result alone does not supply its standing,
backward, lateral and transition confirmation.

## Verified inputs and metrics

Result folder:
`results/wr2_walking/wr2_paper_heading_feedback_20261025_201358_seed3/heading_evaluation_gpu`.

The evaluation used clean commit
`d30f33a426ed81f88b918784978f59942cc7aed2`, the completed run's saved
configuration and the requested +120,176,640/+173,588,480/+200,294,400
checkpoints. Their copied checkpoint hashes match. The local model hash
matches the trained model. Python3.10.19/JAX0.6.2/Brax0.14.1/MJX3.14.0,
Flax0.10.7 and explicit `high` matrix precision match the previous GPU
heading comparison.

Each checkpoint/command uses128 randomized episodes at each of seeds12102
and22102, plus one nominal episode. Commands are .10/.15/.20 TB m/s,
with zero commanded yaw and no external pushes. The27 trace files and
complete summary were checked against their recorded source hashes.

- Evaluator and measurement snapshots match current code and the previous
  GPU comparison byte-for-byte.
- Recomputed per-episode/group summaries agree within3.91e-14. All raw
  data are finite, with the intended824 ticks and valid masks.
- Integrated root-position deltas agree within1.17e-10m. Native XY, yaw
  and orientation rewards reconstruct within4.48e-7.
- Initial root poses match exactly across candidates and the previous300M
  policy for the same randomized draw. Paired comparisons use all256
  completed attempts, with10,000 bootstrap resamples.

The previous GPU evaluation recorded a dirty tree; the user previously
confirmed unchanged code at `d421547`. Its archived evaluator/measurement
sources match this comparison. The current evaluation has clean provenance.

Speeds use TB-equivalent units, `v_WR2 / .8240547565`. Physical durations
are scaled by the same factor; distances by its square. The reported
post-warmup window lasts **15.24 WR2 seconds /18.49 TB seconds**, following
the1.5-TB-second warmup. At .20, commanded WR2 speed is.16481m/s and
commanded forward displacement over that window is2.512m. Heading below
is **absolute net turn during this window**, rather than the standard suite's
mean absolute heading error.

## GPU walking results

Each entry is **requested-direction speed in TB m/s / absolute net turn**.

| Policy | .10 command | .15 command | .20 command |
|---|---:|---:|---:|
| Previous WR2 300M | .0833 /29.4° | .1079 /55.9° | .1025 /79.7° |
| Feedback +120.18M | .0845 /7.1° | .1283 /8.1° | .1543 /13.5° |
| **Feedback +173.59M selected** | **.0864 /6.3°** | **.1332 /13.8°** | **.1624 /28.1°** |
| Feedback +200.29M final | .0814 /18.0° | .1207 /28.1° | .1459 /32.4° |

All three new candidates have0/768 randomized falls across the three
commands. The selected policy improves .20 requested-direction speed
by **58.3%** over the previous300M checkpoint. Its paired change is+.05982
TB m/s,95% interval[+.05420,+.06564]. Net turn decreases51.6°,
interval[−55.6°,−47.6°].

| Policy at .20 | Forward distance | Lateral magnitude | Mean forward lean | Fraction of commanded forward progress |
|---|---:|---:|---:|---:|
| Previous300M | 1.288m | 1.507m | 10.34° | 51.3% |
| +120M | 1.938m | .356m | 7.77° | 77.1% |
| +173M | 2.039m | .664m | 7.29° | 81.2% |
| +200M | 1.833m | .817m | 8.31° | 73.0% |

At .20, body-forward speed changes.1806→.1738 from previous300M to
selected+173M. Thus the additional physical forward advance primarily comes
from reduced direction loss, with slightly lower body-frame speed.

### +120M versus +173M

At .20, moving from+120M to+173M adds:

- **.00807 TB m/s** forward progress,95% interval[+.00633,+.00977].
- **14.6°** net turn, interval[+12.9°, +16.4°].
- **.308m** lateral displacement magnitude, interval[+.268,+.347].

This is approximately10.1cm additional forward travel over15.24 physical
seconds. Unlike the16-draw Mac screen, the larger GPU sample shows a clear
speed advantage for+173M. The straighter-walking advantage for+120M also
persists. At .15, +173M gains.00498 TB m/s but adds5.7° of turn; at .10,
it is slightly faster and slightly straighter than+120M.

Mean turn does not describe the worst episodes:

| .20 candidate | Median net turn | 95th percentile | Episodes >45° |
|---|---:|---:|---:|
| +120M | 10.8° | 34.2° | **2/256** |
| +173M | 25.3° | 56.2° | **28/256** |
| +200M | 31.7° | 52.1° | **33/256** |

The selected+173M policy still has an extreme110° episode. Heading control
has improved substantially; it is not consistently straight at .20.

The final+200M policy is worse than+173M at .20: speed changes−.01642
TB m/s,95% interval[−.01804,−.01475], and turn increases4.3°,
interval[+2.5°, +6.1°]. Preserve the retained candidates rather than using
the last checkpoint as the default result.

## Standard suite and foot contact

Saved seed12102 evaluations use the same15-condition TB comparison suite.
The following metrics average the ten fixed walking conditions:

| Candidate | Reference XY error, TB m/s | Mean absolute heading error | Modeled foot self-contact | Contact-phase agreement |
|---|---:|---:|---:|---:|
| +120M | .04491 | 5.59° | 3.89% | 65.5% |
| +173M | .04369 | 5.47° | 5.31% | 62.2% |
| +200M | .04470 | 7.67° | 2.31% | 59.1% |

+173M has better aggregate tracking, which explains its selection by the
standard suite reaching .10 commands. It also has more modeled foot
self-contact and less reference contact-phase agreement than+120M; paired
intervals exclude zero for both changes. Its held-out confirmation records
5.92% self-contact. The+200M policy's lower self-contact does not compensate
for its worse tracking/turning and phase agreement.

These are existing foot-box contact measurements. The heading traces do
not contain foot landing geometry or step-placement events, so they do not
establish that alternating placement is corrected or identify a joint-level
cause of foot interference. The full+120M confirmation supplies the same
gait metrics before any promotion.

## Reward interpretation and TB comparison

The saved TB locomotion sources again match **`paper-walking-baseline`**,
revision `a4f693d7cc4544f011dad23d6c353e752a6819ab`, byte-for-byte.
Code confirms that TB's XY reward compares body-local velocity with the
command, while orientation reward uses the desired reference quaternion.
TB observes actual yaw/gyro but omits desired-heading error. WR2 keeps the
body-frame reward formula in this experiment and adds explicit actor/critic
heading feedback. Physics, fixed home action origin, moving leg-reference
imitation, qualified landing rule and reward coefficients retain the source
WR2 setup. The result supports keeping the heading observation extension.

The existing native TB200M CPU heading comparison gives .0870/.1227/.1407
TB m/s requested-direction speed at .10/.15/.20, and13.5°/21.1°/37.9°
net turn. It uses16 attempts per command; .20 averages15 completers with one
fall. Both current WR2 intermediate candidates compare favorably in this
limited reference, but WR2's larger GPU evaluation and TB's small CPU/native
runtime are not a matched parity test. Forward lean remains higher for WR2
(7.3–7.8° versus TB's3.8° at .20).

TB's full standard suite still has lower aggregate body-frame XY error:
TB200M .03702 versus selected WR2 .04245, a14.7% WR2 gap. TB has8/1,920
sampled primary falls there; selected WR2 has zero. Robot/randomization
profiles and training seeds differ. These comparisons support improved WR2
walking, rather than complete TB parity.

### Why remaining turning is still acceptable to the objective

At .20, native deterministic return is nearly the same for+120M and+173M:
**24.719 versus24.747 per TB second**. The paired difference+.028 has
95% interval[−.096,+.154], so the native objective does not clearly prefer
one of these frozen policies. The faster+173M policy gains.512 from the
body-frame XY component while losing.178 from yaw and.095 from orientation;
other terms account for the remaining difference. This identifies a reward
tradeoff, not a reconstruction error or a proven causal joint failure.

An offline calculation replacing only the XY term with velocity tracking in
the initial desired-heading frame yields:

| Candidate | Native return/TB second | Goal-frame XY shadow return/TB second | Goal-frame XY error, TB m/s |
|---|---:|---:|---:|
| +120M | 24.719 | **23.669** | **.07719** |
| +173M | 24.747 | 22.370 | .08856 |
| +200M | 24.277 | 21.162 | .09777 |

The calculation reconstructs cached world velocity from the native torso
sample and rotates it by the initial desired heading. It preserves exponent100,
coefficient15 and every other reward term. The+173M shadow return decreases
1.299 relative to+120M,95% interval[−1.504,−1.095], aligning the frozen
policy ranking more closely with straight commanded translation.

This remains an untrained reward counterfactual. With TB's `.97` discount,
reset-discounted shadow returns for+120M and+173M are nearly identical
(16.163/16.165; difference interval includes zero). Therefore the offline
long-window preference does not guarantee a PPO improvement. The result
supports a focused goal-frame XY experiment if directional tracking remains
the next training priority, with the existing heading inputs and TB PPO
settings preserved. It does not establish a need to change discounting,
actuator parameters or several reward coefficients together.

References: [ToddlerBot, Table VII](https://arxiv.org/html/2502.00893v2);
local Paper `mjx_env.py` observation/reward methods;
[WR2 reward implementation](../../wr2/locomotion/paper_rewards.py);
[heading-feedback rationale and public precedent](wr2_heading_feedback_experiment_20261025.md).

## Next step

Complete the full+120M standard evaluation using the existing evaluator and
the saved configuration. This confirms standing, lateral/backward walking,
transitions and foot contact on held-out seed22102, alongside the original
selection seed. The command parses correctly and the checkpoint/config exist:

```bash
uv run --locked --python 3.10 --extra training python -m wr2.locomotion.evaluate \
  --paper-benchmark --no-pushes \
  --config results/wr2_walking/wr2_paper_heading_feedback_20261025_201358_seed3/training_config.yaml \
  --checkpoint results/wr2_walking/wr2_paper_heading_feedback_20261025_201358_seed3/checkpoints/000120176640 \
  --seeds 12102,22102 --num-envs 128 \
  --output results/wr2_walking/wr2_paper_heading_feedback_20261025_201358_seed3/paper_120_confirmation.json
```

If it preserves standard walking and contact quality, +120M is the preferred
candidate for straighter higher-command walking; +173M remains useful where
its additional speed is the priority. Use the confirmed candidate as the
baseline for the next focused training experiment. Retain the existing
`best_params` metadata until that decision is made.

## Artifacts

Recomputed metrics, paired intervals, source/checkpoint hashes, heading tails
and reward counterfactuals are saved under the GPU result's
`review/analysis.json`. The standalone comparison figure is
`review/heading_feedback_gpu_comparison.png` and `.pdf`. Results remain local
artifacts; the training/evaluation implementation was not changed for this review.
