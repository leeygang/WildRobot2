# WR2 PPO learning diagnostic result — 2026-10-24

Run: `results/wr2_walking/wr2_paper_step_20261022_143550_seed3/ppo_learning_diagnostic_v3`.
Protocol/source: [PPO diagnostic guide](wr2_ppo_learning_diagnostic_20261023.md).

## Conclusion

**The completed diagnostic confirms the asymmetric learned posture and shows
that the 120M fresh-Adam probes usually reinforce it.** Under forward commands,
the left target moves slightly backward in **6/8 probes**, while the right
target moves forward in **7/8**. At 80M the left target moves forward in **6/8**;
the later behavior is therefore not a universal property of every checkpoint.

There is **no newly trained walking policy**. These are 16 fresh
optimizer initializations on frozen-checkpoint samples; all probe weights are
discarded. Their data share trajectories, and they are not independent training
seeds or a reconstruction of historical Adam updates.

The most specific new update problem is a **large response to command
resampling at 120M**. Its probe KL rises to **1.658**, versus a median **.0446**
in the other batches. That requires comparison with TB's matching native
probe before selecting a PPO change. The data do not identify a single reward
coefficient that would fix the complete gait.

## 1. Integrity and numerical verification

All **16 batches** complete: two checkpoints, eight batches each, 1024
environments ×20 steps = **327,680 measured transitions**. Warmup is 330
physical ticks per checkpoint. Source is clean `392bb2d`; tool/math hashes
match the exact committed files. Saved configuration, model and actuator
checks pass, and the GPU core packages match the completed training run.

| Check, maximum over all batches | Error |
| --- | ---: |
| Float32-reference parameter gradient | 5.744e−7 |
| Float32-reference output gradient | 7.990e−8 |
| Native output gradient | 7.498e−8 |
| Advantage closure | 1.776e−15 |
| Scalar reward closure | 1.431e−6 |
| Recorded native matrix-kernel gap | 4.229e−4 |

Native loss/entropy metrics match the actual first minibatch of each Adam
probe. The native GPU matrix gap remains recorded; it is the precision effect
isolated by the earlier exact-batch replay, not a failed attribution check.
Sampling, recorded pressures and actual probe updates use the saved default
matrix precision. Only mathematical verification uses `highest`.

There are **9 early-termination transitions at 80M /1 at 120M**, and no
timeouts. These counts cover a short sampled window; they are not the existing
20-second held-out benchmark's episode fall rates.

## 2. The later policy favors stability over forward translation

Forward-command samples are time weighted over all eight recorded batches.
They include a distribution of positive commands, not a fixed `.10` test.
Velocities use TB-equivalent units; reward rates are weighted native return
per TB-equivalent second. Mean forward commands are nearly identical.

| Measure | 80M | 120M |
| --- | ---: | ---: |
| Samples under forward commands | 25,761 | 25,918 |
| Mean commanded forward speed, TB m/s | .10845 | .10812 |
| Mean achieved forward speed, TB m/s | **.05285** | **.04060** |
| Weighted return rate | 19.571 | **20.087** |
| XY tracking contribution | 1.953 | **1.853** |
| Yaw tracking contribution | 3.166 | **3.425** |
| Torso-stability contribution | 1.914 | **2.336** |
| Leg imitation contribution | −.186 | −.203 |
| Air-time contribution | 10.631 | 10.661 |

Achieved forward speed drops approximately **23%**, while total return rises
approximately **2.6%**. Yaw and torso-stability gains account for the positive
tradeoff; the air contribution is nearly unchanged. This extends the earlier
nominal-trajectory observation to noisy, randomized training-like samples.
It is a comparison between two policies' resulting trajectories, not proof
of the sign of each historical PPO gradient.

This supports investigating the stable asymmetric solution and coordinated
translation. It does not select disabling air time, weakening yaw tracking,
or increasing XY weight.

## 3. Actual fresh-Adam target changes

All angles below use physical forward hip directions: WR2 left negative,
right positive. Targets are evaluated on the **same original observations**
before and after four epochs ×four minibatches at LR `1e−4`.

| Forward-command measure | 80M | 120M |
| --- | ---: | ---: |
| Mean left forward target | 6.182° | **3.872°** |
| Mean right forward target | 26.945° | **28.164°** |
| Mean left target change per probe | +.00436° | **−.00597°** |
| Mean right target change per probe | +.01554° | **+.02619°** |
| Left forward-change probes | 6/8 | **2/8** |
| Right forward-change probes | 7/8 | **7/8** |

The changes are small. They must not be multiplied by the remaining number
of training updates to predict a future gait. Optimizer state and policy/data
distributions would evolve during real training.

The right-leading posture is also present across standing, backward, lateral
and turn samples. At 120M the mean left target moves backward in all five
command groups after the probes. This is broader than one nominal forward
recording.

### Sensitivity and exploration

| Left hip, forward samples | 80M | 120M |
| --- | ---: | ---: |
| Mean tanh derivative | .4645 | **.2038** |
| Mean raw Gaussian std | .1186 | **.2193** |
| Actual single-tick request ≥3° farther forward | .408% | **.586%** |

Mean bias reduces deterministic target sensitivity by approximately **56%**.
Raw Gaussian noise increases, and the observed farther-forward request
probability does not disappear. Both probabilities remain below 1%; they are
single-joint, single-tick events, not discovery probabilities for a coordinated
hip/knee/ankle gait. These data do not justify simply increasing raw noise or
entropy weight.

### Foot placement under the original policies

For forward commands, require contact to persist for at least ten recorded
samples after a landing. Unresolved events at the recording's end are excluded.
This is a contact-duration filter, not an exact reconstruction of every
qualified-air eligibility rule.

| Sustained landing | 80M | 120M |
| --- | ---: | ---: |
| Left landings ahead of right | 1/876 | **0/861** |
| Mean left lead, TB mm | −52.99 | **−74.28** |
| Right landings ahead of left | 800/800 | **795/795** |
| Mean right lead, TB mm | 89.99 | **103.35** |

At 120M those mean leads correspond to approximately **−50.4/+70.2 actual
WR2 mm**, using `s²=.6790662418`. The left foot remains behind the right even
when contacts persist. These are landing separation measures, not same-foot
stride lengths. World foot centers are retained, but root heading is not, so
signed forward stride cannot be reconstructed exactly from this dataset.

## 4. Reward-gradient accounting does not isolate one coefficient

Mean projection onto the left forward target, in radians per unit plain-SGD
learning rate:

| Component | 80M | 120M |
| --- | ---: | ---: |
| XY tracking | +.00942 | +.00985 |
| Yaw tracking | +.01102 | +.01863 |
| Torso stability | −.07583 | +.00345 |
| Leg imitation | +.00505 | +.00047 |
| Air time | −1.20969 | −.06503 |
| Other rewards | −.02314 | +.01037 |
| Value/bootstrap residual | +1.14684 | −.02063 |
| Entropy | +.00251 | +.00122 |

The 80M air and value residual almost cancel. Their magnitudes and signs vary
with gait phase and the short rollout's payment/bootstrapping window. The
residual holds the full native critic/bootstrap; reward groups do not have
separately trained component critics. These numbers are an exact sampled-loss
accounting convention, not independent causal reward ablations.

The initial SGD projection points left-backward in **7/8 batches at 80M**,
yet the complete Adam probes move the left forward target forward in **6/8**.
At 120M the initial projection is backward in **5/8**, and the Adam probes
move it backward in **6/8**. Adam preconditioning, four minibatches, critic
updates and clipping across four epochs all intervene; a local SGD projection
is not the actual update.

Native per-output pressure can also differ from the parameter projection:
shared network parameters combine learning signals from all commands. The
projection uses strong-forward observations (`x>.05`, dominant x, no yaw),
while the main forward sample table also includes smaller dominant-x commands.
The difference is explicit, rather than treated as contradictory evidence.

## 5. Command-resampling update sensitivity

The measured window starts at global step 330. The saved interval is 124;
the switch at step **372** falls in zero-based **batch 2**. Direct command
arrays confirm changes in **980/1024 environments at 80M /982 at 120M**.

| Measure | 80M switch batch | 120M switch batch |
| --- | ---: | ---: |
| Native critic loss | .41935 | **.51367** |
| Median critic loss, other batches | .02324 | .01963 |
| Native KL(old || new), last probe epoch | .10604 | **1.65798** |
| Median KL, other batches | .03661 | .04457 |
| Largest post-probe target change | .652° | **3.964°** |

The largest 120M change is **left hip yaw**, verified against the actuator
ordering. In that particular observation its raw Gaussian mean changes by
`.27853`, versus previous std `.015856`: about **17.6 old standard deviations**.
It is a post-update request, not a motion executed in the simulator. There are
no falls in the 120M switch batch.

The value loss can include irreducible variance from a newly randomized
command; the spike alone does not prove a broken critic. The KL spike describes
a **fresh-Adam restart probe**, not the unavailable historical optimizer state.
It nevertheless identifies a concrete transition/update condition to compare
against TB before choosing a fix.

## 6. TB Paper comparison and next priority

Read local **`paper-walking-baseline`** `mjx_env.py`, `paper.gin`, `train_mjx.py`
and the pinned Brax `losses.py` (`1414780`). Code confirms both robots' home-based
`.25-rad` tanh actions, leg-reference imitation weight 5, XY/yaw weights 5,
gamma `.97`, lambda `.95`, 20-step rollouts and four-by-four Adam update recipe.
WR2 adds qualified air credit and collision weight 1 rather than TB's .1.

TB Paper's successful recorded policy retains approximately **14.01°** left
hip request range and tanh sensitivity **.924**; the earlier WR2 nominal
80M/120M values were **2.50°/1.65°**. TB's recorded feet normally alternate
leading at landing. Those native gait results support the posture/coordination
gap, but are not a matched stochastic gradient comparison.

**The matching TB 80M/200M GPU PPO diagnostic is not present in the reviewed
local folders.** The source is confirmed; equivalent learning/update evidence
is still missing. Run/copy the native TB diagnostic using the
[prepared guide](wr2_ppo_learning_diagnostic_20261023.md#gpu-command-native-tb-paper).
Its 400-tick warmup and 150-step command interval also put a scheduled switch
in batch 2, making the comparison particularly useful.

Next priorities:

1. Compare TB's target-change directions and critic/actor update response at
   its corresponding command-resampling batch. Native TB's older loss does not
   export the same KL metric; compare available updates/losses honestly.
2. Audit WR2's value-credit information around switching commands. Source
   confirms the added `supported_air_time`, `pending_air_time`, `stance_steps`
   are absent from `_paper_observation`. That adds reward memory beyond TB's
   immediate air-credit rule. Whether it materially explains the observed
   update spike or posture drift has not been established.
3. Choose one controlled 200–300M training change only after that comparison,
   retaining intermediate checkpoints and using the existing held-out walking
   benchmark. Keep reward weights and physics for this next analysis.

Both source implementations construct observations before assigning the next
resampled command, so the one-tick command-observation timing is shared with
TB and cannot by itself explain the WR2-specific gap. WR2 retains TB's
observation structure while adding the pending qualified-air counters.
Longer training alone is not a
selected solution to the measured mean saturation.

References: [TB paper, Tables VI–VII](https://arxiv.org/html/2502.00893v2),
[PPO](https://arxiv.org/abs/1707.06347),
[GAE](https://arxiv.org/abs/1506.02438),
[earlier saturation/coordination evidence](wr2_hip_pitch_incentive_review_20261023.md).

## Artifacts

Generated review data and figure remain under the ignored results directory:

- `ppo_learning_diagnostic_v3/review_20261024/analysis.json`
- `ppo_learning_diagnostic_v3/review_20261024/ppo_diagnostic_review.png`

Original `summary.json`, `batches.jsonl`, NPZ recordings and provenance are
retained. No CHANGELOG entry or new training recipe is part of this review.
