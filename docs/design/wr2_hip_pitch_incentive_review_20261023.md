# Hip-pitch saturation and walking incentives — 2026-10-23

Follow-up to the [landing analysis](wr2_step_placement_review_20261023.md).
Completed recorded-data analysis and **90 verified matched-state comparison
rollouts**, using retained 80M/120M/300M policies from the qualified-step run.

## Conclusions

1. **The left target loses cyclic motion because its learned mean approaches
   the policy's tanh bound in the backward direction.** The underlying Gaussian
   mean still varies with the gait cycle; squashing increasingly compresses
   that variation. Joint execution attenuates the remaining small requests.
2. **The same actuator model can execute larger left motion**, but adding it
   to the existing joint pattern can produce fore-aft oscillation and yaw costs.
   Greater net translation does not necessarily earn better tracking reward.
3. **Air time contributes to the tradeoff, but an air-weight change alone does
   not reverse the tested larger-step ranking.** At 120M, removing its contribution
   still leaves the larger left-motion control behind by 1.04 discounted return.

The evidence identifies how the left swing disappears and why the tested local
corrections can be unattractive. It does not reconstruct the historical PPO
updates that originally selected this mean, or establish a complete training fix.

## 1. Source and metric verification

Read current WR2 environment, reward, observation, network and action mapping;
read local TB **`paper-walking-baseline` at `a4f693d`** and its pinned Brax
distribution at `1414780`. Both use

`target = q_home + .25 * tanh(mu)`

for deterministic actor evaluation. The native stochastic policy samples its
Gaussian before tanh. TB and WR2 both have **.25 rad** action scale; the fixed
home origin is their existing design. The environment still tracks the moving
leg reference with its imitation term. No action-scale or joint-index discrepancy
was found in the reviewed path.

Kinematic checks confirm a positive hip-pitch degree moves WR2's left foot
backward/right forward, while TB uses the opposite signs. At the recorded
initial poses, forward-X changes are approximately **−3.87/+3.87 mm per degree
WR2**, and **+4.33/−4.33 mm TB**. Subsequent comparisons express pitch as positive
forward, and normalize WR2 distances by `s²=.67906624`, time by `s=.8240547565`.

WR2's 25 per-tick reward components reconstruct total reward within `1e−5`.
Discounted comparisons use the actual **gamma=.97**, unchanged weights,
native controller and **1.31538956** qualified-air compensation. TB's GPU
recording permits exact reconstruction of five phase-dependent terms; its
native episode logs provide all reward terms. Missing TB torque/gyro quantities
are not inferred from the recording.

## 2. The mean moves toward the wrong endpoint

WR2 left hip home is **−16.63°**. Its tanh action range gives target endpoints
approximately **−30.96° /−2.31°**. The latter moves this leg backward.

After 1.5 TB seconds of warmup:

| Checkpoint | Mean raw `mu` | Mean normalized action | Requested left range | Actual left range |
| --- | ---: | ---: | ---: | ---: |
| 20M | .639 | .561 | 3.34° | 2.11° |
| 80M | .930 | .728 | 2.50° | .67° |
| 120M | 1.469 | .894 | 1.65° | .31° |
| 300M | 2.285 | .978 | .44° | .28° |

The mean derivative of the tanh transform, `1-action²`, falls
**.681 →.469 →.199 →.043** over those checkpoints. This compresses changes in
both the Gaussian mean and sampled actions near the endpoint. The left target
has zero physical joint clipping and slew limiting in the selected nominal
recording; the relevant bound here is the policy transform.

The cycle signal itself remains. In the same late full cycle, ticks **600–629**:
raw-mean range grows **.352 at 80M →.540 at 120M →.585 at 300M**, while target
range shrinks **2.38° →1.51° →.40°**. Thus the phase-dependent network output
is being squeezed by its shifted mean.

TB's successful left hip has mean normalized action **−.023**, tanh derivative
**.924**, requested range **14.01°**, and actual range **14.65°**. It retains
usable motion well inside the same action family.

WR2's raw left Gaussian std actually increases .111 →.189 →.390 from
80M to 120M to 300M. Raw std alone therefore cannot measure useful exploration.
Under the recorded nominal states, the mean single-tick probability of requesting
at least 3° further forward is only approximately **.28% / .11% / .14%**.
These are conditional request probabilities, not probabilities of discovering
a coherent multi-step gait or measurements of the randomized training distribution.

## 3. Reward changes during acquisition

On the existing nominal forward recordings, steady weighted return per
TB-normalized second rises **20.98 at 80M →21.99 at 120M**, despite lower
forward speed. Individual rate changes include:

| Component | 120M minus 80M, return per TB second |
| --- | ---: |
| Yaw tracking | +.394 |
| Torso orientation | +.288 |
| Qualified air time | +.156 |
| Angular XY tracking | +.119 |
| Vertical velocity tracking | +.095 |
| XY velocity tracking | **−.093** |

This establishes a trajectory-level tradeoff, not the sign of any particular
historical PPO gradient. Native mixed-command evaluation and this nominal
fixed-forward recording remain distinct measurements.

## 4. Matched-state diagnostic

Three checkpoints ×three start phases ×ten controls yield **90 branches /8910
comparison transitions**. Each branch begins with the same full environment
state as its paired control. Other joint action sequences are held fixed to
their recorded policy outputs. The actor is not allowed to compensate through
those joints during the intervention.

Start ticks are **180, 190, 200**. Treatment lasts two cycles, with six-tick smooth
on/off ramps; all treatment is removed after tick 66. Original action tapes
continue to tick 99, giving **1.98 physical /2.40 TB seconds** per comparison.
Every unloaded interval overlapping treatment resolves through qualifying
payment or cancellation before the recording ends. Remaining pending credit
belongs to later motion; endpoint critic values are reported separately.

Controls comprise baseline/duplicate, both-hip mean correction, separate
left/right mean corrections, waveform-only, and combinations. A mean correction
is **left 3° forward /right 3° backward**. The zero-mean left waveform uses the
native reference's shape. Same-amplitude combinations isolate posture effects;
larger feasible combinations measure available motion after shifting the mean.

The initially proposed 5° waveform cannot fit around the current unshifted
target mean. To preserve its mean and remain inside actor bounds, maximum added
deviation is capped at approximately **3.05° /1.85° /.18°** for 80M/120M/300M.
After moving the left mean forward 3°, the allowed peaks become approximately
**5.00° /4.62° /2.95°**. These are maximum deviations from the mean, not
peak-to-peak ranges. No hidden clipping is used to create an unintended bias.

Independent checks verify each intended hip change, exact unchanged action
entries, action bounds, controller-target reconstruction, component checksums
and survival. All **90 branches survive** without nonfinite values. Duplicate
controls have identical action/contact sequences and discounted total returns.

The nominal reconstruction matches the original 120M/300M recordings closely.
80M has small numerical divergence: maximum foot-center difference **1.75 mm**,
root-position difference **5.37 mm**, contact agreement **99.67%** over the
reconstructed prefix. Its comparisons still use identical paired start states.

### Main results

Means over the three deterministic start phases. Speed differences refer to
the treatment/ramp window; returns include the full continuation.

| Checkpoint and control | Forward speed change, TB m/s | Discounted return change |
| --- | ---: | ---: |
| 80M, feasible waveform only | **+.00905** | **+.04767** |
| 120M, feasible waveform only | −.00032 | +.06026 |
| 300M, feasible waveform only | +.00014 | +.00101 |
| 80M, left mean +larger feasible waveform | **+.01353** | **−1.19093** |
| 120M, left mean +larger feasible waveform | **+.00809** | **−1.60353** |
| 300M, left mean +larger feasible waveform | −.01122 | −.46410 |

Both left-only and right-only mean corrections reduce return in these probes.
The same-amplitude combinations also reduce it. Small waveform-only motion
at 80M has a positive local return/translation direction; the current objective
does not oppose every increase in left movement. At 300M little extra waveform
fits at the existing mean, and its effect is negligible.

These results describe fixed-other-action interventions. They do not measure
the performance of a fully coordinated trained alternative.

## 5. Why the larger 120M motion loses reward

Moving only the left mean forward and adding the larger feasible waveform
improves mean forward movement across the three phases, but not uniformly:
speed differences range **−.00214 to +.01348 TB m/s**. Its discounted return
is lower in every phase, **−2.060 to −.940**, mean **−1.604**.

| Native reward component | Mean discounted change |
| --- | ---: |
| Qualified air time | **−.562** |
| Yaw tracking | **−.486** |
| Combined XY velocity tracking | **−.391** |
| Foot/reference contact matching | −.103 |
| Torso orientation | −.052 |
| Leg-reference imitation | +.0025 |

**The XY loss is predominantly fore-aft tracking, not a larger lateral-factor
loss in this test.** Forward-only Gaussian return falls **.470**; the lateral
factor removes **.079 less** than in the control, leaving the combined loss
.391. This is more specific than the earlier full-reference control's lateral
sway result.

Native body forward-velocity std rises **.032 →.085 TB m/s**, and backward
velocity samples rise **1.5% →19.7%** during treatment. Finite differences of
world root position likewise give std **.032 →.083**, so the effect is not
merely a body-frame yaw conversion. Mean progress increases while motion
becomes much less smooth. The unchanged TB Gaussian correctly penalizes
those instantaneous deviations.

The left sole's loaded pitch range grows **.83° →5.15°**. This accompanies
the fore-aft oscillation and supports examining hip/knee/ankle coordination;
it does not independently establish the force mechanism.

Double support grows **0% →17.2%** over the treatment window. Total physically
qualified air credit drops **1.893 →1.680 s**, with approximately **.067 s**
rejected. The rule therefore contributes through changed support duty and
some failed qualifications, rather than an unexplained reward-accounting loss.

Rescoring these exact trajectories with air coefficient zero leaves the return
change at **−1.041**. Increasing combined XY weight also magnifies the observed
negative XY contribution. Neither operation alone reverses this comparison.

Endpoint critic bootstrapping keeps the larger 120M control negative:
mean difference **−1.672**, versus raw finite-window difference −1.604.
Baseline estimates themselves underpredict the bootstrapped nominal return
by about **1.12–1.22** at 120M. This is not a critic-correctness certificate:
the critic was trained with randomized/noisy observations and stochastic actions;
these probes use nominal states and deterministic recorded action sequences.
The current measurements do not demonstrate a PPO bug.

## 6. Direction before implementation

The verified mechanism for shrinking requested swing is **backward target-mean
drift →tanh compression**. Execution attenuates the remaining small requests;
the learned posture retains trailing left placements. The original cause of
that drift needs a learning/coordination explanation.

The most useful next analysis is the phase relationship of **left hip, knee
and ankle pitch**, including sole loading and fore-aft/yaw reactions. The
current knee/ankle pattern was learned with a nearly fixed hip; larger hip
motion added alone does not constitute a balanced forward step. Compare the
native TB learned joint pattern and WR2 reference before selecting a correction.

Retain 80M as a diagnostic anchor: it admits a modest beneficial left waveform
without a mean change. Its primary evaluation still narrowly fails the 1% gate,
so this does not replace the selected 120M checkpoint. This analysis does not
select a new reward coefficient, action range, distribution or training run.

## Artifacts and sources

[Waveforms and comparison figure](../../results/wr2_walking/wr2_paper_step_20261022_143550_seed3/review_20261023/hip_pitch_diagnostic/hip_pitch_incentives.png),
[verified rollout summary](../../results/wr2_walking/wr2_paper_step_20261022_143550_seed3/review_20261023/hip_pitch_diagnostic/summary.json),
[aggregated effects](../../results/wr2_walking/wr2_paper_step_20261022_143550_seed3/review_20261023/hip_pitch_diagnostic/aggregate.json),
[waveform/reward accounting](../../results/wr2_walking/wr2_paper_step_20261022_143550_seed3/review_20261023/hip_pitch_diagnostic/waveform_rewards.json).
Reproducible analysis sources are alongside these ignored local artifacts.

To view the short paired 120M controls on the Mac, use the existing viewer:

```bash
uv run --no-sync mjpython -m wr2.tools.view_walking \
  --trace results/wr2_walking/wr2_paper_step_20261022_143550_seed3/review_20261023/hip_pitch_diagnostic/120M_start180_baseline.npz \
  --playback-speed 0.25
```

Replace the filename with `120M_start180_left_bias3_swing_feasible.npz` for the
larger left-motion intervention. These are short diagnostic controls starting
at an intermediate state, not newly trained policies. Both viewer checks pass.

Source references: [WR2 reward](../../wr2/locomotion/paper_rewards.py),
[environment](../../wr2/locomotion/walking_env.py),
[network construction](../../wr2/locomotion/ppo.py), local TB branch files
`walk_env.py`, `mjx_env.py`, `mjx_config.py`, `train_mjx.py` and pinned Brax
`distribution.py`; [ToddlerBot Tables VI–VII](https://arxiv.org/html/2502.00893v2).
WR2 diagnostics use Mac CPU Brax .14.2/JAX .6.2/MuJoCo 3.14.0/Flax .11.2,
with high matmul precision. TB comparisons use the existing native GPU
recording/logs. GPU confirmation of these new WR2 interventions is still absent.
