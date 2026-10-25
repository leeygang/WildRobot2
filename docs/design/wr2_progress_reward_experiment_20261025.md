# WR2 progression reward audit and experiment — 2026-10-25

**Completed:** the [300M result](wr2_progress_reward_result_20261025.md) confirms
stronger progression and lower tracking error, with remaining right-leg passing
and foot-contact gaps. The preparation below records the experiment as run.

## Decision

Prepare a **fresh 300M seed-3 run** with one training change:

**`reward_weights.lin_vel_xy: 5 → 15`.**

Use [ppo_walking_progress.yaml](../../wr2/locomotion/configs/ppo_walking_progress.yaml),
version **0.29.0**, through the regular trainer. The completed qualified-step
0.28.8 run is the control. Its configuration remains available in
`ppo_walking_step.yaml`.

This increases the importance of accurate commanded translation relative to
landing and body-stability incentives. It retains TB's instantaneous XY reward
formula, rather than rewarding larger joint requests or prescribing steps.
The supported-air/sustained-landing rule requested by the user remains enabled.

**This is a justified acquisition experiment, not a proved solution or an
optimal coefficient.** Rescoring existing controls changes their incentive;
only a new PPO run can show whether that incentive improves learned walking.

## 1. Source and measurement checks

Read current WR2 `paper_rewards.py`, `_paper_step`, action mapping and training
configuration. Read local TB **`paper-walking-baseline` at `a4f693d`**, including
`paper.gin`, `walk.gin`, `mjx_env.py` and `walk_env.py`. TB's effective Paper
coefficients are XY **5**, yaw **5**, leg imitation **5**, contact **1**, air
**500**, collision **.1**. WR2's existing collision **1** and qualified-landing
treatment are declared departures. The new XY coefficient is an additional
explicit departure; TB's successful recipe remains the comparison.

The shared XY reward is

`exp(-100 * ||v_xy_TB - command_xy_TB||²)`.

WR2 divides physical linear velocity and commands by
`s=.8240547565448791`. Time divides by `s`, and placement distance divides by
`s²=.6790662418`. At the inspected command, `.10 TB m/s` is **.0824055 physical
WR2 m/s**. The command distribution and .60-second physical gait cycle remain
those of the control: matching that command requires about **49.4 physical mm
per same-foot cycle**, rather than a larger commanded speed.

Both implementations imitate actual motor positions against the moving leg
reference with negative mean squared error. Both control legs around a fixed
home plus `.25*tanh(action)`. The reward experiment changes neither that action
family nor WR2's existing reference-derived home. A successful TB policy does
not literally reproduce its complete reference waveform; stronger joint
imitation is not selected here.

Verified recorded per-tick reward sums and WR2's XY formula. Recomputed benchmark
errors from each episode of the ten fixed nonzero conditions; all fifteen
conditions count toward falls. Native TB component metrics include coefficients
but omit `dt`, so multiply by `.02` before comparing returns. Native WR2
`paper_reward` metrics already include `dt`.

For native evaluations with unequal episode lengths, reported rates are
**sum of returns / sum of elapsed TB-equivalent durations**, using ensemble
means. They are not substituted for mean individual episode rates. Nominal
recording rates below use fixed post-warmup windows directly.

Reference: [ToddlerBot Tables VI–VII](https://arxiv.org/html/2502.00893v2).

## 2. Retained PPO progress and TB comparison

No new policy or manual joint intervention is used in this table. Forward speed
is the ensemble episode-mean body velocity under the fixed +.10 TB command;
XY error averages the ten fixed nonzero commands.

| Transitions | WR2 XY error, TB m/s | TB Paper XY error, TB m/s | WR2 forward, TB m/s | TB forward, TB m/s |
| --- | ---: | ---: | ---: | ---: |
| 20M | .09693 | .06202 | .01944 | .03767 |
| 40M | .08594 | .04771 | .03643 | .06197 |
| 80M | .07589 | .03965 | .05906 | .07812 |
| 120M | .07666 | .03952 | .04467 | .08077 |
| 200M | .08241 | .03702 | .01764 | .09053 |
| 300M | .07820 | .03800 | .03326 | .09367 |

WR2 has already received a sufficient **300M** budget for this control. Its
forward response peaks around 80M and deteriorates; extending the same run is
not selected. Selected 120M has 0/1920 primary falls, versus 20/1920 at 80M,
which narrowly misses the existing 1% eligibility gate. TB's selected 200M has
8/1920 falls. These are different robot/runtime/randomization profiles and
training seeds, not a shared-physics causal experiment.

There is an incentive tradeoff in the original nominal recordings. Excluding
the first 62 WR2 transitions (at least 1.5 TB seconds):

| Policy | Net world forward, TB m/s | Body XY error, TB m/s | Native return / TB second |
| --- | ---: | ---: | ---: |
| 80M | .06212 | .06374 | 20.983 |
| 120M | .05186 | .06727 | 21.988 |
| 300M | .03959 | .09282 | 20.899 |

From 80M to 120M, XY reward rate changes **−.0931**, yaw **+.3940**, torso
orientation **+.2882**, angular-XY stability **+.1190**, vertical velocity
**+.0954**, and air **+.1565**. Progress worsens while reward improves.
These are trajectory comparisons, not reconstructed historical PPO gradients.

Native randomized/mixed-command evaluation confirms the broad imbalance at
120M: WR2 XY reward rate is **2.248**, air **10.317**, while TB's are
**4.004 / 6.585** per TB-equivalent second. The higher WR2 air rate partly
reflects its intentional `.97⁻⁹ = 1.3154` qualification-delay compensation.
Its contribution alone does not prove the air coefficient is the cause.

## 3. Why choose XY 15

The corrected centered-reference/full-orientation diagnostic preserves requested
foot height and sole orientation while increasing left forward motion. It has
three matched start phases at each of 80M and 120M, 99 recorded ticks including
treatment and recovery, and the actual control discount **gamma=.97**.

Its actual net speed increases approximately **.063 →.082 TB m/s at 80M** and
**.052 →.065 at 120M**, with larger left placement. Motion is less smooth than
TB, and the left foot still trails. This is a temporary diagnostic control,
not a learned replacement policy.

For each recorded pair, rescoring a weight `w` is exact arithmetic:

`new_return_delta = original_delta + (w/5 - 1) * original_XY_delta`.

No saved critic is used to bootstrap a changed objective.

| Corrected full-orientation control | Mean return delta at XY 5 | Mean XY contribution delta at XY 5 | Mean return delta at XY 15 |
| --- | ---: | ---: | ---: |
| 80M | −.10407 | +.09759 | **+.09112** |
| 120M | −.07139 | +.04543 | **+.01946** |

Mean break-even coefficients are **10.33 /12.86**. Fifteen is the smallest
integer multiple of the existing TB coefficient above both. It is a transparent
test value, not a statistical estimate of an optimum. At each checkpoint only
**2/3 individual start phases** become positive; the unfavorable phase remains
penalized. Duplicate controls agree in contact labels; maximum float differences
are **2.98e−7 qpos /1.49e−6 per-tick reward**.

Two checks prevent interpreting larger steps as sufficient:

- Corrected hip-only controls remain worse at XY 15: mean discounted differences
  **−2.439 at 80M /−3.237 at 120M**.
- The corrected bilateral v2 pattern remains worse in its completed comparison
  windows: rate differences **−7.898 /−9.261 per TB second** at 80M/120M.
  Failed/partial windows remain separate. Its rocking and yaw are not rewarded
  merely because feet exchange the lead.

Thus increasing the existing tracking incentive can favor a tested, better
coordinated local improvement while continuing to reject tested unstable motion.
Reducing air or stability coefficients is not needed to obtain that direction.
The new coefficient also strengthens tracking of backward/lateral commands and
zero translation; it does not encode a preference for one leg.

**Important limit:** XY 15 still ranks the complete nominal 120M policy above
80M (rescored post-warmup rates **27.221 /26.402**). It does not eliminate every
tradeoff or directly require alternating forward placement. The rationale is
improved local acquisition incentives, not a claim that reward accounting has
fully identified why PPO learned the asymmetric posture.

## 4. Stance and placement constrain the interpretation

With current force-based contact labels after the same normalized warmup,
left landings ahead are **0/25** at WR2 80M, 120M and 300M; TB Paper 200M has
**25/27** left and **27/27** right. WR2 left same-foot world-X advances decline
**31.5 →26.1 →19.4 physical mm**. Native TB's 61.1 mm is **41.5 mm at WR2 size**
over this longer recording window. The earlier matched short windows give a
different, explicitly window-specific 44.4 mm value.

Current-contact double support is **1.57% / .13% /5.51%** for those WR2 policies,
versus **13.95%** TB. These use current measured contact, without OR-filtering
the previous contact; they are neither load fractions nor proof of a required
double-support duration. Contact and force observations motivate checking
execution, but do not select a handcrafted load-sharing reward.

The completed bilateral substep audit shows larger leg motion under existing
limits. Reference-timed bilateral replacement worsens propulsion despite the
same left commands. Neither finding proves that physics is irrelevant or that
this new coefficient will recover normal alternation.

## 5. Training and success criteria

Run fresh with the existing seed **3**, **300M**, LR **1e−4**, network,
distribution, randomization, command sampler and physical model. Retain all
sixteen evaluation checkpoints. The existing selector uses independent
`paper_xy_error` with its survival gate, not the changed native reward.
Restoring the compressed late actor and the old-objective critic would mix
initialization/continuation effects into this test.

On the GPU checkout:

```bash
uv run --locked --python 3.10 --extra training python -m wr2.locomotion.train \
  --config wr2/locomotion/configs/ppo_walking_progress.yaml
```

The regular trainer checks the GPU backend and performs the full reference-grid
preflight. Result prefix is **`wr2_paper_progress`**. Compare intermediate policies
with control 80M/selected 120M/final 300M and TB selected 200M.

Success requires improved **independent command tracking and actual progression**,
with larger left/right same-foot advances and more alternating leading landings.
Inspect velocity variation, yaw, standing/stop behavior, falls and self-contact
to check whether progress was purchased through poorer stability. A higher
training reward, bigger requested angles, or improved speed only from the right
leg is not full success. Evaluate the selected checkpoint on the existing
held-out suite and view its original PPO actions; manual overrides remain
diagnostics only. Push performance is a companion metric, separate from the
no-push acquisition comparison.

## 6. Artifacts and verification

The reproducible recorded-data audit is
`results/wr2_walking/wr2_paper_step_20261022_143550_seed3/review_20261025/reward_progress_audit`.
It contains `analyze.py`, `report.json` with input hashes, and
`reward_progress.png`. It reads TB's original local training results and frozen
branch, and makes no writes in the active TB checkout.

The typed configuration comparison verifies that the only training change is
XY 5 →15; remaining differences are version/name/output prefix. Native environment
and PPO integration verification passed on the Mac CPU:

- **20 matched native steps:** exact physics, observations and all other reward
  contributions; XY contribution triples. Maximum total-reward difference from
  the expected coefficient change is **1.04e−7**.
- **160 native Brax transitions**, four environments with short reset episodes:
  actor parameters update, parameters remain finite, evaluation completes.
- Runtime: JAX .6.2, Brax .14.2, MuJoCo/MJX 3.14.0, Flax .11.2.

`verify.py`, `verification.log` and `verification.json` record those checks.
The verification uses a two-command reference grid; full command-grid/GPU
preflight remains part of the regular training command. It is an integration
check, not evidence of a new successful walking policy.

[Comparison figure](../../results/wr2_walking/wr2_paper_step_20261022_143550_seed3/review_20261025/reward_progress_audit/reward_progress.png)
and [audited numerical report](../../results/wr2_walking/wr2_paper_step_20261022_143550_seed3/review_20261025/reward_progress_audit/report.json).
