# Hip action-center training review — 2026-10-29

## Decision

**The model-home hip experiment improves low-command progression and standard
XY tracking, but does not resolve uneven placement or the nearly fixed right
hip.** Retain candidate **+173,588,480** for comparison. The final candidate
regresses in tracking and nominal right-foot placement. Keep the previously
qualified goal-XY parent for higher-command walking while completing the GPU
comparison below.

Do not interpret the better aggregate selector score as a general gait fix.
The candidate still uses a predominantly left-leg hip swing, and its right
landing remains short or behind the left. The hip action-window constraint
was real; removing much of that constraint has not removed this strategy.

## Runs and integrity

| | Control | Model-home candidate |
|---|---|---|
| Result folder | `wr2_paper_hip_control_20261028_191348_seed3` | `wr2_paper_hip_model_home_20261028_224138_seed3` |
| Additional transitions | 200,294,400 | 200,294,400 |
| Selected checkpoint | +106,823,680 | +173,588,480 |
| Time through final training evaluation | 1.90 hours | 1.97 hours |

Both results are under `results/wr2_walking/`. Both record clean source
`d40d383`, identical RTX 5070/Python 3.10.19/JAX .6.2/Brax .14.1/MJX 3.14.0
runtime and default matrix precision. Both restore the qualified goal-XY
`000200294400` actor, critic and normalizer exactly, with fresh Adam/counters.
Their effective saved recipes differ only in hip-center flag and run labels.
Grounded reset/observation home, physics/model hash, rewards, commands,
randomization and network settings match. Only motor indices 4 and 13 change
their action centers.

All fifteen retained checkpoints per run are finite. `best_params` and `params`
match the selected and final checkpoint tensors, respectively. Sixteen inline
standard and sixteen push reports exist per run, plus selected/initial held-out
confirmations. Recomputing **72 reports /1,080 conditions /138,240 attempts**
reproduces selector decisions and per-episode group statistics; largest group
mean difference is 2.94e-6, std difference 5.62e-7. All recorded training and
evaluation metrics are finite.

The audit is saved at
`wr2_paper_hip_model_home_20261028_224138_seed3/review_20261029/analysis.json`.
Results and raw traces remain ignored local artifacts.

## Standard GPU results

This suite covers ten fixed nonzero XY conditions through .10 TB m/s,
standing, and four stop/restart schedules. All conditions command zero yaw.
Observation noise and model/reset/actuator variation are enabled; pushes are
disabled in the primary suite. Speeds/errors below are **TB-equivalent m/s**.
The forward speed uses native cached velocity in the initial desired-heading
frame, averaged over the full episode. It is distinct from geometric landing
lead and the warmup-excluded integrated root speed in the Mac table below.

Selected policies, held-out seed 22102:

| Metric | Qualified parent | Control +106.8M | Candidate +173.6M |
|---|---:|---:|---:|
| Desired-frame XY error | .04528 | .04416 | **.04298** |
| Body-frame XY error | .04490 | .04408 | **.04343** |
| Forward .10 speed | .08417 | .08208 | **.09225** |
| Forward .10 XY error | .04399 | .04494 | **.04354** |
| Forward transition walking speed | .08135 | .08070 | **.09129** |
| Mean absolute heading error, fixed walking | 5.36° | **4.18°** | 5.73° |
| Modeled foot self-contact, fixed walking | 3.83% | 4.01% | **3.08%** |
| Contact-phase agreement, fixed walking | 48.76% | 41.43% | **52.57%** |
| Executed leg-reference RMSE | .1875rad | **.1834rad** | .2106rad |
| Falls across selection + held-out seeds | 0/3,840 | 0/3,840 | 0/3,840 |

Candidate forward speed is 12.4% higher than the selected control, and its
aggregate desired-frame error is 2.7% lower. The paired held-out forward-speed
change is **+.01017** [+.00946,+.01088]; desired-frame error change is
**−.001177** [−.001594,−.000772]. These bootstrap intervals resample 128
episode-index clusters shared across the ten fixed conditions; there is one
training seed per arm.

The parent column uses the control run's repeated initial confirmation of
the same qualified parent tensors. Candidate desired-frame error is 5.1%
lower than that parent. Independent repeats need not give identical simulated
trajectories even when checkpoint/configuration semantics match.

The gain includes a tradeoff. Candidate heading error increases **1.55°**
[+1.20,+1.90], while leg-reference RMSE increases **.02719rad**
[+.02670,+.02770], about 14.8%. Self-contact decreases .93 percentage points
and reference contact-phase agreement improves 11.14 points. Standing desired
XY error is slightly higher, .03050 versus .03019; both remain finite and
complete. Faster forward progression is not evidence of closer joint imitation.

At the equal final 200M budget, candidate/control desired-frame errors are
.04395/.04444 on seed 12102. Their paired difference interval includes zero.
Candidate forward speed remains higher, .0907 versus .0850, but its selected
+173M checkpoint is better than its own final checkpoint. Final checkpoints
have only the inline seed-12102 standard report; held-out confirmations refer
to the selected checkpoints.

The candidate initially starts at .0598 forward speed and .05347 aggregate
error. It recovers much of that loss by +13M, but tracking spikes at +66.8M
and +120.2M. The strongest stable region arrives around +160–187M. Native
logged KL ranges .030–.235, versus .029–.064 for the control; learning rate
remains fixed at 1e-4. These are observed PPO metric fluctuations, not evidence
of a trainer correctness bug or proof that extra training is inherently bad.

Push stress confirmation remains poor: candidate selected 986/1,920 falls,
control selected 951/1,920. Training has no pushes. This stress profile is
outside the present forward-placement qualification and gives no robustness
promotion to either policy.

## Actual placement and motion on the Mac

Fifteen native MJX rollouts compare the parent, both selections and both finals
at .10/.15/.20. **All fifteen complete 824 ticks without falls/nonfinite values.**
They use production policies and mappings, no action overrides, nominal
dynamics/no noise/no pushes, high matrix precision and one episode per command.
Mac uses Python 3.11/Brax .14.2; this is paired local gait evidence, not GPU
high-command qualification. Means exclude 1.5 TB seconds of warmup.

At .20:

| Policy | Integrated forward speed | Mean pitch | Left/right landing lead, TB cm | Left/right stride, TB cm | Right hip near negative action bound |
|---|---:|---:|---:|---:|---:|
| Qualified parent | **.1809** | +7.15° | 9.87/2.50 | 13.14/13.13 | 92.8% |
| Control +106.8M | .1564 | +10.43° | 8.47/1.58 | 10.68/10.76 | 100% |
| Control final | .1791 | +6.47° | 8.62/1.63 | 10.60/10.59 | 81.5% |
| Candidate +173.6M | .1623 | −.54° | 8.23/2.01 | 10.43/10.54 | 68.9% |
| Candidate final | .1579 | −3.53° | 11.59/**−1.10** | 10.56/10.51 | 68.2% |

Landing lead is the landing foot's forward position relative to the other
foot. Stride is displacement between successive landings of the same foot.
Events require supported air, greater than 1mm sole clearance and ten sustained
contact samples. Cached contact/foot-box geometry share the native sampling
point. Both feet advance and contact order alternates; negative right lead
does not mean a stationary right foot. It means that foot lands behind the left.

At .10 the selected candidate reaches .1001 nominal speed, but its right lead
is **−1.02cm**, versus parent −.51cm. Left lead increases 6.56→7.85cm.
Thus its low-speed progression gain accompanies a larger left-dominant stroke,
not corrected alternating placement. At .15 its right lead improves to 1.70cm,
but at .20 it remains well below its 8.23cm left lead. The final candidate's
right lead is negative at all three commands: −2.90/−1.88/−1.10cm.

At .20, the selected candidate's right target p95−p5 span expands
.24°→**1.68°**, but left target span is **27.12°**. Executed right motion
is only **1.44°**, versus left **26.61°**. The right target remains mostly
at the new posterior endpoint, about −7.44° forward angle. The policy has
largely changed its right-leg stance bias rather than learned a comparable
right hip swing. Correct target reconstruction and the previous physical
holding/coverage validation distinguish this outcome from an action-map bug.

Native reference error at .20 also moves in the wrong direction: right hip
RMSE increases **.209→.329rad** from parent to candidate selection, left
.238→.257rad. These errors use executed motor positions against the original
pre-step moving reference. Expanded available range alone has not made the
policy imitate the right hip trajectory.

Candidate .20 torso roll span is 12.25° versus parent 15.37°, but it moves
about 10% slower. Normalized root-height span is essentially unchanged,
11.14 versus 11.10mm. At nearly identical .15 achieved speeds, candidate
root-height variation is larger, 10.40 versus parent 9.02mm. There is no clear
body-height-motion improvement at matched speed.

Raw traces and landing events are in `review_20261029/nominal_gait/`; the
comparison figure is `review_20261029/comparison.png` and `.pdf`.

## ToddlerBot comparison and interpretation

Local TB **`paper-walking-baseline`**, `a4f693d7cc4544f011dad23d6c353e752a6819ab`,
confirms fixed model-home + .25-rad delayed residual targets and executed
leg-position imitation against the moving reference. This candidate moves
only WR2's hip centers toward TB; grounded knee/ankle, reset and observation/
standing-reward home remain as explicitly isolated by the experiment.

TB Paper's best logged 200,089,600 checkpoint has **.03702 body-frame XY error**
and .09053 body-frame forward-.10 speed, with 8/1,920 standard-suite falls.
WR2 candidate selection has .04343 body-frame error and .09127 body-frame
forward speed (.09225 in the desired frame).
Its body-frame tracking error is still about **17% higher** than TB. TB's
mean leg-reference RMSE is .2318rad, greater than WR2's .2106rad; matching that
single joint-error metric cannot establish gait parity. TB has different
native dynamics/randomization and its curve starts from fresh training, while
these WR2 curves are additional steps from an already trained policy.

One remaining objective difference is explicit: TB Paper has XY/leg imitation
weights **5/5**, while these WR2 runs have **15/5**, retaining the prior forward
progression experiment. The right-hip suppression already existed before this
weight change, so this ratio does not establish its original cause. It does
mean current tracking gain can coexist with weak joint imitation. The data
support investigating the combined reward's learning pressure and retained
one-sided strategy; they do not justify blaming joint stops or declaring the
posterior window to be the complete cause.

## Next step

Complete paired GPU .10/.15/.20 validation of both selections and finals before
promoting a policy or choosing another training change. The existing tool
measures integrated progress/heading, torso roll/height and action saturation
with 128 varied episodes per seed. Its heading traces do not measure geometric
landing lead; the nominal foot-geometry evidence above remains a separate scope.
If the right-hip suppression persists, the next targeted investigation is the
existing PPO/reward-pressure diagnostic on these retained policies, measuring
whether the total objective drives a phase-dependent right-hip target away
from its posterior endpoint. That is more specific than changing reward weights
on the assumption that a better selector score fixed placement.

From the updated GPU checkout, run sequentially:

```bash
uv run --locked --python 3.10 --extra training python -m wr2.tools.evaluate_heading \
  --run-dir results/wr2_walking/wr2_paper_hip_control_20261028_191348_seed3 \
  --checkpoint-steps 106823680 200294400 \
  --num-envs 128 --seeds 12102 22102 \
  --output results/wr2_walking/wr2_paper_hip_control_20261028_191348_seed3/heading_evaluation_gpu

uv run --locked --python 3.10 --extra training python -m wr2.tools.evaluate_heading \
  --run-dir results/wr2_walking/wr2_paper_hip_model_home_20261028_224138_seed3 \
  --checkpoint-steps 173588480 200294400 \
  --num-envs 128 --seeds 12102 22102 \
  --output results/wr2_walking/wr2_paper_hip_model_home_20261028_224138_seed3/heading_evaluation_gpu
```

Review also corrected two metadata fields in heading evaluation and visual
logging that still reported observation home as action home. They now record
the actual action center and a separate observation home. Simulation, rewards
and the copied training results are unchanged. Three focused CLI/coordinate/
visual-log tests pass.

References: [TB paper v2](https://arxiv.org/html/2502.00893v2), local TB Paper
`mjx_env.py` and `evaluate_xy.py`, [experiment setup](wr2_hip_action_training_20261028.md),
[original qualification](wr2_goal_xy_final_confirmation_20261027.md).
