# WR2 heading GPU evaluation — progress run 0.29.0

Run: `wr2_paper_progress_20261025_085931_seed3`.
Evaluation: `heading_evaluation_gpu`, source `d421547`.

## 1. Validation and scope

The evaluation is complete: **27 cases**, covering 120,053,760 / 200,089,600 /
300,134,400 checkpoints, .10/.15/.20 TB m/s and zero yaw command. Each
checkpoint has **768 randomized episodes**, 256 per command from seeds
12102/22102, plus three nominal controls. There are no external pushes.

The archived evaluator and measurement source match committed local code.
Checkpoint hashes and saved environment/network/PPO configurations also match.
The summary reports a dirty checkout; the user confirms no files changed on the
GPU and HEAD is d421547. No physics or reward changes are inferred from that flag.

Runtime: Python 3.10.19, JAX .6.2, Brax .14.1, MuJoCo/MJX 3.14.0, GPU,
matrix precision `high`. Physics remains four .005-second substeps per .02-second
control tick. All comparisons use the saved scale **s=.8240547565**.

All primitive trajectories were checked. Native yaw reward reconstructed from
the recorded body gyro agrees within **1.46e-7 per step**; orientation reward
reconstructed against the initial desired heading agrees within **4.19e-7**.
Integrated root displacement agrees within **2.33e-10 m**. Recomputing every
per-episode summary gives maximum error **5.69e-14**. Valid masks are prefixes;
finite terminal samples are retained and no episode resumes after a fall.

Metrics below exclude the first **1.5 TB seconds**: 62 WR2 control ticks,
leaving **15.24 physical /18.494 TB seconds** for completed episodes. Body
velocity uses the native cached torso frame; requested-direction velocity uses
integrated root displacement projected onto the initial desired heading.
Heading change is the absolute net turn over that same window, averaged across
episodes. It is distinct from mean heading error and from instantaneous yaw
oscillation. Distances are physical WR2 metres.

## 2. GPU results

Completed-episode means; failures remain reported separately:

| Command, TB m/s | Checkpoint | Body-forward TB m/s | Requested-direction TB m/s | Absolute heading change | Absolute lateral displacement | Falls /256 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| .10 | 120M | .0751 | .0722 | 22.9° | .233 m | 0 |
| .10 | 200M | .0874 | .0794 | 36.5° | .376 m | 0 |
| .10 | 300M | .0879 | .0833 | 29.4° | .290 m | 0 |
| .15 | 120M | .1203 | .1142 | 24.8° | .411 m | 0 |
| .15 | 200M | .1319 | .1162 | 38.6° | .638 m | 0 |
| .15 | 300M | .1384 | .1079 | 55.9° | .887 m | 0 |
| .20 | 120M | .1617 | .1408 | 45.2° | .734 m | 0 |
| .20 | 200M | .1606 | .1395 | 41.5° | .826 m | 2 |
| .20 | 300M | .1806 | .1025 | 79.7° | 1.507 m | 0 |

At .20, 300M achieves **90.3%** of commanded body speed but **51.3%** of
commanded progress along the requested direction. 120M achieves **70.4%** in
that direction; 200M achieves **69.8%** among its 254 completed episodes.
Completed physical forward displacements are approximately **1.768 /1.752 /
1.288 m** for 120M/200M/300M; the command requests **2.512 m** in this window.

Accounting for every attempted episode and dividing progress achieved before
failure by the full requested horizon gives **.1388 TB m/s** for 200M at .20.
This avoids treating short falling intervals as successful fast walking.
The two falls occur at **2.24 and 8.44 physical seconds**, both seed22102,
episodes8/104. No nonfinite events occur at any checkpoint. Aggregate counts
are **0/768, 2/768 (.26%), 0/768**; the 200M .20 count is **2/256 (.78%)**.
This small fall-count difference does not establish a statistically different
underlying failure rate.

The nominal .20 controls independently reproduce the pattern:

| Checkpoint | Path traveled | Original-direction forward displacement | Net turn |
| --- | ---: | ---: | ---: |
| 120M | 2.233 m | +2.200 m | −17.6° |
| 200M | 2.259 m | +1.524 m | +79.6° |
| 300M | 2.509 m | +.812 m | +108.5° |

The 300M nominal full-episode turn is 119.3°; the table excludes startup.
This is the same curved-walking behavior observed in the Mac/live viewer.

## 3. Paired evidence and checkpoint choice

The same model/reset/noise draws are reused for each WR2 checkpoint.
At .20, 300M minus 120M across all 256 completed pairs gives:

- Body-forward velocity **+.01892 TB m/s**, paired bootstrap 95% interval
  **[+.01795,+.01991]**.
- Requested-direction velocity **−.03827 TB m/s**, interval
  **[−.04601,−.03028]**.
- Absolute net turn **+34.51°**, interval **[+27.87,+40.80]**.
- Absolute lateral displacement **+.774 m**, interval **[+.698,+.845]**.

For 300M minus 200M on 254 common completed pairs, requested-direction velocity
changes **−.03740 TB m/s**, interval **[−.04477,−.02999]**, and turn increases
**38.48°**, interval **[33.12,43.74]**.

However, **200M is not a clear winner over 120M at .20**: its paired progress
difference is **−.00171 TB m/s**, interval **[−.00709,+.00364]**, and heading
difference is **−3.18°**, interval **[−8.51,+2.21]**. Its pitch is about 1.02°
larger and it has two observed falls. At .15 its progress advantage over 120M
is also uncertain, while its turn is reliably about 13.84° larger. At .10 it
does progress faster than 120M, but 300M progresses faster still and leans less.

The Mac suggestion of 200M as a continuation candidate should therefore remain
a tradeoff, not a selected winner. Keep **300M as the speed/posture baseline**
and **120M as the straighter control**. No checkpoint dominates all commands
and measurements. Intervals quantify evaluation draws for one training seed;
they do not measure variation across independently trained policies.

## 4. Comparison with TB Paper

The existing native TB Paper 200M Mac comparison has 16 randomized attempts
per command. Completed means below use 16/16/15 episodes at .10/.15/.20; TB
has one .20 fall. It is a smaller CPU reference, not a matching GPU confirmation.
Both profiles retain their respective native randomization/noise distributions.

| Command | WR2 300M requested-direction TB m/s | TB Paper requested-direction TB m/s | WR2 turn | TB turn |
| --- | ---: | ---: | ---: | ---: |
| .10 | .0833 | .0870 | 29.4° | 13.5° |
| .15 | .1079 | .1227 | 55.9° | 21.1° |
| .20 | .1025 | .1407 | 79.7° | 37.9° |

At .20, WR2 120M/200M have approximately TB's completed requested-direction
progress, while 300M has stronger body speed and poorer direction tracking.
TB also turns, so it does not demonstrate perfect heading control. WR2's
completed .20 mean pitch is **12.59 /13.60 /10.34°** at 120M/200M/300M, versus
approximately **3.81°** in the smaller completed TB reference.

The Mac/GPU WR2 means agree on the principal tradeoff: .20 heading change is
45.1→45.2° at 120M, 40.4→41.5° at 200M, and 85.0→79.7° at 300M. The episode
draws differ when JAX splits a seed into 8 versus 128 keys, and the runtimes
differ, so these are protocol comparisons rather than identical-trajectory
CPU/GPU comparisons.

## 5. What the objective rewards

WR2 `paper_rewards.py` and TB `paper-walking-baseline:mjx_env.py` use body-frame
XY velocity and body gyro Z in their tracking Gaussians. Their full-quaternion
orientation term is `exp(-20 * orientation_error^2)`. Current WR2 changes XY
weight from TB's 5 to 15, with yaw5 and orientation1 retained. Its checkpoint
selector still minimizes body-frame XY error within the finite/fall gate.

At .20, completed native contribution per TB second:

| Checkpoint | XY | Body-Z yaw | Orientation | Total | Only heading-rate input corrected |
| --- | ---: | ---: | ---: | ---: | ---: |
| 120M | 8.067 | 2.500 | .1000 | 22.935 | 22.986 |
| 200M | 7.841 | 2.397 | .0716 | 22.820 | 22.680 |
| 300M | 9.078 | 2.879 | .0247 | 24.911 | 24.654 |

The turning policy receives more XY and yaw reward, despite its much poorer
requested-direction progress. Correcting body-Z rate to actual Euler heading
rate still prefers 300M on these frozen trajectories. This confirms that the
earlier proposed coordinate correction alone does not reverse this objective
preference; it is not proof of the learning outcome under changed rewards.

Mean corrected net heading rate at .20 is approximately **−.0207 /+.0153 /
+.0750 TB rad/s**, while instantaneous heading-rate standard deviation is
**.384 /.418 /.211**. The straighter earlier trajectories have more within-step
yaw motion; the 300M gait has a smoother persistent turn. Its heading is leftward
in **254/256** completed episodes. Increasing the instantaneous yaw coefficient
therefore does not specifically distinguish oscillation from systematic drift.

Frozen rescoring of the existing orientation weight at **3, 5 or 10** still
prefers 300M over both earlier checkpoints at .15/.20. Equalizing the .20 mean
return requires approximately weight **27 versus 120M**, or **46 versus 200M**
on common completed pairs. These are not proposed coefficients. Large changes
would also strongly affect roll/pitch and posture. With gamma=.97, discounted
return from reset also prefers 300M: **15.268 /15.591 /16.335** among completed
.20 episodes. Its early discounted orientation contribution is actually larger,
because early posture and late heading drift affect that term differently.

### Goal-consistent XY rescoring

A more direct candidate preserves the TB XY Gaussian and current coefficient15
while measuring velocity in the **desired-heading reference frame**. The same
cached torso velocity is reconstructed in world coordinates using its quaternion,
then projected onto the initial desired heading. This isolates the frame change
at the native reward sampling point, rather than changing physics or averaging
velocity over a different time interval. The zero yaw command holds that reference
heading constant for these evaluations.

| Command | Checkpoint | Reference-frame XY error, TB m/s | Only XY frame changed: native return/TB second |
| --- | --- | ---: | ---: |
| .10 | 120M | .0570 | 26.099 |
| .10 | 200M | .0611 | 25.325 |
| .10 | 300M | .0512 | 26.719 |
| .15 | 120M | .0696 | 23.985 |
| .15 | 200M | .0821 | 22.809 |
| .15 | 300M | .0967 | 22.359 |
| .20 | 120M | .1055 | 20.334 |
| .20 | 200M | .1099 | 20.092 |
| .20 | 300M | .1693 | 18.584 |

This change gives the straighter 120M trajectory higher return at .15/.20,
while retaining 300M's preference at .10, where its requested-direction progress
is better. All other recorded rewards are unchanged. These frozen scores are
evidence that the objective distinguishes the desired behavior; they do not
predict PPO's improvement or eliminate the need for a continuation experiment.

## 6. Next priority

The stronger GPU evidence supports correcting the mismatch between desired
direction and the learning/evaluation signals. It does not support another
unchanged-objective continuation or selecting 200M solely from the Mac result.

Before a heading-control training change, verify **reference-relative heading
feedback**. Both current actors observe Euler yaw and gyro but do not receive
the accumulated reference heading or its error explicitly. With arbitrary
initial headings and previous turn commands, current yaw alone does not specify
the desired-heading error after the short observation history. The same
limitation exists in TB code; it is not a new WR2 Brax implementation fault.

A relevant published-project precedent is
[Legged Gym's heading feedback](https://github.com/leggedrobotics/legged_gym/blob/8fa29acc6fd1910c3d9659eef6310bdd301cde0a/legged_gym/envs/base/legged_robot.py#L327),
which converts desired-minus-current heading into a yaw-rate command. This
supports checking goal feedback before coefficient tuning. It is a reference
principle, not a controller adopted here; TB's current translation/pure-turn
sampling must be considered before changing command semantics.

The best-supported next candidate is a consistent goal representation:

1. Measure the existing XY reward in the desired-heading frame, preserving
   its Gaussian width and coefficient15.
2. Give actor and critic the reference-relative heading error, so that the
   reference frame used by the reward is observable. Sine/cosine features with
   zero-initialized additional input weights can preserve the initial trained
   policy while making that goal available; checkpoint equivalence needs validation.
3. Evaluate requested-direction XY error and heading when retaining checkpoints.
   Continue from the trained 300M policy for **200M steps** once prepared and
   validated, retaining intermediates and the 120M straighter control.

This is a deliberate extension of TB's body-frame velocity/observation recipe.
The evidence is a larger body-tracking improvement that loses real progress
after 300M, and frame rescoring that restores preference for the straighter gait
without a large coefficient change. It keeps physics, action origin, moving
leg imitation, qualified landing and the other rewards unchanged. No reference
joint actions are substituted for PPO outputs. The new reward frame, observations
and continuation configuration are proposed, not implemented by this review.

References: [ToddlerBot Table VII](https://arxiv.org/html/2502.00893v2), local
`paper-walking-baseline:toddlerbot/locomotion/mjx_env.py`, WR2
`walking_env.py::_paper_observation/_paper_step`, `paper_rewards.py`, and
`paper_evaluation.py::selection_summary`.

## Artifacts

`results/wr2_walking/wr2_paper_progress_20261025_085931_seed3/heading_evaluation_gpu/review`:

- `analysis.json`: recomputed metrics, completed/all-attempt populations,
  paired intervals, failures, coefficient rescoring and discounted-return checks.
- `gpu_heading_comparison.png`: retained WR2 GPU policies and the smaller
  native TB Mac reference.
- `heading_feedback_reference.json`: pinned public-project reference.

The original evaluation summary, traces and archived source are retained.
