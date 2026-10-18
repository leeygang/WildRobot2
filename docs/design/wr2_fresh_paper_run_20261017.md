# WR2 latest result and prepared fresh Paper run — 2026-10-17

**Completed:** the prepared seed-3 run finished 300M. See the
[2026-10-18 result and improvement proposal](wr2_fresh_paper_result_20261018.md)
for the measured tracking/survival regression, actual flight, source comparison
and proposed next experiment. This document records the original preparation.

## Decision

Prepare **0.28.4: fresh seed 3, fixed LR 1e−4, 300M transitions** through the
regular WR2 trainer. This repeats the corrected Paper experiment with an
independent initialization. The previous two continuations inherited the same
seed-0 gait; changing their sampling seeds did not create independent policies.

The new loaded-motion comparison confirms weak motion and greater execution
lag in WR2. It does not identify a measured correction to its fitted physics.
The fresh full-size PPO batch supports TB's initial learning rate. This control
therefore takes priority over changing actuator values, imitation weight or
gait timing without a demonstrated correction.

## Latest completed result

Latest folder: `results/wr2_walking/wr2_paper_20261017_115520_seed2`, recorded
source `1cc55a0`, recipe 0.28.3. It added **300,134,400 transitions** to the
selected seed-1 +60M parent. Final ancestry is **620,277,760 transitions**.

| Checkpoint/profile | TB-equivalent fixed walking XY error, m/s | Falls / 1920 |
| --- | ---: | ---: |
| WR2 continuation start | 0.10639 | 61 (3.18%) |
| WR2 +160,071,680, retained survival control | 0.10544 | 42 (2.19%) |
| WR2 +180,080,640, retained tracking control | **0.10453** | 44 (2.29%) |
| WR2 final +300,134,400 | 0.10727 | 52 (2.71%) |
| TB Paper 200,089,600 | **0.03702** | 8 (0.42%) |
| TB Main final, native comparison profile | 0.08395 | 28 (1.46%) |

The best WR2 checkpoint improves tracking **1.74% versus its start**, remains
**2.82× Paper 200M**, and fails the ≤1% fall gate. The final checkpoint is worse
than the retained tracking control. Epoch KL is 0.0051–0.0106: the smaller
updates addressed the prior optimization instability, while the gait plateau
remains. Main's final error also exceeds the zero-translation baseline 0.07162;
its weak directional response does not support transferring its reward recipe.

The [full progress review](wr2_tb_progress_review_20261017.md) independently
reconstructs 95 suites / 182,400 episode-condition records with zero failures,
checks every new native checkpoint for finite arrays and exact restore/export
identity, and verifies captured sources from both TB runs. Returns are not the
shared comparison metric. Paper and WR2 include native variation and configured
pushes; the [later physical review](wr2_actuator_gap_review_20261018.md) confirms
Paper's pushes are dropped by its `qd`/`qvel` mismatch while WR2 realizes them.
Main's native profile has different variation and no configured pushes.

## New loaded gait comparison

Replayed WR2 +180M and TB Paper 200M nominally for forward, right, left and
standing. Four trajectories per robot complete their full horizons with no
falls or nonfinite samples. WR2 uses 824 ticks, fixed reset and MJX 3.14;
TB uses 1000 ticks, its native random heading/home offset and MJX 3.2.4.
Both use the first of the eight seed-22102 reset keys from the earlier nominal
control. These are individual trajectories, not estimates across model draws.

WR2 velocities divide by `s = 0.824054757`; lengths divide by `s²`. Reference
angles use each implementation's pre-step reward clock. Actual motor and foot
measurements use the end of the 20 ms tick. The delivered target reconstruction
is exact for both robots. Reference lift uses forward kinematics of the
reference motor pose, with the same foot-body definition for both robots.
CoM displacement is measured in the reference heading frame.

| Nominal measurement | WR2 forward | TB forward | WR2 right | TB right | WR2 left | TB left |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Commanded-axis velocity, TB-equivalent m/s | 0.0701 | 0.0980 | −0.0101 | −0.0340 | 0.0050 | 0.0371 |
| Delivered target vs. reference RMSE, rad | 0.1809 | 0.2396 | 0.1797 | 0.2376 | 0.1782 | 0.2395 |
| Actual motor vs. delivered target RMSE, rad | 0.0486 | 0.0238 | 0.0452 | 0.0211 | 0.0477 | 0.0209 |
| Single support | 99.64% | 83.40% | 99.39% | 86.60% | 99.51% | 87.00% |
| Torque/speed-envelope clipping at first substep | 4.93% | 1.18% | 2.89% | 0.18% | 3.70% | 0.16% |
| CoM lateral range relative to foot midpoint, TB-equivalent mm | 6.76 | 43.99 | 11.46 | 36.25 | 13.92 | 38.33 |

WR2's delivered/reference mismatch is about **3.7–4.0×** its motor/delivered
mismatch. The retained policy requests weak motion; execution lag also remains
larger than TB's. Forward sole lift peaks are **6.8/7.8 mm** in TB-equivalent
units, versus TB **10.0/13.5 mm**. The supported foot-body XY speed used by TB's
slip reward is 0.0596/0.0776 m/s for WR2 versus 0.0321/0.0369 for TB. This is a
body-origin proxy that includes foot rocking, not measured contact-point slip.
Standing XY speed is 0.0434 versus TB 0.0270 m/s.

No leg target clips at a joint bound or saturates the tanh action in these
traces. WR2 target slew is active in 1.7–3.1% of moving leg samples; the previous
same-policy slew removal improved nominal errors only 0.3–3.0%. Joint speed at
the first physics substep stays below the nominal limit. These boundary samples
do not characterize all four/five physics substeps or prove full-reference
dynamic feasibility. TB succeeds with greater raw reference RMSE, so increasing
imitation weight or action range solely to reduce that RMSE lacks support.

The [trace audit](../../results/wr2_walking/wr2_paper_20261017_115520_seed2/review_20261017/loaded_gait_audit.json)
reconstructs all eight trajectories / **7,296 sampled control ticks**, with
finite arrays, consistent reference-lift/frame definitions and exact target
reconstruction. Raw traces and per-joint summaries remain local in the run's
`review_20261017` directory.

## Code and optimizer checks

Compared WR2 `walking_env.py`, `paper_rewards.py`, `paper_walk.py`, `ppo.py` and
`train.py` with frozen TB Paper `mjx_env.py`, `walk_zmp_ref.py`,
`mujoco_control.py`, `paper.gin` and its Brax fork `14147806`. Both references
use a static pose at time zero. Fixed action home, one-tick delay, leg imitation
weight 5, Swish/NormalTanh and the 1024 × 20 / four-pass rollout layout remain
aligned. WR2's actual value coefficient is 0.25 × MSE. No new confirmed
reference, initialization or adapter bug was found.

WR2's identified low-load damping mapping remains correct: 0.173622 passive +
0.5 velocity feedback = 0.673622 fitted total. Its larger armature, lower speed
capacity and provisional loaded torque envelope differ from TB; geometry
normalization alone does not establish matching actuator bandwidth. The extra
setpoint slew cap is derived from vendor motor speed, rather than a measured
firmware command-rate limit. See [servo evidence](servo_model.md).

Tested a **fresh seed-3 actor/critic on one real, randomized/noisy WR2 batch of
1024 × 20 transitions**, 256 trajectories per minibatch, four minibatches and
four passes. Both LR probes use identical data, parameters, permutations, RNG
and fresh Adam state, with native Brax 0.14.1 losses.

| Learning rate | Mean pre-gradient joint KL | Post-batch joint KL | Post-batch ratio clip fraction |
| --- | ---: | ---: | ---: |
| **1e−4, prepared** | 0.002444 | 0.003794 | 4.04% |
| 1e−5 | 0.000180 | 0.000312 | 0.00% |

All 20,480 transitions and updated parameters are finite; the batch contains
zero terminal transitions. This supports the fresh initial LR, not a claim
about later optimizer stability or the final walking result. TB's cosine
schedule is effectively constant at 1e−4 across 300M transitions.
`desired_kl=0.01` is inactive under the fixed schedule; review the logged KL
alongside tracking and falls at retained checkpoints.

The [native preflight](../../results/wr2_walking/wr2_paper_20261017_115520_seed2/review_20261017/fresh_native_preflight.json)
also confirms native seed-3 initialization exactly matches the full-physics
probe, performs an 80-transition update with the real network dimensions on
the existing cheap test environment, and reloads its checkpoint exactly.
Two existing native update/restore and evaluator-loading regressions pass.
CLI/config validation confirms a fresh start and the requested budget. These
are CPU checks; GPU compilation remains a host check.

## Start the next run

From the WR2 checkout on the GPU host:

```bash
uv run --extra training python -m wr2.locomotion.train \
  --config wr2/locomotion/configs/ppo_walking_paper.yaml
```

The prepared YAML contains **seed 3, fixed LR 1e−4 and 300M requested
transitions**. Fresh actor, critic, normalizer and Adam are initialized by the
native trainer. Its fixed grounded motor home and physical reset remain the
current corrected Paper home. Config values differ from the original corrected
fresh seed-0 run only in seed and version metadata. Relative to the completed
seed-2 continuation, initialization is fresh and LR returns to the Paper value.

Checkpoints are retained/evaluated about every 20M transitions; native rounding
produces 300,134,400 at the final checkpoint. Evaluate the entire curve against
the retained WR2 +180M/+160M controls and TB Paper 180M/200M. Improvement needs
better signed tracking with survival considered together; qualifying candidates
must meet ≤1% falls and pass held-out seed 22102. Paper parity remains roughly
0.0370 equivalent error with its measured directional response. A fresh run
that reproduces the plateau would strengthen the case for a measured loaded
actuator/reference correction.

The [training guide](wr2_paper_training.md) includes the optional full-batch GPU
check and completed-run reproduction commands. The setup requires only WR2.

## References

- [ToddlerBot, Tables VI/VII](https://arxiv.org/html/2502.00893v2), checked against
  the successful local Paper reconstruction, source `1982ef3`.
- [Schulman et al., PPO](https://arxiv.org/abs/1707.06347).
- [Previous progress and source audit](wr2_tb_progress_review_20261017.md) and
  [servo measurement evidence](servo_model.md).
