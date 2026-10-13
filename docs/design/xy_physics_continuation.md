# Signed XY continuation with corrected randomized physics

## Confirmed 1B review

`results/wr2_walking/wr2_xy_20261012_140728_seed0` added 800,030,720
transitions to the previous 200,007,680-step policy, reaching **1,000,038,400
cumulative transitions / 24,415 PPO updates**. Its saved best is the retained
980,049,920-step policy, not the final policy.

The final inline evaluator completed 1,408 episodes of 1,000 steps without
falls. It disabled actor observation noise and omitted vector model variation;
local servo/reset variation remained enabled. These measurements establish
survival in that evaluation, not robust command following.

| Command `(vx, vy)` m/s | Mean achieved `(vx, vy)` m/s | XY MAE m/s |
|---|---|---:|
| `(+0.10, 0)` | `(+0.0627, +0.0041)` | 0.0621 |
| `(-0.10, 0)` | `(-0.0757, +0.0008)` | 0.0434 |
| `(0, +0.05)` | `(-0.0028, +0.0108)` | 0.0573 |
| `(0, -0.05)` | `(-0.0116, -0.0027)` | 0.0566 |

XY MAE is the mean of each step's vector error. It is different from the error
of the mean velocity. Lateral MAE exceeds the 0.05 m/s error of a stationary
robot. The forward start/stop schedule has 0.0630/0.0578 m/s XY MAE during
the first 0.5 s after command changes. Lateral walking and recovery remain
the main gaps.

The nominal worst-endpoint scores at 400M and 980M are 0.0277506 and
0.0277521, a 0.0055% relative difference. Forward speed improves earlier;
standing and backward motion improve later. Ranking alone does not distinguish
these policies reliably. The final score is lower, at 0.0191188.

## Paired checkpoint comparison

The held-out comparison uses seed **12102**, the corrected physics and full
configured model/observation/servo/reset variation. Each retained policy
(400M, 980M and final 1B) receives the same models and reset keys for **128
episodes in each of 15 conditions**. Measurements stop at the first terminal
step or the 1,000-step limit. Per-step measurements are averaged within each
episode, then across episodes; reported 95% intervals use the episode standard
error. This compares **5,760 episodes** across the three policies.

| Measure | 400M | 980M | Final 1B |
|---|---:|---:|---:|
| Falls across all 1,920 episodes | 10 | 40 | 28 |
| Backward start/stop falls, out of 128 | 1 | 36 | 25 |
| Backward start/stop mean episode length | 993.9 | 786.3 | 852.2 |
| Mean X speed at +0.10 m/s | +0.0859 | +0.0635 | +0.0639 |
| Mean X speed at -0.10 m/s | -0.0771 | -0.0829 | -0.0753 |
| Mean Y speed at +0.05 m/s | +0.0086 | +0.0084 | +0.0114 |
| Mean Y speed at -0.05 m/s | +0.0004 | -0.0036 | -0.0031 |
| Standing-from-reset XY speed, m/s | 0.0374 | 0.0094 | 0.0098 |
| Complete signed XY/recovery qualification | Fail | Fail | Fail |

No nonfinite episode occurs in this comparison. The late policies pass only
the fixed zero-command condition. Every moving endpoint and every transition
condition fails at least one acceptance gate.

| Command `(vx, vy)` m/s | 400M XY MAE | 980M XY MAE | Final XY MAE |
|---|---:|---:|---:|
| `(+0.10, 0)` | 0.0702 ± 0.0031 | 0.0623 ± 0.0010 | 0.0672 ± 0.0024 |
| `(-0.10, 0)` | 0.0592 ± 0.0012 | 0.0422 ± 0.0014 | 0.0474 ± 0.0012 |
| `(0, +0.05)` | 0.0549 ± 0.0006 | 0.0597 ± 0.0037 | 0.0599 ± 0.0010 |
| `(0, -0.05)` | 0.0614 ± 0.0006 | 0.0599 ± 0.0007 | 0.0597 ± 0.0006 |

Values are m/s; ± is the approximate 95% interval half-width across episodes.

The 980M policy falls in **36/128 backward start/stop episodes (28.1%)**,
versus 1/128 for 400M and 25/128 (19.5%) for the final policy. Its backward
transition mean length is 786.3 steps.
One retained trace terminates 207 steps after reset, about 1.1 s after the
first backward launch. This is a launch/recovery gap that the old forward-only
transition evaluator did not measure. Good standing from reset does not
establish successful backward launch. The production transition environment
and standard Brax evaluator reproduce the retained pair exactly: lengths
`[1000, 207]`, with one fall, using identical sampled models and reset keys.

**Continue from 400M.** It has substantially fewer backward launch failures
and better forward response. Its worse standing drift and forward vector
error remain explicit tradeoffs. This selects a learning source under the
corrected physics; it does not promote a qualified walking policy. Neither
the nominal saved best nor the final policy supplies a safer starting point
for the full signed start/stop suite.

The raw comparison, joint traces, source/configuration provenance and smoke
reports are stored locally in the copied run's `review_20261012/` directory.
`candidate_suite.py` records the exact seed, condition order and measurement
procedure; `evaluation_config.yaml` records the configuration used.

## Corrections

1. **Refresh randomized model constants.** WR2 changed mass, inertia and
   armature without recomputing MuJoCo's derived constants. The native oracle
   found inverse-weight errors around 20% and subtree-mass errors of
   16.7–25% for uniform mass changes within the configured range. MJX uses
   inverse weights in constraint rows. Following TB's host preparation,
   `domain_randomization.py` now calls `mj_setConst` once per sampled model
   and copies `body_invweight0`, `dof_invweight0`, `body_subtreemass`,
   `dof_M0`, `actuator_acc0` and `tendon_invweight0` with batched axes.
   The native model is copied, and no host work is added to simulation steps.
2. **Reject invalid/incomplete rankings.** Nonfinite score inputs and missing
   required endpoints cannot promote a checkpoint. Native endpoint evaluation
   requires every configured endpoint; it no longer falls back to the primary
   forward command. Historical Brax evaluations that report only the primary
   command retain their existing behavior.
3. **Invalidate ZMP caches on relevant geometry changes.** The lookup signature
   now includes joint axes/positions, inertial COM positions, site orientations,
   parent/transmission mappings and complete home coordinates. A 0.2-rad
   hip-roll-axis perturbation previously reused stale targets; rebuilding
   changed targets by up to 0.02847 rad (1.63 degrees).

The constant defect was active during training. All recorded endpoint scores
in the completed run were finite, and the selected 980M checkpoint matches
the saved best; the other two defects did not demonstrably affect that run.
A paired frozen-policy probe changed XY MAE by at most 0.00036 m/s after
refreshing constants. Correcting physics is necessary, but this observation
does not establish the cause of weak lateral learning or promise recovery.

All 53 reference geometry cases pass, with at least 8.96 mm planned support
margin. Numerical IK position residuals are about 10 micrometres in the
cardinal probe. Analytic LIPM consistency does not certify executed whole-body
ZMP or actuator feasibility. The reference remains a privileged critic input;
actor targets are residuals around walk-home, and motor imitation is disabled.

## Evaluation and checkpoint qualification

The new [configuration](../../wr2/locomotion/configs/ppo_walking_xy_physics.yaml)
enables observation noise, vector model randomization with refreshed constants,
and the existing local servo/reset variation in inline evaluation. Each
evaluation covers **15 conditions × 128 episodes × 1,000 steps**:

- Ten fixed signed cardinal/diagonal endpoints from the existing XY evaluator.
- A fixed zero command, measured separately from recovery after walking.
- Forward, backward and both lateral start/stop schedules: 3 s standing,
  6 s walking, 3 s standing, 5 s walking, 3 s standing.

For each moving endpoint, log vector XY MAE, X/Y MAE, signed command response
`(command · mean_velocity) / |command|²`, orthogonal mean velocity, survival,
phase/contact, clipping and torque metrics. Each transition category uses its
own positive sample count; standing, walking, the first 0.5 s after starting
and the first 0.5 s after stopping must independently pass.

The initial thresholds explicitly reuse WR2's existing robust-walking goal:
at least 0.90 signed command response, at most 0.02 m/s XY MAE, at most 1%
falls, mean length at least 995, and the existing phase/contact/torque gates.
The 0.02 m/s ceiling also applies separately to reset-standing and every
transition category. Standing is exempt from walking phase/swing gates.
The complete thresholds are in `checkpoints.qualification_goal` in the YAML.

These are **WR2 acceptance goals**, not measured ToddlerBot performance.
They gate saved `best_params.pt` and do not change the PPO objective. A best
file is written only when every endpoint and transition passes. **All retained
checkpoints and final `params.pt` are saved regardless of qualification.**
If no policy qualifies, compare retained policies from `training_metrics.jsonl` and
continue from an explicitly reviewed checkpoint; there may be no best file.
Independent `evaluate` reports scope qualification to the requested commands
and seeds, so a single-command pass does not certify the complete suite.

New run metadata also records the git revision/dirty state, Python and relevant
dependency versions alongside the existing effective configuration, model and
restore hashes.

## Next 200M run

Restore the **400,015,360-step / 9,766-update** checkpoint:
`wr2_xy_20261012_140728_seed0/checkpoints/000200007680.pt`.
The filename counts transitions added within that 800M run; its native
checkpoint counter includes the preceding 200M. Retain the 980M/final policies
for comparison, including their better standing behavior.

The continuation uses 2,048 environments, 20-step rollouts, eight minibatches
and the existing TB-aligned PPO/reward/network settings. Mirror loss stays off.
Gait timing, actuator parameters, action/observation semantics, command ranges,
randomization ranges and reference generation are preserved. The only training
dynamics change is refreshing the randomized constants.

The nominal 200M budget rounds up to **200,007,680 added transitions / 4,883
updates**. Eleven evaluations retain the source at step zero and ten more
checkpoints about 20M transitions apart. Evaluate intermediate policies before
choosing a subsequent source; the final checkpoint is not assumed best.

From the repository root on the GPU host, after syncing this commit and ensuring
the copied source run is present:

```bash
uv sync --extra training
uv run --extra training python -m wr2.locomotion.train \
  --config wr2/locomotion/configs/ppo_walking_xy_physics.yaml \
  --restore-checkpoint results/wr2_walking/wr2_xy_20261012_140728_seed0/checkpoints/000200007680.pt
```

Restore the complete native policy, Adam, learning rate, counters and Torch
RNG state. Do not reset exploration. From the 400M source, the expected final
counter is **600,023,040 cumulative transitions / 14,649 updates**. Compare
against the historical 600M policy at `checkpoints/000400015360.pt` using
the same held-out conditions, as well as all newly retained checkpoints.
The historical run is a budget comparison; environment state was not saved
in its native checkpoints, so it is not an exact paired training replay.
`XLA_PYTHON_CLIENT_PREALLOCATE=false` remains the training startup default.

## Verification

- Full test suite: **242 tests and 77 subtests passed**. The new regressions
  cover native model constants, nonfinite/incomplete promotion, geometry cache
  invalidation, signed projection, independent recovery sample counts and
  preservation of all transition directions. Ruff and diff checks pass.
- The selected 400M checkpoint restores policy, Adam, learning rate, iteration,
  cumulative steps, CPU Torch RNG, robot contract, observation layout and
  mirror setting exactly. An actual MJX/Torch CPU smoke completes 80 added
  transitions and two PPO updates, changing the policy and saving finite model
  and optimizer state. Counters advance from 9,766/400,015,360 to
  9,768/400,015,440. Actor/critic/action dimensions remain 1005/1740/12.
  The 980M checkpoint separately passes the same restoration/update checks.
- The smoke uses shortened 20-step episodes and two environments. It verifies
  restoration and execution; the separate 1,000-step evaluations measure gait.
  It does not verify CUDA RNG replay or GPU memory fit.
- The production randomized Brax evaluator completes two 1,000-step lateral
  transition episodes. It records 450 standing, 550 walking, 50 start and
  50 stop samples per episode, with finite metrics and no falls. The XY gates
  correctly reject the measured recovery errors; evaluator execution passing
  is separate from policy qualification.

Restore-file SHA-256:
`9990ebf7ffb40b6bdfe076bf7ed31133e57f762cd56cd794060c8201621de850`.
Local verification uses JAX/JAXlib 0.6.2, MuJoCo/MJX 3.14.0, Brax 0.14.2,
RSL-RL 2.3.3, Torch 2.8.0, NumPy 2.4.6 and Python 3.11.14. The copied GPU
run does not contain a complete dependency/revision manifest, so this is
verification against current source, not a claim of exact GPU replay.

## ToddlerBot comparison and next decision

Both implementations use 12 leg actions, 20 ms policy periods with four
5 ms physics substeps, one-step action delay, Normal residual actions scaled
by 0.25 rad, and 512/256/128 ELU actor/critic networks. WR2's 2,048 environments
and eight minibatches preserve TB's 5,120 transitions per minibatch. Its
20% standing / 80% XY distribution follows the TB sampler with turning omitted.

The measured hip-to-foot-height ratio is 0.9782; square-root speed/time scaling
is 0.9891. The current 0.72 s gait and 40 mm swing prior are therefore close
under leg-length scaling. WR2's mass is 2.058 kg versus the local TB 2xc model's
3.454 kg. Its actual whole-body walk/home COM height is about 0.209 m versus
0.282 m.
WR2's measured provisional armature/friction are also much higher. Leg length
does not establish complete dynamic similarity, and these differences do not
justify silently replacing WR2's identified servo parameters with TB values.

TB uses preview ZMP planning and analytic IK; WR2 uses a periodic LIPM reference
and model-derived numerical IK for its CAD geometry and joint limits. The
source-confirmed phase/action-history differences are documented in the
[semantic audit](tb_semantic_parity_audit.md). Their effect on this run has not
been measured; keep any planner/control comparison separate from this physics
correction and use executed joint traces to decide whether it is warranted.

If the corrected 200M continuation still fails lateral response or recovery,
compare actual hip-roll/yaw targets, tracking and effort against a healthy
current TB policy before changing rewards, exploration or reference timing.
The only copied TB archive is a historical optional Brax run that becomes
nonfinite near 233M. It cannot supply a valid final TB benchmark. Full TB parity
also requires turning; this continuation's yaw range remains zero.

## References

- [ToddlerBot paper](https://arxiv.org/abs/2502.00893) and
  [public project](https://toddlerbot.github.io/).
- Local source of truth: `~/projects/ToddlerBot/toddlerbot/locomotion/walk.gin`,
  `mjx_env.py`, and `train_mjx.py` (host randomization), plus
  `toddlerbot/algorithms/zmp_walk.py` (planner/IK).
- [MuJoCo `mj_setConst` API](https://mujoco.readthedocs.io/en/stable/APIreference/APIfunctions.html#mj-setconst).
- [WR2 signed XY design and previous continuation](xy_walking_training.md),
  [six-DOF model validation](hip_yaw_model_import.md), and
  [actuator evidence](servo_model.md).
