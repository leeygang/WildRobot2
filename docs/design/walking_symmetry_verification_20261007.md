# Pre-training symmetry and propulsion verification

## Decision

Keep `best_params.pt` from `wr2_ppo_20261007_152924_seed0`. The verification
found no gross joint-sign, joint-limit or ZMP left/right mapping defect. It did
find a numerical contact-solver problem before any new reward should be added:
one Newton iteration leaves substantial momentum-balance residuals. Ten
iterations reduce these residuals by roughly two orders of magnitude; twenty
iterations produce essentially the same gait on the two checked seeds.

The recommended next experiment is **physics convergence first**, preserving
ToddlerBot's Newton solver, 5 ms physics, four substeps, residual action mapping,
Gaussian reward, heading reward and fitted servos. Change only the solver
iteration ceiling to ten, reevaluate the retained checkpoint at all three
commands and during transitions, then run a bounded 20M refinement if those
checks pass. This is a WR2 exception supported by measured numerical error,
not a claim that ToddlerBot's default is generally wrong.

Policy asymmetry remains a separate, measured problem. Consider actor mirror
loss only after the converged-physics experiment. Do not add the standing knee
overrun cost, mirror loss and a solver change together. None of these training
changes are implemented by this verification commit.

## Scope and reproducibility

```bash
uv run --extra training python -m wr2.tools.verify_walking_symmetry \
  --run-dir results/wr2_walking/wr2_ppo_20261007_152924_seed0 \
  --output results/wr2_walking_diagnostics/symmetry_baseline \
  --seeds 101,202,303,404
```

Use a different output directory for each probe:

- Add `--solver-iterations 10` for converged contact physics.
- Add `--solver-iterations 20 --seeds 101,202` for the convergence comparison.
- Add `--project-actor` for a diagnostic mirror-equivariant actor projection.
- Combine `--project-actor --solver-iterations 10` to separate policy symmetry
  from the one-iteration solver artifact.
- `--summarize-only` recomputes summaries from retained NPZ traces, preserving
  the original probe flags and seed metadata.

Measured artifacts are under
`results/wr2_walking_diagnostics/20261007_symmetry_solver_check`,
`20261007_symmetry_solver10`, `20261007_symmetry_solver20`,
`20261007_symmetry_projection`, and
`20261007_symmetry_projection_solver10` (each relative to that results parent).
Each contains checkpoint SHA-256 hashes, model/reference results, complete
control-step NPZ traces and a JSON summary. Results are ignored by Git.

There are three fixed forward commands: 0.05, 0.075 and 0.10 m/s. Both policies
use matched seeds, deterministic actions, nominal model dynamics, configured
reset pose/actuator variation, and noiseless observations. Settled summaries
exclude the first four seconds. Four probes have 24 twenty-second episodes
each, plus twelve episodes for the two-seed convergence check: **108 episodes,
all 1000 steps, no falls or non-finite states**. These are causal diagnostics,
not full randomized deployment qualification. An initial substep-only pilot
is retained separately but is not counted in this matrix.

## 1. Model and reference

Reflect world positions across the sagittal plane, `S = diag(1,-1,1)`.
Angular axes are axial vectors, so their reflection is `-S`. Derive each joint
coordinate sign from the canonical MuJoCo world axes. All WR2 motor signs are
-1 after the left/right swap, including knee, ankle pitch and elbow. Those
signs differ from WR1: copying WR1's transform would be incorrect.

Check physical limits, walk-home targets, neutral axes/anchors, body masses and
world inertias, 64 off-home poses, and the entire five-command/36-phase ZMP grid.

| Check | Measured result |
|---|---:|
| Mirrored joint-limit mismatch | 0 degrees |
| Mirrored walk-home mismatch | 0 degrees |
| Mirrored foot position, random-pose RMS / max coordinate error | 0.104 / 0.345 mm |
| Whole-robot CoM RMS / max coordinate error | 0.142 / 0.267 mm |
| Leg mass-matrix relative RMS / max error | 0.083% / 0.090% |
| ZMP joint-reference RMS / max mismatch | 0.021 / 0.063 degrees |
| ZMP foot-reference RMS / max coordinate mismatch | 0.052 / 0.069 mm |
| Foot collision size / friction mismatch | zero |

These are small but nonzero CAD differences, not evidence of exact geometric
symmetry. Neutral leg axes differ by up to 0.379 degrees; paired leg masses
match. The reference mismatch is much smaller than the measured policy and
gait asymmetry. There is no basis here for altering joint limits, replacing
the reference or imposing exact symmetry on the robot's inertias.

## 2. Actor and stride symmetry

The actor reflection transforms **all fifteen history frames**, not just the
latest frame: phase sin/cos change sign (half-cycle shift), lateral/yaw commands
change sign, joint positions/velocities and previous actions swap with the
model-derived signs, gyro uses axial-vector reflection, gravity uses
polar-vector reflection, and heading sin changes sign while cos is preserved.
Tests verify reflection twice is identity and that transformed actor features
match features generated from a physically reflected MuJoCo state. No critic
features are used for this actor test.

Evaluate `policy(mirror(obs)) - mirror(policy(obs))` on settled rollout states
and express residual error in requested-target degrees using the unchanged
14.3-degree action multiplier:

| Metric | Retained policy | Final policy |
|---|---:|---:|
| Actor mirror mismatch, RMS across commands | 2.10 degrees | 2.28 degrees |
| Actual joint half-cycle mismatch, 0.10 m/s, converged physics | 4.52 degrees | 4.91 degrees |
| Relative-foot half-cycle mismatch, 0.10 m/s, converged physics | 12.52 mm | 11.90 mm |

The foot mismatch includes lateral sway and all three coordinates; it is not
solely a step-length difference. Actual trajectories use a half-cycle
left/right comparison over complete 36-control-step cycles. Each command has
84 complete settled cycles (21 per seed); these correlated cycles are not 84
independent trials.

For converged final-policy physics at 0.10 m/s, quarter-mean forward speeds are
0.0734, 0.1170, 0.0779 and 0.0518 m/s. The corresponding descending-swing
quarters remain very different. Over a full cycle, mean signed forward impulse
is about -0.129 N s on the left and +0.128 N s on the right. This confirms an
uneven propulsion/braking distribution. It does **not** mean a supporting foot
has failed: both carry vertical load, and positive and negative force occur
naturally in walking.

Touchdown events, support fractions, joint traces and relative foot positions
are retained per cycle. Threshold crossings can include contact chatter;
they are event traces, not a falsely precise single touchdown-time score.

## 3. Full-substep forces and solver convergence

Reconstruct the exact delayed, biased, clipped, quantized and slew-limited
servo target, then replay the same four physics substeps. No training dynamics
are modified by instrumentation. Replayed joint positions agree with the
ordinary environment step to within a few millionths in generalized position;
the initial minimal replay was bit-identical. Extra JAX diagnostic operations
can change floating-point fusion slightly.

For each foot, sum the force **on that foot** across every 5 ms substep. Retain
positive and braking impulses separately before cancellation. Compute slip
at supporting contact points, including angular foot motion; unsupported feet
have no slip measurement, rather than zero. Refresh post-integration
kinematics before computing total momentum at every body's inertial center.

Two checks distinguish force bookkeeping from underconverged dynamics:

```text
physical balance = momentum_after - momentum_before
                   - foot_contact_impulse - mass * gravity * control_period
solver balance   = integral((M * qacc - qfrc_smooth - qfrc_constraint)[:3])
```

The first three generalized coordinates are root world translation. Internal
actuator, friction, and joint-limit forces should not create external linear
impulse. Small Euler discretization residuals remain after solver convergence.

| Probe | Retained momentum RMS | Final momentum RMS | Retained solver RMS | Final solver RMS |
|---|---:|---:|---:|---:|
| 1 Newton iteration | 0.08247 N s | 0.07014 N s | 0.08246 N s | 0.07016 N s |
| 10 Newton iterations | 0.000528 N s | 0.000571 N s | 0.00000879 N s | 0.00001015 N s |

This is not only a vertical effect. One-iteration forward residual RMS is
0.0314 N s for the retained policy and 0.0240 N s for the final policy, versus
about 0.00039/0.00043 N s after convergence. At 1.933 kg, the original forward
residual is comparable to 0.012--0.016 m/s of unexplained momentum change over
a control period. This is material relative to the tracking tolerance.

Ten versus twenty iterations differ in per-seed forward-speed MAE by at most
0.000062 m/s on the checked seeds. The ten-iteration momentum residual maximum
is below 0.00181 N s in the complete baseline matrix. This supports ten as a
converged ceiling for these scenarios; it does not establish convergence under
all future randomized contacts or higher speeds.

Do not infer that the left contact impulse directly caused the old speed trace
when the one-iteration momentum balance fails. Converged traces are the source
for the propulsion interpretation above.

## 4. Symmetry intervention probe, without training

Use a diagnostic projection only:

```text
projected_action(obs) = 0.5 * [policy(obs) + mirror(policy(mirror(obs)))]
```

This is exactly mirror-equivariant as a function of observations. It neither
changes checkpoint files nor guarantees symmetric closed-loop motion in an
asymmetric CAD model/reset. It is **not** proposed as a deployment adapter.

Converged-physics forward-speed MAE:

| Command | Retained | Retained projected | Final | Final projected |
|---|---:|---:|---:|---:|
| 0.05 m/s | 0.02280 | 0.02272 | 0.02276 | 0.01937 |
| 0.075 m/s | 0.02480 | 0.02345 | 0.02694 | 0.02014 |
| 0.10 m/s | 0.02436 | 0.02669 | 0.03310 | 0.02537 |

The final policy's actual joint half-cycle mismatch at 0.10 m/s falls from
4.91 to 0.84 degrees and relative-foot mismatch falls from 11.90 to 4.40 mm.
However, projection worsens retained-policy 0.10 m/s tracking. This supports
asymmetry as a contributor to the final regression, not a universal fix or
proof that mirror loss will train successfully. All projected episodes survive.

## ToddlerBot/WR1 comparison and references

- ToddlerBot `locomotion/mjx_config.py::SimConfig` selects Newton; its
  `descriptions/toddlerbot_2xc/toddlerbot_2xc_mjx.xml` sets one iteration and
  four line-search iterations. WR2 currently matches those numbers. Matching
  a fixed solver budget does not guarantee matching accuracy for different
  contacts, inertias and actuator armatures. No equivalent TB trajectory
  convergence matrix was run, so this report does not claim TB is unaffected.
- ToddlerBot `reference/walk_zmp_ref.py` uses phase/command lookup for the critic
  and `locomotion/mjx_env.py` uses physical target clipping. Those strategies
  remain unchanged. The WR2 half-cycle/reference verification tests its own
  five-DOF geometry rather than copying TB's six-DOF joint signs.
- ToddlerBot `locomotion/rsl_rl_config.yml` exposes actor mirror loss but leaves
  the transformation function unset; `train_mjx.py` removes the configuration
  unless `--symmetry` is explicitly selected. It is not the default walking
  objective or evidence of a ready-to-copy transform.
- WR1 `policy_contract/jax/symmetry.py` transforms complete history and uses
  polar/axial reflection; `training/algos/ppo/ppo_core.py` has actor mirror loss.
  Its robot-specific signs are not valid for WR2. This tool shares the
  mathematical method, not its morphology-specific mapping or reward settings.

Public references:
[ToddlerBot project](https://toddlerbot.github.io/),
[ToddlerBot paper](https://arxiv.org/abs/2502.00893), and
[Yu, Turk and Liu, Learning Symmetric and Low-energy Locomotion](https://arxiv.org/abs/1801.08093).
Yu et al. introduce a symmetry **loss**, not an instantaneous left/right force
reward; single support must remain valid. Their paper also uses assisted
curriculum, so its results do not establish that mirror loss alone solves WR2.

Training configuration, model files, rewards, optimizer and checkpoints are
unchanged. Regression tests cover model-derived signs, reference/geometry
parity, all-frame reflection, physically reflected actor observations,
target replay, momentum reference points, full-substep impulse integration,
and unsupported-contact denominators. The full suite passes **134 tests**;
focused checks, Ruff and Git whitespace checks also pass.
