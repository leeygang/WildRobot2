# Six-DOF leg model import — 2026-10-11

The canonical models now come from the updated `onshape_export/wr2.xml`.
Each leg has hip pitch, hip roll, hip yaw, knee pitch, ankle pitch and ankle
roll. WR2 now has the same six independent leg coordinates as ToddlerBot 2xc;
TB uses drive/driven hip-yaw gears, while this WR2 export has direct hip-yaw
joints. Waist yaw is not a substitute for either leg's hip yaw.

## Model and policy contract

There are 21 actuators, 22 hinges (including passive waist yaw), one free root,
and one waist equality constraint: `nq=29`, `nv=28`, `nu=21`.
The canonical order is `wr2/descriptions/wr2/actuator_order.txt`:

| Actuator indices, zero-based | Coordinates | Walking control |
|---|---|---|
| 0 | waist yaw drive | held at home |
| 1–3 | left shoulder pitch/roll, elbow pitch | held at home |
| 4–9 | left hip pitch/roll/yaw, knee pitch, ankle pitch/roll | six residual actions |
| 10–12 | right shoulder pitch/roll, elbow pitch | held at home |
| 13–18 | right hip pitch/roll/yaw, knee pitch, ankle pitch/roll | six residual actions |
| 19–20 | neck yaw, head pitch | held at home |

The leg-only PPO policy has **12 actions**, still using TB's unbounded Normal
residual: `target = clip(walk_home + 0.25 rad * delayed_action, safe_limits)`.
The scale is approximately 14.3 degrees per residual unit; it is not a bound
on the residual. Hip-yaw weighted-pose coefficients are 5.0, matching TB's
active `walk.gin`. No neck/head actions are added to walking PPO.

`wr2_proprio_v5` contains 67 actor values per frame and 116 critic values per
frame. Fifteen newest-first frames give **1005 actor / 1740 critic inputs**.
Joint position/velocity include all 21 coordinates, while previous action
includes only the 12 active legs. IMU mounting, noise, action delay, servo
quantization and residual semantics remain unchanged. Reflection transforms
derive their signs and sizes from the model, including both hip yaws and the
self-reflected neck/head axes.

Old ten-action checkpoints are incompatible with this model. Heading migration
cannot add hip-yaw outputs or repair the changed dynamics. Start fresh.
New native checkpoints record the model XML hash and full/active actuator
orders; restore rejects a different contract even when tensor sizes match.
Legacy evaluation also rejects incompatible observation dimensions.

## Geometry and reference

Post-processing derives each primitive foot box from its CAD foot mesh and
places the foot-center site on the sole. It computes root height separately
for each keyframe so the lowest collision-box point starts 1 mm above ground:

| Quantity | Imported WR2 | TB 2xc |
|---|---:|---:|
| home root height | 0.289479 m | 0.310053 m |
| walk-home root height | 0.288261 m | home used for comparison |
| mean hip-pitch anchor to sole-site vertical distance | 0.238203 m | 0.243500 m |
| whole-model CoM height at that pose | 0.209278 m | 0.282244 m |
| exported total mass | 2.058328 kg | 3.453630 kg |

The leg-height ratio is 0.9782; the corresponding Froude speed/time scale is
0.9891. Keep TB's 0.72 s cycle, 40 mm swing and 20 ms control initially: the
roughly 1% scaling difference is smaller than the unqualified model/dynamics
uncertainty. Physics remains 5 ms with four substeps and the verified WR2
ten-iteration Newton solver.

IK now solves all three position and all three orientation errors. The
LIPM CoM height is measured from the actual model in its reference stance;
root/pelvis height is a separate parameter, as in TB's robot CoM/reference
geometry. The current auto-derived root reference is 0.283261 m, with actual
crouched CoM height 0.205281 m.

The reference uses an additional **5 mm** crouch below walk-home, not the old
10 mm. With full orientation IK, the old offset plus 40 mm swing required
more than the exported 90-degree knee limit: the clipped reference retained
1.36 degrees of orientation error. The 5 mm offset passes without changing
physical ranges or gate tolerances. TB's knees allow about 120 degrees;
WR2's left knee remains -90 to 0 degrees and right knee 0 to 90 degrees.

For the configured signed XY task, all 53 preflight conditions pass (49 lookup
points plus four exact ranked diagonals). Minimum ZMP support margin is
8.96 mm; lookup position error is at most 0.023 mm and full orientation error
0.00052 degrees. Separate forward 0.10/0.15/0.20 m/s checks pass, but the
0.25 m/s diagnostic fails the existing 3 mm stance-ground gate. That speed is
not in the current training distribution and is not qualified by this import.

WR2 still uses its periodic LIPM reference and numerical IK, not TB's complete
preview-controller implementation. The reference is critic context, not an
actor target or ZMP-pretrained policy. XY translation is enabled; pure turning
remains disabled until a turning/path-reference implementation is validated.

## Mass-property correction and qualification

The first export assigned 1e-9 kg to both neck/head bodies. The user's updated
export replaces those placeholders with **5.087 g neck-yaw / 13.467 g head**
and positive full inertias. Post-processing preserves these CAD values exactly;
total mass, CoM, settling and XY references have been rechecked on this update.
Compare total and component masses against hardware before deployment, including
electronics/servo overrides. Holding the head at home removes it from walking
actions, but its real mass still changes balance and dynamics.

The exported neck/head servo class is also a provisional default, not a
qualified HTD-45H or MG996R hardware specification. Identify those servos
before deployment. Installed BNO085 characterization and continuous servo
thermal qualification remain tracked in `docs/TODO.md`.

## Verification and next run

The canonical model compiles and passes two-second home/walk-home settling:
less than 1.8 mm horizontal drift, 0.6 degrees torso tilt and 0.29 mm contact
penetration. Six-DOF IK also independently changes foot yaw by 8 degrees while
preserving position. Symmetry tests cover limits, FK, inertia, references and
all actor history frames.

The updated CAD home soles have 0.2515 mm absolute mirror mismatch. Reference
foot mismatch stays within 0.2532 mm, while **generated displacement relative
to each actual home sole mirrors within 0.0086 mm** across the XY lookup.
The maximum joint-coordinate mismatch is 0.3101 degrees (0.0821 degrees on
the forward grid): unequal CAD geometry can require unequal IK coordinates.
Regressions therefore keep the existing 0.1 mm trajectory-displacement bound
and compare absolute mismatch against measured home geometry. They do not
symmetrize CAD or relax IK, joint-limit, ground or support gates.

A fresh Mac native RSL-RL smoke completes 80 transitions / two PPO updates
with 12 actions, finite policy/Adam state and the new checkpoint metadata.
It skips held-out rollout evaluation; it is not walking qualification or a
full GPU CLI test. No hardware motion or GPU training was started.
All 227 full-suite tests and ten focused model-import/symmetry tests pass;
Ruff and diff checks pass. Repeated post-processing produces identical model
hashes and does not modify the raw export.

Repeat model/reference validation after every CAD update:

```bash
uv run python -m wr2.tools.post_process
uv run python -m wr2.tools.view_legacy_zmp_reference \
  --config wr2/locomotion/configs/ppo_walking_xy.yaml \
  --xy-commands '0.1,0;-0.1,0;0,0.05;0,-0.05;0,0' --validate-only
```

Push the local preparation commit when ready, then on GPU:

```bash
git pull --ff-only
uv run --extra training python -m wr2.locomotion.train \
  --config wr2/locomotion/configs/ppo_walking_xy.yaml \
  --num-timesteps 200000000 --num-evals 11
```

Do not pass an old restore checkpoint. Review intermediate checkpoints and
independent standing/XY-transition evaluations before extending the run.

## Sources of truth

- Local TB `toddlerbot/descriptions/toddlerbot_2xc/toddlerbot_2xc.xml`:
  bilateral hip-yaw gears, six-joint legs, knee ranges and `home` geometry.
- Local TB `toddlerbot/locomotion/walk.gin`: leg-only actions, weighted pose,
  0.25-rad residual scale, 0.72-s cycle, 40-mm swing and signed XY commands.
- Local TB `toddlerbot/descriptions/toddlerbot_2xc/robot.yml`: separate `com_z`.
- Local TB `toddlerbot/reference/walk_zmp_ref.py`: reference/critic strategy.
- [ToddlerBot paper](https://arxiv.org/abs/2502.00893) and
  [project](https://toddlerbot.github.io/).
