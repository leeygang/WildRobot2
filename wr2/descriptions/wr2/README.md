# WR2 robot description

This directory owns the WR2 digital twin and robot-specific configuration.

## Onshape export

The complete robot is exported from one Onshape assembly using
`onshape_export/config.json`. The exporter reads `ONSHAPE_ACCESS_KEY` and
`ONSHAPE_SECRET_KEY` from the shell environment; credentials are never stored
in this repository.

Run from the repository root:

```bash
onshape-to-robot wr2/descriptions/wr2/onshape_export
```

The exporter output is a raw intermediate. It intentionally references the
`htd45hServo` class without embedding its definition, so use the post-process
step below instead of loading `onshape_export/wr2.xml` directly.

## Build the canonical model

Run from the repository root after every Onshape export:

```bash
uv run --no-project --with 'mujoco>=3.3,<4' \
    python -m wr2.tools.post_process
```

The command copies the raw STL assets into this directory, builds the
compile-ready `wr2.xml` and `scene.xml`, and runs a two-second standing/contact
validation. Add `--mass-report` to print every MuJoCo body mass.

The `home` pose uses zero radians for all articulated joints and places the
torso at Z=0.305 m. The `walk_home` training pose bends each hip/ankle by 0.12
rad and each knee by 0.24 rad while keeping the feet flat. Each foot uses one
box-shaped collision geometry and a named foot-center site, following
ToddlerBot's foot-contact convention.

Current generated artifacts:

- `wr2.xml`: canonical free-base, position-servo MuJoCo model.
- `scene.xml`: floor and visualization wrapper for `wr2.xml`.
- `wr2_mjx.xml`: sensor-free MJX training model.
- `scene_mjx.xml`: floor wrapper for `wr2_mjx.xml`.

The later actuator-modeling phase may derive a torque-actuated variant after
loaded HTD-45H system identification is complete.

## Training smoke test

Install the training extra and JIT-compile reset plus 20 policy steps:

```bash
uv run --extra training python -m wr2.locomotion.train --smoke
```

The initial environment uses `scene_mjx.xml`, the `walk_home` keyframe, the
17-channel WR1-compatible policy order, and leg-only actuation. See
`docs/design/training_interface.md` for the observation contract and training
gates.

The HTD-45H model uses a provisional WR2 unit-A static midpoint for `kp`, with
damping, friction loss, armature, and `kv` still transferred from WR1. The
highest short-duration WR2 fixture load observed so far is about 1.016 N m;
this is not a continuous-torque rating. Training uncertainty ranges and reward
weights live in `wr2/locomotion/configs/ppo_walking.yaml`, separate from the
robot description.

Generated XML files must be produced from the raw export plus the post-process
script and must not be edited independently.

## Mass verification

MuJoCo masses come from the Onshape material/density and mass properties stored
in the assembly. In Onshape, open **Mass properties**, select the complete robot
assembly, and compare its total with:

```bash
uv run --no-project --with 'mujoco>=3.3,<4' \
    python -m wr2.tools.post_process --mass-report
```

If the totals differ, inspect parts with missing materials, suppressed parts,
and electronics or fasteners represented only as appearance geometry.
