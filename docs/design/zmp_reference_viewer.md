# Visualize WR2 and ToddlerBot Paper ZMP references

## Commands

From the WR2 repository, install the viewer dependencies once:

```bash
uv sync --extra training --extra dev --extra zmp-viewer
```

On macOS, complete the existing
[MuJoCo library setup](wr2_stance_diagnostic_20261018.md#one-time-macos-library-setup-for-uvs-standalone-python)
if needed. Open either robot separately:

```bash
# WR2: current Paper training reference, generated on the canonical WR2 model.
uv run --no-sync mjpython -m wr2.tools.view_paper_reference --robot wr2

# TB: original Paper lookup and native reference code in the prepared local checkout.
uv run --no-sync mjpython -m wr2.tools.view_paper_reference --robot tb
```

On Linux, replace `mjpython` with `python`. The TB default on this workspace is
`.worktrees/tb-paper-stance`, based on the local `paper-walking-baseline`
commit `a4f693d`. On another workspace, specify that checkout explicitly:

```bash
uv run --no-sync mjpython -m wr2.tools.view_paper_reference \
  --robot tb --tb-root /path/to/toddlerbot-paper
```

The checkout must include its original `toddlerbot/descriptions/toddlerbot/`
model/configuration files, `walk_zmp_lookup_table.pkl` and arm dataset. The
current TB development branch uses a different reference API/cache. The
[existing TB Paper setup guide](tb_stance_diagnostic_20261018.md) describes
the frozen checkout. The viewer prints the source path and selected lookup
command, and verifies the regenerated TB planner against its cached COM,
leg joints and stance before displaying overlays.

### Choose the motion

`--command VX VY YAW` uses TB-equivalent m/s, m/s and rad/s for either robot:

```bash
# Backward, with paused start.
uv run --no-sync mjpython -m wr2.tools.view_paper_reference \
  --robot wr2 --command -0.10 0 0 --paused

# Lateral.
uv run --no-sync mjpython -m wr2.tools.view_paper_reference \
  --robot tb --command 0 0.05 0

# Pure turn. The Paper lookup supports XY walking or pure yaw.
uv run --no-sync mjpython -m wr2.tools.view_paper_reference \
  --robot wr2 --command 0 0 0.5

# Standing.
uv run --no-sync mjpython -m wr2.tools.view_paper_reference \
  --robot tb --command 0 0 0
```

WR2 multiplies XY speeds by its training normalization `s=0.8240547565`
and divides yaw speed by `s`. Default forward speeds are therefore
0.0824055 m/s for WR2 and 0.10 m/s for TB. Each uses its actual training
gait clock: WR2 0.60 s, TB 0.72 s. The viewer shows physical time and
TB-equivalent time (`physical_time / s`). It preserves the native nearest
lookup selection, including ties at commands halfway between lookup keys.

The default playback speed is 0.25. `--playback-speed 1` runs in physical
real time; `--loops 1` closes after one 20-second reference episode. Without
`--loops`, the episode repeats. Use `--config /path/to/train_config.yaml`
to inspect a saved WR2 Paper configuration.

## Controls and overlays

| Control | Action |
| --- | --- |
| Space | Pause/resume |
| `[` / `]` | Pause and step one native 20 ms frame backward/forward |
| R | Restart episode |
| 1 / 2 / 3 / 4 | Front / side / top / perspective |
| Mouse | MuJoCo camera controls |

| Overlay | Meaning |
| --- | --- |
| Red sphere/vertical line | Actual whole-body COM from MuJoCo at the reference pose |
| Yellow sphere | Planned COM; TB uses its native 0.3267 m torso-height proxy |
| Green sphere | Desired ZMP from the footstep plan |
| Magenta sphere | Optimal COP implied by the native LIPM preview controller |
| Blue / gray foot outlines | Projected collision geometry of planned stance / swing feet |
| Green outline | Convex hull of the planned stance feet's projected collision geometry |

Start with the front view to inspect sway and alternating support; use the
side view for swing clearance and the top view for ZMP/support placement.
The robot is transparent by default so COM and ZMP markers remain visible;
MuJoCo's Rendering menu can turn transparency off.
The green outline uses planned stance and projected colliders. It is not a
measurement of loaded contact patches. TB's floor-paired ankle boxes are
both included; WR2 uses its single box per foot.

## Reference fidelity and interpretation

This is kinematic replay: the viewer assigns reference poses and calls
`mj_forward`. It shows the planned walk and its geometry. It does not test
whether the servos or a learned policy can execute it dynamically.

WR2 uses `PaperWalkReference`, the same generator as the Paper training
environment. Only the selected native training lookup row is generated.
Its world root pose, standing frame, 1000 retained frames, extra preview
horizon and nearest-command operation come from training code. The shared
command grid preserves the original constructor's keys byte for byte.

TB uses the frozen `WalkZMPReference.get_state_ref` for cached joint/motor
targets, transmissions, passive joints and static initialization, and
`MotionReference.get_qpos_ref` for the grounded pose. To display the finite
planned walk in world coordinates, it uses the native planner's torso yaw
and replaces the method's fixed XY support anchor with the corresponding
planned support-foot position. Joint angles and grounding remain native.
TB's cached first planning cycle is omitted exactly as in training. Its
960-frame table includes the stopping tail and wraps at **19.2 seconds**;
the viewer preserves that wrap within the 20-second replay. WR2's table
has 1000 retained frames and avoids that finite stopping tail.

TB's planner applies a soft ZMP tracking cost, so optimal COP excursions
outside support are possible. The separate green and magenta markers make
this visible. TB's planned height is a torso proxy; WR2's planner uses its
model-derived COM height. The existing
[WR2/TB Paper setup audit](wr2_tb_paper_setup_audit_20261018.md#reference-construction-and-indexing)
records the geometry, height and timing differences.

For a check without opening a window:

```bash
uv run --no-sync python -m wr2.tools.view_paper_reference --robot wr2 --check-only
uv run --no-sync python -m wr2.tools.view_paper_reference --robot tb --check-only
uv run --no-sync pytest -q tests/test_paper_reference_viewer.py
```

References: [ToddlerBot paper](https://arxiv.org/abs/2502.00893);
[ZMP planning with terminal conditions](https://doi.org/10.1109/HUMANOIDS.2015.7363473).
Code checked: local TB Paper `algorithms/zmp_walk.py`, `zmp_planner.py`,
`reference/walk_zmp_ref.py`, `motion_ref.py`, `sim/robot.py` and original cache;
WR2 [`paper_walk.py`](../../wr2/reference/paper_walk.py) and
[`walking_env.py`](../../wr2/locomotion/walking_env.py).
