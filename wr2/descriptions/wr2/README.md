# WR2 robot description

This directory owns the WR2 digital twin and robot-specific configuration.

## Onshape export

The complete robot is exported from one Onshape assembly using
`onshape_export/config.json`. The exporter reads `ONSHAPE_ACCESS_KEY` and
`ONSHAPE_SECRET_KEY` from the shell environment; credentials are never stored
in this repository.

Run from the repository root:

```bash
onshape-to-robot wildrobot2/descriptions/wr2/onshape_export
```

The initial export intentionally keeps all parts. Add WR2-specific visual and
collision ignore rules only after inspecting the exported part names.

Expected generated model variants:

- `wr2.xml`: free-base torque-actuated MuJoCo model.
- `wr2_pos.xml`: free-base position-actuated debug model.
- `wr2_fixed.xml`: fixed-base torque-actuated model.
- `wr2_mjx.xml`: MJX-compatible training model.
- `scene*.xml`: world and visualization wrappers for each model variant.

The generated XML files must be produced from one canonical model source and
must not be edited independently.
