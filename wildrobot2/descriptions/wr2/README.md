# WR2 robot description

This directory owns the WR2 digital twin and robot-specific configuration.

Expected generated model variants:

- `wr2.xml`: free-base torque-actuated MuJoCo model.
- `wr2_pos.xml`: free-base position-actuated debug model.
- `wr2_fixed.xml`: fixed-base torque-actuated model.
- `wr2_mjx.xml`: MJX-compatible training model.
- `scene*.xml`: world and visualization wrappers for each model variant.

The generated XML files must be produced from one canonical model source and
must not be edited independently.
