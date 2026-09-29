# WR2 actuator model status

WR2 currently uses the HTD-45H position-servo model transferred from WR1:

| Parameter | Nominal value | Evidence status |
|---|---:|---|
| position gain `kp` | 31.902 | fitted from WR1 A1/B1/B2 captures |
| joint damping | 1.10618 N m s/rad | fitted from WR1 A1/B1/B2 captures |
| friction loss | 0.324094 N m | fitted from WR1 A1/B1/B2 captures |
| actuator `kv` | 0.5 | held fixed during fitting |
| armature | 0.024992 kg m^2 | held fixed during fitting |
| force cap | +/-4.0 N m | conservative, vendor-stall-based cap |

The fit replayed quantized 50 Hz commands and reduced mean capture position
RMSE from 0.8051 degrees to 0.4386 degrees. A held-out roughly 0.56 N m capture
had 0.480-degree replay RMSE. This supports use as a nominal software and
initial-training model, but it does not identify continuous torque, the
torque-speed curve, voltage or temperature effects, actuator-to-actuator
variation, or behavior at WR2 leg loads.

The first training environment therefore randomizes friction, damping,
armature, `kp`, and `kv` around the nominal model. It also lowers the force cap
by a uniformly sampled factor from 0.6 to 1.0; it never raises the cap above
the nominal 4 N m. These distributions are provisional uncertainty bounds,
not measurements. Do not use policy success under this cap as evidence that
the physical servo can sustain that torque.

The WR1 fit selected zero additional command-delay samples, but this does not
prove zero end-to-end runtime latency. WR2 currently trains with one 20 ms
control-step delay as a provisional runtime allowance. Measure bus, scheduler,
and feedback timing on WR2 before freezing that setting.

Before deploying a walking policy, repeat the WR1 system-identification
campaign with a representative WR2 leg fixture. Characterize multiple servos
over supply voltage, temperature, speed, both load directions, and the
expected knee/hip torque range. Measure transport latency separately from
closed-loop response lag, then replace the provisional randomization ranges
with fitted distributions. Follow the staged procedure in
[`../hardware/htd45h_deployment_test.md`](../hardware/htd45h_deployment_test.md).

The MG996R considered for head motion is not part of the current 17-actuator
walking model. If it is added later, give it a separate actuator class and
characterization; it should not inherit the HTD-45H constants.
