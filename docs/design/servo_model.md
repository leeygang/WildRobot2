# WR2 actuator model status

WR2 uses a provisional HTD-45H model intended to start policy training while
hardware qualification continues in parallel:

| Parameter | Nominal value | Initial training range | Evidence status |
|---|---:|---:|---|
| position gain `kp` | 16 N m/rad | 8--32 N m/rad | WR2 unit-A static envelope |
| joint damping | 1.10618 N m s/rad | 0.8--1.2x | transferred WR1 fit |
| friction loss | 0.324094 N m | 0.7--1.3x | transferred WR1 fit |
| actuator `kv` | 0.5 | 0.8--1.2x | held fixed in WR1 fitting |
| armature | 0.024992 kg m^2 | 0.8--1.2x | held fixed in WR1 fitting |
| force cap | +/-4.0 N m | 1--4 N m | vendor-based peak bound |
| target bias | 0 rad | +/-2 degrees | WR2 offset/hysteresis allowance |

The HTD-45H manual specifies 11.1 V nominal operation, a 9.6--12.6 V
operating range, 45 kg cm (4.413 N m) stall torque, 3 A stall current,
0.18 s/60 degrees (5.818 rad/s) no-load speed, and 0.2-degree accuracy. Stall
torque is an instantaneous upper bound, not a continuous-duty rating.

The original `kp=31.902`, damping, and friction values came from WR1 dynamic
captures. WR2 unit-A static tests showed substantially softer positive-load
behavior: commands of +8, +10, and +20 degrees settled near +6.0, +7.84, and
+16.63 degrees. Signed tests also showed offset and direction dependence. A
single linear gain cannot describe all of that behavior, so 16 N m/rad is a
provisional training midpoint and the 8--32 N m/rad range deliberately spans
the observed uncertainty. A WR2 dynamic fit is still required before hardware
deployment.

Five-repeat tests reached approximately 0.90 N m positive and 1.02 N m
negative actual fixture torque for three-second holds. These are short-duration
results. In thermal tests at 12.51 V, the torque-enabled near-zero-load case
stabilized around 74 C over ten minutes, while approximately 0.317 N m actual
load reached the inclusive 80 C software cutoff after about 402 seconds. The
continuous boundary remains under investigation and must not be inferred from
the 4 N m peak cap.

The +8-degree, 180-second condition was repeated on 2026-09-30 from a 34 C
start. It reproduced the original run: both completed at 72 C, their final
60-second slopes were 3.712 and 3.661 C/min, and their final-minute positions
averaged 6.006 and 6.008 degrees. The corresponding measured-position fixture
load was about 0.317 N m. Relative to the quantized 7.92-degree command, this
is an apparent load/error stiffness of 9.50 N m/rad and supports the lower end
of the current 8--32 N m/rad training range. It is not a direct `kp` fit because
static friction also carries load. The repeat establishes three-minute
behavior, but the positive slope and the C7 cutoff show that it is not a
continuous rating.

The training command path quantizes targets to the servo's 0.24-degree units
and limits each 20 ms update to 27 units, or 5.655 rad/s. Each episode also
samples an independent target bias for every actuator. One 20 ms command-delay
step remains as a provisional runtime allowance.

Training penalizes squared actuator torque in addition to mechanical power.
This matters because a stationary servo can consume current and heat while
mechanical power is nearly zero. Mean per-step RMS torque, mean per-step peak
torque, and a ten-second exponential torque-exposure proxy are recorded as
training metrics. The exposure is normalized to the 1.016 N m short-duration
WR2 load envelope, so values above one indicate sustained operation beyond that
measured reference. The proxy is not a calibrated temperature model;
deployment must monitor servo temperature directly and shut down at 80 C.
The ten-second exposure window is intentionally a within-episode policy-demand
metric, not a thermal time constant: the present 20-second training episodes
cannot reproduce the several-minute temperature evolution measured by C6--C8.

Intermittent servo-reported voltage readings down to 9.273 V occurred during
motion and hot return transients, despite stable external-supply readings.
The 1--4 N m force-cap randomization and broad gain range provide training
robustness, but do not qualify the BusLinker power path or prove that the
hardware can produce those torques continuously.

The first policy is therefore a training-pipeline baseline. Use its simulated
per-joint torque, speed, and duty-cycle distributions to select the next
hardware conditions, then refit and retrain. Population testing, voltage
sweeps, external current logging, and a low-load WR2 dynamic chirp remain
required before deployment.

The MG996R considered for head motion is not part of the current 17-actuator
walking model. If it is added later, give it a separate actuator class and
characterization; it should not inherit the HTD-45H constants.
