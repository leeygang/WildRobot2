# HTD-45H unit C: run01/run02 review

The two campaigns stopped at the inclusive **55 C cutoff in Q6**. Q1--Q5
completed in both sessions; Q7 was not executed. The valid repeatability and
dynamic captures support a held-out low-load effective model. They do not
complete the seven-stage qualification or establish deployment limits.

Exact values, all 64 raw/derived artifact hashes, calibration limitations,
capture timing checks, and parameter mappings are retained in
[htd45h_unit_c.json](htd45h_unit_c.json). Raw manifests and captures were not
edited. Collection and analysis used Git revision
`0b3ca04cf4f2e36586fc589b651e40eb630c2f3c`; the fit JSON itself does not embed
that revision, so the evidence index records it separately.

## Session and measurement audit

| Metric | Fit run01 | Held-out run02 |
|---|---:|---:|
| Campaign status | failed at Q6 | failed at Q6 |
| Completed captures / available captures | 13 / 14 | 13 / 14 |
| Total archived samples, including aborted Q6 | 4,659 | 4,681 |
| Completed-capture samples | 2,949 | 2,944 |
| CLI cooldown target | 32 C | 32 C |
| +10-degree loaded position mean | 7.9253 degrees | 8.0427 degrees |
| +10-degree repeat std / range | 0.0569 / 0.1600 degrees | 0.0727 / 0.2133 degrees |
| -10-degree loaded position mean | -9.0507 degrees | -9.0187 degrees |
| -10-degree repeat std / range | 0.0272 / 0.0800 degrees | 0.0200 / 0.0533 degrees |
| Q6 positive center absolute loop mean | 0.4985 degrees | 0.4246 degrees |
| Q6 cycle std / range | 0.0452 / 0.1108 degrees | 0.1924 / 0.4615 degrees |
| Q6 last archived temperature / abort reading | 54 / 55 C | 54 / 55 C |
| Q6 last archived profile elapsed time | 34.180 s | 34.719 s |

Repeatability uses all five independent approaches per sign, recomputed from
the JSON preparation traces and checked against the saved summaries. The
requested centers are +/-10 degrees; transmitted centers are +/-10.08 degrees.
The internal position step is 0.24 degree. Smaller mean/std/range statistics
come from averaging quantized readings, not demonstrated sub-resolution output
accuracy. Positive session means differ by +0.1173 degree; negative means by
+0.0320 degree. First-approach zero history remains included: zero observations
span -1.104 to -0.144 degrees in positive run01, rather than supporting one
absolute encoder-zero correction.

All primary command/position/velocity/voltage/temperature values are finite,
array lengths agree, and both monotonic and host wall-clock timestamps increase.
Q3--Q6 maximum sample gaps are 21.704 ms or less for nominal 20 ms sampling,
with no gap above 30 ms. Preparation traces are sampled at roughly 0.11 s;
they must not be represented as 50 Hz profiles. Preparation scheduled-time and
lateness fields are intentionally unavailable, and command-write duration is
NaN when no command was written. A long command age during a held target is
not automatically a dropped command. Internal health values are cached without
per-field acquisition timestamps, so repeated 50 Hz temperature/voltage rows
are not independent instrument samples.

Run02 Q1 repeat05 recorded one 9.508 V warning, recovering to 12.298 V after
about 113 ms. The event remains accepted with a warning, not erased or silently
excluded. No external supply log identifies actual wiring sag versus telemetry
error. There is no archived external current, shaft encoder/torque, or ambient
temperature measurement. The operator supplied 2.650 kg and 0.1204 m; their
calibration uncertainty is not recorded. Fixture mass is 2.722761 kg, and its
hash matches both sessions. EEPROM snapshots report position mode, 0--1000
angle units, an 85 C ceiling, and 4.5--14.0 V limits; software limits remain
stricter.

## Effective dynamics and held-out replay

Only run01 Q3/Q4/Q5 were fitted. All matching run02 captures were held out;
validation did not select parameters or delay. The unchanged fitter uses
equal-weight-per-capture position residuals, a 2 ms fixture timestep, a fixed
4 N m force limit, and extra delays 0--10 capture intervals. The force limit is
not identified by these low-load data.

| Condition | Fit replay RMSE | Held-out replay RMSE | Held-out residual bias |
|---|---:|---:|---:|
| Q3 gravity-neutral | 0.8522 degrees | 0.7309 degrees | +0.6173 degrees |
| Q4 positive load | 0.7516 degrees | 0.8445 degrees | +0.7243 degrees |
| Q5 negative load | 0.6892 degrees | 0.6469 degrees | +0.5428 degrees |
| Mean across captures | 0.7643 degrees | 0.7408 degrees | -- |

Replay residual is predicted minus measured position. Its positive bias is
material; a small mean RMSE does not imply unbiased calibration. The selected
delay is zero additional 20 ms intervals, not zero physical or transport lag.
The zero-delay fit cost is 2.75% lower than the one-delay candidate, whose
armature reaches its lower bound.

| Parameter | Unit C estimate | Approximate conditional 95% interval |
|---|---:|---:|
| Effective position gain | 24.5013 N m/rad | 23.5952--25.4423 N m/rad |
| Effective total velocity damping | 0.852879 N m s/rad | 0.627928--1.158418 N m s/rad |
| Effective Coulomb friction loss | 0.449753 N m | 0.415096--0.487303 N m |
| Reflected armature | 0.036658 kg m^2 | 0.022007--0.061062 kg m^2 |

These local Jacobian intervals assume independent residuals and condition on
the chosen symmetric model/delay. They do not cover correlated quantization,
thermal drift, fixture calibration uncertainty, or population variation.
Armature/damping separation remains weak. With the existing `kv_sim=0.5`
decomposition convention, mapped joint damping would be 0.352879 N m s/rad;
assigning the entire total to joint damping would double-count velocity
feedback. No nominal, randomization, or walking-training configuration changed.

| Low-load model comparison | Unit A | Unit B | Unit C |
|---|---:|---:|---:|
| Position gain, N m/rad | 24.1574 | 24.4438 | 24.5013 |
| Total damping, N m s/rad | 0.673622 | 0.836343 | 0.852879 |
| Friction loss, N m | 0.465471 | 0.449174 | 0.449753 |
| Armature, kg m^2 | 0.020951 | 0.042909 | 0.036658 |
| Held-out mean replay RMSE, degrees | 0.8546 | 0.7558 | 0.7408 |

Three low-load unit fits now exist. Their extrema are observations, not a
qualified population distribution: A/B and C used different starting-temperature
policies, unit C lacks Q7, and electrical/torque instruments are absent.
Replaying the same C held-out traces with the existing A/B/C models gives
0.74143/0.74102/0.74075 degree mean RMSE. The unit-C fit improves over the
current A nominal by only about 0.00068 degree. Different fitted damping and
armature vectors therefore do not establish independently measured physical
variation or justify changing the nominal model or randomization ranges.

The procedure follows ToddlerBot's commanded-trajectory MuJoCo replay and joint
dynamics identification ([local implementation](../../../../toddlerbot/toddlerbot/tools/run_sysID.py)
and [published method](https://arxiv.org/html/2502.00893v4)). WR2 additionally fits
unknown HTD effective position gain; it uses the measured BAM inertia/load and
bounded +/-2-degree, 0.1--4 Hz / 0.1--2 Hz excitation, not ToddlerBot's larger
amplitudes or 10 Hz endpoint. These bounds come from fixture preflight, not a
copied robot-size scaling. WR1's `tools/sysid/fit_servo_dynamics.py` holds
armature fixed because of its weak separation; WR2 retains its current fitter
and reports that uncertainty rather than claiming hardware armature certainty.

## Hysteresis, thermal abort, and parameter mapping

Both Q6 captures contain three complete 8-to-12-degree and 12-to-8-degree sweep
pairs. The abort happened later in the `final_center` return sweep from +8 to
+10 degrees: only 3/30 of its planned 101 samples were saved in run01/run02.
The existing analyzer preserves sweep
metrics but correctly returns overall `status=incomplete` (exit code 2).
Signed loop width is increasing-command branch minus decreasing-command branch
at matching command units; all center widths are negative at unit 542
(+10.08 degrees). Run02's second-cycle width is 0.1662 degree, below telemetry
resolution, so its apparent small deadband is not resolved.

Temperatures rose from 34 to 54 C while the captured profile's inferred gravity
load averaged 0.4096/0.4107 N m. This is not measured motor torque, a controlled
constant load, or continuous-duty equilibrium. Forward/reverse branches also
occur at different temperatures. The 55 C reading is in the exception, not
the array, because the safety check raises before appending it. Teardown code
disables and reads back torque-off; the NPZ contains no post-shutdown torque
state or evidence that an independent power cutoff was operated.

Loaded loop width, signed zero history, friction, gravity, compliance, and
controller response are not separately identified. Physical friction/controller
dynamics must not be duplicated as encoder noise or target bias. Mechanical
deadband and absolute encoder calibration still require external shaft sensing
and unloaded/counterbalanced approaches; see the
[backlash identification survey](https://doi.org/10.1016/S0005-1098(02)00047-X).
WR2's observation backlash still divides torque by its 1.0157 N m exposure
reference, while ToddlerBot uses an independent 0.1 N m activation parameter.
These data identify neither activation value, `brake_torque_limit_nm`, nor
`passive_active_ratio`. No such mapping or thermal time-constant fit was made.

## Reproduction and next operator action

Preserve the existing raw directories. The derived reports are separate:

```bash
uv run --extra sysid python -m wr2.tools.servo_sysid fit \
  results/servo_sysid/unit-c-bam-low-load-fit-run01/03_Q3_gravity_neutral_dynamics.npz \
  results/servo_sysid/unit-c-bam-low-load-fit-run01/04_Q4_loaded_dynamics_plus10.npz \
  results/servo_sysid/unit-c-bam-low-load-fit-run01/05_Q5_loaded_dynamics_minus10.npz \
  --validation-capture results/servo_sysid/unit-c-bam-low-load-validation-run02/03_Q3_gravity_neutral_dynamics.npz \
  --validation-capture results/servo_sysid/unit-c-bam-low-load-validation-run02/04_Q4_loaded_dynamics_plus10.npz \
  --validation-capture results/servo_sysid/unit-c-bam-low-load-validation-run02/05_Q5_loaded_dynamics_minus10.npz \
  --sim-dt 0.002 --force-limit-nm 4 --controller-kv 0.5 \
  --delay-steps 0,1,2,3,4,5,6,7,8,9,10 --max-nfev 100 \
  --output results/servo_sysid/htd45h-unit-c-dynamics-fit-recheck.json

uv run --extra sysid python -m wr2.tools.servo_sysid hysteresis \
  results/servo_sysid/unit-c-bam-low-load-fit-run01/06_Q6_loaded_hysteresis_plus10.npz \
  results/servo_sysid/unit-c-bam-low-load-validation-run02/06_Q6_loaded_hysteresis_plus10.npz \
  --output results/servo_sysid/htd45h-unit-c-hysteresis-recheck.json
```

Use new output names; the hysteresis command intentionally exits 2 for these
failed sessions. Do not restart the full matrix merely to obtain Q7 or raise
the 55 C abort to make Q6 pass. Next, inspect fixture alignment, counter-bearing,
clearance/catcher, power cutoff, and the unresolved voltage event. If the
operator confirms the fixture is safe, collect Q7 separately under the existing
bounded plan and repeat it in an independent session. Preflight first:

```bash
uv run --extra sysid python -m wr2.tools.servo_sysid run \
  --plan bam_low_load_qualification \
  --servo-id 100 --servo-label htd45h-unit-c \
  --board-port /dev/serial/by-id/usb-1a86_USB_Single_Serial_5C4C127022-if00 \
  --run-dir results/servo_sysid/unit-c-bam-hysteresis-minus10-run04 \
  --measured-weight-kg 2.650 --measured-com-radius-m 0.1204 \
  --cooldown-target-c 32 --max-temperature-c 55 \
  --start-at Q7_loaded_hysteresis_minus10 \
  --stop-after Q7_loaded_hysteresis_minus10
```

Only after explicit operator fixture confirmation, rerun with
`--execute --confirm-fixture-safe`; use a new `...minus10-validation-run05`
directory for the held-out repetition. A completed full-positive condition
needs a separately reviewed shorter thermal-budgeted procedure; none was added
or powered during this review. T1/T2 and continuous-duty work remain blocked
on the instrumented/guarded fixtures documented in the deployment test plan.
