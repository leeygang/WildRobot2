# WR2 HTD-45H servo specification and evidence

This document is the traceability index for every HTD-45H value used by WR2.
It distinguishes values used to make training robust from values that have
actually been qualified for deployment. A vendor endpoint, simulation limit,
or single-servo observation is not a deployment rating.

The canonical machine-readable actuator configuration is
[`robot.yml`](../../wr2/descriptions/wr2/robot.yml). The matching nominal
MuJoCo values are duplicated in [`wr2.xml`](../../wr2/descriptions/wr2/wr2.xml)
and checked against `robot.yml` when the robot description is loaded. Training
randomization is defined in
[`ppo_walking.yaml`](../../wr2/locomotion/configs/ppo_walking.yaml). The hardware
procedure and safety rules are in
[`htd45h_deployment_test.md`](../hardware/htd45h_deployment_test.md).
Accepted preliminary observations are preserved separately in the
[machine-readable unit-A evidence log](../hardware/evidence/htd45h_unit_a.json)
so raw run directories can remain untracked.

## Status terms

- **Vendor**: copied from the HTD-45H manual; not independently verified.
- **Transferred**: fitted on WR1 hardware and used provisionally by WR2.
- **Observed**: collected from WR2 hardware, but not sufficient for a rating.
- **Provisional**: deliberately conservative or randomized for training; not a
  measured specification.
- **Qualified**: measured with the required setup, repeated across at least
  three labeled servos, validated on held-out sessions, and accepted with a
  stated safety margin. No WR2 torque, torque-speed, braking, backlash, or
  continuous-duty value is qualified yet.

## Specification traceability

### Electrical, protocol, and safety values

| Specification | Used by | Definition | Current value | How to collect or verify | Required setup | Evidence status |
|---|---|---|---:|---|---|---|
| Rated voltage | Test setup; future voltage-dependent model | `rated_voltage_v` in `robot.yml` | 11.1 V | Set and independently log the supply during every campaign | Calibrated programmable supply or power analyzer | Vendor |
| Operating voltage range | Capture warnings, qualification, and deployment power design | `operating_voltage_range_v` in `robot.yml`; `--min-voltage-v` warning level and `--hard-min-voltage-v` abort floor in `capture.py` | Qualified range 9.6--12.6 V; hard abort below 5.0 V | `capture.py` emits a yellow warning and records each excursion below 9.6 V while continuing; an external logger must independently record bus voltage | BAM or dynamometer plus synchronized voltage logger | Vendor range; 5.0 V is only a software hard floor, not a qualified operating voltage |
| Servo-reported voltage | Capture health evidence | `voltage_v` in every capture NPZ/JSON | Accepted +10-degree repeats: minimum 11.406 V; excluded position repeat aborted at 8.669 V | Any `capture.py`-based test; sampled through `Htd45hBus.read_voltage_v()` | Servo and serial adapter; external logger recommended | Observed on unit A; undervoltage source unresolved |
| Servo temperature | Capture aborts; thermal evidence; deployment shutdown | `temperature_c` capture field; `software_temperature_shutdown_c` in `robot.yml` | Accepted +10-degree repeats: maximum 38 C; shutdown 80 C | Any `capture.py`-based test records temperature and final slope | Servo fixture with reachable power cutoff | Telemetry observed; 80 C is a software safety limit, not a rating |
| Supply current | Thermal, continuous-torque, torque-speed, and power qualification | Not yet represented by a live collector; summary schema is `external_measurements.example.json` | No clock-aligned external artifact in the latest copied BAM run | Add a timestamped instrument adapter; do not substitute `--external-log-label` for data | Logging supply or inline power analyzer | Missing from latest run |
| Stall current | Supply sizing only | `vendor_stall_current_a` in `robot.yml` | 3.0 A | Verify only on a current-limited guarded dynamometer; do not perform an uncontrolled stall | Programmable current-limited supply, torque sensor, guards | Vendor |
| Command range | Training/action conversion and EEPROM preflight | `command_range_units`, `command_range_deg` in `robot.yml`; constants in `core.py` | 0--1000 units, 0--240 degrees; fixture coordinates are -120--+120 degrees | Read EEPROM angle limits before every powered capture | Servo and serial adapter | Protocol/vendor; EEPROM checked at runtime |
| Command/position resolution | Training target quantization; backlash resolution floor | `command_resolution_rad` in `robot.yml`; conversions in `core.py` | 0.00418879 rad = 0.24 degree/unit | Compare commanded units with returned position units | Servo and serial adapter | Protocol-derived |
| Position accuracy | Deployment positioning context only | `vendor_position_accuracy_deg` in `robot.yml` | 0.2 degree vendor value | Verify at multiple loads, directions, angles, and temperatures against an external encoder | Instrumented BAM plus output encoder | Vendor; not verified |

The HTD protocol exposes position, voltage, temperature, torque-enable state,
and EEPROM limits. It does **not** expose motor current or shaft torque.

### Mechanical, control, and thermal values

| Specification | Used by | Definition | Current value or range | Collection process and script | Required setup | Evidence status |
|---|---|---|---:|---|---|---|
| Effective position gain | Training torque controller | `kp_sim` in `robot.yml`; `kp_scale` in `ppo_walking.yaml` | 16 N m/rad nominal; 8--32 N m/rad training range | Capture gravity-neutral and signed loaded dynamics with `capture.py`, then fit with `fit.py`; reserve separate validation captures | BAM for low-load fit; torque sensor preferred for loaded separation | Provisional midpoint from WR2 static behavior |
| Velocity gain | Training torque controller | `kv_sim`; `kv_scale` | 0.5; 0.8--1.2x | Same dynamic capture and fit process | BAM plus accurate output encoder | Held fixed in WR1 fit; provisional |
| Effective damping | Training joint dynamics | `damping`; `damping_scale` | 1.10618 N m s/rad; 0.8--1.2x | `capture.py` signed dynamic captures, then `fit.py` | BAM; torque sensor improves identifiability | Transferred WR1 fit |
| Coulomb friction loss | Training joint dynamics | `frictionloss`; `frictionloss_scale` | 0.324094 N m; 0.7--1.3x | Slow signed sweeps plus dynamic fitting | Instrumented BAM, preferably with external encoder | Transferred WR1 fit |
| Reflected armature | Training joint dynamics | `armature`; `armature_scale` | 0.024992 kg m^2; 0.8--1.2x | Gravity-neutral chirps with known fixture inertia and held-out replay | BAM with accurately modeled inertia | Held fixed in WR1 fit; provisional |
| Whole-response delay | Training action delay and runtime timing budget | `environment.action_delay_steps` in `ppo_walking.yaml`; `fitted_extra_command_delay_steps` in `robot.yml` | Training: 1 x 20 ms step; existing fit field: 0 extra steps | E3 capture with host command/write/read timestamps, then `fit.py` | BAM at gravity-neutral position | Training allowance; WR2 E3 pending |
| Zero offset and target bias | Training target-bias randomization; deployment calibration | `target_bias_rad` in `ppo_walking.yaml`; per-servo deployment calibration is not implemented | Training: +/-2 degrees; accepted +10-degree subset zero mean -0.1680 degree, std 0.1247 degree | Repeated bidirectional approaches with the `bam_position` or `bam_repeatability` plan | BAM; external encoder for absolute output zero | Unit-A observation; training range provisional |
| Position repeatability | Deployment command confidence; training uncertainty selection | Capture result only; not a nominal model field | Accepted +10-degree subset (`n=4`) loaded std 0.0327 degree and range 0.0800 degree | Five or more independent cooldown/approach repeats with the `bam_position` or `bam_repeatability` plan | Current BAM | Preliminary unit-A observation at one angle/direction; sub-resolution result |
| Loaded position error/compliance | Training gain/bias envelope; deployment tracking gates | Capture results; provisional `kp_sim` range | +10.08-degree quantized command settled at 7.8667 degrees mean; mean error 2.2133 degrees and maximum 2.2667 degrees | Repeat at signed angles and loads; retain full trajectories | Current BAM for preliminary evidence; torque sensor for separation | Observed on unit A; not a pure gain measurement |
| Backlash and hysteresis | Training `backlash_rad`; deployment positioning | `backlash_rad` in `ppo_walking.yaml` | 0--2 degrees training range | No dedicated script yet. Add slow unloaded and loaded forward/reverse approaches, fit direction-dependent lost motion, and validate on held-out sweeps | Instrumented BAM; external encoder required below the servo's 0.24-degree telemetry step | Missing; current range provisional |
| Peak/stall torque | Training force-cap envelope; deployment transient limit | `vendor_stall_torque_nm`, `torque_limit_nm` | Vendor stall 4.413 N m; training cap 4.0 N m randomized 1--4 N m | Sweep guarded steady/short-duration points with measured shaft torque and current | Guarded dynamometer with inline torque sensor and encoder | Vendor upper bound; not WR2-qualified |
| Maximum validated fixture load | Training torque-exposure normalization; test planning | `maximum_validated_load_nm` and duration in `robot.yml` | 1.0157 N m for 3 s | Existing signed BAM commissioning captures; repeat across units before adoption as a population limit | Current BAM | Observed short-duration unit-A value |
| Continuous torque | Deployment duty limit; training thermal/exposure constraints | `continuous_actual_load_*` evidence fields; no accepted continuous rating | 0.0833 N m observation did not reach cutoff; 0.317 N m reached 80 C after about 402 s; no continuous rating | Stage increasing static loads, log current/voltage/temperature, and hold until thermal equilibrium or abort. Existing E7 at 2.76 N m must not be used on unit A | Adjustable/instrumented BAM, synchronized current logger, controlled ambient/cooling | Bracket incomplete; unqualified |
| No-load speed | Training speed envelope and command slew | `vendor_no_load_speed_rad_s`, `training_target_speed_limit_rad_s` | Vendor 5.8178 rad/s; command quantization currently permits 27 units/20 ms = 5.655 rad/s | Measure both directions at several voltages and temperatures | Guarded dynamometer and independent encoder | Vendor endpoint; command limit derived |
| Torque-speed plateau/taper | Training asymmetric torque controller; deployment motion limits | `peak_torque_speed_rad_s`, `torque_at_max_speed_nm` | 0 rad/s and 0 N m placeholders define a linear vendor-endpoint taper | No valid collection script yet. Sweep measured steady torque/speed points in both directions, voltages, and temperatures, then fit with uncertainty | Guarded motor dynamometer, inline torque sensor, encoder, current logger | Missing; provisional envelope |
| Active braking torque | Training braking envelope; deployment deceleration limit | `brake_torque_limit_nm`; `brake_torque_scale` | 4.0 N m; 0.5--1.0x training | No valid collection script yet. Drive controlled deceleration and all four torque/velocity quadrants | Guarded bidirectional dynamometer | Missing; provisional/vendor-bounded |
| Passive/active braking ratio | Training direction-dependent torque controller | `passive_active_ratio`; randomization scale | 1.0; currently not randomized | No valid collection script yet. Compare externally back-driven and actively driven quadrants | Guarded bidirectional dynamometer | Missing; placeholder |

Simulation uses these parameters in `WalkingEnv._controller_torque()`. At every
2 ms physics substep it applies PD torque, the provisional acceleration/braking
envelope, and the randomized limits. Deployment hardware does not use this
software torque controller; these values make simulation resemble and bound the
closed-loop position servo.

## Collected WR2 evidence

The following evidence has been collected, but none of it alone qualifies a
population specification:

- Unit A short signed holds reached approximately 0.90 N m positive and
  1.0157 N m negative inferred fixture torque for three seconds.
- At an inferred actual load of about 0.317 N m, repeated 180 s tests ended at
  72 C with final slopes of 3.661--3.712 C/min. A longer test reached the
  inclusive 80 C cutoff after about 402 s. This is a thermal failure bound, not
  a continuous rating.
- The accepted position subset from `unit-a-bam-position-run04` contains the
  first four +10 degree repeats. At the quantized +10.08 degree command it
  observed 7.8667 degree mean loaded position, 0.0327 degree standard
  deviation, -0.1680 degree mean zero, 11.406 V minimum servo voltage, and 38 C
  maximum temperature. Measured-position fixture geometry implies 0.420 N m
  mean torque and an apparent static stiffness of 10.88 N m/rad.
- Repeat 5 is excluded from position statistics because it never reached the
  loaded hold. Its 8.669 V safety abort remains recorded as a power-integrity
  event. The run contains no external supply log, so the event cannot be
  classified as actual bus sag or telemetry error. The -10 degree and E3
  conditions remain uncollected.
- Earlier motion/hot-return tests reported intermittent internal voltage as low
  as 9.273 V while the external supply appeared stable. Without the raw,
  synchronized external log this does not identify whether the drop occurred
  at the servo, BusLinker, wiring, or telemetry path.

The BAM fixture model reports 2.722761 kg total moving mass and uses the
measured 2.650 kg weight at a 0.1204 m center radius. Torque calculated from
this geometry must be named `inferred_fixture_torque`; it is not measured shaft
torque.

## Collection matrix

| Test goal | Hardware collector | Orchestrator today | Analyzer today | Artifact | Readiness |
|---|---|---|---|---|---|
| Voltage, temperature, command, position, derived velocity, timing | `capture.py` | Any caller | Capture summaries | One NPZ trace plus JSON metadata | Available |
| Repeated zero/loaded position | `capture.py --prepare-only` | `campaign.py --plan bam_position` or `bam_repeatability` | `analysis/position.py` | Per-repeat NPZ/JSON plus summary and campaign manifest | Available in the current implementation; captures from older revisions may have empty NPZ traces and JSON-only preparation telemetry |
| Gravity-neutral position dynamics | `capture.py` | E3 in the `bam_position` plan | `fit.py` | NPZ/JSON, fit JSON | Available; unit-A E3 not collected |
| Loaded dynamics | `capture.py` | E4/E5 in the `legacy_deployment` plan | `fit.py` | NPZ/JSON, fit JSON | Script exists, but current E4/E5 loads are unsafe for unit A |
| Backlash/hysteresis | None dedicated | None | None | Required: signed sweep NPZ/JSON and fit report | Missing |
| Low-load continuous torque | `capture.py --constant-hold-s` | E7 in the `legacy_deployment` plan | Thermal gate in `analyze.py` | NPZ/JSON plus external current log | Partly implemented; E7 load is unsafe and instrument control is missing |
| Supply voltage/current | None | Label only | Aggregate JSON ingestion through the `report` command | Required: raw timestamped log plus derived summary | Missing collector |
| Measured shaft torque | None | None | Optional aggregate fields only | Required: raw calibrated torque trace | Missing collector and sensor |
| Torque-speed | None | None | None | Required: quadrant coverage and fitted envelope | Missing; dynamometer required |
| Braking/back-drive | None | None | None | Required: four-quadrant captures and fitted limits | Missing; dynamometer required |

## Script structure

### Implemented structure

```text
wr2/actuation/htd45h.py       one canonical HTD-45H hardware driver
              |
              v
servo_sysid/capture.py        sole hardware motion/telemetry/safety primitive
              ^
              |
servo_sysid/campaign.py       one preflight/run/resume orchestrator
          ^               |
          |               v
plan.py + plans/*.yaml     one campaign manifest and raw artifacts
          |
          +-- bam_position
          +-- bam_repeatability
          +-- legacy_deployment

analysis/position.py          repeatability and loaded-position summaries
fit.py                        effective position-loop dynamics
analyze.py                    legacy deployment gates/specification report
__main__.py                   run, capture, fit, report, and list-plans commands
```

`wr2/actuation/htd45h.py` owns the serial protocol and is not duplicated under
`servo_sysid`. The `ServoBus` protocol in `capture.py` is only a structural
typing and test seam; the hardware implementation is `Htd45hBus` from the
actuation package.

`capture.py` owns the test-specific hardware behavior: EEPROM
preflight, cooldown, torque enable/disable, motion, internal voltage and
temperature polling, qualified-range voltage warnings, hard-floor and other
safety aborts, controlled return, and NPZ/JSON output.
`campaign.py` is the only orchestrator and invokes `capture.py` for each
condition. `commission.py` and `bam_suite.py` remain only as deprecated argument
translators for old commands; they contain no hardware runner or manifest
implementation.

The plan files are versioned inputs. Each condition declares its profile,
repetitions, and hard safety ceilings. A run manifest stores the plan hash,
fixture hash, clean Git revision, configuration, and every condition outcome.

Use the single module entry point:

```bash
uv run python -m wr2.tools.servo_sysid list-plans

uv run python -m wr2.tools.servo_sysid run \
  --plan bam_position \
  --servo-id 100 \
  --servo-label htd45h-unit-a \
  --board-port /dev/serial/by-id/YOUR_ADAPTER \
  --run-dir results/servo_sysid/unit-a-bam-position-run01 \
  --measured-weight-kg 2.650 \
  --measured-com-radius-m 0.1204 \
  --run-all \
  --execute --confirm-fixture-safe
```

### Required future additions

Do not add another servo driver. External supply, torque sensor, encoder, and
load-motor adapters will belong under `servo_sysid/instruments/` because they
are test equipment, not robot actuators. Backlash, thermal, torque-speed, and
braking plan files must not be added until their required profiles,
instruments, synchronization, and offline analyzers exist. The campaign runner
must refuse a plan when any declared instrument is unavailable.

An analyzer must never automatically edit `robot.yml` or PPO ranges. It should
emit a proposed specification with raw evidence references, calibration
records, uncertainty, held-out error, and safety margins for review.

## Acceptance rule

A collected value may replace a provisional training or deployment value only
when its raw artifacts, fixture/instrument calibration, exact Git revision,
servo population, operating-condition coverage, held-out validation, and
conservative margin are recorded. Update `robot.yml`, the matching MJCF values,
and `ppo_walking.yaml` together, then rerun their contract tests and retrain or
revalidate any affected policy.
