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
| Servo-reported voltage | Capture health evidence | `voltage_v` in every capture NPZ/JSON | Accepted +10 and -10 degree repeats: minima 11.406 and 11.999 V; E3 minimum 11.799 V; excluded +10 repeat reported 8.669 V | Any `capture.py`-based test; sampled through `Htd45hBus.read_voltage_v()` | Servo and serial adapter; external logger recommended | Observed on unit A; isolated undervoltage source unresolved |
| Servo temperature | Capture aborts; thermal evidence; deployment shutdown | `temperature_c` capture field; `software_temperature_shutdown_c` in `robot.yml` | Position repeats reached 38 C; E3 rose from 31 to 40 C; shutdown 80 C | Any `capture.py`-based test records temperature and final slope | Servo fixture with reachable power cutoff | Telemetry observed; short E3 rise is not a continuous rating; 80 C is a software safety limit, not a rating |
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
| Effective position gain | Training torque controller | `kp_sim` in `robot.yml`; `kp_scale` in `ppo_walking.yaml` | 24.1574 N m/rad unit-A nominal; approximate fit interval 23.1616--25.1960; conservative training range remains 8--32 | Gravity-neutral and signed loaded `capture.py` traces; fit runs07/10/11 and validate runs08/12/13 with `fit.py` | BAM for low-load fit; torque sensor preferred for loaded separation | Held-out validated for a single servo at low load; accepted as a provisional training nominal only |
| Velocity gain | Training torque controller | `kv_sim`; `kv_scale` | 0.5 N m s/rad | Held fixed while mapping the fitted total damping into active `kv_sim` plus passive joint damping | BAM plus accurate output encoder | Decomposition convention; active and passive damping were not separately identified |
| Effective damping | Training joint dynamics | `fitted_effective_velocity_damping`; `damping`; `kv_sim` | Fitted total 0.673622 N m s/rad; config split is `kv_sim=0.5` plus joint `damping=0.173622`; approximate total interval 0.465151--0.975527 | `capture.py` signed dynamics followed by `fit.py --controller-kv 0.5` | BAM; torque sensor improves identifiability | Held-out validated total for unit A; active/passive split remains conventional |
| Coulomb friction loss | Training joint dynamics | `frictionloss`; `frictionloss_scale` | 0.465471 N m; approximate fit interval 0.428993--0.505051 | Slow signed sweeps plus dynamic fitting | Instrumented BAM, preferably with external encoder | Held-out validated effective unit-A value; not a population range |
| Reflected armature | Training joint dynamics | `armature`; `armature_scale` | 0.020951 kg m^2; approximate fit interval 0.009188--0.047774 | Gravity-neutral chirps with known fixture inertia and held-out replay | BAM with accurately modeled inertia | Held-out replay passed, but the wide uncertainty makes armature weakly identified |
| Whole-response delay | Training action delay and runtime timing budget | `environment.action_delay_steps` in `ppo_walking.yaml`; `fitted_extra_command_delay_steps` in `robot.yml` | Training retains 1 x 20 ms action delay; run14 selected 0 additional fit steps while trace correlation peaked at roughly 100--160 ms | E3 plus signed loaded captures with host timing, then held-out `fit.py` replay | BAM at gravity-neutral and signed low load | No extra discrete delay adopted; response lag is embodied in fitted dynamics and is not a pure transport measurement |
| Zero offset and target bias | Training target-bias randomization; deployment calibration | `target_bias_rad` in `ppo_walking.yaml`; per-servo deployment calibration is not implemented | Training: +/-2 degrees; +10 and -10 repeat zero means were -0.1680 and -1.3248 degrees | Repeated bidirectional approaches with the `bam_position` or `bam_repeatability` plan | BAM; external encoder for absolute output zero | Unit-A direction/session-dependent observation; training range provisional |
| Position repeatability | Deployment command confidence; training uncertainty selection | Capture result only; not a nominal model field | +10 accepted subset (`n=4`): loaded std 0.0327 degree, range 0.0800 degree; -10 (`n=5`): std 0.0850 degree, range 0.2133 degree | Five or more independent cooldown/approach repeats with the `bam_position` or `bam_repeatability` plan | Current BAM | Preliminary unit-A signed observations; +10 has only four accepted repeats and results are near telemetry resolution |
| Loaded position error/compliance | Training gain/bias envelope; deployment tracking gates | Capture results; provisional `kp_sim` range | +10.08-degree command settled at 7.8667 degrees (mean error +2.2133); -10.08-degree command settled at -9.2853 degrees (mean error -0.7947). Requested incremental-travel errors were approximately +2.045 and -2.039 degrees | Repeat at signed angles and loads; retain full trajectories | Current BAM for preliminary evidence; torque sensor for separation | Observed on unit A; contains zero bias, friction, compliance, and fixture effects, not pure gain |
| Backlash and hysteresis | Training `backlash_rad`; deployment positioning | `backlash_rad` in `ppo_walking.yaml` | 0--2 degrees training range; H1 at +8 to +12 degrees measured 0.443 degree mean center loop width (0.026 degree cycle std), 0.612 degree all-command p95, and 0.702 degree maximum | Run `campaign.py --plan bam_hysteresis`; `analysis/hysteresis.py` matches forward/reverse output at identical command units and reports cycle repeatability | Current BAM for coarse loaded hysteresis; external encoder required below the servo's 0.24-degree telemetry step | Preliminary positive-load unit-A observation; H2, held-out repeats, and isolated mechanical backlash remain unqualified |
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
  classified as actual bus sag or telemetry error.
- The five -10 degree repeats from
  `unit-a-bam-position-remaining-run06-low-v` settled at -9.2853 degrees mean
  with 0.0850 degree standard deviation and 0.2133 degree range. Their zero
  mean was -1.3248 degrees, inferred fixture torque averaged -0.5322 N m, and
  all archived voltage samples remained at or above 11.999 V. Reanalysis at
  the qualified 9.6 V minimum passed; the run's nonstandard 5 V execution
  threshold remains recorded as a procedural limitation.
- `unit-a-bam-position-e3-run07` completed 800 samples at 50 Hz with 1.293
  degree tracking RMSE, 2.435 degree p95 error, 3.2 degree maximum error, no
  voltage warnings, and 0.918 ms maximum loop lateness. Servo temperature rose
  from 31 to 40 C. Approximate chirp gain fell from 0.78 near 0.69 Hz to 0.25
  near 3.22 Hz, and direct cross-correlation peaked near 100 ms. These are
  preliminary whole-response observations, not pure transport delay or a
  qualified bandwidth.
- A diagnostic combined fit selected 6 extra 20 ms delay steps but drove
  damping and friction to their upper bounds and armature to its lower bound.
  It is rejected: no training or deployment parameter changes follow from it.
- A second fit used E3 run07 and signed loaded runs10/11, reserving E3 run08
  and signed runs12/13 for held-out replay. Run14 kept all continuous
  parameters inside their bounds, selected no extra delay, achieved 0.8651
  degree mean fit RMSE and 0.8546 degree mean held-out RMSE, and reduced the
  same-trace held-out mean error by about 11 percent relative to the previous
  training nominal. Its fitted total damping is 0.673622 N m s/rad. Training
  holds `kv_sim` at 0.5 and therefore uses 0.173622 N m s/rad joint damping;
  assigning the full fitted total to `damping` would double-count velocity
  feedback. These values are accepted only as a provisional unit-A training
  nominal. Residual replay bias of roughly 0.69--0.85 degree, signed
  hysteresis, missing current/shaft-torque data, and single-servo coverage
  prevent deployment qualification.
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
| Gravity-neutral position dynamics | `capture.py` | E3 in the `bam_position` plan | `fit.py` | NPZ/JSON, fit JSON | Unit-A run07 fit capture and independent run08 validation capture accepted for the provisional training nominal |
| Loaded dynamics | `capture.py` | Direct bounded captures near +/-10 degrees; legacy E4/E5 remain unsafe | `fit.py` | NPZ/JSON, fit JSON | Runs10/11 fit and runs12/13 held-out validation complete at 0.647/0.681 N m predicted peaks |
| Backlash/hysteresis | `capture.py --profile hysteresis` | `campaign.py --plan bam_hysteresis` | `analysis/hysteresis.py`; top-level `hysteresis` command | Signed sweep NPZ/JSON with per-cycle summary; optional multi-capture report | Ready for coarse loaded unit-A capture; external encoder still required for precise mechanical backlash |
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
          +-- bam_hysteresis
          +-- bam_position
          +-- bam_repeatability
          +-- legacy_deployment

analysis/position.py          repeatability and loaded-position summaries
analysis/hysteresis.py        matched-direction loaded loop-width summaries
fit.py                        effective position-loop dynamics and explicit
                              total-damping-to-training mapping
analyze.py                    legacy deployment gates/specification report
__main__.py                   run, capture, fit, hysteresis, report, and
                              list-plans commands
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
are test equipment, not robot actuators. Precise backlash, thermal,
torque-speed, and braking plan files must not be added until their required
instruments, synchronization, and offline analyzers exist. The campaign runner
must refuse a plan when any declared instrument is unavailable. The existing
`bam_hysteresis` plan is explicitly limited to coarse loaded loop width from
the quantized internal encoder.

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
