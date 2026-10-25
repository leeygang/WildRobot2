# Bilateral coordination diagnostic — 2026-10-24

## Completed result

**The corrected diagnostic completed all 24 branches. Larger left steps are
physically executable, but neither tested intervention establishes smooth TB
walking.** The bilateral hip-centering/reference-wave treatment changes which
foot leads while introducing strong fore-aft rocking and reducing net progress.

Use `review_20261024/bilateral_coordination_diagnostic_v2` as the authoritative
result. The first `bilateral_coordination_diagnostic` is retained but superseded:
its redundant IK allowed up to 30 mm of unintended right-foot lateral motion.
The corrected version preserves lateral position, height and sole orientation.

At selected 120M, means over the three post-ramp windows (ticks 72–179):

| Nominal control | Net forward, TB m/s | World-X velocity std, TB m/s | Left same-foot advance, WR2 physical mm | Left landings ahead |
| --- | ---: | ---: | ---: | ---: |
| Original PPO | .05253 | .03265 | 26.20 | 0/10 |
| Left coordinated | .09371 | .09054 | 48.59 | 0/10 |
| Bilateral coordinated | .04976 | .21378 | 20.58 | 10/10 |
| Native TB Paper 200M | .09229 | .01410 | 44.41 at WR2 size | 10/10 |

The bilateral right foot leads on 8/10 landings. Its right same-foot advance is
15.86 physical mm, versus 25.90 in the original policy and 45.31 in the left
control. Thus passing feet and larger joint motion do not imply commanded
translation. All rows describe original or temporary diagnostic controls,
not a new learned PPO policy. TB values use its existing native GPU recording;
WR2 ran on the Mac CPU with its validated MuJoCo 3.14/Brax .14.2 environment.

### Physical execution and body motion

- At 120M, left target range grows **1.68 ->12.96 degrees** and actual range
  **.27 ->10.53 degrees** in the left control. The bilateral actual left range
  is about 11 degrees under the identical left command.
- At 120M, no leg motor torque clipping occurs in the original or left control's
  measurement windows. Bilateral clipping is only .62% at the right knee;
  the other eleven joints have zero clipping in those windows. The tested
  limits do not prevent larger steps in these nominal conditions.
- Whole-body CoM interval velocity variation grows **.0287 ->.0439 ->.1533
  TB m/s** across original/left/bilateral controls. The bilateral rocking is
  therefore not solely a torso-point velocity effect.
- Forward floor-force RMS grows **1.84 ->2.53 ->3.84 N**. The RMS CoM-to-wrench
  ZMP fore-aft displacement grows **17.1 ->22.5 ->43.3 physical mm**. These
  are observations of altered support/propulsion; they do not isolate a unique
  force mechanism or historical PPO update cause.

At 80M, left coordination produces passing left landings on 9/9 events and
larger advances, but again with higher velocity variation and lower reward.
That comparison further weakens a hard mechanical limit as the sole explanation.

There are **two falls in six bilateral cases** across the full 252-tick window:
80M/start180 falls during treatment; 120M/start180 falls after the original
action tape is restored. Original, duplicate-original and left controls all
survive. All three 120M bilateral measurement windows are complete; at 80M,
the incomplete failed window is kept separate rather than averaged with the
two complete windows. The post-ramp window is not a proof of limit-cycle
convergence or long-run stability.

### Reward ranking and next direction

The native WR2 reward rate falls **22.19 ->20.00 ->16.53 per TB-equivalent
second**. The left control's dominant rate changes are XY tracking **-.958**,
yaw **-.539**, torso orientation **-.336**, and angular-XY stability **-.270**;
air time changes only **-.067**. Its mean speed improves, but instantaneous
velocity and angular motion worsen. Increasing XY weight would penalize this
tested trajectory more strongly.

The bilateral losses are principally yaw **-1.884**, XY **-1.801**, and air
**-1.004**. Because it does not provide better command following or smoothness,
these losses do not show that the objective rejects a healthy passing gait.
No reward coefficient, actuator parameter, home origin or training continuation
is selected by this result. Retain the requested sustained-landing treatment.

**The next priority is executed joint timing and stance/weight transfer.** The
left and bilateral controls use identical left targets and similar actual left
motion, while forward response falls when the right pattern changes. This
demonstrates that left tracking amplitude alone does not explain progression.
The right treatment changes its mean and waveform together; distinguish those
effects and compare learned WR2/TB hip/knee/ankle phases and support forces
before choosing a training fix. This particular reference-based bilateral
pattern failing does not prove that bilateral coordination is intrinsically bad.

### Verification and artifacts

Eighteen regression tests and Ruff pass. The run records **2,292 IK solves**,
of which 34 use local reachability projection. Maximum height/lateral position
error is 8.22 micrometres, lateral error 7.37 micrometres, and orientation
error 9.48e-6 rad. Checked nonterminal native physics replay and complete
constraint-force reconstruction errors are zero; actuator mapping error is
below 2.4e-7 Nm.
Duplicate original qpos and reward arrays are exact. There are 5,883 valid
control samples, each with four substep records. No nonfinite event occurs.

The result folder contains `summary.json`, `analysis.json`, all 24 traces,
`full_ik_validation.json`, `output.log`, the reproducible `analyze.py`, and
`bilateral_results.png`. Source implementations are recorded as `dd53058`;
the earlier superseded run used `95c7dfa`. Training checkpoints/configurations
and TB's active development checkout were read only.

[Comparison figure](../../results/wr2_walking/wr2_paper_step_20261022_143550_seed3/review_20261024/bilateral_coordination_diagnostic_v2/bilateral_results.png)

## Question and controls

Test whether redistributing the learned right-leading posture and coordinating
both legs improves executed motion, alternating placement and native reward.
Use the completed qualified-air run at 80M and selected 120M, three identical
start phases (ticks 180/190/200), and .1 TB-equivalent m/s forward command
(.0824055 physical WR2 m/s).

Four controls per start/checkpoint:

1. Original deterministic PPO action tape.
2. Left coordinated: change the left hip pattern and compensate its other joints.
3. Bilateral coordinated: the exact same left targets, plus corresponding right
   hip/leg compensation.
4. Duplicate original tape to verify replay consistency.

For the bilateral hip patterns, preserve the combined mean physical forward
hip target over the applied window. Redistribute it equally between the legs,
then use WR2's centered reference timing with up to 7-degree wave peaks
(approximately TB's learned 14-degree hip swing). Targets remain inside the
existing fixed-home .25-rad actor bounds and mechanical limits.

Each changed leg's IK preserves the original bounded requested foot height,
lateral position and full sole orientation using all five remaining joints.
Forward position is free. Locally infeasible hip requests are projected toward
their original bounded pose; the intended and applied hip angles are recorded.
The desired combined hip mean is preserved before that reachability projection;
the resulting executed means must be measured rather than assumed equal.
These
are temporary diagnostic commands; saved PPO weights and training physics,
reference, reward weights and sustained-landing rule are unchanged.

## Window and measurement checks

Apply six cycles with twelve-tick ramps; record 252 ticks per branch. The
settled comparison is ticks 72–179: two cycles after the initial ramp, before
removal. Report this separately from total discounted transition/continuation
return. A transient cost does not establish a poor settled gait objective.

Record all four native physics substeps: target error, raw/applied torque,
speed-dependent limits, clipping, and friction/contact generalized torques.
Friction is projected from the solver's friction constraint rows. Floor force
and moments include each contact's free moment, with the force sign assigned
to the actual foot endpoint, about the consistently cached subtree CoM.

Validate replay against the native step, actuator torque mapping, complete
constraint-force projection, floor vertical forces, actor bounds, IK residuals,
identical left targets, reward sums and duplicate original controls. Keep
raw terminal physics samples rather than substituting reset poses.

Normalize time/speed by `s=.8240547565`, distance by `s**2=.67906624`.
TB comparison uses local `paper-walking-baseline` at `a4f693d` and its existing
native 200M forward recording; native model/runtime differences remain explicit.
[ToddlerBot Tables VI–VII](https://arxiv.org/html/2502.00893v2) supply the reference
and reward design; the counterfactual controls are an analysis instrument.

## Run

On the Mac with its already validated environment:

```bash
uv run --no-sync python -m wr2.tools.diagnose_leg_coordination \
  --bilateral \
  --run-dir results/wr2_walking/wr2_paper_step_20261022_143550_seed3 \
  --output results/wr2_walking/wr2_paper_step_20261022_143550_seed3/review_20261024/bilateral_coordination_diagnostic_v2
```

The Linux training environment can use the same module/arguments with
`uv run --locked --python 3.10 --extra training python`.

## Interpretation

- Poor target execution: investigate loaded actuator response and constraints.
- Good execution with poor placement/yaw: investigate contact and coordination.
- Better settled command following and placement with lower native reward:
  identify the actual competing terms before considering a reward change.
- Better motion and reward: a feasible attractive control exists in these
  conditions; check acquisition under training variation and native PPO updates
  before claiming an exploration failure.
