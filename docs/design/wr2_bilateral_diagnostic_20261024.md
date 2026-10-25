# Bilateral coordination diagnostic — 2026-10-24

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
