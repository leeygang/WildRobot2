# ZMP reference validation

WR2 validates its analytic walking reference before using it as privileged PPO
critic context. This follows ToddlerBot's practice of replaying its
`WalkZMPReference` kinematically in MuJoCo and checking foot geometry. WR2 adds
an automated support-polygon and LIPM gate so the primary checks do not depend
on visual judgment.

## Headless gate

Run the full command set used by the walking curriculum:

```bash
uv run --extra training python -m wr2.tools.view_zmp_reference \
  --validate-only --commands 0.10,0.15,0.20,0.25 --samples 72
```

For each speed, the command checks one complete gait cycle:

- every nominal stance sole is within 3 mm of the floor;
- a nominal swing sole does not penetrate more than 2 mm;
- the planned ZMP stays inside the active stance-foot support polygon, allowing
  0.5 mm numerical tolerance;
- `zmp = com - h/g * com_acceleration` agrees to 1 micrometre; and
- the largest adjacent joint-reference step is reported for continuity review.

The target commands currently pass with at least 8.5 mm planned-ZMP support
margin. The 0.294 m reference root height is intentional: the taller 0.301 m
posture made the long 0.20 and 0.25 m/s strides lift the nominal stance sole.
The training torso-height reward is disabled, so this does not create a hidden
height objective.

The regression test is:

```bash
uv run --extra training pytest -q tests/test_zmp_reference.py
```

PPO training and `--smoke` invoke this gate automatically for every configured
lookup command plus the exact deterministic-evaluation command. The manual
command remains useful for inspecting additional speeds or obtaining a concise
standalone report.

## Visual audit

Replay one speed in MuJoCo Viewer:

```bash
uv run --extra training python -m wr2.tools.view_zmp_reference \
  --commands 0.20
```

The overlay uses:

- red sphere and line: actual full-model MuJoCo CoM;
- yellow sphere: planned reduced-order LIPM CoM/root-height proxy;
- green sphere: planned ZMP;
- green outline: active support polygon;
- blue foot outline: stance; and
- gray foot outline: swing.

Press Space to pause, `[` or `]` to step backward or forward, and R to reset.
Use `--cycles 2` to stop after two cycles or `--playback-speed 0.25` for slow
motion. The viewer refuses a failed numeric reference unless
`--show-failed` is supplied.

Inspect both frontal and side views. The planned ZMP should transfer between
feet during double support, remain inside the blue stance polygon during
single support, and alternate left and right swings. The red actual CoM need
not coincide exactly with the yellow reduced-order LIPM point because the red
marker includes the complete articulated robot mass distribution. Like
ToddlerBot's planner, WR2 uses the nominal floating-base/root height as the
reduced-order pendulum-height proxy; it is not claiming that this point is the
articulated model's exact CoM. A large or phase-dependent discrepancy is still
useful evidence that the proxy or morphology assumptions need review.

## Scope

The replay calls `mj_forward` on prescribed poses; it deliberately does not
step physics. It establishes periodicity, reachability, foot clearance,
support timing, and internal ZMP consistency. It does not prove that the
unactuated robot can track the path, survive contact impacts, or remain stable.
Those properties remain training and dynamic-rollout acceptance gates.
