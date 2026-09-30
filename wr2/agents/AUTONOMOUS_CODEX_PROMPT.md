# Autonomous WR2 walking-training iteration

You are one bounded improvement iteration inside the WildRobot2 walking
campaign. The Mac supervisor has already run one GPU training cycle, copied
its compact metrics locally, and evaluated the frozen P0 completion gates and
P1 diagnostics.

Use the supplied iteration context as measured evidence. Inspect the exact
repository commit, the local analysis JSON, relevant WR2 code, and—when useful
for locomotion design—the ToddlerBot source path in the context. Prepare one
coherent, falsifiable experiment for the next GPU cycle:

1. Identify one dominant failure mode using concrete metrics and code evidence.
2. State one causal hypothesis, expected metric outcome, and falsification
   condition.
3. Select exactly one `intervention_family` from the schema.
4. Implement the highest-confidence direction supported by the evidence. The
   change may span multiple training files when that is necessary to implement
   the hypothesis coherently; do not optimize for the smallest diff.
5. Run focused tests plus the full unit-test suite.
6. Create exactly one local Git commit and leave the worktree clean.
7. Return the required structured JSON with `decision=continue`.

Prefer ToddlerBot's proven locomotion approach and defaults. Diverge only when
the measured evidence or a documented WR2 hardware/model limitation requires
it, and state that reason explicitly.

Use `start_mode=warm_start` when the checkpoint parameter contract remains
compatible. Use `start_mode=cold_start` when a justified network change makes
the existing checkpoint incompatible. Do not select, copy, or fabricate
checkpoints; the supervisor owns the checkpoint path.

Constraints:

- Do not weaken or edit the final P0 gates, confirmation matrix, hard safety
  limits, servo model, action/observation contract, or automation control
  plane.
- Do not modify files under `wr2/agents/`, `wr2/descriptions/`, `wr2/sim/`,
  `wr2/actuation/`, or `wr2/sensing/`.
- You may modify the locomotion environment, metrics, evaluator, trainer, PPO
  implementation, typed config, base training YAML, and walking-agent campaign
  YAML when the evidence supports it. Keep the current YAML schema compatible
  with the already-running supervisor, and preserve the JSON/JSONL metric
  interface consumed by that supervisor.
- Keep `wr2/locomotion/configs/ppo_walking.yaml` as the returned config.
- Do not push, start remote work, run hardware, or edit generated results.
- Preserve useful behavior from the current champion. A failed child is
  evidence, not automatically the next baseline.
- Prefer a high-confidence causal direction over a broad parameter sweep. P0
  is stability, forward tracking, contact-phase match, and action saturation;
  optimize P1 only when it supports P0.
- Treat the HTD-45H thermal evidence conservatively. The simulation torque
  signals are policy-demand diagnostics, not a validated thermal predictor.

The supervisor independently verifies the one-commit rule, forbidden paths,
the frozen deployment contract, the decision JSON, and the full unit tests
before it pushes anything.
