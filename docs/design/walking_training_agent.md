# WR2 walking training agent

The WR2 agent applies the useful control pattern from WildRobot's training
agent without importing its task-specific reward mutation and recovery stack.
It runs bounded train/evaluate/promote cycles and stops only after independent
confirmation or after exhausting the configured budget.

## Campaign

The default campaign is defined in
`wr2/locomotion/configs/walking_agent.yaml`. Success gates normally terminate it;
each stage has a 2,000-cycle emergency ceiling:

1. `gait_acquisition`: up to 2,000 100M-step cycles under randomized dynamics;
2. `robust_walking`: up to 2,000 100M-step cycles with expanded forward-speed
   coverage, tighter gates, and the same WR2 randomization; and
3. confirmation at 0.10, 0.15, and 0.20 m/s for three independent seeds and
   128 environments per seed/speed combination.

Every cycle uses six evaluations: one initial baseline and five resumable
checkpoints, approximately 20M environment steps apart. The agent reads the
complete metric rows, rejects checkpoints that violate hard simulation
invariants, and warm-starts the next cycle from the best walking checkpoint.

The policy contract, reward equations, joint limits, network shape, and servo
model cannot be changed by stage overrides. The allowed curriculum changes are
limited to forward/standing command sampling and the ranges of the already
enabled domain randomization.

The acquisition gate is deliberately easier because it is only a curriculum
transition: it proves the policy has stopped exploiting standing and has begun
an alternating forward gait. It cannot complete the campaign. Robust-stage
evaluations rotate through 0.10, 0.15, and 0.20 m/s and require achieved speed
to be at least 90% of command with no more than 0.02 m/s absolute error. The
same criteria are then applied at all three confirmation speeds.

## P0 completion gates and P1 diagnostics

Campaign advancement and final confirmation require the P0 signals:

- average episode length and fall rate;
- at least 90% of commanded forward speed and no more than 0.02 m/s error;
- phase/contact agreement; and
- action saturation no greater than 2%.

The following remain reported as P1 diagnostics and warnings, but do not block
completion yet: lateral drift, yaw error, composite walking score, double
support, foot-phase score, the 1.0157 N m actuator RMS target, the provisional
3 N m mean-step peak target, and normalized sustained torque exposure.

Two hard simulation invariants remain non-negotiable even though their tighter
targets are P1:

- no non-finite simulator state; and
- mean per-step peak torque must not exceed the MJCF 4 N m actuator cap.

These diagnostics constrain policy review in simulation. They do not turn
the provisional torque-exposure signal into a thermal model and do not replace
restrained hardware validation or the inclusive 80 C shutdown.

## Run on the GPU machine

```bash
uv run --extra training python -m wr2.agents.walking_training_agent
```

Use `--resume-checkpoint PATH` to warm-start the first cycle. Brax checkpoints
contain the observation normalizer, policy, and critic, but not optimizer
state; cycle boundaries therefore restart Adam while preserving the learned
networks.

## Launch the fixed campaign from a Mac

The Mac control path verifies that both machines are on the exact same clean
Git commit, then streams the same agent process over SSH:

```bash
uv run python -m wr2.agents.walking_training_agent \
  --host linux-pc.local \
  --user leeygang \
  --remote-repo /home/leeygang/projects/WildRobot2
```

For a public host, pass `--host "$LINUX_PUBLIC_IP"` and, if needed,
`--port "$LINUX_PUBLIC_PORT"`. The Mac needs only the base dependencies; JAX,
MJX, and CUDA are loaded by the GPU-side process.

## Autonomous Mac analysis and improvement

Use the separate autonomous entry point when the Mac should own every
train/analyze/change iteration, as in the WR1 training agent:

```bash
uv run python -m wr2.agents.autonomous_walking_loop \
  --host linux-pc.local \
  --user leeygang \
  --remote-repo /home/leeygang/projects/WildRobot2
```

Keep this foreground Mac process running. For each bounded cycle it:

1. requires a clean local `main`, pushes its exact commit, and fast-forwards
   the clean GPU checkout to that SHA;
2. transfers a generated cycle config and runs PPO on the GPU;
3. copies only the effective config and JSON/JSONL metrics to the Mac—the
   checkpoint stays on the GPU;
4. evaluates P0, P1, and hard safety locally and records `analysis.json`;
5. when another experiment is needed, invokes non-interactive `codex exec` on
   the Mac for one high-confidence experiment and one commit; its structured
   decision must identify the matching active ToddlerBot mechanism and justify
   every WR2-specific divergence;
6. allows coherent changes across the locomotion environment, metrics,
   evaluator, trainer, PPO code, typed config, and campaign YAML, then reruns
   all tests and pushes the validated commit; and
7. starts the next GPU cycle from the best hard-safe checkpoint, after the GPU
   has pulled the validated commit.

The coding agent cannot push or select a checkpoint itself. The supervisor
still freezes the hardware/deployment interface, final P0 definitions,
confirmation matrix, hard servo limits, emergency campaign ceilings, and its own
control plane. A network/parameter-contract change must explicitly request a
cold start; compatible changes retain the best hard-safe GPU checkpoint unless
the accepted hypothesis explicitly requires retraining from scratch.
Acquisition can advance directly to robust training without a code change;
robust completion still requires the independent three-speed, three-seed
confirmation.

The Mac must have the Codex CLI authenticated. The GPU host should use SSH-key
authentication because one campaign makes several unattended SSH/SCP calls.
Verify both before a long run:

```bash
codex exec --ephemeral "Reply with OK"
ssh leeygang@linux-pc.local true
```

Use `--dry-run` to validate local configuration and preview the first cycle
without pushing or opening SSH. `--codex-model MODEL` selects a model and
`--codex-timeout-minutes` bounds one improvement turn. A warm-start checkpoint,
when supplied, must be an absolute GPU path under
`/home/leeygang/projects/WildRobot2/results/wr2_walking/`.

The supervisor writes durable state to
`results/wr2_walking_agent/<agent-id>/autonomous_state.json`. If training,
Codex, Git validation, tests, push, or remote synchronization fails, it stops
with `status=failed`; it never skips the failed stage or silently weakens a
gate. The initial version is intentionally foreground and does not adopt a
partially completed remote cycle after the SSH process is interrupted.

Query the newest campaign from another Mac terminal without contacting or
changing the GPU job:

```bash
uv run python -m wr2.agents.autonomous_walking_loop status --latest
```

Select a specific campaign or request machine-readable output with:

```bash
uv run python -m wr2.agents.autonomous_walking_loop status \
  --agent-id wr2_auto_walk_20260930_212129

uv run python -m wr2.agents.autonomous_walking_loop status --latest --json
```

Human-readable status includes the active stage and cycle, run ID, commit,
checkpoint, most recent GPU/Codex log excerpt, and any terminal error.

## Artifacts

The fixed campaign writes `agent_state.json`; the autonomous Mac campaign
writes `autonomous_state.json`, per-cycle analysis, Codex decisions, and logs.
The PPO runs and resumable checkpoints remain under `results/wr2_walking/`.
Copy a GPU-side fixed-supervisor report with:

```bash
./scripts/scp_from_remote.sh --latest walking_agent
```

The final checkpoint is recorded in `agent_state.json` or
`autonomous_state.json`; a policy is not a hardware candidate merely because
the agent reached `complete`.
