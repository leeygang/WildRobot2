# WR2 walking training agent

The WR2 agent applies the useful control pattern from WildRobot's training
agent without importing its task-specific reward mutation and recovery stack.
It runs bounded train/evaluate/promote cycles and stops only after independent
confirmation or after exhausting the configured budget.

## Campaign

The default campaign is defined in
`wr2/locomotion/configs/walking_agent.yaml` and is capped at one billion
environment steps:

1. `gait_acquisition`: at most two 100M-step cycles under nominal dynamics;
2. `robust_walking`: at most eight 100M-step cycles with the measured WR2
   actuator and contact randomization; and
3. confirmation at 0.10, 0.15, and 0.20 m/s for three independent seeds and
   128 environments per seed/speed combination.

Every cycle uses six evaluations: one initial baseline and five resumable
checkpoints, approximately 20M environment steps apart. The agent reads the
complete metric rows, rejects checkpoints that violate hard simulation
invariants, and warm-starts the next cycle from the best walking checkpoint.

The policy contract, reward weights, joint limits, action scale, network shape,
and servo model cannot be changed by stage overrides. The allowed curriculum
changes are limited to command sampling and enabling the already configured
domain randomization.

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

## Launch from a Mac

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

## Artifacts

Each campaign writes `results/wr2_walking_agent/<agent-id>/agent_state.json`,
generated cycle configurations, cycle logs, and the final multi-seed
confirmation report. The PPO runs and resumable checkpoints remain under
`results/wr2_walking/`. Copy the supervisor report with:

```bash
./scripts/scp_from_remote.sh --latest walking_agent
```

The final checkpoint is recorded in `agent_state.json`; a policy is not a
hardware candidate merely because the agent reached `complete`.
