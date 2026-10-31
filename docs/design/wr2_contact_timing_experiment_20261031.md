# Reference-contact timing experiment — 2026-10-31

## Training question

Does restoring TB's relative emphasis on reference contact make WR2 learn a
slower gait with longer, more even forward placement while preserving speed,
heading and body stability?

The [landing-critic review](wr2_landing_critic_result_20261031.md) confirms
correct delivery of the new critic state but no consistent improvement in
landing-payment prediction or cadence. Selected/final still use about 2.06
TB same-foot cycles per second, versus the native reference's 1.37; randomized
walking contact-phase agreement remains about 52%. The existing TB contact
objective is the next focused reward test.

## Exact change and TB rationale

Complete v0.37.0 `ppo_walking_contact3.yaml` changes only
`reward_weights.feet_contact: 1.0 → 3.0` relative to the completed v0.36.0
landing-critic configuration, apart from version/name/output labels.

Both WR2 `paper_rewards.py::paper_reward_terms` and local TB
**`paper-walking-baseline`** `walk_env.py::_reward_feet_contact` compute:

```text
contact_reward = weight × sum(actual_contact == reference_contact) × dt
```

There are two comparisons, one for each foot. WR2's executed contribution
becomes 0, 0.06 or 0.12 per 20 ms control tick, versus 0, 0.02 or 0.04 in the
control. The term applies to both walking and standing, following TB's formula.
Higher standing reward must not be mistaken for a walking improvement.

| Coefficient | TB Paper | Completed WR2 comparison | This experiment |
|---|---:|---:|---:|
| XY tracking | 5 | 15 | 15 |
| Moving leg-reference imitation | 5 | 15 | 15 |
| Reference contact | 1 | 1 | 3 |
| Qualified landing air time | TB first-contact formula, 500 | WR2 qualified rule, 500 | WR2 qualified rule, 500 |

Weight 3 restores TB's contact/XY and contact/leg coefficient ratios of 1/5.
It is a relative-emphasis experiment, not complete TB weight parity or an
additional robot-size scaling rule. The contact/air coefficient ratio becomes
three times TB's; contribution and gradient ratios need not equal coefficient
ratios. The weight is a hypothesis to test, not a demonstrated optimum.

Reference generation, force classification, model/actuators, fixed action
centers and grounded reset are retained. Actor remains 1005 inputs; critic
remains 1521, including current landing credit. Air time still pays only after
the configured ten consecutive contact samples. No stride reward or prescribed
joint target is added.

## Parent and completed comparison

Restore the accepted fresh 200M policy:

```text
results/wr2_walking/wr2_paper_size_aligned_20261030_085111_seed3/checkpoints/000200048640
```

Completed contact-weight-1 comparison:

```text
results/wr2_walking/wr2_paper_landing_critic_20261030_211823_seed3
```

Both start from the same actor/critic/normalizer and seed 3. The regular trainer
automatically appends six zero-weight critic inputs to the 1515-input parent,
preserving initial actor and value predictions. Native continuation resets
Adam/RNG/environment state and counters in both arms. Value predictions begin
unchanged, but the new reward changes the return the critic must learn.

Use the accepted parent rather than the +40M/+160M candidates to preserve the
comparison's initial conditions. Another contact-weight-1 run is unnecessary.

## GPU command

From the updated GPU WR2 checkout:

```bash
uv run --locked --python 3.10 --extra training python -m wr2.locomotion.train \
  --config wr2/locomotion/configs/ppo_walking_contact3.yaml \
  --restore-checkpoint results/wr2_walking/wr2_paper_size_aligned_20261030_085111_seed3/checkpoints/000200048640
```

Output prefix: `wr2_paper_contact3`. The regular trainer performs GPU and full
reference-grid checks, records the effective config/provenance and checkpoint
conversion, and retains all intermediate policies.

Requested 200M rounds to **200,048,640 added transitions / 195,360 Adam updates**;
cumulative **400,097,280 transitions / 390,720 updates**. Sixteen evaluations
retain 15 learned checkpoints about 13.34M transitions apart plus the initial
policy. Selected and initial policies receive held-out confirmation.

All other settings match the completed comparison: 1024 environments,
824-tick episodes, 16-tick rollouts, four epochs × four minibatches, fixed
LR 1e-4, entropy 0.0005, normalized discount/GAE, 512/256/128 SiLU networks,
tanh-normal distribution, noise/randomization and zero training pushes.

Size conversions remain `v_TB=v_WR/s`, `t_TB=t_WR/s` and
`length_TB=length_WR/s²`, with `s=.8240547565448791`. Actual control is 20 ms
with four 5 ms physics substeps; reference is 30 ticks / 0.60 physical seconds.
No speed or timing scaling changes accompany this reward test.

## Validation

The local preflight passed complete-config isolation against the saved
comparison and accepted parent, config round trip, finite parent tensors,
exact selected/final parent equality and zero-weight critic restoration.
Initial actor and critic prediction differences are zero. The parent checkpoint
hash remains `6a4184e6c9d562b43115b6900054014de8f79e047cafe1d9150dd91a50c43e7c`.

On 12 matched native transitions, physics, observations, controller/reward
memory and other metrics match exactly. Contact reward triples, and total
reward difference agrees with the expected extra contribution within 4.48e-8.
The real parent also passes 64 native PPO transitions / 16 Adam minibatch
updates: initial parameters match exactly, updates remain finite, actor
parameters change and saved parameters reload exactly. The regular training
CLI help and saved-config round trip pass.

Development validation uses nominal CPU physics and exact reduced reference
rows; the regular production command retains its full-grid preflight and
configured randomization. Validation parameters are discarded and stored only
in the parent's ignored `contact3_preflight_20261031/` directory.

## Review and promotion criteria

Compare equal added-step checkpoints, selected candidates and the common
initial policy against the completed contact-weight-1 run. Keep the +40M
low-command and +160M higher-command placement candidates as additional
benchmarks. Do not assume the final or automatic XY selection has the best gait.

Use the same force threshold, qualified landing definition, reference rows,
1.5 TB-second warmup and initial command-heading projection at 0.10/0.15/0.20
TB commands. Check:

- Walking reference-contact agreement and same-foot cadence moving toward
  the reference's 1.37 TB Hz.
- Positive left/right landing leads and same-foot stride, with more even
  placement across commands rather than one good nominal sample.
- Desired-direction speed and instantaneous desired-frame XY error, heading,
  standing/transitions, support/self-contact, falls and root/shoulder motion.

Smaller rocking caused by slower translation is not sufficient. Higher raw
return is also insufficient: changing contact weight makes native return and
training value loss directly incomparable across objectives. Physical tracking,
gait and stability metrics remain comparable. The existing selector ranks
learned policies by desired-frame XY error; promotion also requires beating
the initial baseline on the walking priorities.

If phase agreement improves but progression/placement does not, the timing
reward has not solved the target. If phase agreement does not improve after
200M, do not keep increasing the coefficient without examining how PPO's
phase-specific updates and actor exploration respond to the changed objective.

References: local TB **`paper-walking-baseline`** `a4f693d`, `paper.gin` and
`walk_env.py`; [TB paper v2, Tables VI–VII](https://arxiv.org/html/2502.00893v2);
[latest result and source audit](wr2_landing_critic_result_20261031.md).
