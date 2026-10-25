# TB air-time reward and WR2 commanded-speed audit — 2026-10-24

After clarifying the two landing-credit definitions, the user requested keeping
the qualified-landing change. `ppo_walking_step.yaml` retains **0.28.8**,
`air_time_min_stance_s=.1648109513`, air weight **500**, and output prefix
`wr2_paper_step`. The temporary rollback to zero was withdrawn. The completed
qualified-step run remains the relevant current-policy result; the collision-1
run remains its original-air control.

The qualified configuration uses the regular training entrypoint:

```bash
uv run --locked --python 3.10 --extra training python -m wr2.locomotion.train \
  --config wr2/locomotion/configs/ppo_walking_step.yaml
```

## Reward audit and retained landing qualification

Current WR2 already implements TB's moving leg-reference imitation:

`leg_motor_pos = -mean((q_motor[legs] - q_reference(t, command)[legs])**2)`.

The coefficient is **5** and the weighted reward integrates at **dt=.02 s**.
The reference is read at every Paper step. The twelve controlled leg targets
use fixed home plus the policy action, as in local TB `paper-walking-baseline`.
The actor learns reference imitation through PPO rewards and advantages.

The earlier legacy WR2 recipe instead computed
`pose = sum(pose_weights * (q_motor - q_home)**2)` and added `-.5 * pose` in the
XY-physics configuration. That penalty applied during walking and preferred
the same pose at every phase. The Paper recipe introduced in `0a63ae9` replaced
that objective with moving leg imitation. A separate native TB `stand_still`
home penalty remains gated to near-zero commands; it is zero for this forward
test. The old objective is absent from the current Paper reward path.

TB's original air-time rule is:

`sum(feet_air_time * ((feet_air_time > 0) & (contact | previous_contact)))`

Payment is gated to active commands and occurs on the first filtered contact.
Raw air time accumulates while each foot is unloaded; opposite-foot support is
not required. It credits unloading followed by contact, without a forward
placement requirement or sustained-landing test.

WR2 retains its supported-swing and sustained-landing treatment:

1. Accumulate unloaded time only while the opposite foot supports the robot.
2. Save that credit on landing during an active command.
3. Pay once after **ten consecutive contact samples**, spanning nine intervals
   or **180 physical ms / .2184 TB seconds**.
4. Cancel pending credit if contact breaks, the command becomes inactive, or
   the episode resets.

Both definitions pay on or after landing. The experiment's extra distinction is
requiring sustained contact. The retained `gamma**-9` compensation preserves
the discounted landing-time value of eligible credit, not identical GAE or
policy gradients. Moving leg-reference imitation remains weight **5**.
Collision weight **1.0** remains the existing WR2 departure from TB's **.1**.

The qualified treatment produced sustained alternating contacts in the completed
nominal replays but left short, asymmetric placements. Root-cause analysis
therefore continues with the qualified WR2 checkpoints and native TB Paper,
using the original-air control to distinguish the treatment's effects.

## Command speed versus achieved speed

The completed training sampled translation, turns and zero commands, rather
than training only at forward .1 m/s. Physical forward bounds were
**[-.0824055, .164811] m/s**, lateral **[-.0824055, .0824055] m/s**, and yaw
**[-.606756, .606756] rad/s**. These map to TB forward **[-.1, .2] m/s**,
lateral **[-.1, .1] m/s**, and yaw **[-.5, .5] rad/s**. Zero-command probability
was .2, turn probability .3, and command resampling occurred every 124 ticks.

The viewed `forward_010` condition commands **.1 TB-equivalent m/s**, which is
**.0824055 physical WR2 m/s** using `s=.8240547565`.

| Fixed-forward evaluation | Command, TB m/s | Achieved forward, TB m/s | Achieved / command |
| --- | ---: | ---: | ---: |
| WR2 selected 120M, randomized selection | .1 | .044667 | 44.7% |
| WR2 selected 120M, randomized held out | .1 | .044261 | 44.3% |
| WR2 final 300M, randomized selection | .1 | .033256 | 33.3% |
| TB Paper 200M, randomized selection | .1 | .090528 | 90.5% |

Selected WR2 physical forward speed is **.0368 m/s** in randomized selection.
The nominal CPU recording has **.0437 physical m/s** net world-X velocity over
the whole recording. These are different evaluation conditions and velocity
definitions; neither is the mean XY error nor total speed including lateral sway.

Slow commands imply short absolute steps. At the .6 s WR2 cycle, commanded
translation needs approximately **49.4 mm of same-foot advance per cycle**,
or **24.7 mm of body advance per individual alternating step**. These are
different quantities. The observed unloaded foot swings after startup are
**25.7/24.5 physical mm** left/right. TB's **61.9/62.5 mm** native swings map to
**42.0/42.4 WR2 physical mm** using `s**2=.67906624`. Thus the slow command
explains the scale of expected walking, but not WR2's tracking deficit or its
left foot remaining behind the right at every steady landing.

## Sources and verification

- WR2: `walking_env.py::_paper_step`, `paper_rewards.py::paper_reward_terms`;
  legacy source before `0a63ae9`, `ppo_walking_xy_physics.yaml`.
- TB: local `paper-walking-baseline` at `a4f693d`,
  `mjx_env.py::_reward_leg_motor_pos`, `walk_env.py::_reward_feet_air_time`,
  `_reward_stand_still` and `_sample_command`;
  [published Table VII](https://arxiv.org/html/2502.00893v2).
- WR2 raw evaluation JSONs and per-episode velocity arrays, nominal CPU summary,
  and `review_20261023/landings.json`; TB's native `xy_evaluation.jsonl` at 200M.
- Recomputed speeds are retained in the WR2 result's ignored
  `review_20261024/command_speed_audit.json`.
- Seven passing regression checks cover post-landing payment, short-contact rejection,
  cancellation, configuration isolation and real-environment reset handling.
