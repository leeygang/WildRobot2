# WR2 retained-checkpoint heading evaluation — 2026-10-25

## Evaluation contract

Original deterministic PPO policies at **120,053,760**, **200,089,600** and
**300,134,400** steps, with forward commands **.10/.15/.20 TB m/s** and zero
yaw command. Each checkpoint has one nominal episode per command, followed by
eight randomized episodes at each of seeds **12102/22102**: **16 episodes per
command, 48 randomized episodes per checkpoint**. Push amplitudes are zero;
randomized episodes retain each robot's native model, local actuator/reset
variation and observation noise. Policies, rewards and controllers are unchanged.

The shared evaluator measures integrated free-root position and quaternion at
every control boundary. Forward/lateral displacement is projected onto the
initial requested direction, including random initial yaw. Heading is unwrapped
relative to that direction; net heading change and mean absolute heading error
are reported separately. Native reward signals use the cached final-substep
torso frame. Finite fall samples are retained and each episode stops after its
first fall or invalid transition. A synthetic quarter circle and mixed
fall/invalid batch verify these distinctions.

Full-episode and post-warmup windows are saved. The latter excludes
**1.5 TB seconds** (62 WR2 steps / 75 native TB steps). WR2 uses the saved scale
**s=.8240547565**: physical velocity/time/length = TB velocity/time/length
times **s/s/s²**. WR2's 824-step horizon is approximately 20 TB seconds;
TB uses 1000 steps. Distances labeled physical metres use the robot's actual
size; comparisons below convert TB distances to WR2 size.

Sources: WR2 `walking_env.py::_paper_step/_paper_observation`,
`paper_evaluation.py::PaperEvaluator`; TB's verified saved
`paper-walking-baseline` environment, reference and native Brax fork.
The recipe reference is [ToddlerBot Table VII](https://arxiv.org/html/2502.00893v2).
The active TB checkout is not modified.

## WR2 randomized results

All **144 randomized WR2 episodes** finish without a fall or nonfinite event.
Post-warmup means, 16 episodes per row:

| Checkpoint | Command | Body-forward TB m/s | Requested-direction TB m/s | Absolute net heading change | Absolute lateral displacement, physical m |
| --- | ---: | ---: | ---: | ---: | ---: |
| 120M | .10 | .0756 | .0728 | 25.3° | .237 |
| 200M | .10 | .0864 | .0792 | 37.5° | .355 |
| 300M | .10 | .0882 | .0839 | 32.2° | .291 |
| 120M | .15 | .1198 | .1116 | 29.8° | .453 |
| 200M | .15 | .1310 | .1160 | 42.5° | .621 |
| 300M | .15 | .1385 | .1078 | 59.7° | .922 |
| 120M | .20 | .1618 | .1408 | 45.1° | .763 |
| 200M | .20 | .1597 | .1449 | 40.4° | .736 |
| 300M | .20 | .1814 | .0985 | 85.0° | 1.582 |

At .20, 300M's body speed reaches **90.7%** of command, while its progress in
the requested direction reaches **49.3%**. 120M/200M reach **70.4%/72.4%** in
the requested direction. The 300M policy is not uniformly worse: at .10 it
has the best requested-direction progress of these retained candidates.

Within-WR2 comparisons use the same model/reset/noise draws for each saved
policy. At .20, 300M minus 120M gives:

- Body-forward velocity **+.01963 TB m/s**, paired 95% bootstrap interval
  **[+.01513,+.02378]**.
- Requested-direction velocity **−.04227 TB m/s**, interval
  **[−.07228,−.01341]**.
- Absolute heading change **+39.91°**, interval **[+16.56,+65.23]**.
- Absolute lateral displacement **+.819 physical m**, interval **[+.557,+1.080]**.

For 300M minus 200M, requested-direction velocity changes by
**−.04632 TB m/s**, interval **[−.06917,−.02338]**, and heading change increases
**44.65°**, interval **[26.63,62.22]**. These intervals describe 16 evaluation
draws, not variability across independently trained policies.

## Native TB Paper comparison

The frozen Paper 200M checkpoint was evaluated in its native Python 3.10,
JAX .6.2, Brax .10.5 and MuJoCo/MJX 3.2.4 environment. Saved core source,
reference/model hashes and the pinned Brax revision were checked. This run's
saved configuration actually uses **five .004-second physics substeps** per
control step; WR2 uses **four .005-second substeps**. The actual runtime values
are recorded rather than inferred from another TB branch or a default config.

Completed post-warmup episode means:

| Command | Policy | Completed / attempted | Body-forward TB m/s | Requested-direction TB m/s | Absolute net heading change | Absolute lateral displacement at WR2 size |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| .10 | WR2 300M | 16/16 | .0882 | .0839 | 32.2° | .291 m |
| .10 | TB Paper 200M | 16/16 | .0903 | .0870 | 13.5° | .265 m |
| .15 | WR2 300M | 16/16 | .1385 | .1078 | 59.7° | .922 m |
| .15 | TB Paper 200M | 16/16 | .1342 | .1227 | 21.1° | .571 m |
| .20 | WR2 300M | 16/16 | .1814 | .0985 | 85.0° | 1.582 m |
| .20 | TB Paper 200M | 15/16 | .1705 | .1407 | 37.9° | .959 m |

TB has **one fall among 48 randomized episodes**, at .20, seed22102 episode2,
after 100 ticks / 2 physical seconds. Its post-warmup window contains only
.5 seconds, during the fall, with body-forward velocity −.653 TB m/s. It
remains in the raw results and all-episode means; the table explicitly conditions
on completed episodes. Including it gives .1191 body-forward and .0955
requested-direction TB m/s for .20. The short fallen episode's small heading
change is not counted as successful heading tracking.

Mean pitch among completed .20 episodes is **9.77° for WR2 300M versus 3.81°
for TB**; 120M/200M WR2 have 11.74°/12.99°. Thus merely choosing an earlier
checkpoint for straighter travel does not resolve the posture difference.
Both robots preserve their native randomization distributions, which differ;
this comparison does not isolate a particular friction or actuation parameter.

The native yaw reward reconstructed from the recorded body gyro agrees within
**1e-7 per step** for both robots. Source snapshots and 43 input hashes identify
the evaluated code/data. The final evaluator embeds the same zero-yaw nominal
reset locally, so GPU evaluation needs only TB's verified core Paper code.
With identical native wrappers, seed, command and batch size, its reset qpos,
qvel and actor/critic observations are **bit-identical** to the frozen reset.
This verification is saved as `native_reset_validation.json` in the review.

## Conclusion and next decision

The higher-command direction loss persists across randomized WR2 episodes.
300M is better on body-frame velocity and worse on requested-direction progress
than both earlier checkpoints at .20. All 16 .20 episodes turn left at 300M;
earlier policies have mixed turn directions. The recorded objective still
prefers 300M, so further optimization of that objective alone does not establish
better straight walking. Corrected yaw-rate rescoring from the earlier audit
also retains its preference for the turning gait.

At .20, completed TB episodes achieve approximately the requested-direction
velocity of WR2 120M/200M. TB also drifts and has a sampled fall, so its result
does not establish perfect heading control. These findings support evaluating
the strength/saturation of the existing orientation objective and including
commanded-heading accuracy when choosing a continuation checkpoint. They do
not yet select a reward coefficient or a replacement formula.

Both actors observe Euler yaw and gyro, but the accumulated reference heading
used by the orientation reward is not an explicit actor input. The same source
audit should check the availability of the desired-heading error before
strengthening that reward. Training resamples commands about every three TB
seconds; these fixed-command evaluations hold them for 20 TB seconds, exposing
accumulated bias under the same benchmark duration for both robots.

The larger GPU pass is ready for WR2 before choosing that checkpoint. 200M is
a candidate with better .20 direction tracking than final 300M; .10 performance
and its greater pitch remain tradeoffs. The current training selector still
uses body-frame XY tracking within a fall gate; this diagnostic adds measurements
without silently changing the selector or training recipe.

See the [comparison plot](../../results/wr2_walking/wr2_paper_progress_20261025_085931_seed3/heading_evaluation_review_20261025/heading_comparison.png)
and neighboring `comparison.json` for the per-robot means, all/completed episode
populations, source/checkpoint hashes and paired WR2 intervals.

## Reproduce / larger GPU confirmation

From the GPU's WR2 checkout:

```bash
uv run --locked --python 3.10 --extra training python -m wr2.tools.evaluate_heading \
  --run-dir results/wr2_walking/wr2_paper_progress_20261025_085931_seed3 \
  --output results/wr2_walking/wr2_paper_progress_20261025_085931_seed3/heading_evaluation_gpu \
  --num-envs 128
```

This evaluates all three retained checkpoints, both seed sets and all three
commands: **256 randomized episodes per command per checkpoint**, plus nominal
controls. The command requires a GPU; CPU evaluation uses explicit
`--allow-cpu`. It trains no policy. The output must be a new directory.

The Mac command uses `uv run --no-sync python`, `--num-envs 8 --allow-cpu`,
and the output name `heading_evaluation_mac`.

Native TB evaluation runs from the WR2 checkout with TB's saved Python runtime:

```bash
uv run --no-project ~/projects/toddlerbot/.venv-paper-blackwell/bin/python \
  -m wr2.tools.evaluate_heading \
  --tb-root .worktrees/tb-paper-stance \
  --run-dir ~/projects/toddlerbot/results/toddlerbot_walk_ppo_20261015_194347 \
  --output results/toddlerbot_walk_ppo_20261015_194347/heading_evaluation_mac \
  --num-envs 8 --allow-cpu
```

`--tb-root` must be a verified frozen Paper checkout. The evaluator checks core
source bytes, model/reference hashes, pinned Brax and saved package versions;
it does not switch or write to the active TB project.

After GPU completion, copy from the Mac:

```bash
./scripts/scp_from_remote.sh --linux-pc --results \
  wr2_walking/wr2_paper_progress_20261025_085931_seed3/heading_evaluation_gpu
```

## Artifacts and limits

WR2: `results/wr2_walking/wr2_paper_progress_20261025_085931_seed3/heading_evaluation_mac`.
TB: `results/toddlerbot_walk_ppo_20261015_194347/heading_evaluation_mac`.
Each evaluation saves `summary.json`, source snapshots and compact per-case
trajectories. A completed evaluation has **`complete: true`** in its summary.
Arguments, runtime, effective sampling contract, saved training configuration
and source hashes identify the evaluation. Retained checkpoint paths are
explicit; the final evaluator also records their content hashes.

The Mac sample establishes a sizeable direction-tracking loss at higher
commands. Its size does not establish a sub-1% fall rate or broad robot parity.
Use the larger GPU pass before selecting a continuation checkpoint or changing
checkpoint selection. See the earlier
[trajectory/reward audit](wr2_speed_contact_review_20261025.md) for why correcting
the yaw-rate coordinate alone does not reverse the recorded return preference.
