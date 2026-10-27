# Heading-feedback continuation result — 2026-10-26

The subsequent [GPU comparison is complete](wr2_heading_feedback_gpu_review_20261026.md).
It confirms the improvement and resolves the +120M/+173M speed-versus-turn tradeoff.

## Result

Run: `results/wr2_walking/wr2_paper_heading_feedback_20261025_201358_seed3`.

The continuation improves heading control and requested-direction progress
with the existing physics and reward. The selected checkpoint is **+173,588,480
steps**, about **473.72M cumulative steps** including the source run. The final
checkpoint at +200,294,400 is worse. A +120,176,640 intermediate checkpoint
is substantially straighter at higher commands, with similar forward progress
to the selected checkpoint in the small Mac comparison.

Keep the selected `best_params` as the confirmed candidate for the standard
15-condition evaluation. Compare +120M, +173M and +200M on the GPU before
choosing a policy for higher-command walking or another training change.

## What changed and what was verified

The saved YAML differs from the completed 0.29.0 run only in heading
observations, the 200M continuation budget, checkpoint selection metric and
version/output identity. Actor/critic receive sine/cosine of desired-heading
error in each history frame. XY reward still tracks velocity in the current
body frame; its coefficient remains15. Yaw5, orientation1, moving
leg-reference imitation5, qualified landing air-time500, physics, fixed
home-plus-residual action mapping and command/randomization distributions
retain the source setup. Training has no external pushes.

The run used clean commit `74d34d3763fc8e1c0f5975e5bf99caff8ddbf82a`,
Python3.10/JAX0.6.2/Brax0.14.1 on an RTX5070. The restore report confirms
CPU float64 equivalence, with maximum output difference5.33e-15. Native
GPU action/logit differences are zero on the synthetic verification batch;
critic difference is2.20e-4. The saved `initial_params` exactly equal the
original checkpoint after zero-extension. `best_params` exactly match
checkpoint `000173588480`, and final `params` match `000200294400`.
All15 saved checkpoints contain finite parameters. The local model hash
matches the saved training model.

Metrics were checked against current WR2 code and the saved per-episode data:

- All36 primary, push and confirmation reports have the intended15 conditions,
  128 episodes per condition,824 ticks at20ms, and consistent sample/fall counts.
- Recomputed selection summaries agree exactly. Group statistics agree within
  float32 rounding; maximum absolute difference2.78e-6.
- All training/rollout metric values are finite. No evaluation reports contain
  nonfinite transitions. Primary selection has3 falls at +106.82M; initial,
  selected, final and selected held-out reports have zero falls.
- The27 new Mac heading traces reproduce their saved heading/progress summaries
  exactly. Native XY, yaw and orientation rewards reconstruct within3.46e-7.

`restore_checkpoint_sha256` is null because the source checkpoint is an Orbax
directory. The restore report records its parameter-tree SHA256 instead;
the exact parameter comparison above independently confirms the restore.

## Standard walking comparison

XY errors and velocities below are in **TB-equivalent m/s**. Heading here is
**mean absolute heading error during the episode**, across the ten fixed
nonzero walking conditions. These means use1,280 episodes; fall counts cover
all1,920 episodes including standing and transitions. They are different
metrics from the net turn measured in the higher-command comparison below.

| Evaluation | Checkpoint | Body-frame XY error | Reference-frame XY error | Mean absolute heading error | Falls |
|---|---:|---:|---:|---:|---:|
| Selection seed12102 | Restored300M | .04525 | .04893 | 11.71° | 0/1,920 |
| Selection seed12102 | +120.18M | .04340 | .04491 | 5.59° | 0/1,920 |
| Selection seed12102 | **+173.59M selected** | **.04245** | **.04369** | **5.47°** | **0/1,920** |
| Selection seed12102 | +200.29M final | .04306 | .04470 | 7.67° | 0/1,920 |
| Held-out seed22102 | Restored300M | .04582 | .04955 | 11.83° | 0/1,920 |
| Held-out seed22102 | **+173.59M selected** | **.04322** | **.04438** | **5.79°** | **0/1,920** |

Held-out improvements are **10.45% in reference-frame error**, **5.68% in
body-frame error**, and **51.1% in mean absolute heading error**. Paired
bootstrap intervals, resampling128 episode-index clusters shared across the
ten commands, exclude zero for each improvement. For example, physical
reference-frame error changes by−.00427m/s,95% interval[−.00460,−.00393].

For the held-out .10 forward command, body speed changes.08740→.08503,
but requested-direction speed improves.08309→.08491. Reference lateral
speed drops.02002→.00320 and mean heading error17.61°→4.80°.
Much of the improvement is better direction control rather than a faster
body-frame gait. Backward tracking also improves; forward-left diagonal
tracking becomes slightly worse, so the aggregate gain is not universal.

## Higher-command walking: new Mac evaluation

The original native policies were evaluated at .10/.15/.20 TB-equivalent
commands, with zero commanded yaw and no pushes. Each checkpoint/command
has16 randomized attempts, across seeds12102/22102, plus one nominal
episode. All144 randomized WR2 attempts and9 nominal episodes complete
without falls or nonfinite states. This is a CPU screening comparison,
not the larger GPU confirmation.

Both WR2 runs use the same CPU runtime and measurement code. Native TB Paper
uses its previously verified Paper runtime; its .20 result below uses15
completers out of16 attempts, with one fall. This conditional TB mean should
not be treated as a matched stability qualification.

Each table entry is **requested-direction speed / absolute net turn** after
the1.5-TB-second warmup. These windows last15.24 physical WR2 seconds,
or18.49 TB seconds. Commands .10/.15/.20 correspond to physical WR2
speeds.08241/.12361/.16481m/s. Angles require no size normalization.

| Policy | .10 command | .15 command | .20 command |
|---|---:|---:|---:|
| Previous WR2 300M | .0839 /32.2° | .1078 /59.7° | .0985 /85.0° |
| Feedback +120.18M | .0854 /5.0° | .1296 /4.8° | .1587 /11.4° |
| **Feedback +173.59M selected** | **.0870 /6.4°** | **.1340 /14.3°** | **.1624 /29.8°** |
| Feedback +200.29M final | .0816 /19.9° | .1219 /28.7° | .1486 /32.3° |
| Native TB Paper200M | .0870 /13.5° | .1227 /21.1° | .1407 /37.9° |

At .20, previous300M→selected feedback changes physical forward distance
**1.238→2.039m**, lateral displacement magnitude **1.582→.646m**,
and mean forward lean **9.77°→6.81°** over the post-warmup window.
Requested forward distance is2.512m. The selected policy achieves81.2%
of commanded progression, versus49.3% previously. Body-forward speed changes
only.1814→.1750 TB m/s: the larger advance comes mainly from reducing
uncommanded turning.

The paired16-draw .20 improvement in requested-direction speed is+.06382
TB m/s,95% interval[+.04259,+.08730]. Absolute net turn decreases55.2°,
interval[−72.6°,−37.2°]. Thus the small comparison supports a substantial
improvement despite its limited sample size.

The checkpoint tradeoff matters. Selected+173M versus+120M gains just
**.00369 TB m/s**, interval[−.00515,+.01187], but adds **18.4° of net
turn**, interval[+8.5°, +30.0°], at .20. That is approximately4.6cm
additional forward travel over this window, with no clear statistical
progress advantage. The final checkpoint loses forward progress relative
to the selected one. The automatic selector uses the standard command set,
which reaches .10; it does not select for .15/.20 heading performance.

## ToddlerBot comparison and interpretation

The five saved TB locomotion files were compared byte-for-byte with
**`paper-walking-baseline`**, revision
`a4f693d7cc4544f011dad23d6c353e752a6819ab`; all match. Active TB main
development was not used as the source of the Paper comparison.

Code confirms that TB's actor observes actual Euler yaw and gyro, while
reference quaternion enters the orientation reward. Its critic adds motor
reference errors and contact targets but does not observe desired-heading
error. TB's XY reward uses body-local velocity and its yaw reward uses
body-local angular-Z velocity. WR2 retains those reward formulas in this
experiment. Explicit heading error is the intentional observation extension.
Both use a fixed home action origin and moving leg-reference imitation.

The full saved TB Paper run reaches300M. Its best eligible body-frame
tracking checkpoint is200.09M: error.03702 TB m/s and8/1,920 falls
(.42%). TB's final300.13M has error.03800 and134/1,920 falls(6.98%).
Current WR2 selection error.04245 in the same body-frame metric remains
**14.7% higher** than TB200M, with zero sampled primary falls. This is an
unpaired comparison: robot/randomization profiles and training seeds differ.
The reference-frame and heading metrics cannot be recovered from TB's old
standard XY reports; its smaller native heading evaluation supplies the
separate comparison above.

This continuation demonstrates that the existing WR2 physics permits
substantially better forward progression and heading control. It supports
retaining heading feedback. It does not isolate the input change from the
effect of200M additional training and a new Adam state; no matched continuation
without feedback was run. It also does not establish complete TB parity.

Sources: local TB Paper `mjx_env.py` observation and reward methods;
[ToddlerBot, Table VII](https://arxiv.org/html/2502.00893v2);
[experiment rationale and heading-feedback precedent](wr2_heading_feedback_experiment_20261025.md).

## Remaining gaps

1. **Checkpoint-dependent turning at higher commands.** +120M is straighter
   than the selected/final candidates in the CPU screen. Qualify the retained
   candidates on the GPU before selecting a higher-command policy.
2. **Foot contact and gait quality.** Held-out modeled foot self-contact rises
   **3.74%→5.92%** across fixed walking conditions; forward .10 rises
   **3.08%→5.10%**. Contact-phase agreement decreases64.5%→61.6%, and
   torque RMS increases.469→.511Nm. These are regressions despite zero
   primary falls. Self-contact measures the modeled foot boxes, not a new
   measurement of CAD-part overlap or a foot stepping on the other foot.
3. **Speed accuracy and posture.** Selected .20 progression remains below the
   command, and its6.81° forward lean exceeds TB's3.81° in the small comparison.
   A remaining mechanical maximum or the cause of this lean is not established
   by these measurements.
4. **External-push robustness.** Held-out push falls decrease878→814 of1,920
   (45.7%→42.4%), but remain high. This is a separate stress profile; it was
   not used for training or checkpoint selection.

Reported PPO KL is usually.02–.03, with spikes.109 at +26.7M, .064 at
+93.5M and .052 at +106.8M. The last checkpoint's KL is.0275. Training
remains finite, but the final candidate's deterioration reinforces the
need to retain/evaluate intermediates. The log covers about1h58m of training
and evaluation callbacks; final reported training throughput is47.9k steps/s.

## Next comparison

Run the existing evaluator on the GPU, using128 episodes per seed and command
(256 randomized attempts per checkpoint/command), plus nominal episodes:

```bash
uv run --locked --python 3.10 --extra training python -m wr2.tools.evaluate_heading \
  --run-dir results/wr2_walking/wr2_paper_heading_feedback_20261025_201358_seed3 \
  --checkpoint-steps 120176640 173588480 200294400 \
  --num-envs 128 \
  --output results/wr2_walking/wr2_paper_heading_feedback_20261025_201358_seed3/heading_evaluation_gpu
```

Compare these with the previous run's existing GPU evaluation using the same
seeds, runtime and measurement definitions. Prioritize actual forward travel,
net turn, lateral displacement, pitch and falls. The immediate decision is
whether+120M's straighter walking survives the larger draws, and whether the
selected+173M policy's extra speed justifies its larger turn. A +120M promotion
would also need the full held-out15-condition comparison already supplied for
+173M. The previously supported reference-frame XY reward remains a possible
follow-up if persistent turning is still the leading gap after qualification.

Copy the completed comparison from the Mac:

```bash
./scripts/scp_from_remote.sh --linux-pc --results \
  wr2_walking/wr2_paper_heading_feedback_20261025_201358_seed3/heading_evaluation_gpu
```

For visual evaluation of the selected policy on the Mac:

```bash
uv run --no-sync mjpython -m wr2.tools.view_walking \
  --run-dir results/wr2_walking/wr2_paper_heading_feedback_20261025_201358_seed3 \
  --command 0.2 0 0 --playback-speed 1
```

The viewer defaults to `best_params`(+173.59M), and its command is already
in TB-equivalent units. For the earlier candidate, add
`--checkpoint results/wr2_walking/wr2_paper_heading_feedback_20261025_201358_seed3/checkpoints/000120176640`.

## Artifacts

Validated aggregates, paired intervals and source hashes:
`results/wr2_walking/wr2_paper_heading_feedback_20261025_201358_seed3/review_20261026/analysis.json`.
Original-policy traces and complete metadata are in that run's
`heading_evaluation_mac/`. The comparison figure is saved as PNG and PDF in
`review_20261026/heading_feedback_comparison.*`. Results remain local artifacts.
