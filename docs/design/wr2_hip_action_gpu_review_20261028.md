# GPU hip action-center review — 2026-10-28

## Conclusion

The randomized GPU validation confirms that the model-home hip center is
physically usable and expands posterior reference coverage. **Changing the
center under the frozen policy does not fix its walking:** it loses forward
progress, leans backward and retains short right-foot placement. The smaller
torso rocking accompanies slower motion.

This clears the initialization check for a controlled PPO experiment. It does
not establish an improved policy or prove that action coverage is the sole
cause of the remaining asymmetry. The qualified current checkpoint remains
the walking baseline.

## Experiment and verification

Result directory:
`results/wr2_walking/wr2_paper_goal_xy_20261026_193545_seed3/hip_action_validation_gpu/`.
Checkpoint: `checkpoints/000200294400` from that run.

The diagnostic changes only the two hip-pitch **target centers** from the
grounded-reference home, about 16.63° forward, to the raw WR2 model home,
about 6.88° forward. The .25-rad residual scale, grounded actual reset,
position-observation baseline, knee/ankle centers, rewards, references,
network weights and physics remain the same. Policy cases use the original
deterministic actor without converting its outputs; constant-action probes
do not use the actor. This is a test of a changed action interpretation.

Seven cases per origin: zero residual; constant residual holding the original
pose; policy at zero/.10/.15/.20 commands; and a .20 stand/walk/stop/restart
schedule. There are 128 episodes per seed, seeds 12102 and 22102, totaling
**3,584 attempts**. Observation noise and model/reset/actuator variation are
enabled; pushes are disabled. Each complete episode has 824 ticks, 16.48
physical seconds/20 TB seconds. Motion means exclude 1.5 TB seconds of warmup.

Producer: clean commit `5c81132c4fa35fa6f4441e38c620a52d097284c8`, RTX 5070,
Python 3.10.19, JAX/JAXlib .6.2, Brax .14.1, MuJoCo/MJX 3.14.0,
high matrix precision. Saved configuration, diagnostic/model/checkpoint hashes
and runtime packages were verified against local source and training records.

All 28 raw traces were checked. Recomputed summaries agree within 3.6e-15;
root displacements agree within 2.4e-10 m; native nonterminal target
reconstruction error is zero. Validity masks include finite terminal samples
and exclude repeated padding. Corresponding origins have identical initial
actual poses and initial observations. Coverage fractions and target endpoints
match the Mac audit exactly; command decimals differ by at most 2.8e-17.
The reproducible numerical audit is saved in `review/analysis.json`.

## Physical stance and startup

All constant-zero, holding, zero-command policy and transition cases complete
without falls or nonfinite samples. Fixed .10 and .15 policy cases also have
zero falls. The four falls occur only at fixed .20: **current 3/256;
candidate 1/256**. All 3,584 attempts remain finite.

This small difference does not establish a stability gain. The paired
bootstrap interval for the candidate-minus-current fall fraction is
−2.34 to +.78 percentage points, including zero. These instrumented rollouts
also differ from the compiled graph in the earlier GPU heading evaluation;
individual failure identities are not reproducible across those tools.
The baseline's separate standard-suite qualification remains 0/3,840 falls.

The holding residual, approximately −.6813 left/+.6813 right, delivers
identical target traces under both origins. Holding qpos differs by at most
5.3e-6 across GPU traces. Both hold with 100% double-foot support, no modeled
foot self-contact, and mean body pitch −.83°. No completed probe exceeds a
hip-pitch mechanical limit.

Zero residual has a different physical meaning: mean pitch changes from
−.83° to **−11.12°**. Keeping the bent knee/ankle targets while changing the
hips rotates the sole relative to the torso; the supported body settles
backward. The Mac native forward-kinematics check measured about +9.69° sole
pitch at the original body orientation. The preserved actual reset and the
candidate's zero-action stance must not be confused.

## Frozen-policy walking

Speeds below are requested-direction speeds in **TB-equivalent m/s**.
Pitch is body pitch; negative means backward. Means cover completed episodes,
with falls shown separately. Transition speed/pitch use walking windows only.

| Command | Current speed | Candidate speed | Current pitch | Candidate pitch | Falls current/candidate |
|---|---:|---:|---:|---:|---:|
| .10 | .0851 | .0614 | +2.51° | −8.90° | 0/0 |
| .15 | .1331 | .1026 | +4.75° | −7.20° | 0/0 |
| .20 | .1692 | .1425 | +7.14° | −4.33° | 3/1 |
| .20 transitions | .1665 | .1407 | +6.16° | −5.89° | 0/0 |

Fixed-command speed decreases **28%, 23% and 16%**, respectively. At .20,
252 paired common completers give a speed change of −.0268 TB m/s
(bootstrap 95% interval −.0276 to −.0260). Counting all 256 attempts with
zero further progress after a fall gives .1686/current versus .1422/candidate.
The conclusion therefore survives including failed episodes.

This is actual lost translation: .20 net-displacement speed falls
.1698→.1434 TB m/s. Absolute net turn increases 6.29°→9.96°; projection
onto the requested direction is not the main explanation for slower speed.

At .20, torso roll p95−p5 decreases 17.08°→15.29° and normalized root-height
p95−p5 decreases 12.35→11.42 TB mm. These measurements compare equal commands,
with unequal achieved speeds. They cannot establish improved naturalness at
matched speed. Modeled foot self-contact increases at .10 (4.53%→6.87%) and
.15 (.63%→3.53%); it decreases slightly at .20 (.59%→.50%). There is no
consistent contact improvement.

## Placement, right-hip output and reference coverage

Landings require supported air, greater than 1 mm sole clearance, and at least
ten consecutive contact samples. Cached foot geometry and contact force share
the native simulator sampling point. Strides never cross command stops.
Values average each completed episode's geometric events; lengths use the
configured WR2/TB normalization.

| Command | Current left/right landing lead, TB cm | Candidate left/right lead, TB cm | Current left/right stride, TB cm | Candidate left/right stride, TB cm |
|---|---:|---:|---:|---:|
| .10 | 7.19/−1.25 | 6.88/−2.58 | 6.24/6.17 | 4.54/4.44 |
| .15 | 8.50/.69 | 7.47/−.32 | 9.70/9.66 | 7.55/7.43 |
| .20 | 9.62/2.08 | 7.88/1.96 | 12.35/12.32 | 10.38/10.35 |

Lead is the landing foot's forward position relative to the other foot;
stride is displacement between successive landings of the same foot.
Both feet advance and landing order remains about 99% alternating. The gap
is uneven placement within that alternating gait. At .20, the apparent
reduction in left/right lead imbalance comes predominantly from a smaller
left lead; the right lead does not improve. At .10/.15, the candidate's
right foot lands farther behind the left. Cadence remains about 1.37 TB Hz,
so shorter strides accompany the speed loss.

The right-hip actor output remains near its negative bound (action ≤−.99):
**92.46%→82.31%** of post-warmup samples at .20. Its delivered target's
average p95−p5 span remains tiny, .19°→.26°. This span is not the full
min/max excursion reported in the earlier single nominal replay.

The full 227-command×1,000-frame lookup independently confirms a real
representational constraint. For the right hip at .20:

| Coverage measure | Current | Candidate |
|---|---:|---:|
| Quantized forward-angle target window | +2.40°…+30.96° | −7.44°…+21.12° |
| Posterior stance reference outside window | 29.17% | 1.65% |
| Stance outside guaranteed interval across target bias | 33.68% | 5.86% |

Extra training cannot make the current controller issue targets outside its
action window. It can learn a successful gait that differs from the reference;
the audit alone does not prove those excluded frames are necessary for walking.
Moving the window is therefore a justified experiment, not a demonstrated fix.

## ToddlerBot comparison and next training

The local **`paper-walking-baseline`** branch is the source of truth, commit
`a4f693d7cc4544f011dad23d6c353e752a6819ab`. Its archived training sources match
the branch. `mjx_env.py` lines 249–250 derive leg action centers from model-home
joint positions; lines 595–598 apply fixed center + scaled delayed residual.
Lines 1188–1191 reward executed leg position against the moving reference.
TB does not use a moving reference as the action center. WR2 already uses
moving leg-reference imitation; this diagnostic does not change that reward.

TB's hip center is about 8.35° forward, giving a raw .25-rad window of
−5.98°…+22.67°. The candidate moves WR2's hip window toward that design.
However, WR2's grounded knee is about 40.2°, versus its raw model-home 13.8°;
wholesale model-home replacement would make that grounded knee unrepresentable
at the same gain. The two-hip change is a deliberately limited comparison,
not a complete TB initialization transplant. The prior TB native-cache audit
also found some reference stance outside its own window, so perfect coverage
is not a requirement established by TB.

**Recommended next step: compare two 200M PPO continuations from the qualified
`000200294400` checkpoint.** One retains the current target centers; the other
uses the two tested model-home hip centers. Keep the grounded reset,
observation/reward baseline, physics, rewards, command distribution, optimizer
setup and budget identical. Restore the same actor/critic/normalizer in both,
with the same optimizer restart. The candidate intentionally begins with the
changed physical interpretation measured here. Exact preservation of every
old target is impossible because some old anterior requests exceed its new
window; do not label the restore a behavior-preserving conversion.

Retain intermediate checkpoints and compare requested-direction progression,
right landing lead, bilateral strides, right-hip boundary occupancy/target
motion, standing/transitions and falls. Assess rocking at matched achieved
speed. A candidate must recover forward progress and improve placement to be
selected; reduced saturation or smaller rocking alone is insufficient. This
comparison separates the effect of additional training from the changed
action window. It is the next learning test; further frozen-policy center
probes do not answer whether PPO can exploit the added posterior range.

References: [TB paper v2](https://arxiv.org/html/2502.00893v2), local TB Paper
`toddlerbot/locomotion/mjx_env.py`, [Mac validation](wr2_hip_action_validation_20261027.md),
[current policy qualification](wr2_goal_xy_final_confirmation_20261027.md).
