# Hip action-origin validation — 2026-10-27

The [randomized GPU review](wr2_hip_action_gpu_review_20261028.md) is now complete:
3,584 attempts validate supported initialization and reference coverage, while
the frozen candidate remains slower, backward-leaning and uneven in placement.

## Result

The Mac completed the full reference-coverage audit and fourteen native MJX
rollouts. **All fourteen finish 824 ticks with zero height terminations and zero
nonfinite transitions.** The model-home hip center is physically usable, and
the existing grounded pose can be held with identical targets and simulated
poses. It changes the neutral stance and the frozen policy's behavior enough
that it must be treated as a new training experiment.

Production training, saved checkpoints, rewards and model assets are unchanged.
The diagnostic is `wr2/tools/validate_hip_action.py`; it is a standalone module
using the production controller and simulator. It does not replace policy
actions with walking-reference angles.

## What was tested

Run: `wr2_paper_goal_xy_20261026_193545_seed3`, explicit final checkpoint
`checkpoints/000200294400`.

Two diagnostic environments use the same grounded reset, knee/ankle and other
joint centers, position-observation baseline, reward formulas and moving
reference. Only the two hip-pitch action centers differ:

- Current: grounded-reference hip center, approximately 16.63° forward.
- Candidate: WR2 raw model-home hip center, approximately 6.88° forward,
  following TB's use of model-home action centers. Action scale remains .25 rad.

Preserving the position-observation baseline is deliberate: this isolates the
physical action-origin change and retains the existing checkpoint's input
coordinates. It is not a conversion of every home-related field to TB's setup.
The original actual reset poses and initial actor/critic observations match
between corresponding cases.

Each origin runs zero residual, a constant action holding the original pose,
the frozen original PPO policy at zero/.10/.15/.20 TB commands, and a .20
standing/walking transition schedule. Constant probes
do not use the actor; policy probes use the original network weights and
residual outputs without conversion. The holding probe seeds its delayed
action history with the holding action while retaining the grounded reset.

Profile: nominal/no observation noise/no pushes, seed 22102, one episode per
case, 20 TB seconds/16.48 physical seconds, 1.5-TB-second warmup excluded from
motion means. Physics remains 5 ms×4 substeps and 10 Newton iterations.
Local runtime: CPU, JAX 0.6.2, Brax 0.14.2, MuJoCo/MJX 3.14.0, high matrix precision.
GPU training uses Brax 0.14.1. These are paired Mac diagnostic results, not a
GPU qualification or retrained-policy result.

## Coverage

The complete 227-command×1,000-frame native table was audited, with quantized
endpoints and the configured target-bias corners. These fractions cover the
full lookup's stance samples, rather than the earlier post-warmup replay slice.

| Right hip, .20 reference | Current | Model-home hip |
|---|---:|---:|
| Quantized forward-angle target window | +2.40°…+30.96° | −7.44°…+21.12° |
| Posterior stance outside window | 29.17% | **1.65%** |
| Stance outside guaranteed interval across target bias | 33.68% | **5.86%** |

Coverage improves substantially. The remaining nominal outside fraction is
comparable to or smaller than the prior native TB Paper cache reference
(about 8.4% posterior stance outside its raw action window). TB's count uses
its own native cache and mapping; it is not the same quantized/bias-corner audit.
Full coverage is not assumed to be a requirement of the TB recipe.

## Physical stance and initialization

The original grounded pose requires approximately **−.6813 left /+.6813 right**
hip residuals under the new center. Both are inside the action bounds.
The paired holding probes have **bitwise identical qpos traces**, zero target
reconstruction error and 100% double-foot support after warmup. Mean pitch
is −.84° in both; modeled foot self-contact is zero. This directly validates
that initialization can preserve the original supported pose.

**Zero residual has changed physical meaning.** The candidate's zero-action
stance settles at **−10.95° pitch**, versus −.84° for the current center.
Both stand with 100% double-foot support, but the candidate leans backward.
A separate native forward-kinematics check explains this: changing the hips
while keeping knee/ankle targets and the original body orientation gives about
**+9.69° sole pitch**. The free body must rotate to accommodate planted feet.
That hypothetical target pose is not a replacement of the actual reset state.

A whole-model-home substitution would also move knee centers from about 40.2°
to 13.8°. The existing grounded knee angle then exceeds its ±14.3° residual
window. Therefore wholesale home replacement is not validated by this test.

No hip-pitch mechanical-limit violation occurs in any probe. The largest
all-joint overshoots are at the right hip-roll soft stop: .78° in the current
.20 policy versus .52° in the candidate. These existing roll-stop transients
must not be reported as evidence of a hip-pitch stop blocking the new window.
Largest logged final-substep walking torques are 2.92Nm/current and
3.25Nm/candidate at .20; these samples are not a per-control-step torque peak.

## Frozen policy response

Requested-direction speeds and pitch use integrated root poses after warmup.
Negative pitch is backward lean.

| TB command | Current speed | Candidate speed | Current pitch | Candidate pitch |
|---|---:|---:|---:|---:|
| .10 | .0856 | .0665 | +2.28° | −9.36° |
| .15 | .1380 | .1047 | +4.09° | −7.43° |
| .20 | .1809 | .1499 | +7.15° | −4.83° |

Speed falls 17–24%. At .20, right-hip negative-bound actions fall 92.8%→78.5%,
but its delivered target still spans only **.72°**, versus .48° for the current
policy. The nearly fixed right-hip strategy remains present. Better reference
coverage alone has not produced a coordinated right-leg swing in these frozen
policy probes.

The existing policy's zero-command mean body pitch changes **−1.97°→−12.91°**.
Both policies make 25 geometric landings per leg during this nominal
zero-command episode; the constant holding probes make none. This behavior
is already present in the control policy and is not established as a WR2-only
difference from TB. TB Paper’s `get_phase_signal(time)` also keeps its clock
running at zero command; standing behavior must be learned by the actor.
Passing the holding check must not be interpreted as the
restored policy itself holding that pose. The normalized 3/6/3/5/3-second
transition case completes in both origins, with walking-window speed
**.1772/current versus .1459/candidate TB m/s**.

Candidate .20 roll span decreases 15.37°→12.33° and normalized root-height
span 11.10→10.63mm, alongside lower executed speed. These are not evidence of
improved naturalness at equal achieved speed. Modeled foot self-contact rises
3.94%→8.40% at .10 and .26%→3.67% at .15; both .20 nominal probes have zero.
The complete capture includes native cached landing geometry. At .20,
mean left/right landing leads change **9.87/2.50→8.51/1.94 TB cm**, and
same-foot strides change **13.14/13.13→10.76/10.82 TB cm**. The right landing
lead remains much smaller; at .10 it worsens **−.51→−2.91 TB cm**, placing
the right foot behind the left at landing. These measurements require supported
air, more than 1 mm sole clearance and ten sustained contact samples. This
fixed-center remap has not fixed alternating placement.

The earlier static checkpoint audit also found about 27% of current left-hip
requests outside the proposed narrower anterior window. Exact preservation
of the old policy's physical targets is impossible for those requests. Loading
the same weights is a usable perturbed starting policy, not a behavior-preserving
checkpoint conversion. A continuation must state this initialization change.

## Validation and next action

All fourteen traces were checked for finite values, validity/fall masks, 824-tick
completion, zero target reconstruction error, matched actual reset states and
the holding-control equivalence. Three focused unit tests pass: named hip
indices/holding-pose representability; opposite signs/quantized bias coverage;
and finite terminal sample retention with padding excluded from summaries.

Complete Mac results are at
`results/wr2_walking/wr2_paper_goal_xy_20261026_193545_seed3/hip_action_validation_mac_complete/`:
`summary.json`, `coverage.json`, 14 NPZ traces, `review.json` and
`kinematics.json`.
The capture records clean commit `294178b`; producer source, checkpoint and
model hashes were verified. The earlier ten-case capture remains in
`hip_action_validation_mac/` and corresponds to the initial diagnostic source
committed as `0ccc923`. Production physics/reward/model files were unchanged.

The larger native GPU check is ready. It uses matching 128-episode draws per
seed with observation/model/reset/actuator variation, and the actual GPU
runtime. Its 7 cases×2 origins×2 seeds×128 draws cover 3,584 episodes.
It measures whether the candidate's changed startup/feedback causes
additional failures; it is not a training-benefit test. Inspect those results
before choosing the 200M experiment's initialization. Keep the qualified current
policy as the control and retain intermediate training checkpoints.

From the updated GPU WR2 checkout:

```bash
uv run --locked --python 3.10 --extra training python -m wr2.tools.validate_hip_action \
  --run-dir results/wr2_walking/wr2_paper_goal_xy_20261026_193545_seed3 \
  --checkpoint results/wr2_walking/wr2_paper_goal_xy_20261026_193545_seed3/checkpoints/000200294400 \
  --num-envs 128 --seeds 12102 22102 \
  --output results/wr2_walking/wr2_paper_goal_xy_20261026_193545_seed3/hip_action_validation_gpu
```

Copy it with the existing results helper:

```bash
./scripts/scp_from_remote.sh --linux-pc --results \
  wr2_walking/wr2_paper_goal_xy_20261026_193545_seed3/hip_action_validation_gpu
```

References: local TB **`paper-walking-baseline`** `mjx_env.py` (model-home
default action), [TB paper v2](https://arxiv.org/html/2502.00893v2),
[final XY confirmation](wr2_goal_xy_final_confirmation_20261027.md) and
[earlier action-space review](wr2_goal_xy_result_20261027.md).
