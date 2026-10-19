# WR2 and TB Paper GPU stance results — 2026-10-18

## Conclusion and first priority

**The leading measured WR2 problem is its learned inward, contact-constrained
stance.** The matching TB Paper policy executes useful commanded movement,
without foot self-contact. WR2 barely translates, continually requests inward
hip angles, and accumulates most of its joint execution error at those hips.
The reference interventions demonstrate that stance affects execution; they
also demonstrate that prescribing reference angles does not supply the learned
coordination needed for walking.

The additional geometry audit changes the order of the proposed corrections.
**WR2's rectangular foot colliders contact outside the imported CAD surface.**
At every recorded learned-policy pose, both CAD parts of each foot remain
separated, despite box contact on 85–96% of ticks. First validate and correct
this contact approximation, preserving floor support and true foot collision,
before tuning a reward around the box contacts.

**Collider correction alone is not established as a walking solution.** When
only the two hips are placed at their delivered targets, the CAD convex hulls
also overlap on 99.9% of ticks. The policy requests an inward stance that a
CAD-fitted convex collider would constrain too. A corrected model still needs
policy adaptation and coordinated movement. The top priority is this stance
and contact mechanism; a global servo change or tighter reference imitation
has weaker support from these traces.

## 1. Data, source and metric verification

Inputs are the copied TB Paper **200,089,600-step** policy diagnostic and WR2's
selected **260,116,480-step** policy diagnostic: 21 traces per robot, comprising
learned, reference-hip and reference-leg controls at seven nominal commands.
Commands are forward/backward .10, forward/backward .05, left/right .05 and
standing in TB units. Both diagnostics use seed **22102**, deterministic
actions, zero initial yaw and no domain randomization, white observation noise
or pushes. TB retains its native deterministic backlash observation offset.

| Item | TB Paper | WR2 |
| --- | --- | --- |
| Diagnostic source | clean `2e32d8f` | clean `ac91a6a` |
| Verified environment/model source | `paper-walking-baseline`, `a4f693d`; training `1982ef3` | recorded source and current model/controller unchanged |
| Brax | native fork `1414780`, 0.10.5 | 0.14.1 |
| JAX/JAXlib | 0.6.2 | 0.6.2 |
| MuJoCo/MJX | 3.2.4 | 3.14.0 |
| Control period / gait cycle | .020 s / .720 s | .020 s / .600 s |
| Horizon | 1,000 ticks, 20 s | 824 ticks, 16.48 s = 19.9987 TB s |

The independent audit passes **5,063 checks across all 42 traces**:

- Checkpoint/config/provenance hashes match; five archived TB runtime files
  match the frozen Paper branch, as do all **12** model/reference asset hashes.
- Valid sample prefixes, finite termination, commands, clocks, actual joint
  positions and integrated root displacement reconstruct correctly.
- Delivered targets match the recorded native controller pipelines. Independent
  float64 reconstruction differs by at most **7.78e−8 rad**; the diagnostics'
  native reconstruction errors are exactly zero.
- Native floor support, spacing reward, landing air-time credit and summary
  metrics reconstruct. The largest summary reduction discrepancy is
  **2.18e−6**; this does not change the reported conclusions.
- Signed box distances reproduce to **1.25e−16 m** and their signs agree with
  an independent separating-axis test. CAD separation is checked separately
  below using all imported vertices, beyond the box-distance calculation.

XY error is the equal-episode mean of per-tick
`norm(torso_velocity_xy - command_xy)`. WR2 divides speeds by
**s=.8240547565**, distances by **s²=.679066…** and times by `s`.
Net velocity uses integrated root displacement in the initial commanded
heading frame. Standing is excluded from the six-direction aggregate.
Support is summed native floor/foot vertical force **>1 N**; foot self-contact
requires penetrating active constraints with force magnitude **>.1 N**.
Self-contact magnitude does not establish the direction of load transfer.

The robots retain different physics, actuator models and runtime versions.
These are nominal, single-reset measurements, without a multi-seed uncertainty
estimate. They complement the full training benchmark rather than replace it.

Artifacts: [raw audit](../../results/toddlerbot_walk_ppo_20261015_194347/review_20261018/stance_comparison/audit.json),
[recomputed comparison](../../results/toddlerbot_walk_ppo_20261015_194347/review_20261018/stance_comparison/comparison.json),
[audit source](../../results/toddlerbot_walk_ppo_20261015_194347/review_20261018/stance_comparison/analyze.py).

## 2. What the matching TB policy establishes

Six moving commands, speeds in TB-equivalent m/s:

| Control | XY error | Model foot self-contact | Height terminations /6 | Mean observed TB seconds |
| --- | ---: | ---: | ---: | ---: |
| **TB learned** | **.02555** | **0%** | **0** | 20.00 |
| WR2 learned | .06877 | 91.22% | 0 | 20.00 |
| TB reference hips | .21627 | 14.44% | 5 | 6.18 |
| WR2 reference hips | .09918 | 0% | 0 | 20.00 |
| TB reference legs | .43455 | 0% | 6 | .86 |
| WR2 reference legs | .21501 | 0% | 0 | 20.00 |

**The nominal learned-policy tracking gap is 2.69×.** The TB override errors
are over their shorter lifetimes, including the descent to termination; they
are not comparable 20-second walking scores. Their failures establish that
replacing a learned controller's actions with reference angles can destabilize
even the successful TB policy. All standing controls survive the full horizon.

| Learned forward .10 measurement | TB Paper | WR2 |
| --- | ---: | ---: |
| Per-tick XY error | .02381 | .09910 |
| Net forward / cross speed | +.08795 /−.02284 | +.00564 /+.00011 |
| Model foot self-contact | 0% | 95.87% |
| Actual–delivered left/right hip RMSE, rad | .03512 /.02655 | .11230 /.07546 |
| Actual–delivered other ten leg joints RMSE, rad | .02056 | .02734 |
| Hips' share of total leg execution squared error | 31.4% | **71.0%** |
| Actual–reference leg RMSE, rad | **.23478** | .14740 |
| Exact two-foot reference contact-state match | **17.6%** | 67.2% |
| Native per-foot contact reward, maximum 2 | .765 | 1.671 |

TB walks with **larger reference-angle error and poorer exact phase matching**.
Those measurements should not become the first optimization targets. Native
phase reward counts each matching foot separately; exact two-foot matching is
a distinct diagnostic. The other ten joints' execution RMSE differs by only
**1.33×**, compared with the much larger WR2 hip errors. This concentrates the
execution issue at the learned stance rather than demonstrating a uniform
servo deficiency.

WR2's mean delivered hip targets are **+6.20° /−9.98°**, while actual angles
average **+.90° /−5.67°**. Its right hip request hits the physical inward limit
on **99.9%** of ticks in every command, including standing. TB's learned hip
requests do not clip. With WR2's reference-hip intervention, hip execution
RMSE falls to **.01668 /.01792 rad**, but six-direction XY error increases
**44.2%**, with heading drift and worse support timing. This supports a
stance/load contribution and the need for learned coordination; it does not
justify prescribing the hips or copying TB servo constants.

The established full benchmark remains the parity reference:

| Checkpoint | Fixed walking XY error | Height failures /1,920 |
| --- | ---: | ---: |
| WR2 initialization | .07162 | 0 |
| WR2 selected 260M | .07684 | 0 |
| TB Paper retained 180M | .03838 | 3 |
| TB Paper selected 200M | .03702 | 8 |
| TB Paper final 300M | .03800 | 134 |

Its WR2/TB gap is **2.08×**. These evaluations include each recipe's native
variation and a larger condition/reset population. TB's final checkpoint is
less safe than its retained checkpoints; preserve and evaluate intermediates.

## 3. Double check: model foot contact versus CAD contact

WR2's [_replace_foot_collisions](../../wr2/tools/post_process.py) builds one
body-aligned box from each imported foot mesh's full vertex bounds. It therefore
fills the corners and other empty space inside those bounds. The boxes have
the intended current bounds; the approximation itself creates constraints
outside the CAD surface in the observed stance.

The CAD audit uses the **recorded final-substep foot geometry frames**, avoiding
the mismatch with end-of-tick integrated root/joint coordinates. It reconstructs
each foot body's frame and transforms both imported parts: the foot and its
bracket. It checks all four left/right CAD-part pairs at every learned-policy
tick, using native MuJoCo convex-hull distance.

Independently, the closest-point direction defines a separating plane. NumPy
projects **every imported vertex** onto that direction. The right mesh's
minimum projection exceeds the left mesh's maximum for every pair and tick.
Thus the imported CAD parts are separated, including their convex hulls. The
projection certificate agrees with native distances within **2.27e−11 m**.
Home-frame reconstruction agrees within **1e−14** and CAD bounds match the
existing boxes within **1e−8 m**.

| WR2 learned condition | Box contact ticks | CAD convex-hull overlap | Minimum CAD gap, physical mm |
| --- | ---: | ---: | ---: |
| Forward .10 | 95.87% | 0% | +2.177 |
| Backward .10 | 92.96% | 0% | +2.276 |
| Forward .05 | 94.17% | 0% | +1.569 |
| Backward .05 | 89.93% | 0% | +2.468 |
| Left .05 | 89.68% | 0% | +1.629 |
| Right .05 | 84.71% | 0% | +1.729 |
| Standing | 88.47% | 0% | +1.780 |

Every observed box-contact tick has all CAD-part hull pairs clear. This is a
geometric fact about the imported model, not a hardware surface measurement
or a simulation of what the robot would do with a different collider.

The second check replaces only the two hip angles in recorded end-of-tick
`qpos` with the delivered targets, preserving all other coordinates. The CAD
convex hulls then overlap on **823/824 ticks in each case**, with deepest hull
penetration **15.7–17.8 mm**. Convex-hull overlap does not prove overlap of a
concave mesh's triangles. It does show that a CAD-fitted convex collision model
would also block those target configurations. This is a kinematic check,
not a feasible trajectory or a prediction of improved walking.

### Size comparison correction

TB's main sole box is **49 ×42 ×120 mm**. Its lateral width at native home is
**42 mm**; the earlier actuator review used the 49 mm dimension for the lateral
comparison. WR2's local lateral width is **58.8 mm**, with a home world-Y
projection of about **59.2 mm**. WR2 therefore has about **1.41×** TB's projected
sole width despite its **.679** CoM-length ratio. It is not a uniformly scaled
TB contact geometry.

The spacing reward's minimum is **55.728 mm WR2**, versus **50 mm TB**. At home
orientation that minimum is below WR2's projected sole width, while TB's is
above its 42 mm main sole width. Body-origin separation cannot ensure surface
clearance under foot yaw, roll and stagger. This supports a robot-geometry
review, with a CAD-validated contact surface as its first gate.

Artifacts: [CAD audit](../../results/toddlerbot_walk_ppo_20261015_194347/review_20261018/stance_comparison/cad_check.json),
[CAD audit source](../../results/toddlerbot_walk_ppo_20261015_194347/review_20261018/stance_comparison/cad_check.py).

## 4. What the rewards explain, and what remains unproven

Source and saved configurations confirm that both recipes use fixed-home
**.25-rad actions**, one-tick delay, soft moving-reference leg MSE at **5**,
foot contact matching at **1**, spacing at **1**, collision at **.1**, and native
air-time at **500**. The published Paper Tables VI/VII confirm the main PPO
settings and these reward weights. WR2 adds measured quantization/slew and its
own plant; those require physical evidence before changing them.

- **Spacing permits the observed stance.** Both learned policies receive
  maximum spacing reward on every tick. WR2 receives it during box contact.
  The reward checks torso-frame foot-body Y separation, not a surface gap.
- **Air-time permits low-clearance unloading.** Both implementations accrue
  time without native floor support and credit it at filtered first landing.
  They do not require geometric swing clearance or exclude other-foot contact.
  In WR2 forward .10, **15.82 /16.18 s = 97.8%** of air-time credit is awarded
  on box-contact ticks; backward .10 is **81.5%**. TB's fraction is **zero**.
  These WR2 contacts are model box contacts, not demonstrated CAD contact.
- **Air-time is not a complete causal explanation.** All four .05 commands
  and standing have zero air-time credit in both robots, yet WR2 still requests
  the saturated inward stance. Credit/contact coincidence does not prove
  that removing air-time would yield translation.
- **No new target-delivery bug is demonstrated.** The native pipelines and
  metrics reconstruct. This audit does not validate every PPO/plant parameter.
  The geometry approximation is the newly verified implementation concern.

The resulting mechanism is explicit: an inward requested stance, early box
constraints, hip target error, low-clearance unloading that remains rewarded,
and little commanded translation. The traces and interventions support its
components. Their relative contribution to the full training gap still needs
a controlled geometry comparison and learning on the corrected model.

## 5. Focused next plan

1. **Validate a simple foot collider fitted to CAD.** Retain sole/floor contact,
   genuine foot self-collision, material/friction, mass and inertial properties.
   Check its separation against both the recorded learned poses and the full
   walking lookup. Require MJX compatibility, an affordable collision shape,
   correct floor-force/support metrics and the intended sole height. A raw
   full-resolution hull has about 1,961 vertices, so performance needs an
   explicit check when choosing the collider representation.

2. **Compare the fixed 260M policy on original and corrected geometry.** Use
   the same seven nominal commands/resets, and measure inward requests,
   delivered/actual hip error, true/model contact, floor support, net commanded
   progress, cross drift and height failures. This isolates the immediate
   geometry effect. If contact simply moves to the CAD boundary, report that
   result: clearing the box artifact is still a model correction, while stance
   learning remains unresolved. Inspect the spacing objective against actual
   surface clearance before choosing a stance penalty or bound.

3. **Use a fresh 200–300M TB-aligned training run after the geometry gate.**
   Keep the Paper PPO/action/reference recipe and air-time weight **500** for
   this isolated geometry treatment. Retain checkpoints about every 20M steps
   and evaluate with seeds **12102 and 22102**, initialization, the current
   WR2 control and TB 180M/200M. Success requires lower per-tick XY error,
   correct signed commanded translation and controlled drift, together with
   the height-failure gate. Better separation or survival alone is insufficient.
   If the stance persists, the next supported question is how the soft stance
   objective should reflect WR2's physical foot geometry; then examine loaded
   actuator response only where execution error persists without contact.

Defer hard reference actions, tighter phase/reference weights, wider physical
hip limits, global servo retuning and setting air-time to zero. The matching
successful TB controller and these intervention results do not support making
those the first corrections. The previously proposed global 70 mm body-spacing
floor also penalizes valid WR2 reference rows and cannot itself guarantee gap.

This review prepares the decision and evidence. Training source/configuration
has not been changed and no new training has been started.

## Figure and references

![Audited nominal stance comparison](../../results/toddlerbot_walk_ppo_20261015_194347/review_20261018/stance_comparison/stance_comparison.png)

[Standalone PDF](../../results/toddlerbot_walk_ppo_20261015_194347/review_20261018/stance_comparison/stance_comparison.pdf).
The traces and analysis artifacts are local ignored results, not Git assets.

- [ToddlerBot paper, Tables VI/VII](https://arxiv.org/html/2502.00893v2), checked
  directly against the frozen branch and saved run configurations.
- Frozen local TB Paper `mjx_env.py`, `walk_env.py`, `mjx_config.py`, `paper.gin`,
  `evaluate_xy.py`, controller/reference code and saved model assets; the active
  TB development checkout was read only.
- WR2 [environment](../../wr2/locomotion/walking_env.py),
  [Paper rewards](../../wr2/locomotion/paper_rewards.py),
  [model post-processing](../../wr2/tools/post_process.py),
  [diagnostic](../../wr2/tools/diagnose_walking.py),
  [earlier WR2 result](wr2_stance_result_20261018.md) and
  [setup comparison](wr2_tb_paper_setup_audit_20261018.md).
- MuJoCo [mesh assets](https://mujoco.readthedocs.io/en/stable/XMLreference.html#asset-mesh),
  [collision detection](https://mujoco.readthedocs.io/en/stable/computation/index.html#collision-detection)
  and [signed geometry distance](https://mujoco.readthedocs.io/en/stable/APIreference/APIfunctions.html#mj-geomdistance).
