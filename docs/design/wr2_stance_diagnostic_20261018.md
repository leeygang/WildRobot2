# WR2 stance diagnostic — 2026-10-18

**The paired nominal stance controls completed on the GPU.** All 21 episodes
are finite. Reference hip rolls eliminate self-contact, but worsen six-direction
command tracking **44.2%**. See the [GPU result and revised next plan](wr2_stance_result_20261018.md).
The [matching TB Paper 200M diagnostic](tb_stance_diagnostic_20261018.md) has also
completed. See the [paired result and CAD double check](wr2_tb_stance_result_20261018.md)
for the current priorities. The canonical
configuration retains the completed **0.28.5 / air-time weight 500** baseline;
the earlier weight-0 ablation is deferred.

## Recorded GPU command

This output directory is now complete. A repeat needs a fresh `--output`
directory because the tool refuses to overwrite `summary.json`.

```bash
uv run --python 3.10 --extra training python -m wr2.tools.diagnose_walking \
  --paper-stance \
  --run-dir results/wr2_walking/wr2_paper_20261018_121025_seed3 \
  --output results/wr2_walking/wr2_paper_20261018_121025_seed3/stance_diagnostic \
  --seeds 22102
```

The run folder must contain its saved `training_config.yaml` and
`best_params`, already verified to match **260,116,480 steps**. An optional
`--checkpoint` selects another saved parameter file or native Orbax checkpoint
directory. The tool requires a GPU by default, records the actual runtime,
and refuses to overwrite a completed `summary.json`.

Outputs are `summary.json`, `reference_geometry.json` and 21 compressed raw
traces. The reference audit covers **227 commands ×1,000 poses**, rather than
only the commands used in the policy controls. Run artifacts remain in the
ignored results directory.

## What the controls test

| Control | Action sent through WR2's existing controller |
| --- | --- |
| `learned` | Selected policy unchanged |
| `hip_roll_reference` | Replace only left/right hip-roll action coordinates with the current walking reference |
| `leg_reference` | Replace all 12 leg action coordinates with the current walking reference |

Reference angles are converted to bounded actions using
`(q_ref − q_home) / .25`, with fixed home. The normal one-tick delay,
quantization, joint limits, target slew and servo physics apply afterward.
Unreachable reference angles remain bounded; their frequency is reported.
All floor/sole and foot-to-foot collisions stay enabled.

Each control uses the same nominal reset keys and seven fixed commands:
forward/backward .10, forward/backward .05, left/right .05 and standing, in
TB-equivalent m/s. All use nominal model/controller parameters, no reset
variation, observation noise or physical pushes. The configuration records
zero joint/velocity reset noise. Commands are held for **824 ×20 ms**, about
**20 TB-equivalent seconds**. The learned actor responds to each control's
new observations, so the hip intervention can also change its other requested
actions through feedback.

The prescribed controls are diagnostic interventions. Their survival is not
a learned-policy benchmark, and an open-loop reference failure would not
prove that the robot cannot learn walking.

## Measurements and accounting

- **Command tracking:** mean per-tick `‖v_xy − command_xy‖ / s`, torso-frame
  mean velocity and its standard deviation. `s=.824055`.
- **Net translation:** integrated root displacement in the commanded heading
  frame, divided by `s²` for TB-equivalent distance. Net velocity divides
  physical displacement by physical duration and by `s`. Report cross-direction
  drift alongside commanded-axis progress.
- **Stance:** actual single/double/no support, conditional double-support
  matching, signed sole separation, self-contact force, floor vertical force
  and geometric sole clearance.
- **Joint execution:** original actor request, amended current request,
  delayed request, delivered target, actual angle, reference angle, clipping
  and slew per joint. Current-reference errors preserve the production
  reference timing; delivered-to-actual errors preserve the actuator delay.

Support uses the production sum of upward floor/foot forces and its **1 N**
threshold. Self-contact requires penetrating foot-to-foot contact with active
constraint force magnitude **>.1 N**. Signed distance uses the traced MJX
geometry frames from the last physics substep, matching the contact/solver
sampling instant. Re-forwarding integrated qpos can produce a different instant.

A finite terminal tick contributes once, then the episode is frozen. Nonfinite
ticks and later padding contribute no measurements. Double-support matching
uses only valid ticks requesting double support; it is null if none exist.
Air-time credit is the unweighted landing-duration sum, independent of weight.

## Completed CPU diagnostic: findings and limits

The complete 21-episode check is saved in
[stance_diagnostic_cpu_verified/summary.json](../../results/wr2_walking/wr2_paper_20261018_121025_seed3/stance_diagnostic_cpu_verified/summary.json).
All episodes are finite and reach 824 ticks without a height failure.
Reconstructed delivered targets match production targets exactly.

| Control | Forward net speed | Forward cross-direction drift | Backward net speed | Six-direction mean tracking error |
| --- | ---: | ---: | ---: | ---: |
| Learned | .00559 | −.00015 | −.00214 | .06895 |
| Reference hip rolls | .02490 | −.02008 | −.03132 | .10006 |
| Full leg reference | .11521 | +.01040 | −.01152 | .21962 |

Speeds/errors are TB-equivalent m/s. Forward/backward commands in this table
are ±.10. The six-direction score uses the same six constant-command episodes
for all controls, excluding standing. It is a nominal diagnostic score;
the full randomized Paper benchmark is a different evaluation population.

The learned control has foot self-contact on **91.1%** of six-direction ticks
on average. Both reference interventions have **zero**. In the forward
episode, hip-roll clipping falls from **29.2% left /99.9% right** to zero, and
delivered-to-actual RMSE falls from **.112/.076 rad** to **.017/.018 rad**.
Minimum signed separation changes from **−.86 mm** to **+9.78 mm**.

These changes support a contribution from the learned hip-roll stance and
its contact constraints. However, forcing reference hip rolls increases
oscillation and drift, and **worsens command tracking by 45.1%** across the
six directions. More net propulsion alone is insufficient. Actual support
timing also remains poorly matched to the reference. Neither intervention
is ready to become the production action rule.

The full kinematic WR2 reference audit finds **32/227 command rows** with
overlapping soles, up to **2.35 mm**, all at ±.10 lateral speed in TB units.
The overlap also occurs in double support. All nominal reference angles
respect physical joint limits, but **8.30%** of uniformly counted table
joint samples exceed the home-centered .25-rad action range. This statistic
is not weighted by the training command distribution. Hip-roll references
stay inside that action range; knee and ankle pitch account for most excess.
No reference row falls below the existing body-spacing reward minimum.

### TB Paper comparison confirmed from code and assets

Source: local **`paper-walking-baseline` /a4f693d**, with the completed Paper
run's **1982ef3** runtime. Model, scene, config, collision and lookup hashes
match the completed run's `provenance.json`. Its action config also uses
fixed-home **.25 rad**, one-tick delay, **20 ms** control and **.72 s** cycle.

TB's cached teacher also exceeds that action range: nominal forward .10
has approximately **27–30%** knee/hip-pitch reference samples outside it,
and maximum motor excursion is **1.064 rad**. Frozen TB motor and passive
mapping were used for an independent native relative-foot geometry audit.
Both active ankle/foot pairs are clear for the six cardinal benchmark rows;
at ±.10 lateral commands the main pair overlaps by **6.70 mm**, and the
second pair by up to **15.06 mm**. These are cached reference poses, not
measured TB policy contacts. Float32 NumPy/JAX nearest-command selections
agree; actual selected lookup commands are recorded.

Artifacts: [TB reference action reach](../../results/wr2_walking/wr2_paper_20261018_121025_seed3/review_20261018/tb_reference_action_reach.json)
and [TB reference geometry](../../results/wr2_walking/wr2_paper_20261018_121025_seed3/review_20261018/tb_reference_geometry.json).
Reference overlap and imperfect action reach therefore do not uniquely
identify a WR2 training bug. TB's moving joint reference is a soft imitation
objective, and its successful policy can deviate from it.

TB Paper 200M full-benchmark walking error **.037025**, versus WR2 selected
260M **.076838**, remains the parity comparison. The missing measurement is
TB's learned-policy joint execution and support/contact trace under these
same nominal controls. That review used native MuJoCo kinematics and cached
references only. Preparation subsequently found the installed Paper environment
and verified the saved policy in a 21-episode, 100-tick CPU check; the full matching
GPU measurement is ready in the [new guide](tb_stance_diagnostic_20261018.md).

## Following decision and matching TB task

The GPU confirms the stance contribution and the tracking regression under
reference hip rolls. Obtain the matching TB learned-policy traces before
changing the acquisition recipe. Compare
delivered-to-actual joint tracking and phase, clearance and drift, rather
than only each robot's aggregate reference-imitation loss.

A geometry correction must account for actual sole shape and reachable
stance. The previously proposed global **70 mm** body-spacing floor also
penalizes some clear rightward reference poses. Evaluate that tradeoff and
the full lookup coverage explicitly before adopting a bound. The present
evidence supports investigating stance/coordination; it does not validate
hard hip-roll overrides, copying TB actuator constants, widening the action
range or removing the shared air-time term.

The following scope is implemented by the prepared matching TB diagnostic:

> Prepare matching nominal stance diagnostics from `paper-walking-baseline`
> (a4f693d), using the native Paper Brax fork and saved Paper configuration.
> Protect the active development branch; use an isolated worktree if needed.
> Evaluate the 200M policy in `results/toddlerbot_walk_ppo_20261015_194347`,
> retaining its 180M survival checkpoint. Compare learned actions, reference
> hip-roll actions and full leg-reference actions through the same fixed-home,
> bounded action and delayed controller pipeline. Use the same seven fixed
> commands, paired seed 22102, nominal parameters/no noise/no pushes and
> 1,000 control ticks. Record net commanded-frame displacement, torso velocity
> mean/std and per-tick XY error; requested/delivered/actual/ref joint angles,
> clipping, support/conditional double-support match and signed clearance/force
> for both active left/right ankle collider pairs. Stop accounting after the
> finite terminal tick and exclude nonfinite samples. Preserve saved source/
> model/lookup/runtime provenance. Use the WR2 diagnostic guide's metric units
> and report raw traces plus summaries. Do not change TB training rewards or
> physics as part of this measurement task.

Once a correction has evidence, use a fresh **200–300M TB-aligned run**, retain
intermediate checkpoints and compare both evaluation seeds against the
stationary initialization, completed WR2 control and TB Paper. Increased
spacing or survival alone does not establish walking acquisition.

## Verification and references

**26 targeted tests passed**, including real MJX action-delay/target execution,
signed distance with rotated/penetrating boxes, reference time-zero substitution
and fall/nonfinite/conditional-support accounting. Three metric tests were
rerun after adding net displacement. The full nominal diagnostic completed
with no nonfinite samples and zero target reconstruction error. CPU execution
used the existing Mac Python 3.11/JAX 0.6.2 environment and read-only Brax
0.14.1 sources; installed Brax distribution metadata remains 0.14.2. GPU
Python 3.10 uses the established locked Brax 0.14.1/Flax 0.10.7 profile.

Sources: [ToddlerBot paper, Tables VI/VII](https://arxiv.org/html/2502.00893v2),
frozen local TB `walk_env.py`, `mjx_env.py`, `motion_ref.py`, `walk_zmp_ref.py`,
`sim/robot.py`, `paper.gin`, scenes and cached lookup; WR2
[diagnostic](../../wr2/tools/diagnose_walking.py),
[environment](../../wr2/locomotion/walking_env.py),
[reference](../../wr2/reference/paper_walk.py),
[rewards](../../wr2/locomotion/paper_rewards.py) and
[training setup audit](wr2_tb_paper_setup_audit_20261018.md).
See [MuJoCo signed geometry distance](https://mujoco.readthedocs.io/en/stable/APIreference/APIfunctions.html#mj-geomdistance)
and [collision detection](https://mujoco.readthedocs.io/en/stable/computation/index.html#collision-detection).
