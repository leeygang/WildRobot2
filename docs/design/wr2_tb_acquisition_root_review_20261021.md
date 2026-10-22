# Why TB learns walking and WR2 does not: acquisition audit — 2026-10-21

**Update 2026-10-22:** the [reward-direction review](wr2_reward_review_20261022.md)
prioritizes step-credit validity and its competition with motion/stability
incentives. It repeats existing controls for full accounting and rescores
recorded contacts without changing training rewards or physical parameters.

## Current answer

**The complete causal explanation is not established.** The feedback diagnostic
identifies a mechanism sustaining the current WR2 gait. It does not establish
why training selected that gait instead of TB's successful walking strategy.
The proposed CAPS experiment was premature and is deferred. No new training
recipe, rollout intervention or physical-model experiment was run for this audit.

The current evidence shows an **early divergence in learned behavior and reward
allocation**, alongside an independently established difference in modeled
actuator response. It does not isolate physical response, identification error,
remaining implementation differences or training seed as the initiating cause.

## 1. The gap predates autonomous past-action instability

Reanalyze the saved primary forward +.10 TB m/s evaluations. Both use evaluation
seed 12102 and 128 episodes per condition. WR2 velocity is divided by
`s=.8240547565`. Native variation differs between the robots, so this measures
the actual recipe outcomes rather than isolating one causal variable.

| Training steps | WR2 mean forward velocity, TB m/s | TB Paper mean forward velocity, m/s |
| --- | ---: | ---: |
| 20M | **−.0510** | **+.0377** |
| 40M | **−.0640** | **+.0620** |
| 200M | **+.00537** | **+.0905** |

Existing nominal CPU traces independently show WR2 right-ankle request peaks
at **6.30 Hz at 20M**, **6.30 Hz at 40M**, and **6.04 Hz at 140M**. In the tested
static contexts, those early actors' pure past-action recurrence dampens
perturbations; the 240M/300M actors can oscillate autonomously.

Consequently, a local feedback spectral radius below one at 140M does **not**
mean that the actual 140M walking policy is free of the fast mode. Joint/body
feedback can sustain motion even when the isolated past-action subsystem is
stable. The previous 140M warm-start recommendation cannot be justified as a
healthy walking starting point.

## 2. Early reward allocation differs despite common coefficients

The native evaluation logs need a unit correction before comparison:

- TB logs individual weighted reward terms **before** multiplying by `.02`.
- WR2's `paper_reward/...` metrics **already include** that multiplication.
- Exclude `_std` entries from the sum of mean components.

After correction, component sums reproduce reported mean episode rewards
within `.001`. The table describes each run's own native mixed-command
evaluation; different resets, episode lengths and variation remain explicit.

| Steps | Run | Air-time share of total return | XY-tracking share |
| --- | --- | ---: | ---: |
| 20M | WR2 | **67.2%** | **4.95%** |
| 20M | TB Paper | **39.8%** | **20.4%** |
| 40M | WR2 | **67.0%** | **4.73%** |
| 40M | TB Paper | **36.9%** | **21.7%** |
| 200M | WR2 | 44.5% | 8.95% |
| 200M | TB Paper | 34.8% | 21.8% |

TB finds motion that collects tracking reward early. WR2's early strategy
collects substantial air reward while moving in the wrong commanded direction.
This establishes how the outcomes diverge. It does **not** prove why their
policy updates favor different strategies: reward trajectories, achievable
motion and action-to-motion sensitivities depend on the physical model.

Air coefficient 500 is common to both. Neither the formula nor the large
coefficient alone explains the difference. Also, the frozen-policy comparison
against a past-action clamp compares two poor WR2 walking trajectories; it is
not evidence that the objective rejects a good WR2 walking trajectory.

## 3. The independent physical difference is real in simulation

Existing scalar actuator results use matched amplitudes and normalized
frequencies, without learned policy or ground contacts. WR2 physical frequency
1.667 Hz maps to 1.373 TB Hz, near the two reference clocks.

| Ankle-pitch input | WR2 model | TB model |
| --- | ---: | ---: |
| .035 rad amplitude: achieved/requested amplitude | **.628** | **.955** |
| .035 rad amplitude: phase | **−48.1°** | **−8.50°** |
| .25 rad amplitude: achieved/requested amplitude | 1.020 | 1.002 |
| .25 rad amplitude: phase | **−15.9°** | approximately 0° |
| .25 rad amplitude: delivered-target/actual RMSE | **.0498 rad** | **.00644 rad** |

The WR2 model has much greater reflected armature and effective damping,
higher joint friction, a lower speed endpoint and a continuous motoring taper.
TB has different Dynamixel families, transmission mechanisms and a
constant-torque speed region. WR2 additionally limits host target changes.
The controllers and parameters were checked against actual source in the
[actuator review](wr2_actuator_gap_review_20261018.md).

Thus similar small exploratory targets produce different motion and phase in
simulation. This can change how readily the same PPO/reward recipe discovers
coordinated walking. **That learning explanation remains a hypothesis.** The
large-amplitude result also shows that WR2's model can execute substantial
motion at the intended frequency; it is not categorically unable to walk.

The 300M frozen actor retaining its frequency after physical interventions,
or reproducing it without physics, does not exclude physics as the cause of
its acquisition. Those tests hold the already learned weights fixed. The
original physical model can shape the policy during training, after which
that policy can retain the learned motion under other conditions. The same
limitation applies to the floor-geometry intervention.

### Hardware identification does not uniquely identify the relevant dynamics

Existing held-out WR2 captures support two parameterizations:

| Fit | Armature, kg m² | Additional delay | Mean held-out error |
| --- | ---: | ---: | ---: |
| Accepted | .020951 | 0 | .85460° |
| Alternate | .001152 | 20 ms | .86478° |

The second fit has **18.2 times less armature** and only **1.19%** greater
held-out angle error. The fixture has substantial rigid inertia, while
armature accounts for about 97.8% of the modeled WR2 ankle's local inertia.
Consequently, an ambiguity that is small on the fixture can have a large
whole-body simulation effect. Firmware delay and mechanical inertia change
the robot's dynamics differently.

Current captures therefore do not prove that `.020951` represents actual
reflected mechanical inertia. Nor do they establish the loaded torque-speed
curve at the policy's walking rates. This is the leading physical-model
uncertainty to resolve, rather than a reason to guess a smaller armature.

## 4. A confirmed reward implementation difference still needs interpretation

Source was reread from local **`paper-walking-baseline`**, revision `a4f693d`.
TB's global actuator IDs index its 12-value leg action as
`[4,5,6,7,8,9,10,11,11,11,11,11]`. Its smoothness regularizers omit four left-leg
actions and count right ankle pitch five times. WR2 uses all 12 evenly.

Accounting on the same saved physical WR2 action sequence, preserving the
native two-/three-tick reward-history convention, gives:

| WR2 trajectory | Current rate + acceleration return | Using TB's physical-joint multiplicities |
| --- | ---: | ---: |
| 20M forward | −.999 | −1.325 |
| 300M forward | −.185 | −.461 |

These are rescorings, not new rollouts or retraining. The difference is small
relative to the early air return, but its training-gradient effect has not
been measured. It is neither an established root cause nor a difference to
replicate blindly. The common coefficient `.05` does not mean the effective
regularizers are identical.

## 5. Work to complete before a training correction

Keep the existing training recipe while completing the causal audit:

1. **Explain the early acquisition difference.** Inspect motion, action
   execution and reward allocation before/at the first 20M checkpoint in both
   native recipes. The currently copied TB early evaluations contain aggregate
   outcomes; they do not expose its early actor/plant trajectories.
2. **Audit the physical identification against the actual training model.**
   Separate command timing/firmware response from inertia and validate the
   response regime used by walking. Existing captures cannot uniquely settle
   that distinction; a uniquely identified hardware model cannot be inferred
   from the two near-equivalent fits alone.
3. **Finish attribution of implementation differences.** Preserve the correct
   reward units and physical joint mapping. Distinguish known TB bugs,
   WR2 hardware requirements and actual optimization differences. No demonstrated
   PPO adapter bug has been identified.

Only after an initiating cause is supported should a correction be selected.
This audit does not authorize or recommend a new reward, CAPS loss, feedback
input removal, actuator parameter replacement or additional training run.

## Evidence and references

[Accounting audit and source script](../../results/wr2_walking/wr2_paper_collision1_20261020_072851_seed3/review_20261021/acquisition_audit.json)
reconstruct the numbers from existing artifacts. Its adjacent
`acquisition_audit.py` performs no simulation or policy update.

Source checks: WR2 [reward terms](../../wr2/locomotion/paper_rewards.py),
[environment](../../wr2/locomotion/walking_env.py),
[servo identification evidence](servo_model.md); local
`paper-walking-baseline:toddlerbot/locomotion/{mjx_env.py,walk_env.py}`,
`toddlerbot/sim/mujoco_control.py`, and the saved Paper configuration.
The four inspected native environment/controller/reference files match the
frozen diagnostic checkout byte for byte.

References: [ToddlerBot, Tables IV, VI–VII](https://arxiv.org/html/2502.00893v2),
[complete WR2/TB setup audit](wr2_tb_paper_setup_audit_20261018.md), and
[feedback mechanism evidence](wr2_policy_feedback_review_20261021.md).
