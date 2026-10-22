# WR2 hovering foot and short contacts — 2026-10-20

**Update 2026-10-21:** the [GPU confirmation and causal feedback review](wr2_policy_feedback_review_20261021.md)
reproduces the pattern and identifies a learned past-action feedback loop.
The [acquisition audit](wr2_tb_acquisition_root_review_20261021.md) defers
training modifications until the initiating cause is identified. This document
retains the original CPU/CAD observations.

## Revised interpretation

The user's visual evaluation identifies a real gait-quality problem in the
forward recording provided by the viewer command. **WR2 mostly keeps the
right foot unloaded and makes short, tiny contacts. It has not learned TB's
planted stance and forward step.** The earlier 24.5% nominal right-contact
fraction describes collider contacts, not normal steps.

This qualifies the [collision result](wr2_collision_result_20261020.md).
Contact avoidance and aggregate tracking gains remain measured improvements,
but a 200M continuation should first be preceded by matching GPU planting
diagnostics. The visual recording was generated on CPU.

## 1. Contact duration and foot placement

Compare the new WR2 selected 300M forward trace and copied TB Paper 200M
forward trace, command +.10 TB m/s, seed 22102, after 1.5 TB seconds. Both use
native floor force above 1 N. No reference actions replace the learned policy.
Times divide by `s=.8240547565` for WR2; distances divide by `s²`.

| Right-foot measure | WR2 | TB Paper |
| --- | ---: | ---: |
| Fraction of ticks unloaded | **75.5%** | 50.2% |
| Contact onsets in the steady recording | **93** | 27 |
| Contact-onset rate, TB Hz | **5.03** | 1.46 |
| Median contact duration, TB s | **.049** | .340 |
| Median interval between contacts, TB s | **.194** | .700 |
| Median forward box-center advance between contacts, TB mm | **2.95** | 63.53 |
| Configured reference cycle, TB s | .728 | .720 |

WR2's right contact intervals last **.02–.06 physical seconds**, usually two
control ticks. Both legs repeat contacts at approximately 6.1 physical Hz;
the left stays loaded longer, while the right mostly remains up. These are
brief taps and millimeter-scale placement changes, not a conventional stride.

The placement statistic tracks a fixed collider-center point on the foot at
successive contact onsets. It is not an independently qualified stride count.
The unloaded visible foot's median floor clearance is similar in the two
robots: **5.95 mm WR2 /6.59 mm TB**, in physical units. Excessive lift height
does not explain the main difference; duration and horizontal placement do.

## 2. The policy requests the fast oscillation

Recorded actor phase sine/cosine agrees with the configured .6-second WR2
clock, maximum component discrepancy **7.55e−6**. The supplied phase frequency
is **1.67 physical Hz**. A missing or incorrectly scaled phase input has not
been found.

After warmup, the dominant frequency of requested hip-pitch, knee and
ankle-pitch motion is approximately **6.1 physical Hz**. The delivered targets
and right ankle-pitch motion retain that frequency; right-foot height and
torso height also have a dominant component near 6.1 Hz. Some achieved hip
motions retain a weaker response near the reference frequency.

Thus the fast tapping is present in the **policy's requested commands**. It
is not explained solely by the renderer or passive contact-force chatter.
Its exact origin in learned feedback, controller response and objective
tradeoffs remains unisolated. Matching GPU confirmation is the next gate.

## 3. Why the hovering appearance can coexist with reported contact

The viewer reconstructs visible foot meshes from the recorded contact-time
collider/body frames. Projecting compiled CAD vertices onto the floor gives:

| Median height during force-classified contact | WR2 left | WR2 right | TB right |
| --- | ---: | ---: | ---: |
| Lowest visible CAD surface, physical mm | −.52 | **+.68** | −1.37 |

WR2's right visible CAD surface is above the floor on **98.4%** of its loaded
samples. Its right box bottom is nevertheless typically **1.56 mm below the
floor**, and mean force during those intervals is approximately **20.1 N**.
Independent native `mj_geomDistance` mesh/floor queries confirm the positive
CAD clearance at checked loaded frames.

The foot box is generated from the CAD parts' axis-aligned body-frame bounds
in `_replace_foot_collisions`. When tilted, bounding-box corners can extend
beyond the CAD surface. Consequently, collider support can occur while the
visible foot remains slightly above the floor. This is a contact-geometry
approximation, not evidence that a visible sole has been planted. The intended
physical sole/pad and floor-contact representation need checking before a
geometry change; this observation does not prove that geometry caused the
fast policy oscillation.

## 4. The existing reward permits these contacts

Both WR2 and local **`paper-walking-baseline`** use accumulated air time paid
when contact returns, with coefficient **500**. The predicate uses current
or previous contact and positive old air time. It requires neither sustained
ground support nor forward foot placement. A foot held up forever would not
receive its landing credit; the brief contacts seen here allow it to collect
the accumulated duration.

The WR2 forward replay earns **163.40** air-time return, of which **122.80**
comes from the right foot and **40.60** from the left. XY tracking contributes
**39.88**. These are native weighted episode contributions, not proof that
air time alone caused this mode. Repeated contacts do not increase this term
merely by increasing their count: credit depends on accumulated unloaded
time. The measured contacts are eligible despite being poor steps.

TB uses the same rule and achieves much longer load intervals and larger
placement advances. Therefore the common formula alone does not explain the
robot/policy difference, and disabling air time is not an established fix.

## 5. Next action before another training run

1. **Confirm the current nominal gait on the training GPU** using the existing
   diagnostic below. Check contact durations, placement changes and requested
   frequencies, alongside normal XY tracking and survival.
2. **Validate actual floor-contact geometry and diagnose the fast command
   loop.** Distinguish collider-force events from visible/intended sole
   planting; compare requested versus delivered motion and its relationship
   to the reference clock. Preserve required foot-floor collisions.
3. **Choose one controlled correction from that result.** Retain collision
   weight 1.0, whose contact-avoidance effect is established. Treat sustained
   planting and useful foot placement as evaluation requirements. The prior
   additional-200M proposal is conditional on this check; continuing the same
   objective is not yet a demonstrated remedy for this tapping mode.

From the GPU's WR2 checkout:

```bash
uv run --locked --python 3.10 --extra training python -m wr2.tools.diagnose_walking \
  --paper-stance \
  --run-dir results/wr2_walking/wr2_paper_collision1_20261020_072851_seed3 \
  --output results/wr2_walking/wr2_paper_collision1_20261020_072851_seed3/stance_diagnostic_gpu
```

This uses the selected checkpoint, nominal parameters, deterministic inference
and the saved training runtime profile. Its learned trace can replace the CPU
trace in the same viewer. The command also produces the existing reference
interventions; those remain diagnostics rather than trained policies.

## Viewer update and evidence

The viewer now exposes **box floor contact**, **floor force**, and **time in
the current loaded interval**. Loaded duration resets when the force-based
contact label becomes false and uses TB time units. The foot-to-foot box gap
is explicitly labeled to distinguish it from floor clearance. These additions
improve interpretation; they do not change the policy or simulated physics.
Five viewer tests and both native macOS viewers pass after the update.

- [Contact duration and placement comparison](../../results/wr2_walking/wr2_paper_collision1_20261020_072851_seed3/review_20261020/nominal_cpu/foot_landings/comparison.json).
- [Phase/frequency and per-foot air-time audit](../../results/wr2_walking/wr2_paper_collision1_20261020_072851_seed3/review_20261020/nominal_cpu/foot_landings/oscillation.json).
- [CAD/box floor clearance, contacts and forces](../../results/wr2_walking/wr2_paper_collision1_20261020_072851_seed3/review_20261020/nominal_cpu/foot_landings/forward_floor_audit.npz).

WR2 observations come from the supplied nominal CPU trace; TB observations
come from its copied Paper GPU trace. Different native runtimes remain a
limitation, so these new frequency/planting conclusions need GPU reproduction
before changing training. Remote SSH was unavailable from this session.

Sources: [WR2 rewards](../../wr2/locomotion/paper_rewards.py),
[environment](../../wr2/locomotion/walking_env.py),
[foot collider generation](../../wr2/tools/post_process.py),
[viewer](../../wr2/tools/view_walking.py), local
`paper-walking-baseline:toddlerbot/locomotion/{mjx_env.py,walk_env.py,paper.gin}`,
[ToddlerBot, Tables VI–VII](https://arxiv.org/html/2502.00893v2), and
[MuJoCo collision geometry](https://mujoco.readthedocs.io/en/stable/computation/index.html#collision-detection).
