# First right step and subsequent short placements — 2026-10-23

Follow-up to the [completed qualified-step run](wr2_step_reward_result_20261023.md),
prompted by the user's visual evaluation of the selected 120M recording.

## Confirmed pattern

The first right swing is substantially larger than the subsequent swings.
Measure forward displacement of the recorded foot collider center from the
first unloaded control sample to its next loaded sample. These are world
displacements at 20 ms contact sampling, excluding the preceding stance.

| Swing | Actual WR2 distance | TB-equivalent distance |
| --- | ---: | ---: |
| First right swing | **56.30 mm** | 82.91 mm |
| Later left swings, median after 1.5 TB seconds | **25.68 mm** | 37.81 mm |
| Later right swings, median after 1.5 TB seconds | **24.55 mm** | 36.15 mm |

The earlier **38-mm** result described same-foot landing-to-landing advance
in **TB-normalized dimensions**, about **26 physical mm** on WR2. Median
preceding-stance center movement is only .47 mm left /1.25 mm right. Thus the
earlier statistic is consistent with the viewer's short placements; most
advance occurs while unloaded. It does not establish normal forward strides.

TB's median swing advances are 61.92/62.46 native mm. Using the existing
length normalization `s²=.67906624`, those correspond to approximately
**42.04/42.41 physical WR2 mm**, substantially larger than the new policy's
continued swings. Time divides by `s=.8240547565`, as in the main review.

## The stronger defect: the left foot keeps trailing

Project foot-center separation at each landing onto the torso's forward
direction. The selected nominal WR2 policy has alternating support but a
persistent right-foot lead:

- Left lands **48.79 physical mm behind** the right, median; **0/25** landings
  after warmup pass it.
- Right lands **72.41 physical mm ahead** of the left, median; **25/25** lead.
- TB left/right land **30.62/33.66 native mm ahead** of the other foot,
  approximately **20.79/22.86 physical WR2 mm** after scaling. Left leads on
  25/27 landings and right on 27/27.

The existing WR2 full-leg reference recording lands both feet ahead on about
96% of landings, with approximately **31 physical mm** median lead. Its large
lateral sway remains a limitation. This existing prescribed control establishes
that passing steps are achievable in the current model; no new intervention
was run for this follow-up.

## Why the first step looks different

Right hip pitch changes from about **16.6° at reset to 29.8° at its first
landing**, then oscillates around 28.1°. Left hip pitch settles from −16.6°
to approximately −4.2°. The large first placement includes an initial posture
change, rather than demonstrating sustained stride amplitude.

After warmup, the left hip barely swings:

| Left hip-pitch peak-to-peak range | WR2 120M | TB Paper 200M |
| --- | ---: | ---: |
| Actor-requested | **1.65°** | 14.01° |
| Delivered target | **1.68°** | 14.01° |
| Actual joint motion | **.31°** | 14.65° |

The native WR2 reference has **21.17°** of left hip-pitch range. The policy's
request is already small, and execution attenuates it further. This joint has
zero target clipping or slew limiting in the selected nominal recording. Left
knee/ankle motion lifts the foot while hip pitch remains almost fixed.

Requested left hip range shrinks **3.34° at 20M →2.50° at 80M →1.65° at
120M →.44° at 300M**. This is a learned motion change visible in the policy
requests, with further attenuation in the actual motion. The present data do
not fully separate the incentive that reduces requests from physical response
that may affect which requests training favors.

## Consequence for the next investigation

Sustained alternating contact is verified. Normal alternating forward placement
is still missing. The most specific observed defect is **loss of left forward
hip swing and a persistent right-leading posture**. The qualified air rule pays
supported unloading and sustained landing without requiring the left foot to
pass the right. This remains consistent with the reward competition documented
in the main review; it does not prove the exact PPO update responsible.

The [completed hip-pitch incentive analysis](wr2_hip_pitch_incentive_review_20261023.md)
explains shrinking requests through backward mean drift and tanh compression.
Matched-state diagnostics measure why larger isolated left motion can gain
translation yet lose reward through fore-aft/yaw oscillation and air credit.
Hip/knee/ankle phase coordination is the next analysis priority.

## Source and measurement correction

Read current WR2 `paper_reward_terms`, `_paper_step`, reference and contact
measurement code, plus local **`paper-walking-baseline` at `a4f693d`**:
`walk_env.py`, `mjx_env.py`, and the original TB model. The published
[ToddlerBot Table VII](https://arxiv.org/html/2502.00893v2) and local source
retain air 500, XY 5 and leg imitation 5. The native air formula contains no
forward placement requirement; WR2's qualification treatment also has none.

TB geometry axes are **time /collision pair /pair endpoint /XYZ**. The local
analysis previously averaged pair endpoints, mixing left/right geometry. It
now selects the actual sole-box pair, `ank_roll_link_collision` and
`ank_roll_link_2_collision`. TB's earlier right landing-to-landing advance
changes slightly, **63.67 →63.53 mm**. Tracking, contact counts and viewer
playback are unaffected. The corrected geometry is necessary for the new
landing-separation comparison.

[Per-landing figure](../../results/wr2_walking/wr2_paper_step_20261022_143550_seed3/review_20261023/landing_comparison.png)
and [event data](../../results/wr2_walking/wr2_paper_step_20261022_143550_seed3/review_20261023/landings.json)
include startup, stance/swing decomposition, landing order and hip requests.
Reproducible analysis sources are beside these ignored artifacts. WR2 details
are from nominal Mac CPU recordings; TB is the existing native GPU recording.
