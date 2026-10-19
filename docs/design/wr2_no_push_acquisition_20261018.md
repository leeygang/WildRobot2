# WR2 acquisition control without physical pushes — 2026-10-18

**Completed:** `wr2_paper_20261018_121025_seed3` reached **300,134,400
transitions**. Selected 260M has zero height failures on both evaluation seeds,
but error **.07684 TB-equivalent m/s** is **2.08× TB Paper 200M** and worse than
the stationary initial policy. See the
[result review and updated next plan](wr2_no_push_result_20261018.md).
The preparation and commands below preserve the completed experiment.

## Question and controlled change

Can WR2 learn contact-preserving, TB-like walking with its current actuator
model when acquisition starts without physical pushes? The
[actuator and disturbance review](wr2_actuator_gap_review_20261018.md) confirms
that the successful frozen TB Paper run configured root-velocity pushes but
did not apply them to MJX physics. WR2 applied them during training and
evaluation. The published recipe intends physical perturbations; this control
matches the local TB run's effective acquisition exposure.

The approved **0.28.5** experiment used the regular WR2 trainer and Paper
configuration, now archived in the completed run's `training_config.yaml`.
The canonical YAML prepares the [next air-time control](wr2_airtime_ablation_20261018.md).
Relative to completed **0.28.4**, its effective configuration changes only:

| Field | Completed control | Prepared control |
| --- | ---: | ---: |
| Linear push bound, each axis | .4120273783 m/s | 0 |
| Angular push bound, each axis | 1.2135115926 rad/s | 0 |
| Version / name | 0.28.4 / fresh Paper | 0.28.5 / acquisition without pushes |

Keep the **82-tick push interval** so the existing random-key split remains
active. Zeroing the amplitudes also zeros the six critic push features. Frozen
TB instead received random labels for its unrealized pushes; this is a declared
observation difference, so the run cannot isolate that label effect from the
change in physical training exposure.

Fresh initialization, **seed 3**, **300M requested transitions**, fixed
**LR 1e−4**, the 1024 ×20/four-pass PPO layout, network/distribution,
all 25 Paper reward coefficients, observation dimensions, reference, action
home, command timing, noise, model/reset/actuator variation and identified
actuator physics retain the completed control's values. No supported-swing
reward change or critic velocity correction is included.

## Start on the GPU host

From the WR2 checkout:

```bash
uv run --python 3.10 --extra training python -m wr2.locomotion.train \
  --config results/wr2_walking/wr2_paper_20261018_121025_seed3/training_config.yaml
```

Omit the restore option to initialize actor, critic, normalizer and Adam
fresh. The trainer requires a GPU, writes the effective configuration and
provenance, and retains the initial parameters plus every trained checkpoint.
Native rounding gives approximately **300,134,400 transitions**, with 15 trained
checkpoints about every 20M. The timestamped run retains the `wr2_paper` prefix
and seed 3; its version and physical-push metadata identify the experiment.

An optional GPU check exercises the actual training batch and both comparison
profiles:

```bash
uv run --python 3.10 --extra training python -m wr2.locomotion.train \
  --config results/wr2_walking/wr2_paper_20261018_121025_seed3/training_config.yaml \
  --num-timesteps 20480 --num-evals 2
```

Start the full command fresh after that check. A 20K run checks compilation,
update and reporting; it cannot establish learned walking quality.

## Paired evaluation and selection

At step zero and each retained checkpoint, use **15 conditions ×128 episodes**,
selection seed **12102**, deterministic policy and the same model/reset/noise
draws within the two profiles. Both use 824 control ticks at 20 ms, the same
normalized commands and transition schedule:

| Profile | Physical exposure | Role | Artifacts |
| --- | --- | --- | --- |
| Primary | No pushes, native variation/noise | Acquisition comparison and checkpoint selection | `paper_evaluations/<step>.json`, `paper_evaluation_history.jsonl` |
| Pushed | Each linear axis ±.412027 m/s, angular axis ±1.213512 rad/s; every 82 ticks from step zero | Robustness diagnostic | `paper_push_evaluations/<step>.json`, `paper_push_evaluation_history.jsonl` |

The pushed bounds come from frozen TB's configured ±.5 m/s / ±1 rad/s and
2-second interval, normalized by `s=.8240547565`. Overrides affect evaluator
copies, so they do not re-enable physical pushes in the training environment.
Both full reports and histories record whether pushes are physically enabled,
their amplitudes and interval. Existing per-episode measures and normalization
remain unchanged.

Select the lowest primary ten-condition mean instantaneous XY error among
trained checkpoints with a complete finite suite and **≤19 height failures
/1920**. Pushed scores do not choose `best_params`. Confirm the same selected
checkpoint under both profiles on seed **22102**, writing
`paper_confirmation_22102.json` and `paper_push_confirmation_22102.json`.
A nonfinite comparison remains an explicit error with the report and parameters
saved for investigation.

Zero height failures do not establish walking. Compare direction/sign response,
standing/stopping, foot support, forces and nominal geometric flight across the
full checkpoint curve. The native one-newton contact classifier alone cannot
prove a foot is geometrically airborne.

## Recheck saved checkpoints

The regular evaluator can explicitly choose physical exposure while retaining
model/reset/actuator variation and noise:

```bash
uv run --extra training python -m wr2.locomotion.evaluate \
  --config results/wr2_walking/<run>/training_config.yaml \
  --checkpoint results/wr2_walking/<run>/checkpoints/<step> \
  --paper-benchmark --no-pushes --num-envs 128 --seeds 12102 \
  --output results/wr2_walking/<run>/no_push_<step>.json
```

For its matched physical-push diagnostic, replace `--no-pushes` with `--pushes`
and use a different output filename. The latter restores the normalized TB
bounds even when the saved training amplitudes are zero. With neither option,
evaluation uses the saved configuration. Use seed 22102 for held-out checks.
`--nominal` additionally disables variation and noise and cannot be combined
with `--pushes`.

## Preparation verification

- `uv run --locked --extra training --extra dev python -m pytest -q
  tests/test_paper_training.py`: **4 passed**, including native PPO update/
  parameter-only restore, relative checkpoint loading, episode accounting,
  paired model/reset keys and preservation of the source environment.
- JSON-normalized effective configurations agree with completed seed-3 0.28.4
  after removing only the two push amplitudes and version/name metadata.
- Using native Brax **0.14.1 / JAX 0.6.2 / MJX 3.14**, both **128-episode ×824-tick
  forward profiles** reproduce the previously audited pair exactly: maximum
  difference in lengths, failures, invalid counts, sample counts and measurement
  sums is **zero**. This is a focused physical regression check, not a new full
  15-condition policy evaluation.
- The regular training entrypoint was exercised with three synthetic checkpoint
  callbacks. A lower pushed error cannot override the primary candidate;
  both histories and both held-out profiles retain the expected selected
  checkpoint. This checks reporting/selection flow, not physical learning.
- The ordinary train/evaluate CLI help checks pass. The local backend is CPU;
  the full GPU batch check and long acquisition run remain to be executed on
  the GPU host.

Raw validation programs, reports and logs are ignored local artifacts under
`results/wr2_walking/wr2_paper_20261017_195826_seed3/review_20261018/no_push_preparation/`,
including `profile_validation.json` and `callback_validation.json`.

## Comparison and interpretation

Use completed WR2 0.28.4 seed-3 **140M/220M/final** and retained WR2 **+180M**,
plus TB Paper **180M/200M**, as controls. The paired review already evaluated
WR2 220M and TB 200M under explicit exposure:

- WR2 220M without evaluation pushes: **.07658 TB-equivalent m/s**, zero height
  failures, but extensive neither-foot contact and nominal geometric flight.
- Frozen TB 200M with its native, physically absent pushes: **.03707 m/s** in
  the CPU reproduction, versus .03702 in the recorded GPU suite.
- Physically applying TB's configured pushes produces early height terminations
  in all 1920 episodes. Those censored errors are not a shared 20-second
  tracking target, and this policy was not physically trained for that exposure.

Substantial tracking and contact improvement would support disturbance exposure
as an acquisition problem. Persistent hopping would show that removing it is
insufficient. Either outcome leaves actuator-identification ambiguity unresolved.
Keep actual WR2/TB geometry, actuator models and native variation differences
explicit; a single matched WR2 training seed does not establish a general effect.

References: [ToddlerBot paper](https://arxiv.org/html/2502.00893v2), frozen local
TB Paper `1982ef3` / Brax `14147806`, and the source probes and paired audits in
the linked physical review. The active TB project is not part of the runtime
setup.
