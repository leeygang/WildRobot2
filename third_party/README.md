# ToddlerBot walking baseline source

`toddlerbot/` contains the source and assets for the paper-aligned positive
control described in the [training guide](../docs/design/toddlerbot_paper_baseline.md).
Setup and training use this WR2 checkout.

| Source | Pinned version |
| --- | --- |
| ToddlerBot release | `v1.0.0`, `59832cde2be3ab029486958f2e5ba92604f87307` |
| Prepared ToddlerBot snapshot | Recorded in [SOURCE.json](toddlerbot/SOURCE.json) |
| Brax fork | `14147806f6c25724ba830ba36764df3243889b0d` |

Included: the TB Python package and training configuration, original `toddlerbot`
description, meshes referenced by its XML models, motion datasets, ZMP cache,
Brax runtime and packaging files, dependency lock and focused readiness tests.
Unrelated TB descriptions/CAD assemblies and upstream documentation/video assets
are omitted. Exported files retain their source bytes; the preparation changes
are recorded in [the review diff](../docs/design/toddlerbot_paper_baseline.patch).

The runtime records the WR2 Git commit and dirty status, plus the upstream pins.
The original ZMP cache SHA256 is recorded in `SOURCE.json`.

Upstream licenses are included at [TB LICENSE](toddlerbot/LICENSE) and
[Brax LICENSE](toddlerbot/toddlerbot/brax/LICENSE). Public references:
[ToddlerBot paper](https://arxiv.org/abs/2502.00893) and
[project](https://toddlerbot.github.io/).
