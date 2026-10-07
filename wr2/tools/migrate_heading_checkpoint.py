"""Explicit, action-preserving v3 -> v4 RSL checkpoint input migration."""

import argparse
from pathlib import Path

import torch

from wr2.locomotion.rsl_rl_training import expand_heading_inputs


def migrate_checkpoint(source: Path, output: Path) -> None:
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    state = torch.load(source, map_location="cpu", weights_only=False)
    sizes = {"actor.0.weight": 55, "critic.0.weight": 96}
    for name, size in sizes.items():
        state["model_state_dict"][name] = expand_heading_inputs(
            state["model_state_dict"][name], size
        )
    # Adam's first/second moments follow the identical input-column mapping.
    for values in state["optimizer_state_dict"]["state"].values():
        for name in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
            tensor = values.get(name)
            if (
                tensor is not None
                and tensor.ndim == 2
                and tensor.shape[1] in (825, 1440)
            ):
                values[name] = expand_heading_inputs(tensor, tensor.shape[1] // 15)
    state["observation_layout"] = "wr2_proprio_v4"
    state["migration_source"] = str(source.resolve())
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, output)
    print(
        f"Migrated actor 825->855, critic 1440->1470; preserved policy, optimizer, std and LR: {output}"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    migrate_checkpoint(args.checkpoint, args.output)


if __name__ == "__main__":
    main()
