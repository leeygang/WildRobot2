"""Compatibility entry point for the consolidated ``bam_position`` plan."""

from __future__ import annotations

import argparse
import sys
from typing import Sequence

from wr2.tools.servo_sysid.campaign import main as campaign_main
from wr2.tools.servo_sysid.campaign import selected_conditions
from wr2.tools.servo_sysid.plan import load_plan


STAGES = load_plan("bam_position").conditions


def selected_stages(args: argparse.Namespace):
    return selected_conditions(args, load_plan("bam_position"))


def _translate(argv: Sequence[str]) -> list[str]:
    translated = ["--plan", "bam_position"]
    for value in argv:
        if value == "--run-all-safe":
            translated.append("--run-all")
        elif value == "--suite-dir":
            translated.append("--campaign-dir")
        elif value.startswith("--suite-dir="):
            translated.append(value.replace("--suite-dir=", "--campaign-dir=", 1))
        else:
            translated.append(value)
    return translated


def main(argv: Sequence[str] | None = None) -> int:
    print(
        "DEPRECATED: use `python -m wr2.tools.servo_sysid run "
        "--plan bam_position ...`.",
        file=sys.stderr,
    )
    return campaign_main(_translate(list(sys.argv[1:] if argv is None else argv)))


if __name__ == "__main__":
    raise SystemExit(main())
