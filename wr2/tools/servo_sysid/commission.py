"""Compatibility entry point for one repeated BAM position condition."""

from __future__ import annotations

import sys
from typing import Sequence

from wr2.tools.servo_sysid.analysis.position import (
    RepeatMetrics,
    summarize_capture,
    summarize_series,
)
from wr2.tools.servo_sysid.campaign import main as campaign_main


__all__ = ["RepeatMetrics", "summarize_capture", "summarize_series", "main"]


def _translate(argv: Sequence[str]) -> list[str]:
    translated = ["--plan", "bam_repeatability"]
    executing = "--execute" in argv
    for value in argv:
        if value == "--series-dir":
            translated.append("--campaign-dir")
        elif value.startswith("--series-dir="):
            translated.append(value.replace("--series-dir=", "--campaign-dir=", 1))
        else:
            translated.append(value)
    if executing and "--run-all" not in translated:
        translated.append("--run-all")
    return translated


def main(argv: Sequence[str] | None = None) -> int:
    print(
        "DEPRECATED: use `python -m wr2.tools.servo_sysid run "
        "--plan bam_repeatability ...`.",
        file=sys.stderr,
    )
    return campaign_main(_translate(list(sys.argv[1:] if argv is None else argv)))


if __name__ == "__main__":
    raise SystemExit(main())
