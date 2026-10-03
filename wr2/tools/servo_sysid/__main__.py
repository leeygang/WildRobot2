"""Single command-line entry point for HTD-45H characterization."""

from __future__ import annotations

import sys
from typing import Callable, Sequence

from wr2.tools.servo_sysid import analyze, campaign, capture, fit
from wr2.tools.servo_sysid.analysis import hysteresis
from wr2.tools.servo_sysid.plan import available_plans, load_plan


def _usage() -> str:
    return """usage: python -m wr2.tools.servo_sysid COMMAND [ARGS]

commands:
  run          preflight or execute one versioned campaign plan
  capture      expert interface for one hardware condition
  fit          fit effective position-loop dynamics from captures
  hysteresis   summarize bidirectional BAM hysteresis captures
  report       gate legacy deployment campaigns and emit a specification
  list-plans   show installed campaign plans
"""


def _list_plans() -> int:
    for name in available_plans():
        plan = load_plan(name)
        print(f"{name}: {plan.description}")
        print(f"  setup={plan.setup}")
        print("  conditions=" + ", ".join(item.condition_id for item in plan.conditions))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    values = list(sys.argv[1:] if argv is None else argv)
    if not values or values[0] in {"-h", "--help"}:
        print(_usage())
        return 0
    command = values.pop(0)
    handlers: dict[str, Callable[[Sequence[str] | None], int]] = {
        "run": campaign.main,
        "capture": capture.main,
        "fit": fit.main,
        "hysteresis": hysteresis.main,
        "report": analyze.main,
    }
    if command == "list-plans":
        if values:
            raise SystemExit("list-plans does not accept arguments")
        return _list_plans()
    handler = handlers.get(command)
    if handler is None:
        raise SystemExit(f"unknown command {command!r}\n\n{_usage()}")
    return int(handler(values))


if __name__ == "__main__":
    raise SystemExit(main())
