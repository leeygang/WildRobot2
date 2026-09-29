#!/usr/bin/env python3
"""Dump raw Onshape assembly data for diagnosing export topology issues."""

from __future__ import annotations

import argparse
import getpass
import json
import os
from pathlib import Path
from typing import Any

try:
    from dotenv import find_dotenv, load_dotenv
    from onshape_to_robot.config import Config
    from onshape_to_robot.onshape_api.client import Client
except ModuleNotFoundError as exc:
    raise SystemExit(
        "Missing onshape-to-robot. Run this script with:\n"
        "  uv run --no-project --with onshape-to-robot==1.8.3 "
        "python wr2/tools/dump_onshape_raw.py"
    ) from exc


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_DIR = REPO_ROOT / "wr2/descriptions/wr2/onshape_export"


def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {path}")


def ensure_onshape_credentials() -> None:
    """Prompt securely when Onshape credentials are not already configured."""
    os.environ.setdefault("ONSHAPE_API", "https://cad.onshape.com")

    if os.environ.get("ONSHAPE_SECRET_BEARER"):
        return

    if not os.environ.get("ONSHAPE_ACCESS_KEY"):
        os.environ["ONSHAPE_ACCESS_KEY"] = getpass.getpass(
            "Onshape access key: "
        ).strip()
    if not os.environ.get("ONSHAPE_SECRET_KEY"):
        os.environ["ONSHAPE_SECRET_KEY"] = getpass.getpass(
            "Onshape secret key: "
        ).strip()

    if not os.environ["ONSHAPE_ACCESS_KEY"] or not os.environ["ONSHAPE_SECRET_KEY"]:
        raise SystemExit("Both the Onshape access key and secret key are required.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Dump the raw Onshape assembly, feature, and mate-value responses."
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=DEFAULT_MODEL_DIR,
        help="Directory containing the onshape-to-robot config.json.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Dump destination (default: MODEL_DIR/onshape_raw).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_dir = args.model_dir.expanduser().resolve()
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else model_dir / "onshape_raw"
    )

    config_path = model_dir / "config.json"
    if not config_path.is_file():
        raise SystemExit(f"Config file not found: {config_path}")

    load_dotenv(find_dotenv(usecwd=True))
    ensure_onshape_credentials()
    config = Config(str(model_dir))
    client = Client(logging=False, creds=config.config_file)

    wmv = "v" if config.version_id else "w"
    workspace_or_version_id = config.version_id or config.workspace_id
    if workspace_or_version_id is None or config.element_id is None:
        raise SystemExit("The config must identify a workspace/version and element.")

    assembly = client.get_assembly(
        config.document_id,
        workspace_or_version_id,
        config.element_id,
        wmv=wmv,
        configuration=config.configuration,
    )
    microversion_id = assembly["rootAssembly"]["documentMicroversion"]
    features = client.get_features(
        config.document_id,
        microversion_id,
        config.element_id,
        wmv="m",
        configuration=config.configuration,
    )
    matevalues = client.matevalues(
        config.document_id,
        workspace_or_version_id,
        config.element_id,
        wmv=wmv,
        configuration=config.configuration,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "assembly.json", assembly)
    write_json(output_dir / "features.json", features)
    write_json(output_dir / "matevalues.json", matevalues)

    print("Done. The dump contains CAD metadata but no Onshape API keys.")


if __name__ == "__main__":
    main()
