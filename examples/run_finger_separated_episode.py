#!/usr/bin/env python3
"""Run one HumanClawBench episode with the optional finger-separated body."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path

from humanclaw_bench.assets import resolve_agent_asset
from humanclaw_bench.paths import repository_root


def _parser() -> argparse.ArgumentParser:
    """Build the small, user-facing argument parser for this example."""

    parser = argparse.ArgumentParser(
        description=(
            "Validate the optional finger-separated agent asset and run one "
            "episode with the public HumanClawBench CLI."
        )
    )
    parser.add_argument(
        "--model-config",
        default="my_model.json",
        help="configured VLM JSON (default: my_model.json)",
    )
    parser.add_argument(
        "--scene-dataset-config",
        default="data/humanclaw-hssd-val41/hssd-hab.scene_dataset_config.json",
        help="prepared HumanClaw HSSD scene-dataset JSON",
    )
    parser.add_argument(
        "--output",
        default="outputs/finger-separated-smoke",
        help="rollout output directory",
    )
    parser.add_argument(
        "--gpus",
        default="auto",
        help="evaluation GPUs, matching humanclaw-bench run --gpus",
    )
    parser.add_argument("--video", action="store_true", help="also save ego/exo MP4")
    parser.add_argument(
        "--metrics",
        action="store_true",
        help="also compute the benchmark metrics",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate inputs and print the command without starting Habitat",
    )
    return parser


def _input_file(value: str, root: Path) -> Path:
    """Resolve a user path from the current directory, then the release root."""

    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    from_current = (Path.cwd() / path).resolve()
    if from_current.is_file():
        return from_current
    return (root / path).resolve()


def main(argv: list[str] | None = None) -> int:
    """Validate prerequisites and launch the standard one-episode command."""

    parser = _parser()
    args = parser.parse_args(argv)
    root = repository_root()
    model_config = _input_file(args.model_config, root)
    scene_config = _input_file(args.scene_dataset_config, root)
    if not model_config.is_file():
        parser.error(f"model config does not exist: {model_config}")
    if not scene_config.is_file():
        parser.error(f"scene dataset config does not exist: {scene_config}")
    try:
        model_value = json.loads(model_config.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        parser.error(f"model config is not valid JSON: {error}")
    if not isinstance(model_value, dict):
        parser.error(f"model config must contain a JSON object: {model_config}")

    urdf, shift = resolve_agent_asset("finger-separated")
    output = Path(args.output).expanduser()
    if not output.is_absolute():
        output = (Path.cwd() / output).resolve()
    command = [
        sys.executable,
        "-m",
        "humanclaw_bench",
        "run",
        "--episodes",
        "one",
        "--profile",
        "paper_fullval_v1",
        "--model-config",
        str(model_config),
        "--scene-dataset-config",
        str(scene_config),
        "--agent-asset",
        "finger-separated",
        "--gpus",
        args.gpus,
        "--output",
        str(output),
    ]
    if args.video:
        command.append("--video")
    if args.metrics:
        command.append("--metrics")

    report = {
        "agent_asset": "finger-separated",
        "urdf": str(urdf),
        "shift": str(shift),
        "model_config": str(model_config),
        "scene_dataset_config": str(scene_config),
        "command": shlex.join(command),
        "argv": command,
    }
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)
    if args.dry_run:
        return 0
    return subprocess.run(command, cwd=root, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
