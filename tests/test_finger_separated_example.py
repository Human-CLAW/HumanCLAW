"""Exercise the public finger-separated single-episode example."""

from __future__ import annotations

import json
import os
import subprocess
import sys

from humanclaw_bench.paths import repository_root


def test_finger_separated_example_dry_run_builds_the_public_command(tmp_path) -> None:
    """Require the shipped example to validate inputs without starting Habitat."""

    root = repository_root()
    script = root / "examples" / "run_finger_separated_episode.py"
    model = tmp_path / "model.json"
    scene = tmp_path / "hssd-hab.scene_dataset_config.json"
    output = tmp_path / "output"
    model.write_text("{}\n", encoding="utf-8")
    scene.write_text("{}\n", encoding="utf-8")
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(root / "src")
    completed = subprocess.run(
        [
            sys.executable,
            str(script),
            "--model-config",
            str(model),
            "--scene-dataset-config",
            str(scene),
            "--output",
            str(output),
            "--gpus",
            "7",
            "--dry-run",
        ],
        cwd=root,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    report = json.loads(completed.stdout)
    assert report["agent_asset"] == "finger-separated"
    assert report["urdf"].endswith("neutral_beta0_finger_separated/neutral_beta0.urdf")
    assert report["shift"].endswith("neutral_beta0_finger_separated/shift.npy")
    assert report["argv"][report["argv"].index("--agent-asset") + 1] == (
        "finger-separated"
    )
    assert report["argv"][report["argv"].index("--gpus") + 1] == "7"
    assert str(model) in report["argv"]
    assert str(scene) in report["argv"]
