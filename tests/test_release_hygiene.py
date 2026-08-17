import hashlib
import inspect
import json
import subprocess
import sys
from pathlib import Path

from humanclaw_bench.paths import repository_root

GENERATED_TOP_LEVEL = {"data", "outputs", "running", "weights"}


def _is_generated(root: Path, path: Path) -> bool:
    relative = path.relative_to(root)
    return bool(relative.parts and relative.parts[0] in GENERATED_TOP_LEVEL)


def test_public_runtime_has_one_direct_evaluator_entry():
    from humanclaw_bench.evaluation.evaluator import HCFindNavInteractEvaluator

    parameters = list(inspect.signature(HCFindNavInteractEvaluator).parameters)
    assert parameters == ["config"]
    assert hasattr(HCFindNavInteractEvaluator, "check_config_valid")
    assert hasattr(HCFindNavInteractEvaluator, "evaluate_main")

    package = repository_root() / "src" / "humanclaw_bench"
    assert (package / "main.py").is_file()
    assert not (package / "cli.py").exists()
    assert not (package / "rollout.py").exists()


def test_environment_can_be_imported_before_evaluator_without_a_cycle():
    """Exercise the import order that a real Habitat process uses."""

    code = (
        "from humanclaw_bench.envs.find_nav_interact_env import "
        "HCFindNavInteractEnv; "
        "from humanclaw_bench.evaluation import HCFindNavInteractEvaluator; "
        "assert HCFindNavInteractEnv and HCFindNavInteractEvaluator"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=repository_root(),
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_public_names_cover_find_nav_and_interact():
    root = repository_root()
    suffixes = {".py", ".json", ".md", ".toml", ".csv", ".tsv", ".txt"}
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix not in suffixes:
            continue
        if (
            "__pycache__" in path.parts
            or "dist" in path.parts
            or _is_generated(root, path)
        ):
            continue
        text = path.read_text(encoding="utf-8", errors="replace").lower()
        for forbidden in ("object" + "nav", "object" + "_nav", "object" + " nav"):
            assert forbidden not in text, path
    assert not any(path.is_file() for path in (root / "results").glob("*"))


def test_no_old_package_or_user_paths_in_runtime_source():
    source = repository_root() / "src"
    private_home = "/home/" + "lisiyao"
    private_rsc = "/xrcia" + "/"
    for path in source.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        assert "from humanclaw." not in text, path
        assert "import humanclaw." not in text, path
        assert private_home not in text, path
        assert private_rsc not in text, path


def test_weight_manifest_is_exact_and_never_latest():
    path = repository_root() / "resources" / "weights" / "paper_fullval_v1.json"
    manifest = json.loads(path.read_text())
    assert manifest["schema"] == "humanclaw_weight_manifest_v2"
    assert manifest["selection_policy"] == "exact_step_only"
    assert set(manifest["skills"]) == {
        "walk_forward",
        "side_walk",
        "step_back",
        "turn",
        "step_climb_up",
        "step_climb_down",
        "stand",
        "sit",
    }
    assert manifest["base_variants"] == {
        "fp32": {
            "transform": "identity",
            "tensor_sha256": (
                "80375e4eee40fe8003838e2ea7df78cb045812e7377eb432a78290ebd07738e3"
            ),
        },
        "bf16_roundtrip": {
            "transform": "bf16_roundtrip",
            "tensor_sha256": (
                "b582eaef495c4c1673750d4ddc2c385de517682aeb05353229be719d436bd282"
            ),
        },
    }
    expected_rounded = {"walk_forward", "turn", "stand", "sit"}
    actual_rounded = {
        skill
        for skill, entry in manifest["skills"].items()
        if entry["base_variant"] == "bf16_roundtrip"
    }
    assert actual_rounded == expected_rounded
    for entry in manifest["skills"].values():
        assert entry["path"].startswith("skills/")
        assert entry["path"].endswith(".pt")
        assert entry["source_artifact"].endswith("step01500000.pt")
        assert "latest" not in entry["path"]
        assert len(entry["sha256"]) == 64
        assert entry["storage"] == "control_state_v1"


def test_agent_asset_has_no_nested_duplicate_copy():
    root = repository_root() / "resources" / "agent" / "neutral_beta0_handmerged"
    assert not (root / "neutral_beta0_handmerged").exists()


def test_public_text_has_no_private_absolute_paths():
    needles = ("/home/" + "lisiyao", "/data/users/" + "lisiyao", "/xrcia" + "/")
    suffixes = {".py", ".json", ".md", ".toml", ".csv", ".tsv", ".txt", ".patch"}
    root = repository_root()
    for path in root.rglob("*"):
        if (
            not path.is_file()
            or path.suffix not in suffixes
            or ".venv" in path.parts
            or _is_generated(root, path)
        ):
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        assert not any(needle in text for needle in needles), path


def test_habitat_patch_is_pinned():
    root = repository_root()
    patch = root / "patches" / "habitat-sim" / "humanclaw_halfphysics.patch"
    digest = hashlib.sha256(patch.read_bytes()).hexdigest()
    assert digest == "6f57ec8130b4ccca7d208a0754ba691d52e44b469cd6dcee220fa193fcad6766"
    assert "BulletURDFImporter.cpp" in patch.read_text()


def test_runtime_physics_config_is_bundled_and_pinned():
    root = repository_root()
    config = (
        root
        / "src"
        / "humanclaw_bench"
        / "envs"
        / "half_physics"
        / "humanclaw.physics_config.json"
    )
    assert config.is_file()
    assert hashlib.sha256(config.read_bytes()).hexdigest() == (
        "6fa31de4506a2d7c73b04d0026334ec639fa39f345ad6a43dc2fb63cf12ca811"
    )


def test_fullval_half_physics_backend_is_the_validated_standalone_file():
    backend = (
        repository_root()
        / "src"
        / "humanclaw_bench"
        / "envs"
        / "half_physics"
        / "hp.py"
    )
    text = backend.read_text(encoding="utf-8")
    assert hashlib.sha256(backend.read_bytes()).hexdigest() == (
        "af1d2154f941e8f55922e66d87b509497d716e1c21ce4027c3061d0eeb42e320"
    )
    assert "ANGULAR_LIMIT_DEGREES_PER_FRAME = 30.0" in text
    assert "SHOULDER_PJSC_POSITION_GAIN = 0.03" in text
    assert "WRIST_PJSC_POSITION_GAIN = 0.1" in text
    assert "DEFAULT_ROOT_LINEAR_XZ_COMMAND_SUBSTEPS = (0, 2)" in text
    assert "_step_physics_with_movable_object_gravity" in text


def test_motion_runtime_has_no_training_framework_dependency():
    """Keep optional trainers from leaking into rollout-time motion modules."""

    motion = repository_root() / "src" / "humanclaw_bench" / "motion"
    runtime_paths = [
        path
        for path in motion.rglob("*.py")
        if "training" not in path.relative_to(motion).parts
    ]
    text = "\n".join(path.read_text(encoding="utf-8") for path in runtime_paths)
    assert "pytorch_lightning" not in text
    assert "lightning_module" not in text
    assert "from humanclaw." not in text
