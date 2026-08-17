"""Combine HumanClaw's explicit HSSD modifications with an HSSD install.

Most meshes are linked from the official HSSD download.  Instance-specific
baked meshes live in a separately versioned Hugging Face asset so the GitHub
repository stays lightweight.  ``prepare-hssd`` downloads that asset on first
use, verifies it, and reuses the persistent cache through symbolic links.
"""

from __future__ import annotations

import json
import re
import shutil
import tarfile
import tempfile
from pathlib import Path
from typing import Any

from .assets import sha256_file
from .paths import repository_root, resolve_release_path

DEFAULT_MODIFICATIONS = (
    repository_root() / "resources" / "hssd" / "humanclaw-hssd-val41"
)
DEFAULT_OUTPUT = repository_root() / "data" / "humanclaw-hssd-val41"
DEFAULT_ASSET_CACHE = Path.home() / ".cache" / "humanclaw-bench" / "assets"
SUPPLEMENT_MANIFEST_NAME = "supplement.json"
_BLOB_NAME = re.compile(r"[0-9a-f]{64}\.glb")


def _read_json(path: Path) -> dict[str, Any]:
    """Load a JSON file and require an object at its root."""

    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def load_hssd_modifications(
    path: str | Path = DEFAULT_MODIFICATIONS,
) -> dict[str, Any]:
    """Validate the explicit scene and object files included in the release."""

    root = resolve_release_path(path)
    scene_config = root / "hssd-hab.scene_dataset_config.json"
    requirements_path = root / "asset_requirements.json"
    if not scene_config.is_file():
        raise FileNotFoundError(
            f"HumanClaw HSSD scene config not found: {scene_config}"
        )
    if not requirements_path.is_file():
        raise FileNotFoundError(
            f"HumanClaw HSSD asset requirements not found: {requirements_path}"
        )

    requirements = _read_json(requirements_path)
    if requirements.get("schema") != "humanclaw_hssd_asset_requirements_v1":
        raise ValueError(f"Invalid HSSD asset requirements: {requirements_path}")

    scene_paths = sorted((root / "scenes").glob("*.scene_instance.json"))
    object_paths = sorted((root / "objects").rglob("*.object_config.json"))
    expected_scenes = int(requirements["scene_count"])
    expected_objects = int(requirements["object_instance_count"])
    if len(scene_paths) != expected_scenes:
        raise ValueError(
            f"Expected {expected_scenes} HumanClaw scenes, found {len(scene_paths)}"
        )
    if len(object_paths) != expected_objects:
        raise ValueError(
            f"Expected {expected_objects} object configs, found {len(object_paths)}"
        )

    templates: set[str] = set()
    motion_counts: dict[str, int] = {}
    for scene_path in scene_paths:
        scene = _read_json(scene_path)
        for instance in scene.get("object_instances", []):
            templates.add(str(instance["template_name"]))
            motion = str(instance["motion_type"])
            motion_counts[motion] = motion_counts.get(motion, 0) + 1
    config_templates = {
        path.name.removesuffix(".object_config.json") for path in object_paths
    }
    if templates != config_templates:
        raise ValueError("Scene object templates do not match bundled object configs")
    if sum(motion_counts.values()) != expected_objects:
        raise ValueError("HSSD object-instance count does not match scene contents")

    assets = requirements.get("assets")
    if not isinstance(assets, dict) or not isinstance(assets.get("objects"), dict):
        raise ValueError("HSSD asset requirements have no object asset list")
    if len(assets["objects"]) != int(requirements["required_object_asset_count"]):
        raise ValueError("HSSD required-object-asset count mismatch")

    supplement_manifest_path = root / SUPPLEMENT_MANIFEST_NAME
    if not supplement_manifest_path.is_file():
        raise FileNotFoundError(
            f"HumanClaw HSSD supplement manifest not found: {supplement_manifest_path}"
        )
    supplement_manifest = _read_json(supplement_manifest_path)
    if supplement_manifest.get("schema") != "humanclaw_hssd_supplement_v1":
        raise ValueError(f"Invalid HSSD supplement manifest: {supplement_manifest_path}")
    supplement_files = supplement_manifest.get("files")
    if not isinstance(supplement_files, dict):
        raise ValueError("HSSD supplement manifest has no file map")
    expected_supplemental_count = int(
        requirements.get("supplemental_object_asset_count", 0)
    )
    expected_supplemental_size = int(
        requirements.get("supplemental_object_asset_size_bytes", 0)
    )
    if len(supplement_files) != expected_supplemental_count:
        raise ValueError(
            "Expected "
            f"{expected_supplemental_count} supplemental HSSD object assets, "
            f"found {len(supplement_files)} manifest entries"
        )
    if int(supplement_manifest.get("asset_count", -1)) != len(supplement_files):
        raise ValueError("HSSD supplement manifest asset-count mismatch")
    supplemental_size = sum(
        int(dict(spec)["size_bytes"]) for spec in supplement_files.values()
    )
    if supplemental_size != expected_supplemental_size:
        raise ValueError(
            "Supplemental HSSD object-asset size mismatch: "
            f"expected {expected_supplemental_size}, found {supplemental_size}"
        )
    if int(supplement_manifest.get("logical_size_bytes", -1)) != supplemental_size:
        raise ValueError("HSSD supplement manifest logical-size mismatch")
    if supplement_manifest.get("source_hssd_version") != requirements.get(
        "hssd_version"
    ):
        raise ValueError("HSSD supplement was built for a different HSSD version")
    unknown_supplemental = [
        name for name in supplement_files if name not in assets["objects"]
    ]
    if unknown_supplemental:
        raise ValueError(
            "Supplemental HSSD assets are absent from the manifest: "
            f"{unknown_supplemental[:5]}"
        )
    mismatched_supplemental = [
        name
        for name, spec in supplement_files.items()
        if dict(spec) != dict(assets["objects"][name])
    ]
    if mismatched_supplemental:
        raise ValueError(
            "Supplemental HSSD assets disagree with the global requirements: "
            f"{mismatched_supplemental[:5]}"
        )
    referenced_assets: set[str] = set()
    for object_path in object_paths:
        config = _read_json(object_path)
        render_asset = config.get("render_asset")
        if not isinstance(render_asset, str) or not render_asset:
            raise ValueError(f"Object config has no render asset: {object_path}")
        referenced_assets.add(render_asset)
        collision_asset = config.get("collision_asset")
        if collision_asset:
            referenced_assets.add(str(collision_asset))
    if referenced_assets != set(assets["objects"]):
        raise ValueError(
            "Object configs do not match the required HSSD object asset list"
        )

    legacy_supplemental_root = root / "supplemental_objects"
    legacy_supplemental_paths = sorted(
        path for path in legacy_supplemental_root.glob("*") if path.is_file()
    )
    return {
        "root": root,
        "scene_config": scene_config,
        "scene_paths": scene_paths,
        "object_paths": object_paths,
        "supplement_manifest_path": supplement_manifest_path,
        "supplement_manifest": supplement_manifest,
        "legacy_supplemental_root": legacy_supplemental_root,
        "legacy_supplemental_paths": legacy_supplemental_paths,
        "requirements": requirements,
        "motion_counts": motion_counts,
    }


def _hssd_root(value: str | Path) -> Path:
    """Validate and normalize the user's official HSSD installation root."""

    path = Path(value).expanduser().resolve()
    if path.is_file() and path.name == "hssd-hab.scene_dataset_config.json":
        path = path.parent
    if not (path / "hssd-hab.scene_dataset_config.json").is_file():
        raise FileNotFoundError(
            "Expected an HSSD root containing hssd-hab.scene_dataset_config.json: "
            f"{path}"
        )
    for directory in ("objects", "stages", "semantics"):
        if not (path / directory).is_dir():
            raise FileNotFoundError(f"HSSD directory is missing {directory}/: {path}")
    return path


def _matching_asset(
    candidates: list[Path],
    spec: dict[str, Any],
    *,
    label: str,
    verify_hashes: bool,
) -> Path:
    """Choose the official HSSD asset matching pinned size and optional SHA-256."""

    expected_size = int(spec["size_bytes"])
    sized = [path for path in candidates if path.stat().st_size == expected_size]
    if not sized:
        raise FileNotFoundError(f"Required HSSD asset is missing: {label}")
    if not verify_hashes:
        return sorted(sized)[0]
    expected_hash = str(spec["sha256"])
    for path in sorted(sized):
        if sha256_file(path) == expected_hash:
            return path
    raise ValueError(f"HSSD asset checksum mismatch: {label}")


def _verify_flat_supplement(
    root: Path,
    files: dict[str, Any],
    *,
    verify_hashes: bool,
) -> dict[str, Path]:
    """Validate the legacy flat supplement and map every asset name to a file."""

    assets: dict[str, Path] = {}
    for name, raw_spec in files.items():
        path = root / name
        assets[name] = _matching_asset(
            [path] if path.is_file() else [],
            dict(raw_spec),
            label=f"supplement/{name}",
            verify_hashes=verify_hashes,
        )
    return assets


def _verify_blob_supplement(
    root: Path,
    files: dict[str, Any],
    *,
    verify_hashes: bool,
) -> dict[str, Path]:
    """Validate content-addressed blobs once and expand their filename mapping."""

    blobs = root / "blobs"
    verified: dict[str, Path] = {}
    assets: dict[str, Path] = {}
    for name, raw_spec in files.items():
        spec = dict(raw_spec)
        digest = str(spec["sha256"])
        path = blobs / f"{digest}.glb"
        if digest not in verified:
            verified[digest] = _matching_asset(
                [path] if path.is_file() else [],
                spec,
                label=f"supplement/blobs/{digest}.glb",
                verify_hashes=verify_hashes,
            )
        assets[name] = verified[digest]
    return assets


def _supplement_assets(
    root: Path,
    manifest: dict[str, Any],
    *,
    verify_hashes: bool,
) -> dict[str, Path]:
    """Resolve either the content-addressed or legacy flat supplement layout."""

    files = dict(manifest["files"])
    if (root / "blobs").is_dir():
        return _verify_blob_supplement(root, files, verify_hashes=verify_hashes)
    if (root / "supplemental_objects").is_dir():
        root = root / "supplemental_objects"
    return _verify_flat_supplement(root, files, verify_hashes=verify_hashes)


def _extract_supplement_archive(
    archive_path: Path,
    manifest: dict[str, Any],
    *,
    verify_hashes: bool,
) -> Path:
    """Safely extract an immutable supplement archive into the user cache."""

    archive_spec = dict(manifest["archive"])
    if archive_path.stat().st_size != int(archive_spec["size_bytes"]):
        raise ValueError(f"HSSD supplement archive size mismatch: {archive_path}")
    if verify_hashes and sha256_file(archive_path) != str(archive_spec["sha256"]):
        raise ValueError(f"HSSD supplement archive checksum mismatch: {archive_path}")

    version = str(manifest["version"])
    digest = str(archive_spec["sha256"])
    destination = DEFAULT_ASSET_CACHE / f"{version}-{digest[:12]}"
    if (destination / "blobs").is_dir():
        return destination

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{version}-", dir=str(destination.parent))
    )
    expected = {str(dict(spec)["sha256"]) for spec in manifest["files"].values()}
    found: set[str] = set()
    try:
        with tarfile.open(archive_path, mode="r:gz") as archive:
            for member in archive.getmembers():
                parts = Path(member.name).parts
                if (
                    len(parts) != 2
                    or parts[0] != "blobs"
                    or _BLOB_NAME.fullmatch(parts[1]) is None
                    or not member.isfile()
                ):
                    raise ValueError(
                        f"Unsafe or unexpected HSSD supplement member: {member.name}"
                    )
                blob_digest = parts[1].removesuffix(".glb")
                if blob_digest not in expected or blob_digest in found:
                    raise ValueError(
                        f"Unknown or duplicate HSSD supplement blob: {member.name}"
                    )
                source = archive.extractfile(member)
                if source is None:
                    raise ValueError(f"Cannot read HSSD supplement member: {member.name}")
                target = temporary / "blobs" / parts[1]
                target.parent.mkdir(parents=True, exist_ok=True)
                with source, target.open("wb") as handle:
                    shutil.copyfileobj(source, handle)
                found.add(blob_digest)
        if found != expected:
            raise ValueError("HSSD supplement archive is missing expected blobs")
        _supplement_assets(temporary, manifest, verify_hashes=verify_hashes)
        try:
            temporary.replace(destination)
        except FileExistsError:
            shutil.rmtree(temporary, ignore_errors=True)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return destination


def _download_supplement_archive(manifest: dict[str, Any]) -> Path:
    """Download the immutable supplement archive from its Hugging Face dataset."""

    try:
        from huggingface_hub import hf_hub_download
    except ImportError as error:
        raise RuntimeError(
            "Automatic HSSD supplement download requires huggingface_hub. "
            "Install the package or pass --supplement with a local archive/directory."
        ) from error
    hf = dict(manifest["hf"])
    return Path(
        hf_hub_download(
            repo_id=str(hf["repo_id"]),
            repo_type=str(hf.get("repo_type", "dataset")),
            filename=str(hf["filename"]),
        )
    ).resolve()


def _resolve_supplement(
    modifications: dict[str, Any],
    value: str | Path | None,
    *,
    verify_hashes: bool,
) -> tuple[Path, dict[str, Path]]:
    """Resolve a local supplement or download and cache the pinned HF archive."""

    manifest = modifications["supplement_manifest"]
    if value is None:
        legacy_paths = modifications["legacy_supplemental_paths"]
        if legacy_paths:
            root = modifications["legacy_supplemental_root"]
        else:
            archive = _download_supplement_archive(manifest)
            root = _extract_supplement_archive(
                archive, manifest, verify_hashes=verify_hashes
            )
    else:
        supplied = Path(value).expanduser().resolve()
        if supplied.is_file():
            root = _extract_supplement_archive(
                supplied, manifest, verify_hashes=verify_hashes
            )
        elif supplied.is_dir():
            root = supplied
        else:
            raise FileNotFoundError(f"HSSD supplement not found: {supplied}")
    return root, _supplement_assets(root, manifest, verify_hashes=verify_hashes)


def _resolve_required_assets(
    hssd_root: Path,
    required: dict[str, Any],
    *,
    supplemental_assets: dict[str, Path],
    verify_hashes: bool,
) -> dict[str, Path]:
    """Resolve each mesh from official HSSD or the verified external supplement."""

    object_specs = dict(required["objects"])
    candidates: dict[str, list[Path]] = {name: [] for name in object_specs}
    for path in (hssd_root / "objects").rglob("*"):
        if path.is_file() and path.name in candidates:
            candidates[path.name].append(path)
    for name, path in supplemental_assets.items():
        candidates[name].append(path)
    objects = {
        name: _matching_asset(
            candidates[name],
            spec,
            label=f"objects/{name}",
            verify_hashes=verify_hashes,
        )
        for name, spec in object_specs.items()
    }

    for group in ("stages", "semantics"):
        for name, spec in dict(required[group]).items():
            path = hssd_root / group / name
            _matching_asset(
                [path] if path.is_file() else [],
                spec,
                label=f"{group}/{name}",
                verify_hashes=verify_hashes,
            )
    return objects


def _write_json(path: Path, value: Any) -> None:
    """Write stable indented UTF-8 JSON with a trailing newline."""

    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def prepare_hssd(
    hssd_root: str | Path,
    output_root: str | Path = DEFAULT_OUTPUT,
    *,
    modifications_root: str | Path = DEFAULT_MODIFICATIONS,
    supplement: str | Path | None = None,
    verify_hashes: bool = True,
) -> dict[str, Any]:
    """Create the complete val41 dataset while reusing official HSSD meshes."""

    source = _hssd_root(hssd_root)
    output = resolve_release_path(output_root)
    if output.exists():
        raise FileExistsError(
            f"Prepared HSSD output already exists: {output}. "
            "Choose a new --output path."
        )
    modifications = load_hssd_modifications(modifications_root)
    requirements = modifications["requirements"]
    supplement_root, supplemental_assets = _resolve_supplement(
        modifications, supplement, verify_hashes=verify_hashes
    )
    object_assets = _resolve_required_assets(
        source,
        requirements["assets"],
        supplemental_assets=supplemental_assets,
        verify_hashes=verify_hashes,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.", dir=str(output.parent))
    )
    try:
        shutil.copy2(
            modifications["scene_config"],
            temporary / "hssd-hab.scene_dataset_config.json",
        )
        shutil.copytree(modifications["root"] / "scenes", temporary / "scenes")
        shutil.copytree(modifications["root"] / "objects", temporary / "objects")

        object_root = temporary / "objects" / "humanclaw"
        for name, source_path in object_assets.items():
            (object_root / name).symlink_to(source_path.resolve())

        (temporary / "stages").symlink_to(
            (source / "stages").resolve(), target_is_directory=True
        )
        (temporary / "semantics").symlink_to(
            (source / "semantics").resolve(), target_is_directory=True
        )
        supplement_manifest = modifications["supplement_manifest"]
        marker = {
            "schema": "humanclaw_prepared_hssd_v1",
            "source_hssd_root": str(source),
            "scene_count": len(modifications["scene_paths"]),
            "object_instance_count": len(modifications["object_paths"]),
            "motion_counts": modifications["motion_counts"],
            "linked_object_asset_count": len(object_assets),
            "supplement_version": supplement_manifest["version"],
            "supplement_root": str(supplement_root),
            "supplemental_object_asset_count": len(supplemental_assets),
            "supplemental_object_asset_size_bytes": int(
                supplement_manifest["logical_size_bytes"]
            ),
            "hashes_verified": bool(verify_hashes),
        }
        _write_json(temporary / "prepared.json", marker)
        temporary.replace(output)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise

    return {
        **marker,
        "scene_dataset_config": str(output / "hssd-hab.scene_dataset_config.json"),
        "output_root": str(output),
    }
