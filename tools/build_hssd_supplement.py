#!/usr/bin/env python3
"""Build the content-addressed HumanClaw HSSD supplement for Hugging Face."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import tarfile
from pathlib import Path
from typing import Any


ARCHIVE_NAME = "humanclaw-hssd-val41-supplement-v1.tar.gz"
MANIFEST_NAME = "humanclaw-hssd-val41-supplement-v1.manifest.json"


def _sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    """Return the SHA-256 digest of one file without loading it into memory."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    """Load a JSON object from ``path``."""

    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _write_json(path: Path, value: Any) -> None:
    """Write deterministic, human-readable JSON with a trailing newline."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_archive(path: Path, blobs: dict[str, Path]) -> None:
    """Write a deterministic gzip-compressed tar containing unique mesh blobs."""

    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as zipped:
            with tarfile.open(fileobj=zipped, mode="w") as archive:
                for digest, source in sorted(blobs.items()):
                    info = tarfile.TarInfo(name=f"blobs/{digest}.glb")
                    info.size = source.stat().st_size
                    info.mode = 0o644
                    info.mtime = 0
                    info.uid = 0
                    info.gid = 0
                    info.uname = ""
                    info.gname = ""
                    with source.open("rb") as handle:
                        archive.addfile(info, handle)
    temporary.replace(path)


def build_supplement(
    modifications_root: Path,
    source_root: Path,
    output_root: Path,
    *,
    repo_id: str,
    release_manifest: Path | None,
) -> dict[str, Any]:
    """Validate flat baked meshes and package their unique contents once."""

    requirements = _read_json(modifications_root / "asset_requirements.json")
    object_specs = dict(requirements["assets"]["objects"])
    source_paths = sorted(path for path in source_root.iterdir() if path.is_file())
    expected_count = int(requirements["supplemental_object_asset_count"])
    expected_size = int(requirements["supplemental_object_asset_size_bytes"])
    if len(source_paths) != expected_count:
        raise ValueError(
            f"Expected {expected_count} supplemental meshes, found {len(source_paths)}"
        )

    files: dict[str, dict[str, Any]] = {}
    blobs: dict[str, Path] = {}
    logical_size = 0
    for source in source_paths:
        if source.suffix != ".glb":
            raise ValueError(f"Supplement contains a non-GLB file: {source}")
        if source.name not in object_specs:
            raise ValueError(f"Supplement mesh is absent from requirements: {source}")
        spec = dict(object_specs[source.name])
        size = source.stat().st_size
        digest = _sha256(source)
        if size != int(spec["size_bytes"]) or digest != str(spec["sha256"]):
            raise ValueError(f"Supplement mesh does not match requirements: {source}")
        files[source.name] = {"sha256": digest, "size_bytes": size}
        blobs.setdefault(digest, source)
        logical_size += size
    if logical_size != expected_size:
        raise ValueError(
            f"Expected {expected_size} logical bytes, found {logical_size}"
        )

    output_root.mkdir(parents=True, exist_ok=True)
    archive_path = output_root / ARCHIVE_NAME
    _write_archive(archive_path, blobs)
    unique_size = sum(path.stat().st_size for path in blobs.values())
    manifest = {
        "schema": "humanclaw_hssd_supplement_v1",
        "version": "humanclaw-hssd-val41-supplement-v1",
        "description": (
            "Content-addressed per-instance HSSD meshes required to reproduce "
            "the HumanClawBench val41 physics and rendering setup."
        ),
        "source_hssd_version": requirements["hssd_version"],
        "hf": {
            "repo_id": repo_id,
            "repo_type": "dataset",
            "filename": f"hssd/{ARCHIVE_NAME}",
        },
        "archive": {
            "format": "tar.gz",
            "sha256": _sha256(archive_path),
            "size_bytes": archive_path.stat().st_size,
        },
        "asset_count": len(files),
        "logical_size_bytes": logical_size,
        "unique_blob_count": len(blobs),
        "unique_blob_size_bytes": unique_size,
        "files": files,
    }
    manifest_path = output_root / MANIFEST_NAME
    _write_json(manifest_path, manifest)
    if release_manifest is not None:
        _write_json(release_manifest, manifest)
    return {
        "archive": str(archive_path),
        "manifest": str(manifest_path),
        "archive_sha256": manifest["archive"]["sha256"],
        "archive_size_bytes": manifest["archive"]["size_bytes"],
        "asset_count": len(files),
        "logical_size_bytes": logical_size,
        "unique_blob_count": len(blobs),
        "unique_blob_size_bytes": unique_size,
    }


def _parse_args() -> argparse.Namespace:
    """Parse the maintainer-facing supplement builder arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--modifications", required=True, type=Path)
    parser.add_argument(
        "--source",
        required=True,
        type=Path,
        help="flat directory containing the 1,693 validated baked GLB files",
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--repo-id", default="HumanCLAW/HumanCLAW-HSSD")
    parser.add_argument("--release-manifest", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    """Build the supplement and print its immutable release metadata."""

    args = _parse_args()
    result = build_supplement(
        args.modifications.expanduser().resolve(),
        args.source.expanduser().resolve(),
        args.output.expanduser().resolve(),
        repo_id=args.repo_id,
        release_manifest=(
            args.release_manifest.expanduser().resolve()
            if args.release_manifest is not None
            else None
        ),
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
