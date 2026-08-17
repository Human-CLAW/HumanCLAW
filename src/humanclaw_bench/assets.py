"""Checksum validation for bundled assets and externally supplied weights."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

from .paths import repository_root, resolve_release_path


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    """Compute a file's SHA-256 digest by streaming fixed-size chunks."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest(path: str | Path) -> dict[str, Any]:
    """Load a release-relative JSON manifest and require an object root."""

    resolved = resolve_release_path(path)
    value = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Manifest must be an object: {resolved}")
    return value


def verify_entries(
    entries: Iterable[dict[str, Any]], root: Path
) -> list[dict[str, Any]]:
    """Verify file existence, size, and SHA-256 against manifest entries."""

    results: list[dict[str, Any]] = []
    for entry in entries:
        relative = str(entry["path"])
        path = (root / relative).resolve()
        expected = str(entry["sha256"]).lower()
        if not path.is_file():
            results.append({"path": str(path), "ok": False, "error": "missing"})
            continue
        actual = sha256_file(path)
        expected_size = entry.get("size_bytes")
        size_ok = expected_size is None or path.stat().st_size == int(expected_size)
        results.append(
            {
                "path": str(path),
                "ok": actual == expected and size_ok,
                "sha256": actual,
                "size_bytes": path.stat().st_size,
                "expected_sha256": expected,
                "expected_size_bytes": expected_size,
            }
        )
    return results


def sha256_tree(root: Path) -> tuple[str, int, int]:
    """Match the deterministic tree digest documented in resources/assets.json."""
    digest = hashlib.sha256()
    files = sorted(path for path in root.rglob("*") if path.is_file())
    total_size = 0
    for path in files:
        relative = "./" + path.relative_to(root).as_posix()
        file_digest = sha256_file(path)
        digest.update(f"{file_digest}  {relative}\n".encode("utf-8"))
        total_size += path.stat().st_size
    return digest.hexdigest(), len(files), total_size


def verify_trees(entries: Iterable[dict[str, Any]], root: Path) -> list[dict[str, Any]]:
    """Verify deterministic tree digest, file count, and total byte size."""

    results: list[dict[str, Any]] = []
    for entry in entries:
        path = (root / str(entry["path"])).resolve()
        if not path.is_dir():
            results.append({"path": str(path), "ok": False, "error": "missing"})
            continue
        actual, count, size = sha256_tree(path)
        expected_count = int(entry["file_count"])
        expected_size = int(entry["size_bytes"])
        expected = str(entry["sha256"]).lower()
        results.append(
            {
                "path": str(path),
                "ok": actual == expected
                and count == expected_count
                and size == expected_size,
                "sha256": actual,
                "file_count": count,
                "size_bytes": size,
                "expected_sha256": expected,
                "expected_file_count": expected_count,
                "expected_size_bytes": expected_size,
            }
        )
    return results


def verify_bundled_assets() -> list[dict[str, Any]]:
    """Verify every file and directory tree pinned by resources/assets.json."""

    root = repository_root()
    manifest = load_manifest(root / "resources" / "assets.json")
    return verify_entries(manifest.get("files", []), root) + verify_trees(
        manifest.get("trees", []), root
    )


def weight_entries(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    """Flatten base and skill checkpoint records from a weight manifest."""

    entries = [dict(manifest["base"])]
    entries.extend(dict(value) for value in manifest.get("skills", {}).values())
    return entries


def verify_weights(
    manifest_path: str | Path, weights_root: str | Path
) -> list[dict[str, Any]]:
    """Verify all externally downloaded checkpoints against the pinned manifest."""

    manifest = load_manifest(manifest_path)
    return verify_entries(
        weight_entries(manifest), Path(weights_root).expanduser().resolve()
    )
