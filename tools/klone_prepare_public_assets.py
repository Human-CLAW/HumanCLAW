"""Download and verify the public assets required by the pinned HumanCLAW release."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import shutil
import tarfile

from huggingface_hub import HfApi, hf_hub_download


# The v0.2.5 tag differs from the release manifest for 104 required assets.
# This official revision matches the manifest; downloaded bytes are also hashed.
HSSD_REVISION = "4369cb9876214c7fbebcf552eb532380e4d287e4"
WEIGHTS_SHA256 = "3b3c0c1b232af4c462301655de909a4bc54fd4756bcd42c22e7965fccb650667"


def digest(path: Path) -> str:
    """Hash a file without loading checkpoint contents into Python objects."""
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def prepare_weights(root: Path, repo: Path) -> None:
    """Extract only the nine allowlisted checkpoints after archive validation."""
    archive = root / "data/downloads/humanclaw-weights/HumanCLAW_pretrained_weights_paper_fullval_v1_20260816.tar.gz"
    if digest(archive) != WEIGHTS_SHA256:
        raise ValueError("Motion-weight archive checksum mismatch")
    manifest = json.loads((repo / "resources/weights/paper_fullval_v1.json").read_text())
    entries = [manifest["base"], *manifest["skills"].values()]
    destination = root / "data/weights/paper_fullval_v1"
    verified = []
    with tarfile.open(archive, "r:gz") as source:
        members = {member.name: member for member in source.getmembers()}
        for entry in entries:
            relative = Path(entry["path"])
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("Unsafe checkpoint path")
            matches = [member for name, member in members.items()
                       if name.endswith("/weights/paper_fullval_v1/" + relative.as_posix())]
            if len(matches) != 1 or not matches[0].isfile():
                raise ValueError("Archive checkpoint missing or ambiguous: " + str(relative))
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            if not (target.is_file() and target.stat().st_size == entry["size_bytes"]
                    and digest(target) == entry["sha256"]):
                temporary = target.with_suffix(target.suffix + ".partial")
                with source.extractfile(matches[0]) as input_stream, temporary.open("wb") as output_stream:
                    shutil.copyfileobj(input_stream, output_stream)
                if temporary.stat().st_size != entry["size_bytes"] or digest(temporary) != entry["sha256"]:
                    raise ValueError("Extracted checkpoint checksum mismatch: " + str(relative))
                temporary.replace(target)
            verified.append(relative.as_posix())
    for name in ("base", "skills"):
        link = repo / "weights/paper_fullval_v1" / name
        target = destination / name
        if link.is_symlink() and link.resolve() == target.resolve():
            continue
        if link.exists() or link.is_symlink():
            raise FileExistsError("Preserving existing weight path: " + str(link))
        link.symlink_to(target, target_is_directory=True)
    print(json.dumps({"step": "weights", "verified_files": len(verified),
                      "archive_sha256": WEIGHTS_SHA256, "weights_root": str(destination)}), flush=True)


def prepare_official_hssd(root: Path, repo: Path, *, token: str | bool = False,
                          workers: int = 4) -> None:
    """Fetch public assets matching the release's exact per-file checksums."""
    mods = repo / "resources/hssd/humanclaw-hssd-val41"
    requirements = json.loads((mods / "asset_requirements.json").read_text())
    supplement = json.loads((mods / "supplement.json").read_text())
    info = HfApi(token=token).repo_info("hssd/hssd-hab", repo_type="dataset",
                                     revision=HSSD_REVISION, files_metadata=True)
    by_name = {}
    by_path = {entry.rfilename: entry for entry in info.siblings}
    for entry in info.siblings:
        if entry.rfilename.startswith("objects/"):
            by_name.setdefault(Path(entry.rfilename).name, []).append(entry)
    selected = {}
    for group, assets in requirements["assets"].items():
        for name, spec in assets.items():
            if group == "objects" and name in supplement["files"]:
                continue
            candidates = by_name.get(name, []) if group == "objects" else [by_path.get(group + "/" + name)]
            candidates = [entry for entry in candidates if entry is not None and entry.size == spec["size_bytes"]
                          and (not entry.lfs or entry.lfs.get("sha256") == spec["sha256"])]
            if not candidates:
                raise ValueError("Required official asset absent at pinned revision: " + group + "/" + name)
            chosen = sorted(candidates, key=lambda entry: entry.rfilename)[0]
            selected[chosen.rfilename] = spec
    config_name = "hssd-hab.scene_dataset_config.json"
    if config_name not in by_path:
        raise ValueError("Official HSSD scene config is missing")
    target = root / ("data/datasets/hssd-hab-" + HSSD_REVISION[:12])
    plan = {"repo": "hssd/hssd-hab", "revision": HSSD_REVISION,
            "file_count": len(selected) + 1,
            "size_bytes": sum(spec["size_bytes"] for spec in selected.values()) + by_path[config_name].size,
            "destination": str(target)}
    print(json.dumps({"step": "hssd_download_plan", **plan}), flush=True)
    # These are already exact filenames. Avoid snapshot_download's quadratic
    # comparison of thousands of allow-patterns with the full repository tree.
    pool = ThreadPoolExecutor(max_workers=workers)
    try:
        futures = [pool.submit(hf_hub_download, "hssd/hssd-hab", name,
                               repo_type="dataset", revision=HSSD_REVISION,
                               local_dir=target, token=token)
                   for name in [config_name, *selected]]
        for count, future in enumerate(as_completed(futures), 1):
            future.result()
            if count % 500 == 0 or count == len(futures):
                print(json.dumps({"step": "hssd_download_progress", "files": count,
                                  "total": len(futures)}), flush=True)
    finally:
        pool.shutdown(wait=True, cancel_futures=True)
    for name, spec in selected.items():
        path = target / name
        if path.stat().st_size != spec["size_bytes"] or digest(path) != spec["sha256"]:
            raise ValueError("Official HSSD checksum mismatch: " + name)
    marker = {**plan, "verified_assets": len(selected), "supplement_included": False}
    (target / "humanclaw-download.json").write_text(json.dumps(marker, indent=2) + "\n")
    print(json.dumps({"step": "hssd_public_complete", **marker}), flush=True)


def main() -> None:
    """Prepare a selected asset group under the user's private scratch root."""
    parser = argparse.ArgumentParser()
    parser.add_argument("step", choices=("weights", "hssd"))
    parser.add_argument("--root", type=Path, default=Path("/gscratch/stf/hengyil6"))
    parser.add_argument("--repo", type=Path, help="Source checkout; defaults to ROOT/HumanCLAW")
    parser.add_argument("--use-hf-token", action="store_true",
                        help="Use HF_TOKEN from this environment or the source checkout's .env")
    parser.add_argument("--workers", type=int, default=4,
                        help="Concurrent HSSD file downloads (1–16; default: 4)")
    args = parser.parse_args()
    if not 1 <= args.workers <= 16:
        parser.error("--workers must be between 1 and 16")
    root = args.root.resolve()
    repo = args.repo.resolve() if args.repo else root / "HumanCLAW"
    if args.step == "weights":
        prepare_weights(root, repo)
    else:
        token = os.environ.get("HF_TOKEN") if args.use_hf_token else False
        if args.use_hf_token and not token:
            from dotenv import dotenv_values

            token = dotenv_values(repo / ".env").get("HF_TOKEN")
        if args.use_hf_token and not token:
            parser.error("--use-hf-token requires HF_TOKEN in the environment or source checkout's .env")
        prepare_official_hssd(root, repo, token=token, workers=args.workers)


if __name__ == "__main__":
    main()
