import hashlib
import io
import json
import tarfile

from humanclaw_bench import hssd
from humanclaw_bench.hssd import load_hssd_modifications, prepare_hssd


def _spec(payload: bytes) -> dict:
    return {
        "size_bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def test_release_hssd_modifications_are_complete():
    modifications = load_hssd_modifications()
    requirements = modifications["requirements"]
    assert len(modifications["scene_paths"]) == 41
    assert len(modifications["object_paths"]) == 14537
    assert modifications["motion_counts"] == {"STATIC": 11616, "DYNAMIC": 2921}
    assert requirements["required_object_asset_count"] == 12112
    assert requirements["supplemental_object_asset_count"] == 1693
    assert requirements["supplemental_object_asset_size_bytes"] == 184310072

    root = modifications["root"]
    assert (root / "hssd-hab.scene_dataset_config.json").is_file()
    supplement = modifications["supplement_manifest"]
    assert supplement["asset_count"] == 1693
    assert supplement["logical_size_bytes"] == 184310072
    assert len(supplement["files"]) == 1693
    assert supplement["hf"] == {
        "filename": "hssd/humanclaw-hssd-val41-supplement-v1.tar.gz",
        "repo_id": "HumanCLAW/HumanCLAW-HSSD",
        "repo_type": "dataset",
    }
    assert modifications["legacy_supplemental_paths"] == []
    assert not (root / "supplemental_objects").exists()
    encoded = "\n".join(
        path.read_text(encoding="utf-8")
        for path in [modifications["scene_config"], *modifications["scene_paths"]]
    ).lower()
    assert "object" + "nav" not in encoded
    assert "/home/" not in encoded
    assert "/data/users/" not in encoded


def test_sealed_shower_is_removed_without_shifting_object_ids():
    """Keep scene object IDs stable while removing the sealed shower stall."""

    modifications = load_hssd_modifications()
    scene_path = (
        modifications["root"]
        / "scenes"
        / "104862384_172226319.scene_instance.json"
    )
    instances = json.loads(scene_path.read_text(encoding="utf-8"))["object_instances"]

    assert len(instances) == 317
    assert instances[166]["template_name"].startswith(
        "hcbv3r_104862384_172226319_0166_8baa20f9"
    )
    assert instances[166]["translation"] == [
        -9.624489784240723,
        -100.0,
        -0.5027099548155078,
    ]
    # Episode 8 targets the bed at Habitat object ID 223. Keeping the shower
    # entry in place guarantees this ID and every later object ID stay fixed.
    assert instances[223]["template_name"].startswith(
        "hcbv3r_104862384_172226319_0223_f2b9051c"
    )


def test_prepare_hssd_combines_official_and_supplemental_meshes(
    tmp_path,
):
    source = tmp_path / "official-hssd"
    (source / "objects" / "a").mkdir(parents=True)
    (source / "stages").mkdir()
    (source / "semantics" / "scenes").mkdir(parents=True)
    (source / "hssd-hab.scene_dataset_config.json").write_text("{}\n")

    files = {
        source / "objects" / "a" / "asset.glb": b"object-mesh",
        source / "stages" / "scene.glb": b"stage-mesh",
        source / "stages" / "scene.stage_config.json": b"{}\n",
        source / "semantics" / "hssd-hab_semantic_lexicon.json": b"{}\n",
        source / "semantics" / "objects.csv": b"id,name\n",
        source / "semantics" / "scenes" / "scene.semantic_config.json": b"{}\n",
    }
    for path, payload in files.items():
        path.write_bytes(payload)

    modifications = tmp_path / "modifications"
    (modifications / "scenes").mkdir(parents=True)
    (modifications / "objects" / "humanclaw").mkdir(parents=True)
    (modifications / "supplemental_objects").mkdir()
    (modifications / "supplemental_objects" / "generated.glb").write_bytes(
        b"generated-object-mesh"
    )
    (modifications / "hssd-hab.scene_dataset_config.json").write_text(
        json.dumps(
            {
                "stages": {"paths": {".json": ["stages"]}},
                "objects": {"paths": {".json": ["objects/*"]}},
                "scene_instances": {"paths": {".json": ["scenes"]}},
            }
        )
    )
    (modifications / "scenes" / "scene.scene_instance.json").write_text(
        json.dumps(
            {
                "stage_instance": {"template_name": "stages/scene"},
                "object_instances": [
                    {
                        "template_name": "humanclaw_scene_0000",
                        "motion_type": "DYNAMIC",
                    }
                ],
            }
        )
    )
    (
        modifications
        / "objects"
        / "humanclaw"
        / "humanclaw_scene_0000.object_config.json"
    ).write_text(
        json.dumps(
            {
                "render_asset": "generated.glb",
                "collision_asset": "asset.glb",
                "mass": 1.0,
            }
        )
    )
    (modifications / "asset_requirements.json").write_text(
        json.dumps(
            {
                "schema": "humanclaw_hssd_asset_requirements_v1",
                "hssd_version": "test",
                "scene_count": 1,
                "object_instance_count": 1,
                "required_object_asset_count": 2,
                "supplemental_object_asset_count": 1,
                "supplemental_object_asset_size_bytes": len(
                    b"generated-object-mesh"
                ),
                "assets": {
                    "objects": {
                        "asset.glb": _spec(b"object-mesh"),
                        "generated.glb": _spec(b"generated-object-mesh"),
                    },
                    "stages": {
                        "scene.glb": _spec(b"stage-mesh"),
                        "scene.stage_config.json": _spec(b"{}\n"),
                    },
                    "semantics": {
                        "hssd-hab_semantic_lexicon.json": _spec(b"{}\n"),
                        "objects.csv": _spec(b"id,name\n"),
                        "scenes/scene.semantic_config.json": _spec(b"{}\n"),
                    },
                },
            }
        )
    )
    generated_spec = _spec(b"generated-object-mesh")
    (modifications / "supplement.json").write_text(
        json.dumps(
            {
                "schema": "humanclaw_hssd_supplement_v1",
                "version": "test-supplement-v1",
                "source_hssd_version": "test",
                "hf": {
                    "repo_id": "test/repository",
                    "repo_type": "dataset",
                    "filename": "hssd/test.tar.gz",
                },
                "archive": {
                    "format": "tar.gz",
                    "sha256": "0" * 64,
                    "size_bytes": 0,
                },
                "asset_count": 1,
                "logical_size_bytes": len(b"generated-object-mesh"),
                "unique_blob_count": 1,
                "unique_blob_size_bytes": len(b"generated-object-mesh"),
                "files": {"generated.glb": generated_spec},
            }
        )
    )

    output = tmp_path / "prepared"
    summary = prepare_hssd(
        source,
        output,
        modifications_root=modifications,
        supplement=modifications / "supplemental_objects",
    )
    assert summary["scene_count"] == 1
    assert summary["motion_counts"] == {"DYNAMIC": 1}
    assert summary["supplemental_object_asset_count"] == 1
    assert summary["supplemental_object_asset_size_bytes"] == len(
        b"generated-object-mesh"
    )
    assert summary["supplement_version"] == "test-supplement-v1"
    assert summary["hashes_verified"] is True
    assert (output / "stages").is_symlink()
    assert (output / "semantics").is_symlink()
    linked_asset = output / "objects" / "humanclaw" / "asset.glb"
    assert linked_asset.is_symlink()
    assert linked_asset.read_bytes() == b"object-mesh"
    generated_asset = output / "objects" / "humanclaw" / "generated.glb"
    assert generated_asset.is_symlink()
    assert generated_asset.read_bytes() == b"generated-object-mesh"
    assert (
        output / "objects" / "humanclaw" / "humanclaw_scene_0000.object_config.json"
    ).is_file()


def test_content_addressed_supplement_archive_is_verified_and_cached(
    monkeypatch, tmp_path
):
    payload = b"one-baked-mesh"
    digest = hashlib.sha256(payload).hexdigest()
    archive_path = tmp_path / "supplement.tar.gz"
    with tarfile.open(archive_path, mode="w:gz") as archive:
        info = tarfile.TarInfo(f"blobs/{digest}.glb")
        info.size = len(payload)
        archive.addfile(info, io.BytesIO(payload))
    manifest = {
        "version": "test-content-addressed-v1",
        "archive": {
            "sha256": hashlib.sha256(archive_path.read_bytes()).hexdigest(),
            "size_bytes": archive_path.stat().st_size,
        },
        "files": {
            "instance.glb": {
                "sha256": digest,
                "size_bytes": len(payload),
            }
        },
    }
    monkeypatch.setattr(hssd, "DEFAULT_ASSET_CACHE", tmp_path / "cache")

    root = hssd._extract_supplement_archive(
        archive_path, manifest, verify_hashes=True
    )
    assets = hssd._supplement_assets(root, manifest, verify_hashes=True)

    assert assets["instance.glb"].read_bytes() == payload
    assert hssd._extract_supplement_archive(
        archive_path, manifest, verify_hashes=True
    ) == root
