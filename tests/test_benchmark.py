import csv
import gzip
import json
import math
from collections import Counter

from humanclaw_bench.batch import (
    _least_loaded_device,
    _load_episode_subset,
    _rollout_complete,
    _task_name,
    resolve_devices,
)
from humanclaw_bench.benchmark.episodes import (
    list_episode_specs,
    load_episode,
    start_rotation_to_habitat_yaw_deg,
)
from humanclaw_bench.config import load_config
from humanclaw_bench.paths import resolve_release_path


def _all_raw_episodes(dataset_dir, split):
    for content_path in sorted((dataset_dir / split / "content").glob("*.json.gz")):
        with gzip.open(content_path, "rt", encoding="utf-8") as handle:
            raw = json.load(handle)
        yield from raw["episodes"]


def test_bundled_fullval_has_one_complete_source_of_1218_episodes():
    profile = load_config("paper_fullval_v1").data
    benchmark = profile["benchmark"]
    assert "manifest" not in benchmark
    assert "spawn_repairs" not in benchmark

    dataset_dir = resolve_release_path(benchmark["dataset_dir"])
    specs = list_episode_specs(dataset_dir, benchmark["split"])
    assert len(specs) == 1218
    assert len({(row["scene_id"], row["episode_id"]) for row in specs}) == 1218
    assert len({row["scene_id"] for row in specs}) == 41
    assert Counter(row["object_category"] for row in specs) == {
        "bed": 218,
        "chair": 231,
        "couch": 231,
        "potted_plant": 205,
        "toilet": 148,
        "tv": 185,
    }

    raw_episodes = list(_all_raw_episodes(dataset_dir, benchmark["split"]))
    assert len(raw_episodes) == 1218
    assert all(len(row["start_position"]) == 3 for row in raw_episodes)
    assert all(len(row["start_rotation"]) == 4 for row in raw_episodes)
    assert all(len(row["init_offset"]) == 3 for row in raw_episodes)
    assert all(isinstance(row["init_yaw"], (int, float)) for row in raw_episodes)

    episode = load_episode(
        benchmark_dataset_dir=dataset_dir,
        split=benchmark["split"],
        scene_id="102343992",
        scene_dataset_config=resolve_release_path(benchmark["scene_dataset_config"]),
        episode_id="0",
        object_category="bed",
        max_steps=benchmark["max_steps"],
    )
    assert episode.episode_id == "0"
    assert episode.object_category == "bed"
    assert episode.max_steps == 100
    assert episode.init_offset == (
        -15.35628729342458,
        5.436051628972242,
        0.40680947780609134,
    )
    assert math.isclose(episode.init_yaw, 132.68120705254887, abs_tol=1e-12)


def test_val100_is_a_transparent_unique_subset_of_fullval():
    profile = load_config("paper_fullval_v1").data
    benchmark = profile["benchmark"]
    full = list_episode_specs(
        resolve_release_path(benchmark["dataset_dir"]), benchmark["split"]
    )
    subset = _load_episode_subset("resources/benchmark/val100.json", full)
    assert len(subset) == 100
    assert len({(row["scene_id"], row["episode_id"]) for row in subset}) == 100
    assert Counter(row["scene_id"] for row in subset) == {
        "104348028_171512877": 20,
        "104862384_172226319": 20,
        "105515184_173104128": 20,
        "106878858_174886965": 20,
        "108736689_177263340": 20,
    }
    assert Counter(row["object_category"] for row in subset) == {
        "bed": 14,
        "chair": 20,
        "couch": 25,
        "potted_plant": 10,
        "toilet": 14,
        "tv": 17,
    }


def test_resume_requires_every_requested_final_artifact(tmp_path):
    episode = {
        "scene_id": "scene",
        "episode_id": "7",
        "object_category": "chair",
    }
    rollout = tmp_path / _task_name(episode) / "rollout_00"
    rollout.mkdir(parents=True)
    required = (
        "replay_manifest.json",
        "trajectory_before.npz",
        "trajectory_after.npz",
        "ego.mp4",
        "exo.mp4",
        "metrics.json",
    )
    for name in required[:-1]:
        (rollout / name).write_bytes(b"x")
    assert not _rollout_complete(
        tmp_path, episode, save_video=True, compute_metrics=True
    )
    (rollout / required[-1]).write_bytes(b"x")
    assert _rollout_complete(
        tmp_path, episode, save_video=True, compute_metrics=True
    )


def test_batch_assigns_the_least_loaded_device_and_rotates_ties():
    devices = ("0", "1")
    assert _least_loaded_device(devices, ["0"] * 9 + ["1"] * 4, 0) == "1"
    assert _least_loaded_device(devices, ["0", "1"], 0) == "0"
    assert _least_loaded_device(devices, ["0", "1"], 1) == "1"
    assert _least_loaded_device((), [], 0) is None


def test_public_gpu_selection_respects_the_visible_device_boundary():
    assert resolve_devices("0,2", environ={}) == ("0", "2")
    assert resolve_devices("auto", environ={"CUDA_VISIBLE_DEVICES": "4,6"}) == (
        "4",
        "6",
    )
    assert resolve_devices("auto", environ={"CUDA_VISIBLE_DEVICES": "-1"}) == ()


def test_final_spawn_values_are_materialized_in_the_episode_shards():
    profile = load_config("paper_fullval_v1").data
    benchmark = profile["benchmark"]
    dataset_dir = resolve_release_path(benchmark["dataset_dir"])
    raw_episodes = list(_all_raw_episodes(dataset_dir, benchmark["split"]))

    history_path = resolve_release_path(
        "resources/provenance/spawn_repair_history_20260806_v2.csv"
    )
    with history_path.open(newline="", encoding="utf-8") as handle:
        history = list(csv.DictReader(handle))
    assert len(history) == 308
    assert Counter(row["final_value_revision"] for row in history) == {
        "20260719-first-pass": 270,
        "20260720-refined-1cm": 37,
        "20260806-v2": 1,
    }

    repaired_names = {row["episode_name"] for row in history}
    episode_names = set()
    for row in raw_episodes:
        scene_id = str(row["scene_id"])
        name = f"hc_find_nav_interact_{scene_id}_ep{row['episode_id']}"
        episode_names.add(name)
        assert math.isclose(
            start_rotation_to_habitat_yaw_deg(row["start_rotation"]),
            float(row["init_yaw"]),
            abs_tol=1e-9,
        )
    assert repaired_names <= episode_names

    followup = load_episode(
        benchmark_dataset_dir=dataset_dir,
        split=benchmark["split"],
        scene_id="105515379_173104395",
        scene_dataset_config=resolve_release_path(benchmark["scene_dataset_config"]),
        episode_id="27",
        object_category=None,
        max_steps=benchmark["max_steps"],
    )
    assert followup.init_offset == (
        -1.2030187424180205,
        -2.4387146481768736,
        0.11006425941586861,
    )
