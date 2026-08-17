"""Load the fixed HSSD episodes used by HumanClawBench.

The source files follow Habitat's dataset schema, but the public benchmark task
is Find/Nav/Interact rather than a navigation-only task.
"""

from __future__ import annotations

import gzip
import json
import math
import os
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Optional, Sequence

from humanclaw_bench.paths import repository_root

REPO_ROOT = repository_root()
DEFAULT_BENCHMARK_DATASET_DIR = REPO_ROOT / "resources" / "benchmark" / "episodes"
HSSD_SMALL_DATASET_CONFIG = (
    REPO_ROOT / "resources" / "scenes" / "hssd-hab.scene_dataset_config.json"
)

CATEGORY_LABELS = {
    "potted_plant": "potted plant",
    "tv": "TV",
    "couch": "couch",
    "chair": "chair",
    "bed": "bed",
    "toilet": "toilet",
}


@dataclass(frozen=True)
class HCFindNavInteractEpisode:
    """Normalized task, spawn, scene, and target metadata for one episode."""

    name: str
    task_type: str
    instruction: str
    scene_id: str
    scene_label: str
    scene_dataset_config: str
    episode_id: str
    object_category: str
    object_label: str
    init_offset: tuple[float, float, float]
    init_yaw: float
    max_steps: int
    goals: list[dict[str, Any]]
    viewpoint_positions: list[list[float]]
    goal_objects: list[dict[str, Any]]
    dataset_start_position: list[float] = field(default_factory=list)
    dataset_start_rotation_xyzw: list[float] = field(default_factory=list)
    geodesic_distance: Optional[float] = None


def default_scene_dataset_config() -> Path:
    """Return the configured prepared-HSSD scene dataset, honoring the environment override."""

    override = os.environ.get("HUMANCLAW_HSSD_SCENE_DATASET_CONFIG")
    if override:
        return Path(override).expanduser().resolve()
    return HSSD_SMALL_DATASET_CONFIG


def category_label(category: str) -> str:
    """Convert an internal category key to the instruction's display label."""

    return CATEGORY_LABELS.get(category, category.replace("_", " "))


def find_nav_interact_instruction(category: str) -> str:
    """Build the base Find/Nav instruction for one target category."""

    label = category_label(category)
    return (
        f"Find the {label} and move until your body is touching the target, "
        "with zero distance to the target object."
    )


def apply_instruction_version(
    episode: HCFindNavInteractEpisode,
    version: str = "v0",
) -> HCFindNavInteractEpisode:
    """Apply the selected instruction wording, including Interact tasks in v1."""

    key = str(version or "v0").strip().lower()
    if not key.startswith("v"):
        key = f"v{key}"
    if key == "v0":
        return episode
    if key != "v1":
        raise ValueError(f"Unknown Find/Nav/Interact instruction version {version!r}")
    if episode.object_category not in {"bed", "couch", "toilet"}:
        return episode
    suffix = f" Finally, sit on the {episode.object_label}."
    if episode.instruction.endswith(suffix.strip()):
        return episode
    return replace(episode, instruction=episode.instruction + suffix)


def load_scene_content(path: Path) -> dict[str, Any]:
    """Load one gzipped Habitat episode shard as a JSON object."""

    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def resolve_scene_id(scene_id: str, scene_dataset_config: str) -> str:
    """Resolve a short scene ID against the prepared HSSD scene directory."""

    scene_path = Path(scene_id)
    if scene_path.is_file():
        return str(scene_path.resolve())
    dataset_dir = Path(scene_dataset_config).resolve().parent
    candidate = dataset_dir / "scenes" / f"{scene_id}.scene_instance.json"
    if candidate.is_file():
        return str(candidate)
    return scene_id


def scene_label_of(scene_id: str) -> str:
    """Convert a scene path or ID into its stable benchmark label."""

    path = Path(scene_id)
    if path.name.endswith(".scene_instance.json"):
        return path.name[: -len(".scene_instance.json")]
    return scene_id.replace("/", "_").replace(":", "_")


def short_scene_id(raw_scene_id: str, content_path: Path) -> str:
    """Extract the short HSSD scene ID stored in benchmark indexes."""

    name = Path(str(raw_scene_id)).name
    if name.endswith(".scene_instance.json"):
        name = name[: -len(".scene_instance.json")]
    if name.endswith(".glb"):
        name = name[: -len(".glb")]
    if not name:
        name = content_path.name.removesuffix(".json.gz")
    return name


def start_position_to_init_offset(
    start_position: Sequence[float],
) -> tuple[float, float, float]:
    """Convert Habitat start coordinates to the motion runner's episode offset convention."""

    if len(start_position) < 3:
        raise ValueError(f"Expected 3D start_position, got {start_position!r}")
    x, _y, z = (
        float(start_position[0]),
        float(start_position[1]),
        float(start_position[2]),
    )
    return (-x, z, 0.0)


def _quat_rotate_vector_xyzw(q: Sequence[float], v: Sequence[float]) -> list[float]:
    """Rotate a three-vector by a normalized xyzw quaternion."""

    x, y, z, w = [float(value) for value in q]
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm <= 0:
        raise ValueError(f"Invalid zero quaternion: {q!r}")
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    vx, vy, vz = [float(value) for value in v]
    tx = 2.0 * (y * vz - z * vy)
    ty = 2.0 * (z * vx - x * vz)
    tz = 2.0 * (x * vy - y * vx)
    return [
        vx + w * tx + (y * tz - z * ty),
        vy + w * ty + (z * tx - x * tz),
        vz + w * tz + (x * ty - y * tx),
    ]


def start_rotation_to_habitat_yaw_deg(
    start_rotation_xyzw: Sequence[float],
) -> float:
    """Convert Habitat's start quaternion to the release root-yaw convention."""

    forward = _quat_rotate_vector_xyzw(start_rotation_xyzw, [0.0, 0.0, -1.0])
    yaw = math.degrees(math.atan2(float(forward[0]), float(forward[2]))) - 180.0
    while yaw <= -180.0:
        yaw += 360.0
    while yaw > 180.0:
        yaw -= 360.0
    return yaw


def _template_hash_of(object_name: Any) -> str:
    """Strip instance suffixes from a Habitat goal object's template name."""

    name = str(object_name or "").strip()
    if ":" in name:
        name = name.split(":", 1)[0]
    return name.rstrip("_")


def goal_viewpoints_and_objects_for(
    scene_raw: dict[str, Any], scene_key: str, category: str
) -> tuple[list[list[float]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Extract goal viewpoints, target object centers, and raw goals for one category."""

    goals = scene_raw.get("goals_by_category", {}).get(f"{scene_key}_{category}", [])
    viewpoints: list[list[float]] = []
    objects: list[dict[str, Any]] = []
    for goal in goals:
        for vp in goal.get("view_points", []) or []:
            pos = (vp.get("agent_state", {}) or {}).get("position")
            if isinstance(pos, list) and len(pos) >= 3:
                viewpoints.append([float(pos[0]), float(pos[1]), float(pos[2])])
        obj_pos = goal.get("position")
        if isinstance(obj_pos, list) and len(obj_pos) >= 3:
            objects.append(
                {
                    "object_name": goal.get("object_name"),
                    "template_hash": _template_hash_of(goal.get("object_name")),
                    "object_id": goal.get("object_id"),
                    "center": [float(obj_pos[0]), float(obj_pos[1]), float(obj_pos[2])],
                }
            )
    return viewpoints, objects, list(goals)


def _select_episode(
    episodes: Sequence[dict[str, Any]],
    *,
    episode_id: Optional[str],
    episode_index: Optional[int],
    object_category: Optional[str],
) -> dict[str, Any]:
    """Select one episode deterministically by category, ID, or sorted index."""

    pool = list(episodes)
    if object_category is not None:
        pool = [ep for ep in pool if str(ep.get("object_category")) == object_category]
        if not pool:
            raise ValueError(f"No episodes with object_category={object_category!r}")
    if episode_id is not None:
        for ep in pool:
            if str(ep.get("episode_id")) == str(episode_id):
                return ep
        raise ValueError(f"episode_id={episode_id!r} not found")
    pool.sort(key=lambda ep: int(ep.get("episode_id", "0")))
    index = int(episode_index or 0)
    if index < 0 or index >= len(pool):
        raise ValueError(f"episode_index {index} out of range (0..{len(pool) - 1})")
    return pool[index]


def list_episode_specs(
    benchmark_dataset_dir: Path = DEFAULT_BENCHMARK_DATASET_DIR,
    split: str = "val",
) -> list[dict[str, str]]:
    """List every episode directly from the canonical scene shards."""

    content_dir = Path(benchmark_dataset_dir) / split / "content"
    if not content_dir.is_dir():
        raise FileNotFoundError(f"Benchmark episode directory not found: {content_dir}")
    rows: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for content_path in sorted(content_dir.glob("*.json.gz")):
        raw = load_scene_content(content_path)
        for episode in raw.get("episodes") or []:
            scene_id = short_scene_id(str(episode.get("scene_id") or ""), content_path)
            episode_id = str(episode["episode_id"])
            key = (scene_id, episode_id)
            if key in seen:
                raise ValueError(
                    f"Duplicate benchmark episode: {scene_id}/ep{episode_id}"
                )
            seen.add(key)
            rows.append(
                {
                    "scene_id": scene_id,
                    "episode_id": episode_id,
                    "object_category": str(episode["object_category"]),
                }
            )
    return rows


def load_episode(
    *,
    benchmark_dataset_dir: Path = DEFAULT_BENCHMARK_DATASET_DIR,
    split: str = "val",
    scene_id: str = "103997919_171031233",
    scene_dataset_config: Optional[Path] = None,
    episode_id: Optional[str] = None,
    episode_index: Optional[int] = None,
    object_category: Optional[str] = "toilet",
    max_steps: int = 100,
) -> HCFindNavInteractEpisode:
    """Load and validate one canonical shard entry as an HCFindNavInteractEpisode."""

    content_path = benchmark_dataset_dir / split / "content" / f"{scene_id}.json.gz"
    if not content_path.is_file():
        raise FileNotFoundError(f"Benchmark episode content not found: {content_path}")
    raw = load_scene_content(content_path)
    episodes = raw.get("episodes") or []
    if not episodes:
        raise ValueError(f"No episodes in {content_path}")
    scene_key = short_scene_id(
        str(episodes[0].get("scene_id") or scene_id), content_path
    )

    episode = _select_episode(
        episodes,
        episode_id=episode_id,
        episode_index=episode_index,
        object_category=object_category,
    )
    category = str(episode.get("object_category") or object_category or "")
    episode_name = f"hc_find_nav_interact_{scene_key}_ep{episode.get('episode_id')}"
    start_position = [
        float(value) for value in (episode.get("start_position") or [0.0, 0.0, 0.0])
    ]
    start_rotation = episode.get("start_rotation") or [0.0, 0.0, 0.0, 1.0]
    info = episode.get("info") if isinstance(episode.get("info"), dict) else {}

    raw_offset = episode.get("init_offset")
    if raw_offset is None:
        init_offset = start_position_to_init_offset(start_position)
    elif not isinstance(raw_offset, list) or len(raw_offset) != 3:
        raise ValueError(f"Invalid init_offset in {episode_name}")
    else:
        init_offset = tuple(float(value) for value in raw_offset)
    init_yaw = float(
        episode.get("init_yaw", start_rotation_to_habitat_yaw_deg(start_rotation))
    )

    config_path = str(
        (scene_dataset_config or default_scene_dataset_config()).resolve()
    )
    resolved_scene = resolve_scene_id(scene_key, config_path)
    viewpoints, goal_objects, goals = goal_viewpoints_and_objects_for(
        raw, scene_key, category
    )
    if not viewpoints:
        raise ValueError(f"No goal viewpoints for {scene_key}/{category}")
    if not goal_objects:
        raise ValueError(f"No goal objects for {scene_key}/{category}")

    return HCFindNavInteractEpisode(
        name=episode_name,
        task_type="find_nav_interact",
        instruction=find_nav_interact_instruction(category),
        scene_id=resolved_scene,
        scene_label=scene_label_of(resolved_scene),
        scene_dataset_config=config_path,
        episode_id=str(episode.get("episode_id")),
        object_category=category,
        object_label=category_label(category),
        init_offset=init_offset,
        init_yaw=init_yaw,
        max_steps=int(max_steps),
        goals=goals,
        viewpoint_positions=viewpoints,
        goal_objects=goal_objects,
        dataset_start_position=start_position,
        dataset_start_rotation_xyzw=[float(v) for v in start_rotation],
        geodesic_distance=(
            float(info["geodesic_distance"])
            if info.get("geodesic_distance") is not None
            else None
        ),
    )
