"""Human-readable tables for an aggregated HumanClawBench metric summary."""

from __future__ import annotations

from typing import Any, Iterable


def _cell(value: Any, digits: int, suffix: str = "") -> str:
    """Format one numeric cell while preserving unavailable values explicitly."""

    if value is None:
        return "--"
    return f"{float(value):.{digits}f}{suffix}"


def _markdown_table(headers: Iterable[str], values: Iterable[str]) -> str:
    """Render one compact, copyable Markdown table with right-aligned values."""

    header_cells = [str(item) for item in headers]
    value_cells = [str(item) for item in values]
    return "\n".join(
        (
            "| " + " | ".join(header_cells) + " |",
            "|" + "|".join("---:" for _ in header_cells) + "|",
            "| " + " | ".join(value_cells) + " |",
        )
    )


def format_metric_summary(summary: dict[str, Any]) -> str:
    """Format the paper main table, variants, and collision body groups."""

    counts = dict(summary.get("counts") or {})
    success = dict(summary.get("high_level_success_percent") or {})
    body = dict(summary.get("body_scene") or {})
    groups = dict(body.get("collision_by_body_group_percent") or {})
    quality = dict(summary.get("action_quality") or {})
    cost = dict(summary.get("cost") or {})

    main = _markdown_table(
        (
            "FindSR",
            "NavSR@20cm",
            "InteractSR",
            "Coll.",
            "#Dtb/ep",
            "dDtb(m)",
            "Motion Jerk",
            "avg steps",
            "in tok/step",
            "out tok/step",
        ),
        (
            _cell(success.get("find_sr"), 1, "%"),
            _cell(success.get("nav_sr_20cm"), 1, "%"),
            _cell(success.get("interact_sr"), 1, "%"),
            _cell(body.get("collision_step_percent"), 1, "%"),
            _cell(body.get("disturbed_objects_per_episode"), 2),
            _cell(body.get("disturbed_object_path_length_mean_m"), 2),
            _cell(quality.get("motion_jerk_m_s3"), 2),
            _cell(cost.get("average_steps"), 1),
            _cell(cost.get("input_tokens_per_step"), 0),
            _cell(cost.get("visible_output_tokens_per_step"), 0),
        ),
    )
    variants = _markdown_table(
        ("GeoFindSR", "GeoNavSR@20cm", "NavSR@1m", "GeoInteractSR"),
        (
            _cell(success.get("geo_find_sr"), 1, "%"),
            _cell(success.get("geo_nav_sr_20cm"), 1, "%"),
            _cell(success.get("nav_sr_1m"), 1, "%"),
            _cell(success.get("geo_interact_sr"), 1, "%"),
        ),
    )
    collision = _markdown_table(
        ("Arm/Hand", "Torso", "Leg/Foot", "Head"),
        (
            _cell(groups.get("hand_arm"), 1, "%"),
            _cell(groups.get("torso"), 1, "%"),
            _cell(groups.get("leg"), 1, "%"),
            _cell(groups.get("head"), 1, "%"),
        ),
    )
    token_sources = ", ".join(str(item) for item in cost.get("token_sources") or ())
    count_line = (
        f"Episodes: {int(counts.get('episodes') or 0)} | "
        f"Interact: {int(counts.get('interact_episodes') or 0)} | "
        f"Physical: {int(counts.get('physical_metric_episodes') or 0)} | "
        "Initial-penetration excluded: "
        f"{int(counts.get('initial_penetration_excluded') or 0)}"
    )
    sections = [
        "HumanClawBench paper metrics",
        count_line,
        "",
        "Main",
        main,
        "",
        "Variants",
        variants,
        "",
        "Collision by body group",
        collision,
    ]
    if token_sources:
        sections.extend(("", f"Token source: {token_sources}"))
    return "\n".join(sections)


__all__ = ["format_metric_summary"]
