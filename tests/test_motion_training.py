"""Validate the compact, optional motion-training release surface."""

from __future__ import annotations

import csv
import json
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

from humanclaw_bench.paths import repository_root


EXPECTED_SKILL_ROWS = {
    "walk_forward": 16_023,
    "side_walk": 2_012,
    "step_back": 2_294,
    "turn": 6_773,
    "step_climb_up": 2_086,
    "step_climb_down": 2_094,
    "stop": 4_011,
    "sit": 5_912,
}

EXPECTED_SEGMENT_ROWS = {
    "walk_forward": 493,
    "side_walk": 67,
    "step_back": 129,
    "turn": 436,
    "step_climb_up": 96,
    "step_climb_down": 125,
    "stop": 111,
    "sit": 116,
}


def _training_profile() -> dict[str, object]:
    """Read the release-pinned training profile without importing PyTorch."""

    path = (
        repository_root()
        / "resources"
        / "motion"
        / "training"
        / "paper_training_v1.json"
    )
    return json.loads(path.read_text(encoding="utf-8"))


def test_motion_training_uses_one_shared_recipe() -> None:
    """Keep optimization settings centralized instead of copying eight trainers."""

    profile = _training_profile()
    assert profile["schema"] == "humanclaw_motion_training_profile_v1"
    architecture = profile["architecture"]
    assert architecture == {
        "x_dim": 219,
        "hidden_dim": 512,
        "num_layers": 10,
        "num_heads": 8,
        "mlp_ratio": 2.0,
        "n_history": 5,
        "n_future": 15,
        "nfe_steps": 30,
        "use_qk_norm": False,
    }
    base = profile["base_training"]
    assert base["batch_size"] == 512
    assert base["learning_rate"] == 1e-4
    assert base["max_epochs"] == 135
    assert base["reference_iterations"] == 109_620
    control = profile["control_defaults"]
    assert control["batch_size"] == 2_048
    assert control["learning_rate"] == 3e-4
    assert control["max_steps"] == 1_500_000

    skills = profile["skills"]
    assert set(skills) == set(EXPECTED_SKILL_ROWS)
    per_skill_optimizer_fields = {
        "batch_size",
        "learning_rate",
        "max_steps",
        "parameter_dtype",
        "data_dtype",
        "autocast_dtype",
    }
    for name, expected_rows in EXPECTED_SKILL_ROWS.items():
        entry = skills[name]
        assert entry["expected_samples"] == expected_rows
        assert entry["expected_segments"] == EXPECTED_SEGMENT_ROWS[name]
        assert entry["segments"] == (
            f"resources/motion/training/segments/{name}.csv"
        )
        assert not per_skill_optimizer_fields.intersection(entry)


def test_motion_training_manifests_are_exact_and_portable() -> None:
    """Pin every selected source chunk without embedding machine-local paths."""

    root = repository_root()
    profile = _training_profile()
    for skill, expected_rows in EXPECTED_SKILL_ROWS.items():
        relative = Path(profile["skills"][skill]["manifest"])
        assert not relative.is_absolute()
        path = root / relative
        assert path.is_file(), f"missing {skill} manifest: {relative}"
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        assert len(rows) == expected_rows
        for row in rows:
            source = Path(row["rel_pkl"])
            source_index = int(row["source_chunk_idx"])
            assert not source.is_absolute()
            assert ".." not in source.parts
            assert source.suffix == ".pkl"
            assert source_index >= 0


def test_motion_segment_manifests_expose_reviewed_time_boundaries() -> None:
    """Publish BABEL segment times without leaking review-machine paths."""

    root = repository_root() / "resources" / "motion" / "training" / "segments"
    index = json.loads((root / "index.json").read_text(encoding="utf-8"))
    assert index["schema"] == "humanclaw_motion_segment_manifests_v1"
    assert set(index["skills"]) == set(EXPECTED_SEGMENT_ROWS)

    for skill, expected_rows in EXPECTED_SEGMENT_ROWS.items():
        entry = index["skills"][skill]
        assert entry["final_used_segments"] == expected_rows
        assert entry["final_training_chunks"] == EXPECTED_SKILL_ROWS[skill]
        path = root / entry["path"]
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        assert len(rows) == expected_rows
        for row in rows:
            assert row["skill"] == skill
            sequence = Path(row["amass_sequence"])
            assert not sequence.is_absolute()
            assert ".." not in sequence.parts
            assert sequence.suffix == ".pkl"
            start_s = Decimal(row["segment_start_s"])
            end_s = Decimal(row["segment_end_s"])
            for seconds_field, milliseconds_field in (
                ("segment_start_s", "segment_start_ms"),
                ("segment_end_s", "segment_end_ms"),
                ("first_used_chunk_start_s", "first_used_chunk_start_ms"),
                ("last_used_chunk_end_s", "last_used_chunk_end_ms"),
            ):
                expected_ms = (Decimal(row[seconds_field]) * 1000).quantize(
                    Decimal("1"), rounding=ROUND_HALF_UP
                )
                assert Decimal(row[milliseconds_field]) == expected_ms
            assert Decimal(0) <= start_s < end_s
            assert row["babel_sid"]
            assert row["provenance"]
        assert sum(int(row["used_chunk_count"]) for row in rows) == (
            EXPECTED_SKILL_ROWS[skill]
        )

    assert index["skills"]["step_climb_up"]["accepted_source_segments"] is None
