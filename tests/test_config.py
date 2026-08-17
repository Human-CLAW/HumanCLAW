import json
from pathlib import Path

from humanclaw_bench.config import REQUIRED_SECTIONS, load_config
from humanclaw_bench.paths import repository_root


def test_profiles_are_complete_and_non_inheriting():
    config = load_config("paper_fullval_v1")
    assert REQUIRED_SECTIONS <= config.data.keys()
    assert "extends" not in config.data
    assert config.data["vlm"]["max_tokens_policy"] == "model_specific"
    assert "max_tokens" not in config.data["vlm"]
    physics = config.data["physics"]
    assert physics["backend"] == "hp"
    assert physics["pjsc_lambda_by_link"]["left_shoulder"] == 0.03
    assert physics["pjsc_lambda_by_link"]["right_shoulder"] == 0.03
    assert physics["pjsc_lambda_by_link"]["left_wrist"] == 0.1
    assert physics["pjsc_lambda_by_link"]["right_wrist"] == 0.1
    assert physics["root_linear_xz_command_substeps"] == [0, 2]
    assert physics["friction"] == 0.4
    assert config.data["metrics"]["find_pixel_threshold"] == 100
    assert (
        config.data["metrics"]["collision_contact_source"]
        == "post_physics_30hz"
    )
    assert config.data["metrics"]["fixed_contact_min_height_m"] == 0.0205
    assert config.data["metrics"]["jerk_stride"] == 8


def test_runtime_paths_are_release_relative():
    data = load_config("paper_fullval_v1").data
    values = [
        data["benchmark"]["dataset_dir"],
        data["benchmark"]["scene_dataset_config"],
        data["motion"]["weights_manifest"],
        data["motion"]["weights_root"],
        data["motion"]["seed_pt"],
        data["physics"]["agent_urdf"],
        data["physics"]["agent_shift_npy"],
        data["physics"]["physics_config"],
        data["metrics"]["jerk_neutral_body22"],
    ]
    assert all(not Path(value).is_absolute() for value in values)
    assert data["benchmark"]["scene_dataset_config"] == (
        "data/humanclaw-hssd-val41/hssd-hab.scene_dataset_config.json"
    )


def test_model_examples_contain_no_real_credentials():
    for path in (repository_root() / "configs" / "models").glob("*.json"):
        value = json.loads(path.read_text())
        assert not any("secret" in str(item).lower() for item in value.values())


def test_paper_model_invocations_match_audited_values():
    path = repository_root() / "configs" / "paper_table_model_invocations.json"
    value = json.loads(path.read_text())
    rows = {row["id"]: row for row in value["models"]}
    assert len(rows) == 9
    assert rows["gemma4"]["max_tokens"] == 1200
    assert rows["qwen36_27b"]["max_tokens"] == 4096
    assert rows["internvl35_38b"]["max_tokens"] == 2048
    assert rows["claude48"]["historical_backend"] == "filesystem_queue"
    assert rows["gpt55low"]["historical_backend"] == "azure_openai"


def test_vlm_factory_has_no_sampling_fallback(tmp_path):
    from humanclaw_bench.vlm.factory import build_model

    try:
        build_model({"backend": "openai_compatible", "model": "example"}, tmp_path)
    except ValueError as error:
        assert "max_tokens" in str(error)
        assert "temperature" in str(error)
    else:
        raise AssertionError("incomplete model config was accepted")
