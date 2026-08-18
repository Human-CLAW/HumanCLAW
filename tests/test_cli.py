import json

from humanclaw_bench import batch
from humanclaw_bench import hssd
from humanclaw_bench.main import _build_parser, main


def test_rollout_cli_accepts_bounded_max_steps():
    args = _build_parser().parse_args(
        [
            "rollout",
            "--model-config",
            "model.json",
            "--scene-id",
            "102343992",
            "--max-steps",
            "1",
        ]
    )
    assert args.max_steps == 1


def test_config_cli_smoke(capsys):
    assert main(["config", "paper_fullval_v1"]) == 0
    value = json.loads(capsys.readouterr().out)
    assert value["profile"] == "paper_fullval_v1"
    assert value["agent"]["prompt_version"] == "v4"


def test_prepare_hssd_cli_forwards_an_offline_supplement(
    monkeypatch, capsys, tmp_path
):
    captured = {}

    def fake_prepare(source, output, **options):
        captured.update({"source": source, "output": output, **options})
        return {"output_root": output}

    monkeypatch.setattr(hssd, "prepare_hssd", fake_prepare)
    archive = tmp_path / "supplement.tar.gz"
    assert (
        main(
            [
                "prepare-hssd",
                "--hssd-root",
                str(tmp_path / "hssd"),
                "--output",
                str(tmp_path / "prepared"),
                "--supplement",
                str(archive),
            ]
        )
        == 0
    )
    assert captured["supplement"] == str(archive)
    assert captured["verify_hashes"] is True
    assert json.loads(capsys.readouterr().out)["output_root"] == str(
        tmp_path / "prepared"
    )


def test_run_cli_maps_public_options_to_the_episode_dispatcher(
    monkeypatch, capsys, tmp_path
):
    captured = {}

    monkeypatch.setattr(batch, "resolve_devices", lambda _value: ("2", "3"))

    def fake_run_batch(config):
        captured.update(config)
        return {
            "selected": 100,
            "completed": 100,
            "failed": 0,
        }

    monkeypatch.setattr(batch, "run_batch", fake_run_batch)
    assert (
        main(
            [
                "run",
                "--episodes",
                "val100",
                "--model-config",
                str(tmp_path / "model.json"),
                "--output",
                str(tmp_path / "out"),
                "--gpus",
                "auto",
                "--workers-per-gpu",
                "3",
                "--video",
                "--metrics",
                "--resume",
                "--agent-asset",
                "finger-separated",
            ]
        )
        == 0
    )
    assert captured["episode_list"] == "resources/benchmark/val100.json"
    assert captured["devices"] == ("2", "3")
    assert captured["max_parallel"] == 6
    assert captured["save_video"] is True
    assert captured["compute_metrics"] is True
    assert captured["resume"] is True
    assert captured["agent_asset"] == "finger-separated"
    assert json.loads(capsys.readouterr().out)["completed"] == 100


def test_metrics_cli_prints_tables_and_optionally_writes_json(capsys, tmp_path):
    episode = tmp_path / "shard00" / "episode"
    episode.mkdir(parents=True)
    metric = {
        "success": {
            "find_sr": True,
            "geo_find_sr": True,
            "nav_sr_20cm": True,
            "geo_nav_sr_20cm": True,
            "nav_sr_1m": True,
            "is_interact_episode": True,
            "interact_sr": True,
            "geo_interact_sr": True,
        },
        "body_scene": {
            "physical_metrics_eligible": True,
            "collision_step_fraction": 0.25,
            "by_body_group_step_fraction": {
                "hand_arm": 0.1,
                "torso": 0.2,
                "leg": 0.3,
                "head": 0.4,
            },
            "affected_dynamic_object_count": 2,
            "mapped_affected_dynamic_object_count": 2,
            "affected_object_path_length_sum_m": 3.0,
        },
        "action_quality": {"motion_jerk_m_s3": 5.0},
        "cost": {
            "decision_steps": 10,
            "input_tokens": 100,
            "visible_output_tokens": 20,
            "token_source": "provider_exact",
        },
    }
    (episode / "metrics.json").write_text(json.dumps(metric))

    assert main(["metrics", str(tmp_path)]) == 0
    assert "HumanClawBench paper metrics" in capsys.readouterr().out
    assert not (tmp_path / "metrics_summary.json").exists()

    assert main(["metrics", str(tmp_path), "--json", "--write-json"]) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["counts"]["episodes"] == 1
    assert (tmp_path / "metrics_summary.json").is_file()
