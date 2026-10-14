"""Exercise dotenv bootstrapping using only disposable, synthetic credentials."""

from __future__ import annotations

import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import textwrap
from types import SimpleNamespace

import pytest

from humanclaw_bench.environment import load_project_environment
from humanclaw_bench.vlm import api_check, factory


_KEY = "HUMANCLAW_DOTENV_TEST_KEY"
_MODEL = "HUMANCLAW_DOTENV_TEST_MODEL"
_ENDPOINT = "HUMANCLAW_DOTENV_TEST_ENDPOINT"
_REFERENCE = "HUMANCLAW_DOTENV_TEST_REFERENCE"


@pytest.fixture
def dotenv_root(tmp_path, monkeypatch):
    """Enable dotenv only after selecting a temporary release root."""
    root = tmp_path / "release"
    root.mkdir()
    (root / "pyproject.toml").write_text("[project]\nname='test-release'\n", encoding="utf-8")
    monkeypatch.setenv("HUMANCLAW_BENCH_HOME", str(root))
    for name in (_KEY, _MODEL, _ENDPOINT, _REFERENCE):
        # Register restoration even when the variable was initially absent.
        monkeypatch.setenv(name, "temporary-fixture-value")
        monkeypatch.delenv(name)
    monkeypatch.delenv("PYTHON_DOTENV_DISABLED", raising=False)
    return root


def _model_config(root):
    """Write a complete adapter config referring to isolated environment names."""
    config = {
        "backend": "openai_compatible",
        "model": "fallback-deployment",
        "model_env": _MODEL,
        "base_url": "https://example.invalid/openai/v1/",
        "api_key_env": _KEY,
        "max_tokens": 256,
        "temperature": 0.0,
        "response_format": {"type": "json_object"},
    }
    path = root / "model.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    return path, config


def test_loads_explicit_release_root_and_standard_dotenv_syntax(
    dotenv_root, monkeypatch, capsys
):
    (dotenv_root / ".env").write_text(
        f"export {_KEY}='synthetic # value'\n"
        f'{_MODEL}="dotenv-deployment"\n'
        f"{_REFERENCE}=${{{_MODEL}}}\n",
        encoding="utf-8",
    )
    other = dotenv_root.parent / "unrelated-workdir"
    other.mkdir()
    (other / ".env").write_text(f"{_KEY}=wrong-working-directory\n", encoding="utf-8")
    monkeypatch.chdir(other)

    assert load_project_environment() is True

    assert os.environ[_KEY] == "synthetic # value"
    assert os.environ[_MODEL] == "dotenv-deployment"
    assert os.environ[_REFERENCE] == "dotenv-deployment"
    output = capsys.readouterr()
    assert not output.out and not output.err


def test_existing_environment_wins_including_interpolation_and_repeated_loads(
    dotenv_root, monkeypatch
):
    monkeypatch.setenv(_KEY, "shell-value")
    monkeypatch.setenv(_MODEL, "shell-deployment")
    monkeypatch.setenv(_ENDPOINT, "")
    path = dotenv_root / ".env"
    path.write_text(
        f"{_KEY}=dotenv-value\n{_MODEL}=dotenv-deployment\n"
        f"{_ENDPOINT}=https://example.invalid/\n{_REFERENCE}=${{{_MODEL}}}\n",
        encoding="utf-8",
    )

    load_project_environment()
    path.write_text(f"{_KEY}=changed-value\n{_REFERENCE}=changed-reference\n", encoding="utf-8")
    load_project_environment()

    assert os.environ[_KEY] == "shell-value"
    assert os.environ[_MODEL] == "shell-deployment"
    assert os.environ[_ENDPOINT] == ""
    assert os.environ[_REFERENCE] == "shell-deployment"


def test_missing_root_dotenv_does_not_search_parent_or_working_directory(
    dotenv_root, monkeypatch
):
    (dotenv_root.parent / ".env").write_text(f"{_KEY}=parent-value\n", encoding="utf-8")
    nested = dotenv_root / "nested"
    nested.mkdir()
    (nested / ".env").write_text(f"{_KEY}=working-directory-value\n", encoding="utf-8")
    monkeypatch.chdir(nested)

    assert load_project_environment() is False
    assert _KEY not in os.environ


def test_standard_disable_switch_prevents_loading(dotenv_root, monkeypatch):
    (dotenv_root / ".env").write_text(f"{_KEY}=must-not-load\n", encoding="utf-8")
    monkeypatch.setenv("PYTHON_DOTENV_DISABLED", "1")

    assert load_project_environment() is False
    assert _KEY not in os.environ


def test_main_bootstraps_before_environment_dependent_dispatch(
    dotenv_root, monkeypatch, capsys
):
    cli = importlib.import_module("humanclaw_bench.main")
    (dotenv_root / ".env").write_text(f"{_MODEL}=dotenv-deployment\n", encoding="utf-8")
    calls = []

    def load_config(profile):
        calls.append((profile, os.environ.get(_MODEL)))
        return SimpleNamespace(data={"loaded": True})

    monkeypatch.setattr(cli, "load_config", load_config)

    assert cli.main(["config", "test-profile"]) == 0

    assert calls == [("test-profile", "dotenv-deployment")]
    assert json.loads(capsys.readouterr().out) == {"loaded": True}


def test_api_check_bootstraps_before_missing_credential_check(
    dotenv_root, monkeypatch, capsys
):
    (dotenv_root / ".env").write_text(f"{_KEY}=synthetic-credential\n", encoding="utf-8")
    path, _ = _model_config(dotenv_root)
    built = []
    client = SimpleNamespace(with_options=lambda **kwargs: client, close=lambda: None)
    model = SimpleNamespace(
        model_name="test-deployment", client=client, last_usage=None,
        respond=lambda messages: '{"status":"ok","image_color":"red"}',
    )

    def build(config, output_dir):
        assert os.environ[_KEY] == "synthetic-credential"
        built.append(config)
        return model

    monkeypatch.setattr(api_check, "build_model", build)

    report = api_check.check_model_api(path)

    assert report["passed"] is True
    assert len(built) == 1
    assert "synthetic-credential" not in json.dumps(report)
    captured = capsys.readouterr()
    assert not captured.out and not captured.err


def test_direct_factory_bootstraps_before_resolving_model_override(
    dotenv_root, monkeypatch, capsys
):
    (dotenv_root / ".env").write_text(
        f"{_KEY}=synthetic-credential\n{_MODEL}=dotenv-deployment\n", encoding="utf-8"
    )
    _, config = _model_config(dotenv_root)
    captured = {}

    def adapter(**kwargs):
        assert os.environ[kwargs["api_key_env"]] == "synthetic-credential"
        captured.update(kwargs)
        return SimpleNamespace(model_name=kwargs["model"])

    monkeypatch.setattr(factory, "OpenAICompatibleModel", adapter)

    model = factory.build_model(config, dotenv_root)

    assert model.model_name == "dotenv-deployment"
    assert captured["api_key_env"] == _KEY
    assert captured["output_dir"] == dotenv_root
    output = capsys.readouterr()
    assert not output.out and not output.err


def test_fresh_cli_loads_dotenv_with_real_factory_and_fake_sdk(dotenv_root):
    """Check the complete bootstrap in a child with no inherited credentials."""
    (dotenv_root / ".env").write_text(
        f"{_KEY}=synthetic-credential\n{_MODEL}=dotenv-deployment\n", encoding="utf-8"
    )
    path, _ = _model_config(dotenv_root)
    source = Path(__file__).resolve().parents[1] / "src"
    program = textwrap.dedent('''
        import sys
        from types import SimpleNamespace

        class FakeClient:
            def __init__(self, **kwargs):
                assert kwargs["api_key"] == "synthetic-credential"
                self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))
            def with_options(self, **kwargs):
                assert kwargs["max_retries"] == 0
                return self
            def create(self, **kwargs):
                assert kwargs["model"] == "dotenv-deployment"
                return SimpleNamespace(
                    choices=[SimpleNamespace(message=SimpleNamespace(
                        content='{"status":"ok","image_color":"red"}'
                    ))], usage=None
                )
            def close(self):
                pass

        sys.modules["openai"] = SimpleNamespace(OpenAI=FakeClient)
        from humanclaw_bench.main import main
        raise SystemExit(main(["api-check", "--model-config", sys.argv[1]]))
    ''')

    process = subprocess.run(
        [sys.executable, "-c", program, str(path)],
        cwd=dotenv_root.parent,
        env={
            "PYTHONPATH": str(source), "PYTHONDONTWRITEBYTECODE": "1",
            "HUMANCLAW_BENCH_HOME": str(dotenv_root),
        },
        text=True, capture_output=True, timeout=20, check=False,
    )

    assert process.returncode == 0, process.stderr
    report = json.loads(process.stdout)
    assert report["passed"] is True
    assert report["deployment"] == "dotenv-deployment"
    assert report["usage"] is None
    assert process.stderr == ""
    assert "synthetic-credential" not in process.stdout
