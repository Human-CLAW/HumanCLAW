"""Offline coverage of the public model API smoke check; never call a provider."""

from __future__ import annotations

import base64
import builtins
import io
import json
from pathlib import Path
import re
import subprocess
import sys
from types import SimpleNamespace

from PIL import Image
import pytest

from humanclaw_bench.main import main
from humanclaw_bench.vlm import api_check


_KEY_ENV = "HUMANCLAW_SMOKE_TEST_KEY"
_DUMMY_KEY = "dummy-secret-must-never-appear-in-output"
_VALID_RESPONSE = '{"status":"ok","image_color":"red"}'


@pytest.fixture
def model_config(tmp_path, monkeypatch):
    """Create a complete config using an isolated dummy credential variable."""
    monkeypatch.setenv(_KEY_ENV, _DUMMY_KEY)
    config = {
        "backend": "openai_compatible",
        "model": "test-vision-deployment",
        "base_url": "https://example.invalid/openai/v1/",
        "api_key_env": _KEY_ENV,
        "max_tokens": 4096,
        "max_tokens_parameter": "max_completion_tokens",
        "temperature": 0.0,
        "send_temperature": False,
        "response_format": {"type": "json_object"},
    }
    path = tmp_path / "model.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    return path, config


class _FakeClient:
    """Retain timeout/retry settings and closure without any network transport."""

    def __init__(self):
        self.options = None
        self.closed = False

    def with_options(self, **kwargs):
        self.options = kwargs
        return self

    def close(self):
        self.closed = True


class _FakeModel:
    """Return a configurable text response or a controlled provider failure."""

    def __init__(self, raw=_VALID_RESPONSE, *, usage=None, error=None):
        self.model_name = "effective-deployment"
        self.client = _FakeClient()
        self.last_usage = usage
        self.raw = raw
        self.error = error
        self.calls = []

    def respond(self, messages):
        self.calls.append(messages)
        if self.error is not None:
            raise self.error
        return self.raw


def _install_model(monkeypatch, model):
    """Replace factory construction and retain its exact config and directory."""
    builds = []

    def build(config, output_dir):
        builds.append((config, output_dir))
        return model

    monkeypatch.setattr(api_check, "build_model", build)
    return builds


def _assert_image_probe(messages):
    """Verify actual image pixels and that text does not give away the answer."""
    assert len(messages) == 1
    assert messages[0]["role"] == "user"
    parts = messages[0]["content"]
    text = " ".join(part["text"] for part in parts if part["type"] == "text")
    assert not re.search(r"\bred\b", text, re.IGNORECASE)
    assert "JSON" in text and "image_color" in text
    image_parts = [part for part in parts if part["type"] == "image_url"]
    assert len(image_parts) == 1
    prefix, encoded = image_parts[0]["image_url"]["url"].split(",", 1)
    assert prefix == "data:image/png;base64"
    with Image.open(io.BytesIO(base64.b64decode(encoded, validate=True))) as image:
        assert image.format == "PNG"
        assert image.mode == "RGB"
        assert image.size == (64, 64)
        assert image.getextrema() == ((255, 255), (0, 0), (0, 0))


def test_success_forwards_config_and_checks_image_without_persistent_logs(
    monkeypatch, model_config
):
    path, config = model_config
    model = _FakeModel(usage={"prompt_tokens": 10, "completion_tokens": 4})
    builds = _install_model(monkeypatch, model)
    before = set(path.parent.rglob("*"))

    report = api_check.check_model_api(path, timeout=17.5)

    assert builds == [(config, path.parent)]
    assert model.client.options == {"timeout": 17.5, "max_retries": 0}
    assert model.client.closed
    assert len(model.calls) == 1
    _assert_image_probe(model.calls[0])
    assert report["passed"] is True
    assert report["request_ok"] is True
    assert report["json_ok"] is True
    assert report["image_check_ok"] is True
    assert report["deployment"] == "effective-deployment"
    assert report["usage"] == {"prompt_tokens": 10, "completion_tokens": 4}
    assert report["usage_available"] is True
    assert report["sdk_max_retries"] == 0
    assert report["timeout_seconds"] == 17.5
    assert report["latency_ms"] >= 0
    assert "warning" not in report
    assert _DUMMY_KEY not in json.dumps(report)
    assert set(path.parent.rglob("*")) == before


def test_real_factory_and_adapter_use_configured_sdk_endpoint_and_deployment(
    monkeypatch, model_config
):
    path, config = model_config
    captured = {}

    class SDKClient(_FakeClient):
        """Exercise the actual rollout adapter behind a fake SDK boundary."""

        def __init__(self, **kwargs):
            super().__init__()
            captured["constructor"] = kwargs
            captured["client"] = self
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

        def create(self, **kwargs):
            captured["request"] = kwargs
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content=_VALID_RESPONSE))],
                usage=SimpleNamespace(model_dump=lambda: {"total_tokens": 21}),
            )

    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=SDKClient))

    report = api_check.check_model_api(path, timeout=8.0)

    assert report["passed"] is True
    assert report["deployment"] == config["model"]
    assert report["usage"] == {"total_tokens": 21}
    assert captured["constructor"] == {
        "base_url": config["base_url"], "api_key": _DUMMY_KEY,
    }
    request = captured["request"]
    assert request["model"] == config["model"]
    assert request["max_completion_tokens"] == 4096
    assert request["response_format"] == {"type": "json_object"}
    assert "temperature" not in request
    assert "max_tokens" not in request
    _assert_image_probe(request["messages"])
    assert captured["client"].options == {"timeout": 8.0, "max_retries": 0}
    assert captured["client"].closed


def test_missing_credential_does_not_construct_model(monkeypatch, model_config):
    path, _ = model_config
    monkeypatch.delenv(_KEY_ENV)

    def forbid_build(*args, **kwargs):
        pytest.fail("Missing credentials must fail before constructing a provider")

    monkeypatch.setattr(api_check, "build_model", forbid_build)
    report = api_check.check_model_api(path)
    assert report["passed"] is False
    assert report["request_ok"] is False
    assert report["usage"] is None
    assert report["latency_ms"] is None
    assert report["error"]["type"] == "MissingCredential"
    assert _KEY_ENV in report["error"]["hint"]


@pytest.mark.parametrize("raw", ["", "{", "```json\n" + _VALID_RESPONSE + "\n```"])
def test_malformed_json_fails_after_successful_request(monkeypatch, model_config, raw):
    path, _ = model_config
    model = _FakeModel(raw, usage={"total_tokens": 15})
    _install_model(monkeypatch, model)

    report = api_check.check_model_api(path)

    assert report["request_ok"] is True
    assert report["passed"] is False
    assert report["json_ok"] is False
    assert report["image_check_ok"] is False
    assert report["error"]["type"] == "InvalidJSON"
    assert report["usage"] == {"total_tokens": 15}
    assert model.client.closed


@pytest.mark.parametrize("raw,json_ok", [
    ('{"status":"ok","image_color":"blue"}', True),
    ('{"status":"wrong","image_color":"red"}', True),
    ('{"status":"ok"}', True),
    ("[]", False),
    ("null", False),
])
def test_valid_json_without_correct_image_answer_fails(
    monkeypatch, model_config, raw, json_ok
):
    path, _ = model_config
    _install_model(monkeypatch, _FakeModel(raw))

    report = api_check.check_model_api(path)

    assert report["request_ok"] is True
    assert report["json_ok"] is json_ok
    assert report["image_check_ok"] is False
    assert report["passed"] is False
    assert report["error"]["type"] == "ImageCheckFailed"


@pytest.mark.parametrize("usage", [
    None,
    {},
    {"provider_metadata": _DUMMY_KEY},
    {"prompt_tokens_details": {}},
    {"output_tokens_details": {"reasoning_tokens": True, "raw": _DUMMY_KEY}},
])
def test_missing_recognizable_usage_remains_unknown(monkeypatch, model_config, usage):
    path, _ = model_config
    _install_model(monkeypatch, _FakeModel(usage=usage))

    report = api_check.check_model_api(path)

    assert report["passed"] is True
    assert report["usage"] is None
    assert report["usage_available"] is False
    assert report["warning"]
    assert _DUMMY_KEY not in json.dumps(report)


def test_usage_only_reports_recognized_numeric_token_counts(monkeypatch, model_config):
    path, _ = model_config
    usage = {
        "prompt_tokens": 13,
        "completion_tokens": 0,
        "total_tokens": True,
        "input_tokens": "13",
        "secret": _DUMMY_KEY,
        "prompt_tokens_details": {"cached_tokens": 4, "secret": _DUMMY_KEY},
        "completion_tokens_details": {"reasoning_tokens": 0, "audio_tokens": False},
    }
    _install_model(monkeypatch, _FakeModel(usage=usage))

    report = api_check.check_model_api(path)

    assert report["usage"] == {
        "prompt_tokens": 13,
        "completion_tokens": 0,
        "prompt_tokens_details": {"cached_tokens": 4},
        "completion_tokens_details": {"reasoning_tokens": 0},
    }
    assert report["usage_available"] is True
    assert _DUMMY_KEY not in json.dumps(report)


@pytest.mark.parametrize("status", [401, 404])
def test_provider_error_is_actionable_without_exposing_exception_body(
    monkeypatch, model_config, capsys, status
):
    path, _ = model_config

    class ProviderError(Exception):
        """Include sensitive provider diagnostics that must not reach output."""

        status_code = status

    error = ProviderError("Authorization: " + _DUMMY_KEY)
    error.body = {"api_key": _DUMMY_KEY}
    model = _FakeModel(error=error, usage={"total_tokens": 7, "secret": _DUMMY_KEY})
    _install_model(monkeypatch, model)

    report = api_check.check_model_api(path)

    assert report["passed"] is False
    assert report["request_ok"] is False
    assert report["error"]["http_status"] == status
    assert report["error"]["type"] == "ProviderError"
    assert report["error"]["hint"]
    assert report["usage"] == {"total_tokens": 7}
    assert report["latency_ms"] >= 0
    assert model.client.closed
    assert _DUMMY_KEY not in json.dumps(report)
    captured = capsys.readouterr()
    assert not captured.out and not captured.err


@pytest.mark.parametrize("raw,expected_exit", [
    (_VALID_RESPONSE, 0),
    ('{"status":"ok","image_color":"blue"}', 1),
])
def test_cli_smoke_returns_status_without_gpu_or_simulator_imports(
    monkeypatch, model_config, capsys, raw, expected_exit
):
    path, _ = model_config
    model = _FakeModel(raw)
    _install_model(monkeypatch, model)
    original_import = builtins.__import__
    forbidden = ("torch", "habitat_sim", "humanclaw_bench.evaluation", "humanclaw_bench.motion")

    def guarded_import(name, *args, **kwargs):
        if any(name == prefix or name.startswith(prefix + ".") for prefix in forbidden):
            pytest.fail("API smoke check must not load GPU/simulator modules: " + name)
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)

    assert main(["api-check", "--model-config", str(path), "--timeout", "9"]) == expected_exit

    report = json.loads(capsys.readouterr().out)
    assert report["check"] == "api_smoke"
    assert report["passed"] is (expected_exit == 0)
    assert len(model.calls) == 1
    assert model.client.options == {"timeout": 9.0, "max_retries": 0}
    assert _DUMMY_KEY not in json.dumps(report)
    assert sorted(entry.name for entry in path.parent.iterdir()) == ["model.json"]


def test_module_cli_missing_key_returns_nonzero_without_provider_or_gpu(tmp_path):
    """Run the actual module entry point in a clean environment without secrets."""
    path = tmp_path / "model.json"
    path.write_text(json.dumps({
        "backend": "openai_compatible", "model": "test-deployment",
        "api_key_env": _KEY_ENV,
    }), encoding="utf-8")
    source = Path(__file__).resolve().parents[1] / "src"

    process = subprocess.run(
        [sys.executable, "-m", "humanclaw_bench", "api-check", "--model-config", str(path)],
        cwd=tmp_path,
        env={
            "PYTHONPATH": str(source), "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHON_DOTENV_DISABLED": "1",
        },
        text=True, capture_output=True, timeout=20, check=False,
    )

    assert process.returncode == 1, process.stderr
    report = json.loads(process.stdout)
    assert report["passed"] is False
    assert report["error"]["type"] == "MissingCredential"
    assert report["request_ok"] is False
    assert process.stderr == ""
    assert sorted(entry.name for entry in tmp_path.iterdir()) == ["model.json"]


@pytest.mark.parametrize("timeout", [0, -1, float("inf"), float("nan")])
def test_invalid_timeout_fails_before_model_construction(monkeypatch, model_config, timeout):
    path, _ = model_config
    builds = _install_model(monkeypatch, _FakeModel())

    with pytest.raises(ValueError, match="finite and positive"):
        api_check.check_model_api(path, timeout=timeout)

    assert not builds
