from __future__ import annotations

from types import SimpleNamespace

import pytest

openai = pytest.importorskip("openai")

from humanclaw_bench.vlm.factory import build_model
from humanclaw_bench.vlm.openai_compatible import OpenAICompatibleModel


class _FakeCompletions:
    def __init__(self) -> None:
        self.request = None

    def create(self, **kwargs):
        self.request = kwargs
        usage = SimpleNamespace(model_dump=lambda: {"total_tokens": 7})
        message = SimpleNamespace(content='{"status":"ok"}')
        return SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=usage)


class _FakeClient:
    def __init__(self, **kwargs) -> None:
        self.constructor = kwargs
        self.chat = SimpleNamespace(completions=_FakeCompletions())


def test_azure_o_series_request_uses_provider_specific_parameters(
    monkeypatch, tmp_path
):
    clients = []

    def fake_azure_openai(**kwargs):
        client = _FakeClient(**kwargs)
        clients.append(client)
        return client

    monkeypatch.setattr(openai, "AzureOpenAI", fake_azure_openai)
    monkeypatch.setenv("TEST_AZURE_KEY", "key")
    monkeypatch.setenv("TEST_AZURE_ENDPOINT", "https://example.openai.azure.com")
    monkeypatch.setenv("TEST_AZURE_VERSION", "2025-03-01-preview")
    model = OpenAICompatibleModel(
        model="deployment",
        output_dir=tmp_path,
        client_type="azure_openai",
        api_key_env="TEST_AZURE_KEY",
        azure_endpoint_env="TEST_AZURE_ENDPOINT",
        api_version_env="TEST_AZURE_VERSION",
        max_tokens=256,
        max_tokens_parameter="max_completion_tokens",
        temperature=0.0,
        send_temperature=False,
        reasoning_effort="low",
        response_format={"type": "json_object"},
    )

    assert model.respond([{"role": "user", "content": "test"}]) == '{"status":"ok"}'
    assert model.last_usage == {"total_tokens": 7}
    assert clients[0].constructor == {
        "azure_endpoint": "https://example.openai.azure.com",
        "api_version": "2025-03-01-preview",
        "api_key": "key",
    }
    assert clients[0].chat.completions.request == {
        "model": "deployment",
        "messages": [{"role": "user", "content": "test"}],
        "max_completion_tokens": 256,
        "reasoning_effort": "low",
        "response_format": {"type": "json_object"},
    }


def test_factory_resolves_azure_deployment_from_environment(monkeypatch, tmp_path):
    monkeypatch.setattr(openai, "AzureOpenAI", _FakeClient)
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "key")
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://example.openai.azure.com")
    monkeypatch.setenv("AZURE_OPENAI_API_VERSION", "2025-03-01-preview")
    monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT", "live-deployment")
    model = build_model(
        {
            "backend": "azure_openai",
            "model": "placeholder",
            "model_env": "AZURE_OPENAI_DEPLOYMENT",
            "max_tokens": 128,
            "max_tokens_parameter": "max_completion_tokens",
            "temperature": 0.0,
            "send_temperature": False,
            "response_format": {"type": "json_object"},
        },
        tmp_path,
    )
    assert model.model_name == "live-deployment"


def test_invalid_max_token_parameter_is_rejected(tmp_path):
    try:
        OpenAICompatibleModel(
            model="model",
            output_dir=tmp_path,
            api_key="key",
            max_tokens_parameter="token_limit",
        )
    except ValueError as error:
        assert "max_tokens_parameter" in str(error)
    else:
        raise AssertionError("invalid token parameter was accepted")
