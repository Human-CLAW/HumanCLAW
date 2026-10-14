"""Check a configured model's image/JSON interface without starting a simulator."""

from __future__ import annotations

import base64
import io
import json
import math
import os
from pathlib import Path
import time
from typing import Any

from ..environment import load_project_environment
from .factory import build_model


def _messages() -> list[dict[str, Any]]:
    """Ask about a synthetic image without disclosing its expected color in text."""
    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGB", (64, 64), (255, 0, 0)).save(buffer, format="PNG")
    url = "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")
    return [{"role": "user", "content": [
        {"type": "text", "text": (
            'Inspect the attached image. Return only a JSON object with '
            '"status": "ok" and "image_color": the dominant color you observe, '
            'as one lowercase English color word.'
        )},
        {"type": "image_url", "image_url": {"url": url}},
    ]}]


def _safe_error(exc: Exception) -> dict[str, Any]:
    """Describe failures without copying provider response bodies or credentials."""
    status = getattr(exc, "status_code", None)
    hints = {
        400: "请求参数或图像/JSON格式不被此部署支持；请核对模型的接口要求。",
        401: "认证失败，请核对 API key 是否属于当前 Azure 资源。",
        403: "访问被拒绝，请检查资源权限、网络访问策略或内容策略。",
        404: "未找到接口或部署，请核对 endpoint 和 Azure 部署名称。",
        429: "请求受到配额或速率限制，请检查 Azure 配额。",
    }
    return {"type": type(exc).__name__, "http_status": status if isinstance(status, int) else None,
            "hint": hints.get(status, "请求未完成，请检查网络、超时和模型服务状态。")}


def check_model_api(model_config: str | Path, *, timeout: float = 60.0) -> dict[str, Any]:
    """Send one image/JSON probe with SDK retries disabled and report separate checks.

    Uses the real rollout adapter, but creates no simulator, benchmark result or
    persistent log. Credentials come from the environment or release-root .env.
    Provider errors are summarized without raw exception text or response bodies.
    """
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("timeout must be finite and positive")
    path = Path(model_config).expanduser().resolve()
    config = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("Model config must be a JSON object")
    if config.get("backend") not in {"openai_compatible", "azure_openai"}:
        raise ValueError("api-check supports direct OpenAI-compatible/Azure adapters")
    load_project_environment()
    key_env = str(config.get("api_key_env") or (
        "AZURE_OPENAI_API_KEY" if config.get("backend") == "azure_openai" else "OPENAI_API_KEY"
    ))
    report: dict[str, Any] = {
        "check": "api_smoke", "passed": False, "deployment": config.get("model"),
        "request_ok": False, "json_ok": False, "image_check_ok": False,
        "usage_available": False, "usage": None, "latency_ms": None,
        "sdk_max_retries": 0, "timeout_seconds": timeout,
    }
    if not (config.get("api_key") or os.environ.get(key_env)):
        report["error"] = {"type": "MissingCredential", "hint": (
            f"当前进程未读取到 {key_env}。请在项目根目录 .env 中设置，或在已设置密钥的终端运行此命令。"
        )}
        return report
    model = None
    try:
        model = build_model(config, path.parent)
        model.client = model.client.with_options(timeout=timeout, max_retries=0)
        report["deployment"] = model.model_name
        messages = _messages()
        started = time.perf_counter()
        try:
            raw = model.respond(messages)
        finally:
            report["latency_ms"] = round((time.perf_counter() - started) * 1000, 3)
            usage = dict(getattr(model, "last_usage", {}) or {})
            # Only recognized numeric token fields are safe diagnostic output.
            token_names = {
                "prompt_tokens", "completion_tokens", "input_tokens", "output_tokens",
                "total_tokens", "cached_tokens", "reasoning_tokens", "audio_tokens",
                "accepted_prediction_tokens", "rejected_prediction_tokens",
            }
            safe_usage = {key: value for key, value in usage.items()
                          if key in token_names and isinstance(value, int) and not isinstance(value, bool)}
            for key in ("prompt_tokens_details", "completion_tokens_details", "input_tokens_details", "output_tokens_details"):
                if isinstance(usage.get(key), dict):
                    details = {name: value for name, value in usage[key].items()
                               if name in token_names and isinstance(value, int) and not isinstance(value, bool)}
                    if details:
                        safe_usage[key] = details
            report.update(usage=safe_usage or None, usage_available=bool(safe_usage))
        report["request_ok"] = True
        try:
            value = json.loads(raw)
        except (ValueError, TypeError):
            report["error"] = {"type": "InvalidJSON", "hint": (
                "接口已返回，但回复为空或不是完整 JSON；请检查输出预算和部署的 JSON 支持。"
            )}
            return report
        report["json_ok"] = isinstance(value, dict)
        report["image_check_ok"] = (
            isinstance(value, dict) and value.get("status") == "ok"
            and str(value.get("image_color", "")).strip().lower() == "red"
        )
        report["passed"] = report["json_ok"] and report["image_check_ok"]
        if not report["passed"]:
            report["error"] = {"type": "ImageCheckFailed", "hint": (
                "接口有回复，但结构或颜色识别不符合测试要求；尚不能确认图像通道正常。"
            )}
        if not report["usage_available"]:
            report["warning"] = "服务未返回可识别的 token 用量，费用统计将保持未知。"
        return report
    except ImportError:
        report["error"] = {"type": "MissingDependency", "hint": "当前 Python 环境需要安装 openai 和 Pillow。"}
        return report
    except Exception as exc:
        report["error"] = _safe_error(exc)
        return report
    finally:
        if model is not None:
            try:
                model.client.close()
            except Exception:
                pass


__all__ = ["check_model_api"]
