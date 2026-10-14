from __future__ import annotations

import json
import threading
import time
from pathlib import Path

from humanclaw_bench.vlm.filesystem_queue import FilesystemQueueModel
import pytest


def _model(queue_dir: Path) -> FilesystemQueueModel:
    return FilesystemQueueModel(
        model="test-model",
        queue_dir=queue_dir,
        output_dir=queue_dir / "unused",
        timeout_s=2.0,
        poll_interval_s=0.01,
    )


def test_response_waits_for_json_body_after_filename_is_visible(tmp_path: Path) -> None:
    """An empty network-visible response file must not consume a VLM retry."""

    queue_dir = tmp_path / "queue"
    model = _model(queue_dir)

    def worker() -> None:
        pending = queue_dir / "pending"
        request_dir: Path | None = None
        deadline = time.time() + 1.0
        while time.time() < deadline and request_dir is None:
            request_dir = next(pending.iterdir(), None)
            if request_dir is None:
                time.sleep(0.01)
        assert request_dir is not None

        response_dir = queue_dir / "done" / request_dir.name
        response_dir.mkdir(parents=True)
        response_path = response_dir / "response.json"
        response_path.write_bytes(b"")
        time.sleep(0.05)
        response_path.write_text(
            json.dumps({"content": '{"action_id": 1}', "usage": {}}),
            encoding="utf-8",
        )

    thread = threading.Thread(target=worker)
    thread.start()
    result = model.respond([{"role": "user", "content": "test"}])
    thread.join(timeout=1.0)

    assert result == '{"action_id": 1}'
    assert model.call_index == 1
    assert not any((queue_dir / "done").iterdir())


@pytest.mark.parametrize("usage", [
    {"prompt_tokens": 100, "completion_tokens": 20, "num_reasoning_tokens": 30, "total_tokens": 150},
    "Usage(prompt_tokens=100, completion_tokens=20, num_reasoning_tokens=30, total_tokens=150)",
])
def test_queue_trace_preserves_separate_reasoning_accounting(tmp_path, monkeypatch, usage):
    from humanclaw_bench.experiments.integration import call_model
    from humanclaw_bench.experiments.recorder import TraceRecorder
    from humanclaw_bench.experiments.report import summarize_run

    model = _model(tmp_path / "queue")

    def publish(messages):
        destination = model.queue_dir / "done" / "request"
        destination.mkdir(parents=True)
        (destination / "response.json").write_text(json.dumps({
            "content": '{"action_id": 1}', "usage": usage,
        }))
        return "request"

    monkeypatch.setattr(model, "_write_request", publish)
    with TraceRecorder(tmp_path / "traces", {}) as recorder:
        assert call_model(model, [{"role": "user", "content": "test"}]) == '{"action_id": 1}'
    summary = summarize_run(recorder.directory)
    assert summary["model_usage"]["tokens"]["output_tokens"]["total"] == 50
    assert summary["model_usage"]["tokens"]["reasoning_tokens"]["total"] == 30
    assert summary["model_usage"]["tokens"]["total_tokens"]["total"] == 150
    assert model.last_usage["output_tokens"] == 20  # Original paper accounting stays unchanged.

