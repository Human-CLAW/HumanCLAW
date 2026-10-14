import asyncio
import base64
from concurrent.futures import ThreadPoolExecutor
import contextvars
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image
import pytest

from humanclaw_bench.experiments.recorder import TraceRecorder, event, get_recorder, span


def _events(recorder):
    return [json.loads(line) for line in (recorder.directory / "events.jsonl").read_text().splitlines()]


def test_disabled_helpers_are_noops_and_never_suppress_exceptions(tmp_path):
    class Unserializable:
        def __repr__(self):
            raise AssertionError("Disabled instrumentation must not inspect inputs")

    assert get_recorder() is None
    assert event("tool.requested", data=Unserializable()) is None
    with span("disabled", input=Unserializable()) as operation:
        assert operation.span_id is None
        operation.set_output(Unserializable())
        operation.set_usage({"input_tokens": 1})
        operation.set_attribute("x", Unserializable())
        operation.fail(Unserializable())
    with pytest.raises(KeyboardInterrupt):
        with span("disabled"):
            raise KeyboardInterrupt()
    assert not list(tmp_path.iterdir())


def test_nested_trace_records_actual_calls_usage_and_inherited_step(tmp_path):
    recorder = TraceRecorder(tmp_path, {"benchmark": "test", "model": "test-model"})
    with recorder:
        with span("episode", kind="episode", step=7) as episode:
            with span("request", kind="model", input={"messages": [{"content": "full prompt"}]}) as request:
                event("tool.exposed", kind="tool", name="observe", data={"tools": ["observe"]})
                request.set_usage({"provider": "test", "input_tokens": 12, "output_tokens": 4, "details": {"cached_tokens": 3}})
                request.set_output("full answer")
                request.set_attribute("attempt", 1)
            with span("execute", kind="environment", step=8):
                pass
        recorder.finish(outcome={"success": True})
    rows = _events(recorder)
    assert [row["sequence"] for row in rows] == list(range(1, len(rows) + 1))
    assert len({row["event_id"] for row in rows}) == len(rows)
    assert {row["schema"] for row in rows} == {"humanclaw_trace_v1"}
    assert {row["run_id"] for row in rows} == {recorder.run_id}
    assert {row["session_id"] for row in rows} == {recorder.session_id}
    assert {row["episode_id"] for row in rows} == {recorder.episode_id}
    assert all(row["timestamp_utc"].endswith("Z") for row in rows)
    assert [row["monotonic_ns"] for row in rows] == sorted(row["monotonic_ns"] for row in rows)
    request_rows = [row for row in rows if row["span_id"] == request.span_id]
    assert all(row["parent_span_id"] == episode.span_id for row in request_rows)
    assert all(row["step"] == 7 for row in request_rows)
    end = next(row for row in request_rows if row["event"] == "span.end")
    assert end["data"]["output"] == "full answer"
    assert end["data"]["usage"]["details"]["cached_tokens"] == 3
    assert end["data"]["attributes"]["attempt"] == 1
    assert end["data"]["duration_ms"] >= 0
    assert rows[-1]["event"] == "run.end"
    assert recorder.manifest["outcome"] == {"success": True}
    assert json.loads((recorder.directory / "run.json").read_text()) == recorder.manifest
    assert get_recorder() is None


def test_event_without_payload_has_valid_empty_object(tmp_path):
    with TraceRecorder(tmp_path, {}) as recorder:
        event("marker")
    marker = next(row for row in _events(recorder) if row["event"] == "marker")
    assert marker["data"] == {}


@pytest.mark.parametrize(
    ("exception", "status"),
    [(RuntimeError("failed call"), "failed"), (KeyboardInterrupt("cancelled"), "interrupted"), (SystemExit(2), "interrupted")],
)
def test_exceptions_and_base_exceptions_flush_endings_and_propagate(tmp_path, exception, status):
    recorder = TraceRecorder(tmp_path, {})
    with pytest.raises(type(exception)):
        with recorder:
            with span("outer"):
                with span("inner"):
                    raise exception
    rows = _events(recorder)
    starts = [row for row in rows if row["event"] == "span.start"]
    ends = [row for row in rows if row["event"] == "span.end"]
    assert {row["span_id"] for row in starts} == {row["span_id"] for row in ends}
    assert all(row["data"]["status"] == status for row in ends)
    assert all(row["data"]["error"]["type"] == type(exception).__name__ for row in ends)
    assert all("traceback" in row["data"]["error"] for row in ends)
    assert recorder.manifest["status"] == status
    assert rows[-1]["event"] == "run.end"
    assert get_recorder() is None


def test_caught_call_failure_does_not_fail_successful_retried_run(tmp_path):
    with TraceRecorder(tmp_path, {}) as recorder:
        with span("attempt", attributes={"attempt": 1}) as first:
            first.fail({"type": "ParseError", "message": "invalid JSON"})
        with span("attempt", attributes={"attempt": 2}) as second:
            second.set_output({"valid": True})
    ends = [row["data"] for row in _events(recorder) if row["event"] == "span.end"]
    assert [row["status"] for row in ends] == ["failed", "completed"]
    assert recorder.manifest["status"] == "completed"


def test_concurrent_copied_contexts_preserve_parent_and_serialized_events(tmp_path):
    with TraceRecorder(tmp_path, {}) as recorder:
        with span("parallel", step=3) as parent:
            def worker(index):
                with span(f"worker-{index}") as child:
                    event("tool.executed", kind="tool", name="sample", data={"index": index})
                    with span("nested"):
                        pass
                    child.set_output(index)
                return child.span_id

            with ThreadPoolExecutor(max_workers=6) as pool:
                futures = [pool.submit(contextvars.copy_context().run, worker, i) for i in range(24)]
                child_ids = {future.result() for future in futures}
    rows = _events(recorder)
    assert [row["sequence"] for row in rows] == list(range(1, len(rows) + 1))
    starts = {row["span_id"]: row for row in rows if row["event"] == "span.start"}
    ends = {row["span_id"]: row for row in rows if row["event"] == "span.end"}
    assert starts.keys() == ends.keys()
    assert all(starts[child]["parent_span_id"] == parent.span_id for child in child_ids)
    assert all(row["step"] == 3 for row in starts.values())
    assert all(row["parent_span_id"] in child_ids for row in starts.values() if row["name"] == "nested")


def test_async_tasks_do_not_cross_nest(tmp_path):
    async def run_tasks():
        async def worker(index):
            with span(f"async-{index}", step=index) as outer:
                await asyncio.sleep(0)
                with span("async-child") as child:
                    await asyncio.sleep(0)
                return outer.span_id, child.span_id

        return await asyncio.gather(worker(1), worker(2))

    with TraceRecorder(tmp_path, {}) as recorder:
        pairs = asyncio.run(run_tasks())
    starts = {row["span_id"]: row for row in _events(recorder) if row["event"] == "span.start"}
    for parent, child in pairs:
        assert starts[child]["parent_span_id"] == parent
        assert starts[child]["step"] == starts[parent]["step"]
        assert starts[parent]["parent_span_id"] is None


def test_artifacts_are_lossless_deduplicated_and_not_embedded_in_events(tmp_path):
    array = np.arange(24, dtype=np.float32).reshape(2, 4, 3)
    picture = Image.new("RGB", (4, 3), color=(10, 20, 30))
    png = io.BytesIO()
    picture.save(png, format="PNG")
    encoded = base64.b64encode(png.getvalue()).decode("ascii")
    image_url = "data:image/png;base64," + encoded
    with TraceRecorder(tmp_path, {}) as recorder:
        with span("multimodal", input={"nested": [{"image": image_url}], "array": array}) as operation:
            operation.set_output({"image": picture, "raw": png.getvalue(), "text": "data:text/plain,hello%20world"})
    start = next(row for row in _events(recorder) if row["event"] == "span.start")
    end = next(row for row in _events(recorder) if row["event"] == "span.end")
    image_ref = start["data"]["input"]["nested"][0]["image"]
    array_ref = start["data"]["input"]["array"]
    assert image_ref["artifact"] == end["data"]["output"]["image"]["artifact"]
    assert image_ref["artifact"] == end["data"]["output"]["raw"]["artifact"]
    assert np.array_equal(np.load(recorder.directory / array_ref["artifact"], allow_pickle=False), array)
    image_path = recorder.directory / image_ref["artifact"]
    assert image_path.read_bytes() == png.getvalue()
    assert hashlib.sha256(image_path.read_bytes()).hexdigest() == image_ref["sha256"]
    text_ref = end["data"]["output"]["text"]
    assert (recorder.directory / text_ref["artifact"]).read_text() == "hello world"
    assert encoded not in (recorder.directory / "events.jsonl").read_text()
    assert len(list((recorder.directory / "artifacts").iterdir())) == 3


def test_metadata_credentials_redacted_without_redacting_prompts_or_usage(tmp_path):
    metadata = {
        "api_key": "secret-a",
        "config": {"OPENAI_API_KEY": "secret-b", "Authorization": "Bearer secret-c", "client_secret": "secret-d"},
        "prompt": "Analyze the api_key field; this prompt must stay verbatim.",
        "max_tokens": 100,
    }
    with TraceRecorder(tmp_path, metadata, pricing={"input_per_million": 1.5, "api_token": "secret-e"}) as recorder:
        with span("model") as operation:
            operation.set_usage({"input_tokens": 10, "output_tokens": 2})
    text = (recorder.directory / "run.json").read_text() + (recorder.directory / "events.jsonl").read_text()
    for secret in ("secret-a", "secret-b", "secret-c", "secret-d", "secret-e"):
        assert secret not in text
    assert recorder.manifest["metadata"]["prompt"] == metadata["prompt"]
    assert recorder.manifest["metadata"]["max_tokens"] == 100
    assert recorder.manifest["pricing"]["input_per_million"] == 1.5
    assert metadata["api_key"] == "secret-a"


def test_unique_run_directories_keep_previous_results_and_finish_is_idempotent(tmp_path):
    with TraceRecorder(tmp_path, {"episode": "same"}) as first:
        event("custom", data={"value": 1})
        first.finish(outcome={"success": False})
    original = (first.directory / "run.json").read_bytes()
    first.finish(status="failed", outcome={"success": True})
    with TraceRecorder(tmp_path, {"episode": "same"}) as second:
        event("custom", data={"value": 2})
    assert first.directory != second.directory
    assert first.session_id != second.session_id
    assert first.episode_id != second.episode_id
    assert (first.directory / "run.json").read_bytes() == original
    assert len([row for row in _events(first) if row["event"] == "run.end"]) == 1


def test_nested_recorder_restores_outer_context_without_cross_run_parent(tmp_path):
    with TraceRecorder(tmp_path, {}) as outer:
        with span("outer") as parent:
            with TraceRecorder(tmp_path, {}) as inner:
                with span("inner"):
                    assert get_recorder() is inner
            assert get_recorder() is outer
            with span("outer-child"):
                pass
    inner_start = next(row for row in _events(inner) if row["event"] == "span.start")
    child_start = next(row for row in _events(outer) if row["event"] == "span.start" and row["name"] == "outer-child")
    assert inner_start["parent_span_id"] is None
    assert child_start["parent_span_id"] == parent.span_id


def test_finishing_with_open_span_marks_interruption_without_duplicate_end(tmp_path):
    with TraceRecorder(tmp_path, {}) as recorder:
        with span("unfinished"):
            recorder.finish()
    rows = _events(recorder)
    ends = [row for row in rows if row["event"] == "span.end"]
    assert len(ends) == 1
    assert ends[0]["data"]["status"] == "interrupted"
    assert recorder.manifest["status"] == "interrupted"
    assert recorder.manifest["unfinished_span_ids"] == [ends[0]["span_id"]]


def test_abrupt_process_exit_preserves_unmatched_start_and_running_manifest(tmp_path):
    code = """
import os
import sys
from humanclaw_bench.experiments.recorder import TraceRecorder, span
with TraceRecorder(sys.argv[1], {}) as recorder:
    with span('killed-call', kind='model'):
        os._exit(23)
"""
    env = dict(os.environ)
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")
    result = subprocess.run([sys.executable, "-c", code, str(tmp_path)], env=env, capture_output=True)
    assert result.returncode == 23, result.stderr.decode()
    directory = next((tmp_path / "experiment_runs").iterdir())
    rows = [json.loads(line) for line in (directory / "events.jsonl").read_text().splitlines()]
    assert [row["event"] for row in rows] == ["run.start", "span.start"]
    assert json.loads((directory / "run.json").read_text())["status"] == "running"


def test_atomic_json_writes_stay_within_run_and_preserve_previous_valid_file(tmp_path):
    with TraceRecorder(tmp_path, {}) as recorder:
        path = recorder.write_json("reports/check.json", {"complete": True})
        with pytest.raises(ValueError):
            recorder.write_json("../outside.json", {"bad": True})
        with pytest.raises(ValueError):
            recorder.write_json(str(tmp_path / "outside.json"), {"bad": True})
        assert json.loads(path.read_text()) == {"complete": True}
        assert not list(path.parent.glob(".trace-*"))
