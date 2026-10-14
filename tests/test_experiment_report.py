"""Accounting tests for incomplete traces and provider-specific token semantics."""

import json

import pytest

from humanclaw_bench.experiments.report import (
    format_timeline,
    normalize_model_usage,
    read_events,
    summarize_run,
    summarize_runs,
    validate_trace,
)


PRICE = {
    "version": "test-2026-10", "currency": "USD",
    "input_per_million": 2, "cached_input_per_million": 0.5,
    "output_per_million": 8,
}


class Run:
    def __init__(self, root, name="run-a", *, status="completed", success=True, pricing=PRICE):
        self.directory = root / name
        self.directory.mkdir(parents=True)
        self.events = []
        self.manifest = {
            "schema": "humanclaw_experiment_run_v1", "run_id": name,
            "session_id": "session-1", "episode_id": "episode-1", "status": status,
            "pricing": pricing, "metadata": {"capability_coverage": {"tools": "complete", "modules": "partial"}},
            "outcome": {"task_success": success, "termination_reason": "active_stop", "steps": 1},
        }

    def event(self, event, ms, *, span=None, parent=None, kind="run", name="run", data=None):
        sequence = len(self.events) + 1
        if event == "run.end" and data is None:
            data = {"status": self.manifest["status"]}
        self.events.append({
            "schema": "humanclaw_trace_v1", "event": event, "event_id": f"event-{sequence}",
            "sequence": sequence, "run_id": self.manifest["run_id"],
            "session_id": "session-1", "episode_id": "episode-1", "step": 0,
            "timestamp_utc": "2026-10-13T00:00:00+00:00", "monotonic_ns": int(ms * 1e6),
            "span_id": span, "parent_span_id": parent, "kind": kind, "name": name,
            "data": {} if data is None else data,
        })
        return self

    def call(self, span, start, end, usage, *, parent=None, status="completed", kind="model", name="planner"):
        self.event("span.start", start, span=span, parent=parent, kind=kind, name=name, data={"input": {}, "attributes": {}})
        self.event("span.end", end, span=span, parent=parent, kind=kind, name=name, data={
            "output": {}, "usage": usage, "attributes": {}, "status": status,
            "error": {"type": "TimeoutError", "message": "provider timeout"} if status == "failed" else None,
            "duration_ms": end - start,
        })
        return self

    def save(self, tail=""):
        (self.directory / "run.json").write_text(json.dumps(self.manifest))
        (self.directory / "events.jsonl").write_text("".join(json.dumps(event) + "\n" for event in self.events) + tail)
        return self.directory


def _usage(prompt=100, output=30, cached=20, reasoning=10):
    return {
        "prompt_tokens": prompt, "completion_tokens": output, "total_tokens": prompt + output,
        "prompt_tokens_details": {"cached_tokens": cached},
        "completion_tokens_details": {"reasoning_tokens": reasoning},
    }


@pytest.mark.parametrize("usage", [
    _usage(),
    {"input_tokens": 100, "output_tokens": 30, "total_tokens": 130,
     "input_tokens_details": {"cached_tokens": 20}, "output_tokens_details": {"reasoning_tokens": 10}},
    {"promptTokenCount": 100, "candidatesTokenCount": 20, "thoughtsTokenCount": 10,
     "cachedContentTokenCount": 20, "totalTokenCount": 130},
    {"prompt_tokens": 100, "completion_tokens": 20, "num_reasoning_tokens": 10,
     "cached_input_tokens": 20, "total_tokens": 130},
    {"prompt_tokens": 100, "visible_output_tokens": 20, "reasoning_tokens": 10,
     "cached_input_tokens": 20, "total_tokens": 130},
])
def test_provider_usage_keeps_reasoning_and_cached_tokens_as_subsets(usage):
    normalized = normalize_model_usage(usage)
    assert normalized == {
        "input_tokens": 100, "cached_input_tokens": 20,
        "output_tokens": 30, "reasoning_tokens": 10,
        "visible_output_tokens": 20, "total_tokens": 130, "warnings": [],
    }


def test_gemini_missing_thoughts_does_not_silently_become_zero():
    result = normalize_model_usage({"promptTokenCount": 100, "candidatesTokenCount": 20})
    assert result["visible_output_tokens"] == 20
    assert result["output_tokens"] is None
    assert result["reasoning_tokens"] is None
    with_total = normalize_model_usage({"usageMetadata": {
        "promptTokenCount": 100, "candidatesTokenCount": 20, "totalTokenCount": 130,
    }})
    assert with_total["output_tokens"] == 30
    assert with_total["reasoning_tokens"] == 10


def test_failed_provider_attempts_are_billed_and_wrapper_is_not_counted(tmp_path):
    run = Run(tmp_path).event("run.start", 0)
    run.event("span.start", 10, span="wrapper", kind="model", name="retry_wrapper")
    run.call("attempt-1", 20, 40, _usage(), parent="wrapper", status="failed")
    run.call("attempt-2", 45, 70, _usage(), parent="wrapper")
    run.event("span.end", 80, span="wrapper", kind="model", name="retry_wrapper", data={"status": "completed", "usage": _usage(200, 60, 40, 20)})
    path = run.event("run.end", 100, data={"status": "completed"}).save()
    summary = summarize_run(path, write=True)
    assert summary["trace_diagnostics"]["ok"]
    assert summary["model_usage"]["leaf_model_calls"] == 2
    assert summary["model_usage"]["status_counts"] == {"failed": 1, "completed": 1}
    assert summary["model_usage"]["tokens"]["output_tokens"]["total"] == 60
    assert summary["model_usage"]["tokens"]["input_tokens"]["total"] == 200
    assert summary["cost"]["total_usd"] == pytest.approx(2 * (80 * 2 + 20 * 0.5 + 30 * 8) / 1e6)
    assert summary["cost"]["coverage"] == 1
    assert (path / "summary.json").is_file()
    assert "[failed]" in format_timeline(path)
    assert "TimeoutError" in format_timeline(path)


def test_unknown_usage_and_prices_preserve_known_subtotals(tmp_path):
    run = Run(tmp_path).event("run.start", 0)
    run.call("known", 1, 3, _usage())
    run.call("missing", 3, 5, None, status="failed")
    path = run.event("run.end", 6, data={"status": "failed"}).save()
    summary = summarize_run(path)
    assert summary["cost"]["total_usd"] is None
    assert summary["cost"]["known_subtotal_usd"] > 0
    assert summary["cost"]["coverage"] == 0.5
    tokens = summary["model_usage"]["tokens"]["input_tokens"]
    assert tokens == {"total": None, "known_subtotal": 100, "known_calls": 1, "unknown_calls": 1, "coverage": 0.5}
    run.manifest["pricing"] = None
    summary = summarize_run(run.save())
    assert summary["cost"]["total_usd"] is None
    assert summary["cost"]["known_subtotal_usd"] == 0
    assert summary["model_usage"]["tokens"]["input_tokens"]["known_subtotal"] == 100


def test_missing_cache_breakdown_keeps_discounted_input_cost_unknown(tmp_path):
    run = Run(tmp_path).event("run.start", 0)
    run.call("provider", 1, 2, {"input_tokens": 100, "output_tokens": 30})
    path = run.event("run.end", 3).save()
    summary = summarize_run(path)
    assert summary["cost"]["total_usd"] is None
    assert summary["cost"]["known_subtotal_usd"] == pytest.approx(30 * 8 / 1e6)
    run.manifest["pricing"] = {**PRICE, "cached_input_per_million": 2}
    assert summarize_run(run.save())["cost"]["total_usd"] == pytest.approx((100 * 2 + 30 * 8) / 1e6)


def test_latency_unions_nested_and_overlapping_intervals(tmp_path):
    run = Run(tmp_path).event("run.start", 0)
    run.event("span.start", 10, span="outer", kind="module", name="planner")
    run.event("span.start", 20, span="a", parent="outer", kind="module", name="a")
    run.event("span.start", 30, span="b", parent="outer", kind="module", name="b")
    run.event("span.end", 60, span="a", parent="outer", kind="module", name="a", data={"status": "completed"})
    run.event("span.end", 70, span="b", parent="outer", kind="module", name="b", data={"status": "completed"})
    run.event("span.end", 90, span="outer", kind="module", name="planner", data={"status": "completed"})
    path = run.event("run.end", 100).save()
    latency = summarize_run(path)["latency"]
    assert latency["run_wall_ms"] == 100
    assert latency["by_kind"]["module"]["inclusive_ms"] == 160
    assert latency["by_kind"]["module"]["union_ms"] == 80
    assert latency["by_kind"]["module"]["mean_ms"] == pytest.approx(160 / 3)
    assert latency["by_kind"]["module"]["p50_ms"] == 40
    assert latency["by_kind"]["module"]["p95_ms"] == pytest.approx(76)
    assert latency["by_kind"]["module"]["max_ms"] == 80
    assert latency["all_spans_union_ms"] == 80


def test_interrupted_tail_keeps_unfinished_call_unknown_and_is_inspectable(tmp_path):
    run = Run(tmp_path, status="running", success=None).event("run.start", 0)
    run.call("known", 1, 3, _usage())
    run.event("span.start", 4, span="ongoing", kind="model", name="planner")
    path = run.save(tail='{"event":"span.end","span_id":')
    events = read_events(path)
    assert isinstance(events, list)
    assert len(events) == 4
    assert "truncated" in events.diagnostics["warnings"][0]
    validation = validate_trace(path)
    assert validation["ok"]
    assert any("incomplete" in warning for warning in validation["warnings"])
    summary = summarize_run(path)
    assert summary["latency"]["run_wall_ms"] is None
    assert summary["latency"]["observed_elapsed_ms"] == 4
    assert summary["latency"]["incomplete_spans"] == 1
    assert summary["model_usage"]["leaf_model_calls"] == 2
    assert summary["cost"]["total_usd"] is None
    assert summary["outcome"]["task_success"] is None
    assert "INCOMPLETE span=ongoing" in format_timeline(path)


def test_invalid_interior_records_and_broken_span_links_are_reported(tmp_path):
    run = Run(tmp_path).event("run.start", 0)
    run.call("orphan", 2, 1, _usage(), parent="missing")
    path = run.event("run.end", 3).save()
    with (path / "events.jsonl").open("a") as stream:
        stream.write("{broken}\n")
    validation = validate_trace(path)
    assert not validation["ok"]
    assert any("Invalid record" in error for error in validation["errors"])
    assert any("end precedes start" in error for error in validation["errors"])
    assert any("parent missing" in error for error in validation["errors"])


def test_trace_validation_detects_missing_or_escaping_artifacts(tmp_path):
    run = Run(tmp_path).event("run.start", 0, data={"input": {"artifact": "artifacts/missing.json"}})
    run.event("run.end", 2, data={"output": {"artifact": "../outside.json"}})
    errors = validate_trace(run.save())["errors"]
    assert any("Missing artifact" in error for error in errors)
    assert any("escapes" in error for error in errors)


def test_capabilities_distinguish_registration_exposure_requests_and_execution(tmp_path):
    run = Run(tmp_path).event("run.start", 0)
    for event in ("capability.registered", "capability.exposed", "tool.requested", "tool.rejected"):
        run.event(event, 1, kind="tool", name="recover")
    run.event("capability.registered", 2, kind="module", name="state_estimator", data={"available": True})
    run.event("capability.registered", 2, kind="module", name="metrics", data={"available": False})
    run.call("state", 3, 5, None, kind="module", name="state_estimator")
    capabilities = summarize_run(run.event("run.end", 6).save())["capabilities"]
    entries = {entry["name"]: entry for entry in capabilities["entries"]}
    assert entries["recover"]["registered"] == 1
    assert entries["recover"]["exposed"] == 1
    assert entries["recover"]["requested"] == 1
    assert entries["recover"]["rejected"] == 1
    assert entries["recover"]["executed"] == 0
    assert entries["state_estimator"]["executed"] == 1
    assert entries["state_estimator"]["completed"] == 1
    assert entries["state_estimator"]["available"] is True
    assert entries["metrics"]["available"] is False
    assert entries["metrics"]["executed"] == 0
    assert entries["recover"]["available"] is None
    assert capabilities["coverage"]["modules"] == "partial"


def test_mixed_runs_keep_failed_interrupted_and_unscored_denominators(tmp_path):
    for name, status, success in (
        ("success", "completed", True), ("task-failure", "completed", False),
        ("provider-failure", "failed", None), ("interrupted", "interrupted", None),
        ("unscored", "completed", None),
    ):
        run = Run(tmp_path, name, status=status, success=success).event("run.start", 0)
        if status != "interrupted":
            run.event("run.end", 10, data={"status": status})
        run.save()
    result = summarize_runs(tmp_path, write=True)
    summary = result["aggregate"]
    assert summary["runs"] == 5
    assert summary["scored_runs"] == 2
    assert summary["unscored_runs"] == 3
    assert summary["task_success_rate_all_runs"] == 0.2
    assert summary["task_success_rate_scored_runs"] == 0.5
    assert summary["outcome_coverage"] == 0.4
    assert summary["status_counts"] == {"completed": 3, "failed": 1, "interrupted": 1}
    assert summary["run_wall_ms"]["total"] is None
    assert (tmp_path / "experiment_summary.json").is_file()


def test_invalid_usage_subsets_and_non_numeric_values_are_not_silently_charged():
    result = normalize_model_usage({
        "input_tokens": 100, "output_tokens": 20,
        "cached_input_tokens": 101, "reasoning_tokens": 21,
    })
    assert result["cached_input_tokens"] is None
    assert result["reasoning_tokens"] is None
    assert len(result["warnings"]) == 2
    missing = normalize_model_usage({"input_tokens": True, "output_tokens": -1})
    assert missing["input_tokens"] is None
    assert missing["output_tokens"] is None


def test_missing_trace_end_invalidates_complete_bill_even_with_closed_model_calls(tmp_path):
    run = Run(tmp_path, status="running", success=None).event("run.start", 0)
    run.call("provider", 1, 2, _usage())
    summary = summarize_run(run.save())
    assert summary["cost"]["total_usd"] is None
    assert summary["cost"]["known_subtotal_usd"] > 0
    assert not summary["cost"]["trace_accounting_complete"]
    assert summary["model_usage"]["tokens"]["input_tokens"]["total"] is None
    aggregate = summarize_runs(tmp_path)["aggregate"]
    assert aggregate["cost"]["total_usd"] is None


def test_empty_usage_coverage_does_not_claim_zero_tokens_or_bill(tmp_path):
    path = Run(tmp_path).event("run.start", 0).event("run.end", 1).save()
    summary = summarize_run(path)
    assert summary["cost"]["total_usd"] is None
    assert summary["cost"]["coverage"] is None
    assert summary["model_usage"]["tokens"]["input_tokens"]["total"] is None


def test_sequence_gap_invalidates_total_bill_but_retains_known_usage(tmp_path):
    run = Run(tmp_path).event("run.start", 0)
    run.call("provider", 1, 2, _usage()).event("run.end", 3)
    # Simulate losing an entire model attempt, leaving valid JSON on every line.
    run.events[-1]["sequence"] += 2
    path = run.save()
    validation = validate_trace(path)
    assert validation["ok"]
    assert any("sequence gap" in warning for warning in validation["warnings"])
    summary = summarize_run(path)
    assert not summary["cost"]["trace_accounting_complete"]
    assert summary["cost"]["total_usd"] is None
    assert summary["cost"]["known_subtotal_usd"] > 0
    assert summary["model_usage"]["tokens"]["input_tokens"]["total"] is None
    assert summary["model_usage"]["tokens"]["input_tokens"]["known_subtotal"] == 100


def test_manifest_final_status_must_agree_with_terminal_event(tmp_path):
    run = Run(tmp_path, status="completed").event("run.start", 0)
    run.call("provider", 1, 2, _usage())
    path = run.event("run.end", 3, data={"status": "failed"}).save()
    validation = validate_trace(path)
    assert not validation["ok"]
    assert "Manifest status differs from run.end status" in validation["errors"]
    assert not summarize_run(path)["cost"]["trace_accounting_complete"]


@pytest.mark.parametrize("duplicate", ["run.start", "run.end"])
def test_duplicate_run_boundaries_invalidate_accounting(tmp_path, duplicate):
    run = Run(tmp_path).event("run.start", 0)
    run.call("provider", 1, 2, _usage()).event("run.end", 3)
    path = run.event(duplicate, 4).save()
    validation = validate_trace(path)
    assert not validation["ok"]
    assert f"Duplicate {duplicate} events" in validation["errors"]
    assert summarize_run(path)["cost"]["total_usd"] is None


def test_model_end_without_start_invalidates_accounting(tmp_path):
    run = Run(tmp_path).event("run.start", 0)
    run.event("span.end", 2, span="missing-start", kind="model", name="provider",
              data={"status": "completed", "usage": _usage()})
    path = run.event("run.end", 3).save()
    validation = validate_trace(path)
    assert not validation["ok"]
    assert any("end has no start" in error for error in validation["errors"])
    assert summarize_run(path)["cost"]["total_usd"] is None
