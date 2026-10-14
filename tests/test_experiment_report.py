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
    for name in ("recover", "metrics"):
        assert entries[name]["failure_rate"] is None
        assert entries[name]["failure_rate_denominator"] == 0
        assert entries[name]["failure_rate_coverage"] is None
        assert entries[name]["latency"]["spans"] == 0
        assert entries[name]["latency"]["inclusive_ms"] == 0
        assert entries[name]["latency"]["union_ms"] == 0
        for field in ("mean_ms", "p50_ms", "p95_ms", "max_ms", "coverage"):
            assert entries[name]["latency"][field] is None
    assert entries["state_estimator"]["failure_rate"] == 0
    assert entries["state_estimator"]["failure_rate_coverage"] == 1
    assert entries["state_estimator"]["latency"]["mean_ms"] == 2


@pytest.mark.parametrize("kind", ["tool", "module"])
def test_capability_failures_and_latency_cover_nested_parallel_and_partial_calls(tmp_path, kind):
    run = Run(tmp_path, status="interrupted", success=False).event("run.start", 0)
    if kind == "tool":
        for event in ("tool.requested", "tool.rejected", "tool.cancelled"):
            run.event(event, 1, kind=kind, name="recover")
    # One name is recursively nested and also executes in parallel.
    run.event("span.start", 10, span="outer", kind=kind, name="recover")
    run.call("child", 20, 60, None, kind=kind, name="recover", parent="outer", status="failed")
    run.call("parallel", 30, 70, None, kind=kind, name="recover")
    run.event("span.end", 90, span="outer", kind=kind, name="recover", data={"status": "completed"})
    run.call("interrupted", 80, 100, None, kind=kind, name="recover", status="interrupted")
    run.event("span.start", 105, span="open", kind=kind, name="recover")
    other_kind = "module" if kind == "tool" else "tool"
    run.call("different-kind", 1, 6, None, kind=other_kind, name="recover")
    path = run.event("run.end", 110).save()
    summary = summarize_run(path, write=True)
    entries = {(entry["kind"], entry["name"]): entry for entry in summary["capabilities"]["entries"]}
    entry = entries[kind, "recover"]
    assert entry["executed"] == 5
    assert entry["completed"] == 2
    assert entry["failed"] == entry["interrupted"] == entry["incomplete"] == 1
    assert entry["failure_rate"] == pytest.approx(1 / 3)
    assert entry["failure_rate_denominator"] == 3
    assert entry["failure_rate_coverage"] == 0.6
    assert entry["latency"] == {
        "spans": 5, "closed_spans": 4, "incomplete_spans": 1, "invalid_spans": 0,
        "coverage": 0.8, "inclusive_ms": 180, "union_ms": 90,
        "mean_ms": 45, "p50_ms": 40, "p95_ms": pytest.approx(74), "max_ms": 80,
    }
    assert entries[other_kind, "recover"]["executed"] == 1
    assert entries[other_kind, "recover"]["failure_rate"] == 0
    assert entries[other_kind, "recover"]["latency"]["mean_ms"] == 5
    assert json.loads((path / "summary.json").read_text())["capabilities"] == summary["capabilities"]


@pytest.mark.parametrize("bad_time", [None, True, "invalid", -1])
def test_capability_invalid_timing_is_not_a_zero_duration_sample(tmp_path, bad_time):
    run = Run(tmp_path).event("run.start", 0)
    run.call("zero-duration", 1, 1, None, kind="module", name="observe")
    run.call("reversed", 5, 4, None, kind="module", name="observe")
    run.call("invalid-time", 10, 20, None, kind="module", name="observe")
    run.events[-1]["monotonic_ns"] = bad_time
    run.event("span.start", 25, span="open", kind="module", name="observe")
    entries = summarize_run(run.event("run.end", 30).save())["capabilities"]["entries"]
    latency = entries[0]["latency"]
    assert latency == {
        "spans": 4, "closed_spans": 1, "incomplete_spans": 1, "invalid_spans": 2,
        "coverage": 0.25, "inclusive_ms": 0, "union_ms": 0,
        "mean_ms": 0, "p50_ms": 0, "p95_ms": 0, "max_ms": 0,
    }


def test_capability_orphan_and_unknown_status_do_not_become_resolved_executions(tmp_path):
    run = Run(tmp_path).event("run.start", 0)
    run.event("span.end", 1, span="orphan", kind="tool", name="recover", data={"status": "failed"})
    run.call("unknown", 2, 3, None, kind="module", name="observe", status=None)
    entries = {entry["name"]: entry for entry in summarize_run(run.event("run.end", 4).save())["capabilities"]["entries"]}
    assert entries["recover"]["executed"] == entries["recover"]["failed"] == 0
    assert entries["recover"]["orphan_ends"] == 1
    assert entries["recover"]["latency"]["incomplete_spans"] == 1
    assert entries["recover"]["latency"]["mean_ms"] is None
    assert entries["observe"]["executed"] == entries["observe"]["unknown_status"] == 1
    assert entries["observe"]["latency"]["mean_ms"] == 1
    for entry in entries.values():
        assert entry["failure_rate"] is None
        assert entry["failure_rate_denominator"] == 0


def test_collection_capabilities_pool_calls_and_sum_only_per_run_unions(tmp_path):
    first = Run(tmp_path, "first").event("run.start", 0)
    first.event("capability.registered", 0, kind="tool", name="turn", data={"available": True})
    first.event("capability.registered", 0, kind="tool", name="unused", data={"available": True})
    first.call("a", 1, 3, None, kind="tool", name="turn", status="failed")
    first.call("b", 2, 12, None, kind="tool", name="turn")
    first.event("run.end", 15).save()
    # Same span IDs and monotonic origin on a different run/host must not merge.
    second = Run(tmp_path, "second", status="interrupted").event("run.start", 0)
    second.event("capability.registered", 0, kind="tool", name="turn", data={"available": True})
    second.event("capability.registered", 0, kind="tool", name="unused", data={"available": False})
    second.call("a", 1, 21, None, kind="tool", name="turn")
    second.event("span.start", 30, span="b", kind="tool", name="turn")
    second.event("run.end", 40).save()
    Run(tmp_path, "unobserved").event("run.start", 0).event("run.end", 1).save()
    result = summarize_runs(tmp_path, write=True)
    capabilities = result["aggregate"]["capabilities"]
    entries = {entry["name"]: entry for entry in capabilities["entries"]}
    turn = entries["turn"]
    assert capabilities["runs"] == 3
    assert turn["runs_with_entry"] == turn["runs_with_execution"] == 2
    assert turn["executed"] == 4
    assert turn["failure_rate"] == pytest.approx(1 / 3)
    assert turn["failure_rate_denominator"] == 3
    assert turn["failure_rate_coverage"] == 0.75
    assert turn["latency"] == {
        "spans": 4, "closed_spans": 3, "incomplete_spans": 1, "invalid_spans": 0,
        "coverage": 0.75, "inclusive_ms": 32, "union_ms": 31,
        "mean_ms": pytest.approx(32 / 3), "p50_ms": 10, "p95_ms": 19, "max_ms": 20,
    }
    assert turn["availability_runs"] == {"available": 2, "unavailable": 0, "unknown": 0}
    assert entries["unused"]["runs_with_entry"] == 2
    assert entries["unused"]["runs_with_execution"] == 0
    assert entries["unused"]["availability_runs"] == {"available": 1, "unavailable": 1, "unknown": 0}
    assert entries["unused"]["failure_rate"] is None
    assert entries["unused"]["latency"]["p95_ms"] is None
    saved = json.loads((tmp_path / "experiment_summary.json").read_text())
    assert saved == result


def test_collection_without_traces_has_no_capability_observations(tmp_path):
    capabilities = summarize_runs(tmp_path)["aggregate"]["capabilities"]
    assert capabilities["runs"] == 0
    assert capabilities["entries"] == []


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
