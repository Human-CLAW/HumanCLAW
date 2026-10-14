"""Read experiment traces without importing simulators or provider clients.

Token totals include failed provider attempts. Unknown usage, prices and task
outcomes stay unknown; observed subtotals are never presented as complete bills.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
from typing import Any


class TraceEvents(list):
    """A normal event list carrying non-fatal reader diagnostics."""

    def __init__(self) -> None:
        """Start with no events and separate reader error/warning lists."""
        super().__init__()
        self.diagnostics: dict[str, list[str]] = {"errors": [], "warnings": []}


def _mapping(value: Any) -> dict[str, Any]:
    """Accept JSON objects and treat other optional payloads as absent."""
    return value if isinstance(value, dict) else {}


def _number(value: Any) -> float | None:
    """Accept finite nonnegative numeric measurements, excluding booleans."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if math.isfinite(value) and value >= 0 else None


def _first_number(value: dict[str, Any], *keys: str) -> int | None:
    """Read the first valid integral token count among provider aliases."""
    for key in keys:
        number = _number(value.get(key))
        if number is not None and number.is_integer():
            return int(number)
    return None


def read_events(run_dir: str | Path) -> TraceEvents:
    """Read JSONL, tolerating only a malformed, unterminated final record.

    Inspect ``result.diagnostics`` for errors and warnings. Invalid interior
    records are reported and skipped so partial runs remain inspectable.
    """

    events = TraceEvents()
    path = Path(run_dir) / "events.jsonl"
    try:
        lines = path.read_bytes().splitlines(keepends=True)
    except OSError as exc:
        events.diagnostics["errors"].append(f"Cannot read events.jsonl: {exc}")
        return events
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            event = json.loads(line)
            if not isinstance(event, dict):
                raise ValueError("event must be a JSON object")
        except (ValueError, UnicodeDecodeError) as exc:
            tail = index == len(lines) - 1 and not line.endswith((b"\n", b"\r"))
            level = "warnings" if tail else "errors"
            prefix = "Ignored truncated final record" if tail else "Invalid record"
            events.diagnostics[level].append(f"{prefix} at line {index + 1}: {exc}")
            continue
        events.append(event)
    return events


def _manifest(run_dir: Path, diagnostics: dict[str, list[str]]) -> dict[str, Any]:
    """Read run metadata without hiding missing or malformed manifests."""
    try:
        value = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError("manifest must be an object")
        return value
    except (OSError, ValueError) as exc:
        diagnostics["errors"].append(f"Cannot read run.json: {exc}")
        return {}


def _spans(events: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Pair first start/end events by span ID, preserving partial pairs."""
    spans: dict[str, dict[str, Any]] = {}
    for event in events:
        if event.get("event") not in ("span.start", "span.end"):
            continue
        span_id = event.get("span_id")
        if not isinstance(span_id, str) or not span_id:
            continue
        record = spans.setdefault(span_id, {})
        record.setdefault("start" if event["event"] == "span.start" else "end", event)
    return spans


def _artifact_refs(value: Any):
    """Yield nested content-addressed artifact references from JSON payloads."""
    if isinstance(value, dict):
        if isinstance(value.get("artifact"), str):
            yield value
        for child in value.values():
            yield from _artifact_refs(child)
    elif isinstance(value, list):
        for child in value:
            yield from _artifact_refs(child)


def _validate(
    run_dir: Path, events: TraceEvents, manifest: dict[str, Any],
    diagnostics: dict[str, list[str]],
) -> dict[str, Any]:
    """Check event identity, nesting, timing and artifact integrity."""
    errors, warnings = diagnostics["errors"], diagnostics["warnings"]
    seen_events: set[str] = set()
    previous_sequence = 0
    span_counts: Counter = Counter()
    for event in events:
        sequence = event.get("sequence")
        label = f"Event {sequence!r}"
        if event.get("schema") != "humanclaw_trace_v1":
            errors.append(f"{label}: unsupported trace schema")
        event_id = event.get("event_id")
        if not isinstance(event_id, str) or not event_id:
            errors.append(f"{label}: missing event_id")
        elif event_id in seen_events:
            errors.append(f"{label}: duplicate event_id {event_id}")
        else:
            seen_events.add(event_id)
        if not isinstance(sequence, int) or isinstance(sequence, bool) or sequence <= previous_sequence:
            errors.append(f"{label}: sequence must strictly increase")
        else:
            if sequence != previous_sequence + 1:
                warnings.append(f"{label}: sequence gap after {previous_sequence}")
            previous_sequence = sequence
        if not isinstance(event.get("data"), dict):
            errors.append(f"{label}: data must be an object")
        if _number(event.get("monotonic_ns")) is None:
            errors.append(f"{label}: invalid monotonic_ns")
        if manifest.get("run_id") and event.get("run_id") != manifest["run_id"]:
            errors.append(f"{label}: run_id differs from manifest")
        if event.get("event") in ("span.start", "span.end"):
            span_id = event.get("span_id")
            if not isinstance(span_id, str) or not span_id:
                errors.append(f"{label}: missing span_id")
            else:
                key = (span_id, event["event"])
                span_counts[key] += 1
                if span_counts[key] > 1:
                    errors.append(f"{label}: duplicate {event['event']} for {span_id}")
    spans = _spans(events)
    for span_id, pair in spans.items():
        start, end = pair.get("start"), pair.get("end")
        if start is None:
            errors.append(f"Span {span_id}: end has no start")
            continue
        if end is None:
            warnings.append(f"Span {span_id}: incomplete (no end)")
        else:
            start_ns, end_ns = _number(start.get("monotonic_ns")), _number(end.get("monotonic_ns"))
            if start_ns is not None and end_ns is not None and end_ns < start_ns:
                errors.append(f"Span {span_id}: end precedes start")
            if start.get("kind") != end.get("kind") or start.get("name") != end.get("name"):
                errors.append(f"Span {span_id}: end kind/name differs from start")
        visited = {span_id}
        parent = start.get("parent_span_id")
        while parent:
            if not isinstance(parent, str):
                errors.append(f"Span {span_id}: parent_span_id must be a string or null")
                break
            if parent in visited:
                errors.append(f"Span {span_id}: parent cycle")
                break
            visited.add(parent)
            if parent not in spans or "start" not in spans[parent]:
                errors.append(f"Span {span_id}: parent {parent} has no start")
                break
            parent = spans[parent]["start"].get("parent_span_id")
    run_starts = [event for event in events if event.get("event") == "run.start"]
    run_ends = [event for event in events if event.get("event") == "run.end"]
    if not run_starts:
        warnings.append("Missing run.start")
    elif len(run_starts) > 1:
        errors.append("Duplicate run.start events")
    if not run_ends:
        warnings.append("Missing run.end; run may be interrupted or still active")
    else:
        if len(run_ends) > 1:
            errors.append("Duplicate run.end events")
        end_status = _mapping(run_ends[-1].get("data")).get("status")
        if end_status not in ("completed", "failed", "interrupted"):
            errors.append("run.end has missing or invalid terminal status")
        if manifest.get("status") != end_status:
            errors.append("Manifest status differs from run.end status")
    checked: set[str] = set()
    for reference in _artifact_refs([manifest, events]):
        relative = reference["artifact"]
        if relative in checked:
            continue
        checked.add(relative)
        path = (run_dir / relative).resolve()
        if Path(relative).is_absolute() or not path.is_relative_to(run_dir.resolve()):
            errors.append(f"Artifact reference escapes run directory: {relative}")
            continue
        try:
            content = path.read_bytes()
        except OSError:
            errors.append(f"Missing artifact: {relative}")
            continue
        digest = reference.get("sha256")
        if digest and hashlib.sha256(content).hexdigest() != digest:
            errors.append(f"Artifact checksum mismatch: {relative}")
    return {"ok": not errors, "errors": errors, "warnings": warnings}


def validate_trace(run_dir: str | Path) -> dict[str, Any]:
    """Validate trace relationships and referenced artifacts, including partial runs."""

    directory = Path(run_dir)
    events = read_events(directory)
    diagnostics = {key: list(value) for key, value in events.diagnostics.items()}
    manifest = _manifest(directory, diagnostics)
    return _validate(directory, events, manifest, diagnostics)


def normalize_model_usage(value: Any) -> dict[str, Any]:
    """Normalize raw provider usage, keeping totals and their subsets separate.

    OpenAI completion/output totals already include reasoning. Gemini candidate
    totals and queue-worker visible totals exclude thoughts; combine them only
    when the separate thought count is known. Cached tokens are an input subset.
    """

    usage = _mapping(value)
    nested = _mapping(usage.get("usageMetadata") or usage.get("usage_metadata"))
    if nested:
        usage = nested
    prompt = _first_number(usage, "input_tokens", "prompt_tokens", "prompt_token_count", "promptTokenCount")
    reasoning = _first_number(usage, "reasoning_tokens", "num_reasoning_tokens", "thoughts_token_count", "thoughtsTokenCount")
    if reasoning is None:
        reasoning = _first_number(_mapping(usage.get("completion_tokens_details") or usage.get("output_tokens_details")), "reasoning_tokens")
    cached = _first_number(usage, "cached_input_tokens", "cached_tokens", "cached_content_token_count", "cachedContentTokenCount")
    if cached is None:
        cached = _first_number(_mapping(usage.get("prompt_tokens_details") or usage.get("input_tokens_details")), "cached_tokens")
    visible = _first_number(usage, "visible_output_tokens", "candidates_token_count", "candidatesTokenCount")
    output = _first_number(usage, "output_tokens", "completion_tokens")
    if "num_reasoning_tokens" in usage and "completion_tokens" in usage and visible is None:
        # Historical Gemini queue workers called candidate-only tokens completion.
        visible = _first_number(usage, "completion_tokens")
        output = None
    total = _first_number(usage, "total_tokens", "total_token_count", "totalTokenCount")
    warnings: list[str] = []
    if visible is not None:
        if reasoning is not None and output is None:
            output = visible + reasoning
        elif reasoning is not None and output != visible + reasoning:
            warnings.append("visible plus reasoning differs from inclusive output total")
        elif output is None and total is not None and prompt is not None and total >= prompt + visible:
            output = total - prompt
            reasoning = output - visible
    elif output is not None and reasoning is not None and reasoning <= output:
        visible = output - reasoning
    if cached is not None and prompt is not None and cached > prompt:
        warnings.append("cached input exceeds input total")
        cached = None
    if reasoning is not None and output is not None and reasoning > output:
        warnings.append("reasoning exceeds output total")
        reasoning = None
        visible = None
    if total is None and prompt is not None and output is not None:
        total = prompt + output
    if total is not None and prompt is not None and output is not None and total != prompt + output:
        warnings.append("provider total differs from input plus output")
    return {
        "input_tokens": prompt, "cached_input_tokens": cached,
        "output_tokens": output, "reasoning_tokens": reasoning,
        "visible_output_tokens": visible, "total_tokens": total,
        "warnings": warnings,
    }


def _price_usage(usage: dict[str, Any], pricing: Any) -> dict[str, Any]:
    """Price known inclusive token totals using an explicit versioned rate card."""
    price = _mapping(pricing)
    input_price = _number(price.get("input_per_million"))
    cache_price = _number(price.get("cached_input_per_million"))
    output_price = _number(price.get("output_per_million"))
    valid_price = bool(price.get("version")) and price.get("currency") == "USD"
    reasons: list[str] = []
    input_cost = output_cost = None
    if not valid_price:
        reasons.append("missing versioned USD pricing")
    else:
        prompt, cached, output = usage["input_tokens"], usage["cached_input_tokens"], usage["output_tokens"]
        if prompt == 0:
            input_cost = 0.0
        elif prompt is not None and input_price is not None:
            if cached == 0 or cache_price == input_price:
                input_cost = prompt * input_price / 1_000_000
            elif cached is not None and cache_price is not None:
                input_cost = ((prompt - cached) * input_price + cached * cache_price) / 1_000_000
            else:
                reasons.append("unknown cached input quantity or cache price")
        else:
            reasons.append("missing input usage or input price")
        if output == 0:
            output_cost = 0.0
        elif output is not None and output_price is not None:
            output_cost = output * output_price / 1_000_000
        else:
            reasons.append("missing inclusive output usage or output price")
    known = sum(value for value in (input_cost, output_cost) if value is not None)
    return {
        "total_usd": input_cost + output_cost if input_cost is not None and output_cost is not None else None,
        "known_subtotal_usd": known,
        "input_usd": input_cost, "output_usd": output_cost,
        "unknown_reasons": reasons,
    }


def _interval(pair: dict[str, Any]) -> tuple[float, float] | None:
    """Return a closed span's monotonic interval in milliseconds when valid."""
    start = _number(_mapping(pair.get("start")).get("monotonic_ns"))
    end = _number(_mapping(pair.get("end")).get("monotonic_ns"))
    return (start / 1e6, end / 1e6) if start is not None and end is not None and end >= start else None


def _union_ms(intervals: list[tuple[float, float]]) -> float:
    """Measure covered wall time without double counting nested/overlapping spans."""
    result = 0.0
    previous_end = -math.inf
    for start, end in sorted(intervals):
        result += max(0.0, end - max(start, previous_end))
        previous_end = max(previous_end, end)
    return result


def _duration(start: Any, end: Any) -> float | None:
    """Compute a nonnegative ISO timestamp difference as a timing fallback."""
    try:
        value = (datetime.fromisoformat(end.replace("Z", "+00:00")) - datetime.fromisoformat(start.replace("Z", "+00:00"))).total_seconds() * 1000
        return value if value >= 0 else None
    except (AttributeError, TypeError, ValueError):
        return None


def _percentile(values: list[float], fraction: float) -> float | None:
    """Compute an interpolated percentile over closed span durations."""
    if not values:
        return None
    ordered = sorted(values)
    index = fraction * (len(ordered) - 1)
    low, high = math.floor(index), math.ceil(index)
    return ordered[low] + (ordered[high] - ordered[low]) * (index - low)


def _latency(events: list[dict[str, Any]], spans: dict[str, Any], manifest: dict[str, Any]) -> dict[str, Any]:
    """Summarize run wall time and per-kind duration distributions and coverage."""
    intervals: dict[str, list[tuple[float, float]]] = defaultdict(list)
    started: Counter = Counter()
    incomplete: Counter = Counter()
    for pair in spans.values():
        event = pair.get("start") or pair.get("end") or {}
        kind = str(event.get("kind", "unknown"))
        started[kind] += 1
        interval = _interval(pair)
        if interval:
            intervals[kind].append(interval)
        else:
            incomplete[kind] += 1
    starts = [event for event in events if event.get("event") == "run.start"]
    ends = [event for event in events if event.get("event") == "run.end"]
    run_interval = _interval({"start": starts[0] if starts else {}, "end": ends[-1] if ends else {}})
    wall = run_interval[1] - run_interval[0] if run_interval else _number(manifest.get("duration_ms"))
    if wall is None:
        wall = _duration(manifest.get("started_at_utc"), manifest.get("finished_at_utc"))
    observed = [_number(event.get("monotonic_ns")) for event in events]
    observed = [value for value in observed if value is not None]
    by_kind = {}
    for kind in sorted(started):
        durations = [end - start for start, end in intervals[kind]]
        by_kind[kind] = {
            "spans": started[kind], "closed_spans": len(intervals[kind]),
            "incomplete_spans": incomplete[kind],
            "inclusive_ms": sum(durations),
            "union_ms": _union_ms(intervals[kind]),
            "mean_ms": sum(durations) / len(durations) if durations else None,
            "p50_ms": _percentile(durations, 0.5),
            "p95_ms": _percentile(durations, 0.95),
            "max_ms": max(durations) if durations else None,
        }
    return {
        "run_wall_ms": wall,
        "observed_elapsed_ms": (max(observed) - min(observed)) / 1e6 if observed else None,
        "by_kind": by_kind,
        "all_spans_union_ms": _union_ms([interval for group in intervals.values() for interval in group]),
        "incomplete_spans": sum(incomplete.values()),
    }


def _model_records(spans: dict[str, Any], pricing: Any) -> list[dict[str, Any]]:
    """Normalize only leaf model attempts, excluding accounting wrapper spans."""
    models = {span_id for span_id, pair in spans.items() if (pair.get("start") or pair.get("end") or {}).get("kind") == "model"}
    nonleaf: set[str] = set()
    for span_id in models:
        parent = _mapping(spans[span_id].get("start")).get("parent_span_id")
        visited = {span_id}
        while isinstance(parent, str) and parent in spans and parent not in visited:
            visited.add(parent)
            if parent in models:
                nonleaf.add(parent)
            parent = _mapping(spans[parent].get("start")).get("parent_span_id")
    records = []
    for span_id, pair in spans.items():
        if span_id not in models - nonleaf:
            continue
        event = pair.get("start") or pair.get("end") or {}
        end = _mapping(pair.get("end"))
        data = _mapping(end.get("data"))
        usage = normalize_model_usage(data.get("usage"))
        records.append({
            "span_id": span_id, "name": event.get("name"), "step": event.get("step"),
            "status": data.get("status", "incomplete"), "closed": bool(end),
            "usage": usage, "cost": _price_usage(usage, pricing),
        })
    return records


_TOKEN_FIELDS = ("input_tokens", "cached_input_tokens", "output_tokens", "reasoning_tokens", "visible_output_tokens", "total_tokens")


def _usage_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Keep complete totals distinct from known subtotals and field coverage."""
    tokens = {}
    for field in _TOKEN_FIELDS:
        known = [record["usage"][field] for record in records if record["usage"][field] is not None]
        tokens[field] = {
            "total": sum(known) if records and len(known) == len(records) else None,
            "known_subtotal": sum(known), "known_calls": len(known),
            "unknown_calls": len(records) - len(known),
            "coverage": len(known) / len(records) if records else None,
        }
    return {
        "leaf_model_calls": len(records), "closed_calls": sum(record["closed"] for record in records),
        "status_counts": dict(Counter(record["status"] for record in records)),
        "tokens": tokens, "calls": records,
    }


def _cost_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate priced components without turning unknown bills into zeros."""
    complete = [record["cost"]["total_usd"] for record in records if record["cost"]["total_usd"] is not None]
    return {
        "currency": "USD", "total_usd": sum(complete) if records and len(complete) == len(records) else None,
        "known_subtotal_usd": sum(record["cost"]["known_subtotal_usd"] for record in records),
        "priced_calls": len(complete), "unpriced_calls": len(records) - len(complete),
        "coverage": len(complete) / len(records) if records else None,
        "is_estimate": True,
    }


_CAPABILITY_COUNTS = (
    "registered", "exposed", "requested", "rejected", "cancelled", "executed",
    "completed", "failed", "interrupted", "incomplete", "unknown_status", "orphan_ends",
)
_CAPABILITY_SEMANTICS = {
    "count_semantics": "Recorded events only; absent instrumentation does not imply no activity. "
                       "Exposure/request counts for internal modules are not applicable unless instrumented.",
    "failure_rate_semantics": "failed / (completed + failed), or null with no resolved executions. "
                              "Interrupted, incomplete, unknown-status and orphan spans, rejected/cancelled "
                              "requests are excluded. Coverage is resolved executions / started executions. "
                              "Execution completion does not imply benchmark task success.",
    "latency_semantics": "Milliseconds over valid paired monotonic intervals, including failed/interrupted "
                         "executions. Missing/invalid intervals are excluded, not zero-duration samples. "
                         "inclusive_ms sums durations; union_ms merges overlaps within each run and name. "
                         "Across runs, union_ms sums per-run unions, not collection wall time. "
                         "Do not add different names or kinds to infer end-to-end latency. "
                         "Percentiles use linear interpolation over pooled call durations.",
}


def _failure_stats(counts: dict[str, int]) -> dict[str, Any]:
    """Keep unresolved executions outside the failure-rate denominator."""
    resolved = counts["completed"] + counts["failed"]
    return {
        "failure_rate": counts["failed"] / resolved if resolved else None,
        "failure_rate_denominator": resolved,
        "failure_rate_coverage": resolved / counts["executed"] if counts["executed"] else None,
    }


def _capability_latency(
    durations: list[float], *, union_ms: float, spans: int, incomplete: int, invalid: int,
) -> dict[str, Any]:
    """Report the observed timing subtotal and coverage without inventing samples."""
    return {
        "spans": spans, "closed_spans": len(durations),
        "incomplete_spans": incomplete, "invalid_spans": invalid,
        "coverage": len(durations) / spans if spans else None,
        "inclusive_ms": sum(durations), "union_ms": union_ms,
        "mean_ms": sum(durations) / len(durations) if durations else None,
        "p50_ms": _percentile(durations, 0.5), "p95_ms": _percentile(durations, 0.95),
        "max_ms": max(durations) if durations else None,
    }


def _capabilities(
    events: list[dict[str, Any]], spans: dict[str, Any], metadata: dict[str, Any],
) -> tuple[dict[str, Any], dict[tuple[str, str], list[float]]]:
    """Summarize capability executions; retain samples in memory for collection statistics."""
    rows: dict[tuple[str, str], Counter] = defaultdict(Counter)
    availability: dict[tuple[str, str], bool] = {}
    intervals: dict[tuple[str, str], list[tuple[float, float]]] = defaultdict(list)
    incomplete: Counter = Counter()
    invalid: Counter = Counter()
    for event in events:
        kind = str(event.get("kind", "unknown"))
        if kind not in {"tool", "module"}:
            continue
        name = str(event.get("name", "unknown"))
        event_type = event.get("event")
        if event_type == "capability.registered":
            available = _mapping(event.get("data")).get("available")
            if isinstance(available, bool):
                availability[(kind, name)] = available
        if event_type in ("capability.registered", "capability.exposed", "tool.requested", "tool.rejected", "tool.cancelled"):
            rows[(kind, name)][str(event_type).split(".")[-1]] += 1
    for pair in spans.values():
        event = pair.get("start") or pair.get("end") or {}
        if event.get("kind") not in {"tool", "module"}:
            continue
        key = (event["kind"], str(event.get("name", "unknown")))
        row = rows[key]
        if "start" not in pair:
            row["orphan_ends"] += 1
        else:
            row["executed"] += 1
            status = _mapping(_mapping(pair.get("end")).get("data")).get("status")
            if "end" not in pair:
                row["incomplete"] += 1
            elif status in ("completed", "failed", "interrupted"):
                row[status] += 1
            else:
                row["unknown_status"] += 1
        interval = _interval(pair)
        if interval is not None:
            intervals[key].append(interval)
        elif "start" not in pair or "end" not in pair:
            incomplete[key] += 1
        else:
            invalid[key] += 1
    samples = {}
    entries = []
    for key, counts in sorted(rows.items()):
        kind, name = key
        durations = samples[key] = [end - start for start, end in intervals[key]]
        entries.append({
            "kind": kind, "name": name, "available": availability.get(key),
            **{column: counts[column] for column in _CAPABILITY_COUNTS}, **_failure_stats(counts),
            "latency": _capability_latency(
                durations, union_ms=_union_ms(intervals[key]),
                spans=counts["executed"] + counts["orphan_ends"],
                incomplete=incomplete[key], invalid=invalid[key],
            ),
        })
    return {
        "coverage": metadata.get("capability_coverage", metadata.get("instrumentation_coverage", metadata.get("observability", "unknown"))),
        **_CAPABILITY_SEMANTICS,
        "entries": entries,
    }, samples


def _aggregate_capabilities(
    results: list[tuple[dict[str, Any], dict[tuple[str, str], list[float]]]],
) -> dict[str, Any]:
    """Pool calls, not per-run means/percentiles or unrelated monotonic clocks."""
    counts: dict[tuple[str, str], Counter] = defaultdict(Counter)
    timings: dict[tuple[str, str], Counter] = defaultdict(Counter)
    samples: dict[tuple[str, str], list[float]] = defaultdict(list)
    availability: dict[tuple[str, str], Counter] = defaultdict(Counter)
    for summary, durations in results:
        for entry in summary["capabilities"]["entries"]:
            key = (entry["kind"], entry["name"])
            row = counts[key]
            row.update({column: entry[column] for column in _CAPABILITY_COUNTS})
            row["runs_with_entry"] += 1
            row["runs_with_execution"] += int(entry["executed"] > 0)
            availability[key][str(entry["available"]).lower()] += 1
            timings[key].update({field: entry["latency"][field] for field in (
                "spans", "incomplete_spans", "invalid_spans", "union_ms",
            )})
            samples[key].extend(durations[key])
    entries = []
    for key, row in sorted(counts.items()):
        timing = timings[key]
        entries.append({
            "kind": key[0], "name": key[1],
            **{column: row[column] for column in _CAPABILITY_COUNTS}, **_failure_stats(row),
            "runs_with_entry": row["runs_with_entry"], "runs_with_execution": row["runs_with_execution"],
            "availability_runs": {
                "available": availability[key]["true"], "unavailable": availability[key]["false"],
                "unknown": availability[key]["none"],
            },
            "latency": _capability_latency(
                samples[key], union_ms=timing["union_ms"], spans=timing["spans"],
                incomplete=timing["incomplete_spans"], invalid=timing["invalid_spans"],
            ),
        })
    return {
        "coverage": "See each run's capabilities.coverage; missing rows do not establish zero activity.",
        "scope": "All discovered trace attempts, including previous batches; not the current batch denominator.",
        "runs": len(results), **_CAPABILITY_SEMANTICS, "entries": entries,
    }


def summarize_run(run_dir: str | Path, write: bool = False) -> dict[str, Any]:
    """Summarize a complete, failed or interrupted run; optionally save summary.json."""
    return _summarize_run(run_dir, write)[0]


def _summarize_run(
    run_dir: str | Path, write: bool,
) -> tuple[dict[str, Any], dict[tuple[str, str], list[float]]]:
    """Read once, returning private duration samples alongside the serializable report."""

    directory = Path(run_dir)
    events = read_events(directory)
    diagnostics = {key: list(value) for key, value in events.diagnostics.items()}
    manifest = _manifest(directory, diagnostics)
    validation = _validate(directory, events, manifest, diagnostics)
    spans = _spans(events)
    records = _model_records(spans, manifest.get("pricing"))
    metadata = _mapping(manifest.get("metadata"))
    outcome = _mapping(manifest.get("outcome"))
    # A process completing is not evidence that the embodied task succeeded.
    outcome = {**outcome, "task_success": outcome.get("task_success") if isinstance(outcome.get("task_success"), bool) else None}
    run_end = next((event for event in reversed(events) if event.get("event") == "run.end"), {})
    status = manifest.get("status") or _mapping(run_end.get("data")).get("status") or "unknown"
    capabilities, capability_samples = _capabilities(events, spans, metadata)
    result = {
        "schema": "humanclaw_experiment_summary_v1", "run_dir": str(directory.resolve()),
        "run_id": manifest.get("run_id") or (events[0].get("run_id") if events else directory.name),
        "session_id": manifest.get("session_id"), "episode_id": manifest.get("episode_id"),
        "status": status, "outcome": outcome, "metadata": metadata, "pricing": manifest.get("pricing"),
        "latency": _latency(events, spans, manifest), "model_usage": _usage_summary(records),
        "cost": _cost_summary(records), "capabilities": capabilities,
        "trace_diagnostics": validation,
    }
    accounting_complete = (
        bool(run_end) and not validation["errors"]
        and not validation["warnings"] and all(record["closed"] for record in records)
    )
    result["model_usage"]["trace_accounting_complete"] = accounting_complete
    result["cost"]["trace_accounting_complete"] = accounting_complete
    if not accounting_complete:
        result["cost"]["total_usd"] = None
        for tokens in result["model_usage"]["tokens"].values():
            tokens["total"] = None
    if write:
        (directory / "summary.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return result, capability_samples


def summarize_runs(root: str | Path, write: bool = False) -> dict[str, Any]:
    """Aggregate every discovered trace, retaining unsuccessful/unscored runs."""

    directory = Path(root)
    candidates = {path.parent for path in directory.rglob("run.json")}
    candidates.update(path.parent for path in directory.rglob("events.jsonl"))
    results = [_summarize_run(path, write) for path in sorted(candidates)]
    runs = [summary for summary, _ in results]
    scored = [run for run in runs if isinstance(run["outcome"].get("task_success"), bool)]
    successful = sum(run["outcome"]["task_success"] for run in scored)
    records = [record for run in runs for record in run["model_usage"]["calls"]]
    wall_values = [run["latency"]["run_wall_ms"] for run in runs if run["latency"]["run_wall_ms"] is not None]
    aggregate = {
        "runs": len(runs), "status_counts": dict(Counter(run["status"] for run in runs)),
        "task_successes": successful, "scored_runs": len(scored), "unscored_runs": len(runs) - len(scored),
        "task_success_rate_all_runs": successful / len(runs) if runs else None,
        "task_success_rate_scored_runs": successful / len(scored) if scored else None,
        "outcome_coverage": len(scored) / len(runs) if runs else None,
        "model_usage": _usage_summary(records), "cost": _cost_summary(records),
        "capabilities": _aggregate_capabilities(results),
        "run_wall_ms": {
            "known_subtotal": sum(wall_values),
            "total": sum(wall_values) if len(wall_values) == len(runs) else None,
            "known_runs": len(wall_values), "unknown_runs": len(runs) - len(wall_values),
        },
        "invalid_traces": sum(not run["trace_diagnostics"]["ok"] for run in runs),
    }
    accounting_complete = bool(runs) and all(run["model_usage"]["trace_accounting_complete"] for run in runs)
    aggregate["model_usage"]["trace_accounting_complete"] = accounting_complete
    aggregate["cost"]["trace_accounting_complete"] = accounting_complete
    if not accounting_complete:
        aggregate["cost"]["total_usd"] = None
        for tokens in aggregate["model_usage"]["tokens"].values():
            tokens["total"] = None
    result = {"schema": "humanclaw_experiment_collection_v1", "root": str(directory.resolve()), "aggregate": aggregate, "runs": runs}
    if write:
        (directory / "experiment_summary.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return result


def format_timeline(run_dir: str | Path) -> str:
    """Format span nesting, terminal status and trace gaps without loading payloads."""

    events = read_events(run_dir)
    spans = _spans(events)
    origin = next((_number(event.get("monotonic_ns")) for event in events if _number(event.get("monotonic_ns")) is not None), None)
    lines = [f"Experiment timeline: {Path(run_dir).resolve()}"]
    for event in events:
        parent = event.get("parent_span_id")
        visited: set[str] = set()
        depth = 0
        while isinstance(parent, str) and parent in spans and parent not in visited:
            visited.add(parent)
            depth += 1
            parent = _mapping(spans[parent].get("start")).get("parent_span_id")
        when = _number(event.get("monotonic_ns"))
        timestamp = f"{(when - origin) / 1e6:10.3f} ms" if when is not None and origin is not None else "          ? ms"
        data = _mapping(event.get("data"))
        suffix = ""
        if event.get("event") in ("span.end", "run.end"):
            suffix = f" [{data.get('status', 'unknown')}]"
            duration = _number(data.get("duration_ms"))
            if duration is not None:
                suffix += f" {duration:.3f} ms"
            if data.get("error"):
                suffix += " error=" + json.dumps(data["error"], ensure_ascii=False)[:240]
        step = f" step={event['step']}" if event.get("step") is not None else ""
        lines.append(f"{timestamp} {'  ' * depth}{event.get('event', '?')} {event.get('kind', '?')}:{event.get('name', '?')}{step}{suffix}")
    for span_id, pair in spans.items():
        if "end" not in pair:
            lines.append(f"INCOMPLETE span={span_id} {pair['start'].get('name', '?')}")
    for level, messages in events.diagnostics.items():
        lines.extend(f"{level.upper()}: {message}" for message in messages)
    return "\n".join(lines)


__all__ = ["read_events", "validate_trace", "normalize_model_usage", "summarize_run", "summarize_runs", "format_timeline"]
