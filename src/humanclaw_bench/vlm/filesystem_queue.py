"""Credential-free filesystem transport used by the paper Gemini/Claude runs."""

from __future__ import annotations

import base64
import json
import os
import re
import shutil
import time
import uuid
from pathlib import Path
from typing import Any


def _decode_data_url(url: str) -> tuple[str, bytes] | None:
    """Decode an inline image data URL into its media type and raw bytes."""

    match = re.match(r"^data:([^;]+);base64,(.*)$", url, flags=re.S)
    return None if not match else (match.group(1), base64.b64decode(match.group(2)))


def _stage(messages: list[dict[str, Any]]) -> str:
    """Extract a filesystem-safe stage label from the prompt metadata."""

    text = json.dumps(messages, ensure_ascii=False)
    if "plan-and-skill chooser" in text:
        return "planner_skill"
    if "verifier for the proposed action" in text:
        return "verifier"
    return "unknown"


def _parse_usage(value: Any) -> dict[str, Any]:
    """Parse either JSON usage or the queue worker's ``Usage(...)`` string.

    The production Gemini worker reports ``completion_tokens`` as visible
    candidate tokens and ``num_reasoning_tokens`` separately.  We therefore
    normalize the former to ``output_tokens`` so downstream accounting does
    not subtract reasoning a second time.
    """

    if isinstance(value, dict):
        fields = dict(value)
        # Some queue workers serialize the Usage object as a mapping instead
        # of its repr.  In that API, completion_tokens means visible Gemini
        # candidate tokens (reasoning is a separate field), unlike OpenAI's
        # inclusive completion total.  Rename it at this provider boundary.
        if (
            "completion_tokens" in fields
            and "num_reasoning_tokens" in fields
            and "output_tokens" not in fields
            and "visible_output_tokens" not in fields
        ):
            fields["output_tokens"] = fields.pop("completion_tokens")
            fields["reasoning_tokens"] = fields.pop("num_reasoning_tokens")
        return fields
    if not isinstance(value, str):
        return {}
    fields: dict[str, Any] = {}
    patterns = {
        "prompt_tokens": r"\bprompt_tokens=(\d+)",
        "output_tokens": r"\bcompletion_tokens=(\d+)",
        "reasoning_tokens": r"\bnum_reasoning_tokens=(\d+)",
        "total_tokens": r"\btotal_tokens=(\d+)",
    }
    for key, pattern in patterns.items():
        match = re.search(pattern, value)
        if match:
            fields[key] = int(match.group(1))
    return fields


class FilesystemQueueModel:
    """Exchange VLM requests and responses through an externally served filesystem queue."""

    def __init__(
        self,
        *,
        model: str,
        queue_dir: str | Path,
        output_dir: str | Path,
        max_tokens: int = 4096,
        temperature: float = 0.0,
        reasoning_budget_tokens: int | None = None,
        response_format: dict[str, Any] | None = None,
        timeout_s: float = 900.0,
        poll_interval_s: float = 1.0,
    ) -> None:
        """Configure queue directories, polling timeout, and per-call usage state."""

        self.model_name = model
        self.queue_dir = Path(queue_dir)
        self.max_tokens = int(max_tokens)
        self.temperature = float(temperature)
        self.reasoning_budget_tokens = reasoning_budget_tokens
        self.response_format = response_format
        self.timeout_s = float(timeout_s)
        self.poll_interval_s = float(poll_interval_s)
        self.current_episode_step = -1
        self.call_index = 0
        self.last_usage: dict[str, Any] = {}
        del output_dir
        for name in ("pending", "done", "tmp"):
            (self.queue_dir / name).mkdir(parents=True, exist_ok=True)

    def _write_request(self, messages: list[dict[str, Any]]) -> str:
        """Persist one atomic request JSON plus deduplicated image payloads."""

        call_id = f"{self.call_index:06d}_step{self.current_episode_step:04d}_{_stage(messages)}_{uuid.uuid4().hex[:10]}"
        temporary = self.queue_dir / "tmp" / call_id
        pending = self.queue_dir / "pending" / call_id
        temporary.mkdir(parents=True, exist_ok=False)
        stored = json.loads(json.dumps(messages, ensure_ascii=False))
        images = []
        for message in stored:
            for item in (
                message.get("content", [])
                if isinstance(message.get("content"), list)
                else []
            ):
                image = (
                    item.get("image_url") if item.get("type") == "image_url" else None
                )
                decoded = _decode_data_url(str((image or {}).get("url") or ""))
                if decoded is None:
                    continue
                media_type, payload = decoded
                suffix = (
                    ".png"
                    if media_type == "image/png"
                    else ".jpg"
                    if media_type == "image/jpeg"
                    else ".bin"
                )
                filename = f"image_{len(images) + 1:02d}{suffix}"
                (temporary / filename).write_bytes(payload)
                item["image_url"] = {
                    "stored_image_file": filename,
                    "media_type": media_type,
                }
                images.append(
                    {
                        "image_file": filename,
                        "media_type": media_type,
                        "bytes": len(payload),
                    }
                )
        request = {
            "schema": "humanclaw_vlm_queue_request_v1",
            "call_id": call_id,
            "model": self.model_name,
            "messages": stored,
            "images": images,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "reasoning_budget_tokens": self.reasoning_budget_tokens,
            "response_format": self.response_format,
            "stage": _stage(messages),
            "episode_step": self.current_episode_step,
            "created_time": time.time(),
            "pid": os.getpid(),
        }
        (temporary / "request.json").write_text(
            json.dumps(request, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        os.replace(temporary, pending)
        self.call_index += 1
        return call_id

    def respond(self, messages: list[dict[str, Any]]) -> str:
        """Submit one queued request, await its response, and expose provider usage."""

        self.last_usage = {}
        call_id = self._write_request(messages)
        response_path = self.queue_dir / "done" / call_id / "response.json"
        deadline = time.time() + self.timeout_s
        response: dict[str, Any] | None = None
        last_read_error: OSError | UnicodeError | json.JSONDecodeError | None = None
        while time.time() < deadline:
            # Queue workers may publish through a network filesystem.  On
            # SSHFS/NFS the destination filename can become visible before
            # its small JSON body has finished crossing the wire.  Treat that
            # as an in-progress response, not as a failed model call; otherwise
            # one valid response can spuriously consume all planner retries.
            if response_path.is_file():
                try:
                    decoded = json.loads(response_path.read_text(encoding="utf-8"))
                    if not isinstance(decoded, dict):
                        raise ValueError(
                            f"Queued VLM response must be a JSON object: {call_id}"
                        )
                    response = decoded
                    break
                except (OSError, UnicodeError, json.JSONDecodeError) as exc:
                    last_read_error = exc
            time.sleep(self.poll_interval_s)
        if response is None:
            detail = (
                "response file remained incomplete"
                if last_read_error is not None
                else "response file did not appear"
            )
            raise TimeoutError(
                f"Timed out waiting for queued VLM response ({detail}): {call_id}"
            ) from last_read_error
        # Queue workers in the archived paper runs used both a nested ``usage``
        # object and flat token fields.  Preserve either shape before removing
        # the temporary request/response directory.
        usage = _parse_usage(response.get("usage"))
        if usage:
            self.last_usage = usage
        else:
            token_keys = (
                "input_tokens",
                "output_tokens",
                "visible_output_tokens",
                "reasoning_tokens",
                "num_reasoning_tokens",
                "thoughts_token_count",
                "total_tokens",
                "prompt_tokens",
                "completion_tokens",
                "completion_tokens_details",
            )
            self.last_usage = {
                key: response[key] for key in token_keys if key in response
            }
        for state in ("done", "in_progress", "pending", "tmp"):
            shutil.rmtree(self.queue_dir / state / call_id, ignore_errors=True)
        if response.get("error"):
            raise RuntimeError(str(response["error"]))
        text = str(
            response.get("content")
            or (
                ((response.get("choices") or [{}])[0].get("message") or {}).get(
                    "content"
                )
            )
            or ""
        )
        return text
