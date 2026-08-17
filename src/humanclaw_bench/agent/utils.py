"""Prompt, image, and JSON utilities for HumanClawBench ego agents."""

from __future__ import annotations

import base64
import io
import json
import re
from pathlib import Path
from typing import Any

from PIL import Image


def image_to_data_url(image: Image.Image) -> str:
    """PNG-encode a PIL image and return an inline base64 data URL."""

    buf = io.BytesIO()
    image.save(buf, format="PNG")
    encoded = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


def image_path_to_data_url(path: str) -> str:
    """Read an image path and encode it as an inline data URL."""

    data = Path(path).read_bytes()
    encoded = base64.b64encode(data).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


def parse_json_loose(text: str) -> dict[str, Any]:
    """Parse provider text as JSON, tolerating markdown fences and surrounding prose."""

    clean = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    clean = clean.replace("```json", "").replace("```", "").strip()
    try:
        parsed = json.loads(clean)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", clean):
        try:
            obj, _ = decoder.raw_decode(clean[match.start() :])
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            return obj
    raise ValueError(f"Could not parse JSON from model output: {text[:300]}")


def clip_text(value: object, limit: int) -> str:
    """Bound diagnostic text length while preserving a clear truncation marker."""

    text = "" if value is None else str(value).strip().replace("\n", " ")
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


__all__ = [
    "clip_text",
    "image_path_to_data_url",
    "image_to_data_url",
    "parse_json_loose",
]
