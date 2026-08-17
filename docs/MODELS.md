<div align="right"><a href="MODELS_zh.md">中文</a></div>

# Model invocation contract

All rollout backbones use temperature 0, planner prompt v4, verifier v3,
instruction v1, a six-step planning horizon, and a 100-environment-step cap.
`max_tokens` remains model-specific.

| Model | Historical backend | `max_tokens` | Reasoning budget | JSON request |
|---|---:|---:|---:|---:|
| GPT-5.5 | Azure OpenAI direct | 4096 | 512 | no explicit argument |
| Gemini-3.1 | filesystem queue | 1200 | 512 | yes |
| Gemini-2.5 | filesystem queue | 1200 | 512 | yes |
| Claude-4.8 | filesystem queue | 1200 | 0 | yes |
| Gemma-4-31B | vLLM | 1200 | unset | yes |
| Qwen3.6-27B | vLLM | 4096 | unset | yes |
| Qwen3.6-35B-A3B | vLLM | 4096 | unset | yes |
| Qwen3.5-27B | vLLM | 4096 | unset | yes |
| InternVL3.5-38B | vLLM | 2048 | unset | yes |

The machine-readable provenance is
`configs/paper_table_model_invocations.json`. It intentionally omits private
endpoints and credentials.

## OpenAI-compatible adapter

```json
{
  "backend": "openai_compatible",
  "model": "served-model-name",
  "base_url": "http://127.0.0.1:8100/v1",
  "api_key_env": "OPENAI_API_KEY",
  "max_tokens": 4096,
  "temperature": 0.0,
  "response_format": {"type": "json_object"},
  "extra_body": {}
}
```

Local OpenAI-compatible endpoints need no credential. For a remote endpoint,
export the named environment variable; do not store secrets in this file.

## Filesystem queue adapter

The rollout atomically moves each request directory into
`<queue_dir>/pending/`. A worker writes its response to
`<queue_dir>/done/<call_id>/response.json`:

```json
{"content": "{...model JSON response...}"}
```

The request and response directories are removed after the response is read.
They are transport state and are not part of the saved episode.

For exact cost metrics, the response may additionally contain a structured
`usage` object or the worker's `Usage(...)` string. The adapter accepts
OpenAI-style totals as well as Gemini's separate prompt, visible candidate,
and reasoning-token counts. Hidden reasoning is never folded into the paper's
visible-output column. If no usage is returned, `metrics.json` labels the
character-count fallback as an estimate.
