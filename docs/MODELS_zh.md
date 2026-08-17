<div align="right"><a href="MODELS.md">English</a></div>

# 模型调用约定

所有 rollout backbone 都使用 temperature 0、planner prompt v4、verifier
v3、instruction v1、六步 planning horizon，以及最多 100 个 environment
step。`max_tokens` 由模型分别设置。

| 模型 | 历史 backend | `max_tokens` | Reasoning budget | JSON request |
|---|---:|---:|---:|---:|
| GPT-5.5 | Azure OpenAI direct | 4096 | 512 | 未显式传参 |
| Gemini-3.1 | filesystem queue | 1200 | 512 | 是 |
| Gemini-2.5 | filesystem queue | 1200 | 512 | 是 |
| Claude-4.8 | filesystem queue | 1200 | 0 | 是 |
| Gemma-4-31B | vLLM | 1200 | 未设置 | 是 |
| Qwen3.6-27B | vLLM | 4096 | 未设置 | 是 |
| Qwen3.6-35B-A3B | vLLM | 4096 | 未设置 | 是 |
| Qwen3.5-27B | vLLM | 4096 | 未设置 | 是 |
| InternVL3.5-38B | vLLM | 2048 | 未设置 | 是 |

机器可读的 provenance 位于
`configs/paper_table_model_invocations.json`。其中有意省略了私有 endpoint
和 credential。

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

本地 OpenAI-compatible endpoint 不需要 credential。远程 endpoint 请导出
对应环境变量，不要把密钥写进配置文件。

## Filesystem queue adapter

Rollout 将每个 request 目录原子地移动到 `<queue_dir>/pending/`。Worker 将
response 写入 `<queue_dir>/done/<call_id>/response.json`：

```json
{"content": "{...model JSON response...}"}
```

读取 response 后会删除 request 和 response 目录。它们是 transport state，
不属于保存的 episode。

如果要得到准确的 cost metric，response 还可以包含结构化 `usage` 对象或
worker 的 `Usage(...)` 字符串。Adapter 同时接受 OpenAI 风格的总量，以及
Gemini 分开的 prompt、visible candidate 和 reasoning token 数。隐藏 reasoning
不会计入论文的 visible-output 列。如果没有 usage，`metrics.json` 会把基于
字符数的估算明确标为 estimated。
