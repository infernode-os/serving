# lucibridge — per-tool routing for multi-backend serving

**Owning ticket:** INFR-79. **Cross-repo dependency:** the bridge
implementation lives in
[`infernode-os/infernode`](https://github.com/infernode-os/infernode);
this file is the **operational schema and config** that lives in the
serving repo so deploy decisions stay with the deploy artefacts. The
agentlib changes in infernode that consume this schema are tracked on
INFR-79 itself.

---

## Why this exists

Per V4-PLAN's production strategy: **gpt-oss is for dispatch / tool
calls**, **devstral-limbo is for Limbo authoring**. After INFR-77
unblocks gpt-oss on SGLang and the spike (INFR-68) measured ~3×
concurrent throughput vs Ollama, the box has two viable backends:

| Backend | URL | Model strengths |
|---|---|---|
| Ollama | `http://127.0.0.1:11434/v1` | single-user latency, Devstral GGUF, Limbo authoring |
| SGLang | `http://127.0.0.1:30000/v1` | concurrent fan-out, gpt-oss-20b, xgrammar tool-call grounding |

A single fixed `LLM_BACKEND_URL` env var no longer captures the
intent. `lucibridge` needs to **route per request**, and the routing
key is the tool-category (which the agentlib already attaches to each
dispatch).

---

## Routing table

The canonical route table for v1 (Hephaestus, Veltro-on-SGLang). All
URLs are relative to the configured backend prefix; `model` is the
model selector accepted by both backends' `/v1/chat/completions`.

| Tool category | Backend | Model | Notes |
|---|---|---|---|
| `limbo_authoring` | Ollama | `devstral-limbo-v3` (or v4 once daedalus lands) | Single-user fluency; no concurrent fan-out |
| `dispatch` | SGLang | `openai/gpt-oss-20b` | Fast tool-call dispatch, xgrammar-friendly |
| `tool_call` | SGLang | `openai/gpt-oss-20b` | Per-tool args, may be grammar-constrained |
| `memory` | SGLang | `openai/gpt-oss-20b` | Fan-out heavy, benefits from RadixAttention |
| `task` | SGLang | `openai/gpt-oss-20b` | Same as above |
| `chat` / unset / unknown | Ollama | `LLM_DEFAULT_MODEL` | Fallback — backward compatible with v0 |

Every row is overridable by config (see below); the table is the
**default** when no override is supplied. Unknown categories must
fall back to Ollama — the dev path stays unchanged.

---

## Config schema

Stored at `/etc/lucibridge/routing.json` on Hephaestus, owned by
`root:root`, mode `0644` (no secrets):

```json
{
  "$schema": "infernode.lucibridge.routing/v1",
  "backends": {
    "ollama":  { "base_url": "http://127.0.0.1:11434/v1" },
    "sglang":  { "base_url": "http://127.0.0.1:30000/v1" }
  },
  "default_backend": "ollama",
  "default_model":   "devstral-limbo-v3",
  "routes": [
    { "category": "limbo_authoring", "backend": "ollama",  "model": "devstral-limbo-v3" },
    { "category": "dispatch",        "backend": "sglang",  "model": "openai/gpt-oss-20b" },
    { "category": "tool_call",       "backend": "sglang",  "model": "openai/gpt-oss-20b",
      "extra": { "grammar_backend": "xgrammar" } },
    { "category": "memory",          "backend": "sglang",  "model": "openai/gpt-oss-20b" },
    { "category": "task",            "backend": "sglang",  "model": "openai/gpt-oss-20b" }
  ],
  "fallback": {
    "on_backend_unreachable": "default_backend",
    "on_unknown_category":    "default_backend",
    "log_decisions":          true
  }
}
```

### Field reference

| Field | Type | Meaning |
|---|---|---|
| `backends` | map\<name, {base_url}\> | named pool of OpenAI-compatible endpoints |
| `default_backend` | string | backend used when no route matches; also the fallback target on unreachable backend |
| `default_model` | string | model passed to `default_backend` when none specified |
| `routes` | list\<{category, backend, model, extra?}\> | first-match per-category route; ordering matters only for diagnostics |
| `routes[].extra` | map | passed through as extra fields on the upstream `/v1/chat/completions` body (e.g. `grammar_backend`, `lora_name` for INFR-77 multi-LoRA) |
| `fallback.on_backend_unreachable` | enum | `default_backend` \| `fail` |
| `fallback.on_unknown_category` | enum | `default_backend` \| `fail` |
| `fallback.log_decisions` | bool | emit a structured log line per routing decision (recommended on) |

### Backwards compatibility

Bridges from before INFR-79 spoke a single `LLM_BACKEND_URL` env var.
The post-INFR-79 bridge MUST still honour that env var as a
configuration fallback: if `/etc/lucibridge/routing.json` is missing,
the bridge constructs an implicit config with one backend (the
`LLM_BACKEND_URL`) and the default model from `LLM_DEFAULT_MODEL`, and
routes every request to it. v0 deployments don't need to change.

---

## env-var bridging from `serve-llm.sh`

The serve-llm launcher exports the env vars that the bridge resolves
into the implicit / explicit config. For Hephaestus dual-backend mode:

```sh
# /etc/serve-llm.env  (sourced by serve-llm.service)
LLM_BACKEND_DEFAULT=http://127.0.0.1:11434/v1
LLM_BACKEND_SGLANG=http://127.0.0.1:30000/v1
LLM_DEFAULT_MODEL=devstral-limbo-v3
LUCIBRIDGE_ROUTING_CONFIG=/etc/lucibridge/routing.json
```

The bridge uses `LUCIBRIDGE_ROUTING_CONFIG` if set, falling back to
the implicit single-backend mode (using `LLM_BACKEND_DEFAULT`) when
the file is absent. This matches the §8 mode-switching pattern in
`runbooks/hephaestus-deploy.md`.

---

## Observability — what to log

Every routing decision emits one structured log line at INFO. Format
(JSON; one line per request):

```json
{
  "ts": "2026-05-14T03:14:15Z",
  "event": "lucibridge.route",
  "request_id": "req_…",
  "category": "tool_call",
  "matched_route_index": 2,
  "backend": "sglang",
  "backend_url": "http://127.0.0.1:30000/v1",
  "model": "openai/gpt-oss-20b",
  "fallback_used": false,
  "fallback_reason": null
}
```

`fallback_used: true` with `fallback_reason: "unknown_category"` or
`"backend_unreachable"` is the signal for routing problems. Alert on
sustained `fallback_used: true` (>5% of decisions over a 5-minute
window).

---

## Test plan (lives in `infernode-os/infernode:agentlib_test/`)

Once the bridge code change is implemented, the tests that must exist:

1. **Single-backend v0 compat:** routing.json absent, `LLM_BACKEND_URL`
   set → every request goes to that URL. No regression vs pre-INFR-79.
2. **Per-category routing:** routing.json present, two backends mocked
   → a `limbo_authoring` request hits Ollama mock; a `tool_call`
   request hits SGLang mock; verify the URL + model dispatched.
3. **Fallback on unreachable backend:** SGLang mock returns 503 →
   request falls back to default backend, log line shows `fallback_used: true, fallback_reason: "backend_unreachable"`.
4. **Fallback on unknown category:** `category: "frobnicate"` →
   default backend used, fallback log emitted.
5. **`extra` passthrough:** route with `extra: {grammar_backend: "xgrammar"}`
   → the outgoing body includes that field.
6. **Decision-log emission:** with `log_decisions: true`, every
   request produces exactly one structured log line.

These are pre-existing `lucibridge_test` shape; the diff is the new
fixture file and the new test functions.

---

## Acceptance for INFR-79

| Criterion | Where verified |
|---|---|
| A configured Veltro session routes `limbo_authoring` to Ollama and `tool_call` to SGLang, both succeed | `infernode-os/infernode` agentlib_test suite + manual run on Hephaestus |
| Fallback documented and tested | tests 3 + 4 above |
| Existing single-URL configs still work unchanged | test 1 above |
| No regression in existing `lucibridge_test` suite | CI run on `infernode-os/infernode` |

---

## Open issues

* **Per-tool LoRA selection** — gpt-oss-20b will serve multiple
  adapters (`gpt-oss-limbo-v3` plus future v4) layered on one resident
  base. The route's `extra` field should carry `lora_name` once SGLang
  multi-LoRA is wired up. Tracked as a follow-up under INFR-77's
  multi-LoRA validation.
* **Bridge config hot-reload** — v1 reloads on `SIGHUP`; document
  exact behaviour once the agentlib PR lands.
* **Cross-host routing** — current schema assumes Hephaestus-local
  backends. If we add a second Jetson, `backends[].base_url` already
  takes any URL; document the firewall/zerotier story before exposing.
