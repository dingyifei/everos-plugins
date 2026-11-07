# Hermes Memory — EverOS

A native [Hermes Agent](https://github.com/NousResearch/hermes-agent) memory
provider backed by a local [EverOS](https://github.com/EverMind-AI/EverOS)
server: markdown files as the source of truth, hybrid (vector + BM25 + scalar)
recall, and two memory tracks — your profile/episodes and the agent's distilled
cases/skills. Recall is automatic every turn. The model also gets two tools:
`everos_search` (look further than the automatic recall; returns episode ids)
and `everos_forget` (delete episodes by id). Profile corrections need no tool:
stating the correct fact in conversation updates the profile.

Optional, per profile: a [Jev](https://opencode.ai) relevance filter over the
recalled items (one request per turn, fail-open), and notes recall from a local
search socket (e.g. an Obsidian vault index).

Design details: [`docs/DESIGN_DOC.md`](./docs/DESIGN_DOC.md) · build/test/release:
[`docs/HANDOFF.md`](./docs/HANDOFF.md).

## Install

```bash
hermes plugins install EverMind-AI/plugins/hermes    # → ~/.hermes/plugins/memory/everos/
hermes memory setup                                  # pick "everos"; answer the prompts
hermes memory status                                 # verify: provider active, EverOS healthy
```

One-time, outside the plugin: EverOS needs `everos init` once and API keys in
`~/.everos/everos.toml`. If EverOS isn't running, the provider starts it
automatically on agent startup (configure `everos_dir` / `start_cmd`).

**Updating:** subfolder installs can't use `hermes plugins update` — re-run the
install command above.

## Configuration (`$HERMES_HOME/everos.json`, written by `hermes memory setup`)

| Key | Default | Purpose |
|---|---|---|
| `base_url` | `http://127.0.0.1:8000` | Where EverOS listens (its port is EverOS config) |
| `user_id` | OS account | Developer identity for the user track |
| `agent_id` | `hermes` | Constant pooled agent identity |
| `query_max_units` | `500` | Weighted recall-query clip (CJK char = 2 units) |
| `everos_dir` | — | EverOS checkout, for auto-start from its venv |
| `start_cmd` | `everos server start` | Auto-start command (quote spaced paths) |
| `autostart` | `true` | `false` when the server is managed elsewhere (a container or a service unit): the plugin never starts one |
| `jev_enabled` | `false` | Filter recalled items with Jev before injecting them (key from `$OPENCODE_GO_API_KEY`) |
| `jev_url` | OpenCode Zen System One | Jev endpoint |
| `jev_model` | `jev-1.13` | Jev model |
| `jev_cutoff` | `0.5` | Keep items Jev scores at or above this (the best one is always kept) |
| `jev_timeout_s` | `4.0` | Past this, every item is kept (fail-open) |
| `agent_method` | server default (`hybrid`) | Agent-track search method; `vector` or `keyword` skip EverOS's cross-encoder rerank of skills |
| `vault_socket` | — | Unix socket of a notes search (`{"query","top_k"}` in, `{"results":[{path,heading,text}]}` out, one JSON line each) |
| `vault_top_k` | `20` | Notes fetched per turn (Jev filters them) |
| `vault_unfiltered_k` | `8` | Notes kept when Jev is off or fails |

Each recall logs one line:
`[everos] recall session=<id8> eps=<got>/<kept> agent=<got>/<kept> vault=<got>/<kept> top=<best Jev score|-> jev=<ok|off|failopen:reason> ms=<total>`.

## What data leaves the device

The provider talks to your local EverOS (`127.0.0.1`) and, with `jev_enabled`,
sends each turn's recall query and the recalled items to the Jev endpoint. EverOS itself sends
conversation text (each completed turn, including tool calls/results; built-in
memory edits; delegation summaries) to **the LLM and embedding providers you
configured in EverOS** for extraction — cloud services unless you point EverOS
at local models. Nothing else leaves the machine; the memory store
(Markdown/SQLite/LanceDB) is local.

## License

[Apache-2.0](../LICENSE)
