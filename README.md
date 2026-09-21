# jev-rs

**System One judgments from any LLM, in one prefill.** A Rust engine that
answers typed questions about a piece of state — `noul` (yes/no), `choice`
(one of N), `score` (ordered scale) — with probabilities, never generated
text. Wire-compatible with TypeSafe's Jev
[`POST /v1/systemone`](https://docs.typesafe.ai/api), and exposed to coding
agents as an MCP tool.

```
Claude Code / Codex / Grok Build / OpenCode ──MCP stdio──▶ jev ──▶ llama-server (any GGUF)
TypeSafe SDKs (TYPESAFE_BASE_URL) ──────────POST /v1/systemone──▶ jev serve ──▶ llama-server
```

## Install

```sh
curl -fsSL https://raw.githubusercontent.com/yijunyu/jev-rs/main/install.sh | sh
```

Installs a prebuilt `jev` into `~/.local/bin` (macOS arm64/x86_64, Linux
x86_64/arm64) or builds from source with cargo if no binary matches.
With a Rust toolchain, `cargo install jev-rs` works too. Then
start any GGUF model behind `llama-server` (macOS: `brew install llama.cpp`):

```sh
llama-server -hf Qwen/Qwen3-4B-GGUF --port 8089 -np 2 -c 8192
```

Try it:

```sh
jev --backend http://127.0.0.1:8089 ask \
  --state "We were billed twice for March. Refund the duplicate today or we cancel." \
  --choice "dept=Which team handles this?|billing:refunds and invoices,technical:bugs,sales:pricing" \
  --score  "urgency=How urgent?|not urgent,soon,blocking" \
  --noul   "churn=Does the customer threaten to leave?"
```

```json
{
  "dept":    {"type":"choice","choice":"billing","probabilities":{"billing":1.0,"technical":0.0,"sales":0.0},"confidence":1.0},
  "urgency": {"type":"score","score":1.98,"legend":{"0":"not urgent","1":"soon","2":"blocking"},"probabilities":{"0":0.0,"1":0.02,"2":0.98},"confidence":0.98},
  "churn":   {"type":"noul","noul":1.0}
}
```

Set `JEV_BACKEND_URL` once to drop the `--backend` flag. Use
`--template gemma|llama3|raw` for non-ChatML models.

## Use with coding agents

`jev mcp` is an MCP server over stdio with one tool, `judge`. The agent
sends a state and typed questions and gets probabilities back instead of
prompting a big model to classify and parsing its prose. Typical uses inside
an agent session: routing a task to a skill, triaging tool output, gating a
risky command, ranking candidates, yes/no checks on a diff.

**Claude Code**

```sh
claude mcp add jev -e JEV_BACKEND_URL=http://127.0.0.1:8089 -- jev mcp
```

or in `.mcp.json` at the project root:

```json
{"mcpServers": {"jev": {"command": "jev", "args": ["mcp"], "env": {"JEV_BACKEND_URL": "http://127.0.0.1:8089"}}}}
```

**Codex** (`~/.codex/config.toml`)

```toml
[mcp_servers.jev]
command = "jev"
args = ["mcp"]
env = { JEV_BACKEND_URL = "http://127.0.0.1:8089" }
```

**Grok Build** (`.grok/mcp.json` in the project, or `~/.grok/mcp.json`)

```json
{"servers": {"jev": {"command": "jev mcp", "env": {"JEV_BACKEND_URL": "http://127.0.0.1:8089"}}}}
```

or `grok mcp add jev --command "jev mcp" --env JEV_BACKEND_URL=http://127.0.0.1:8089`.

**OpenCode** (`opencode.json`)

```json
{"mcp": {"jev": {"type": "local", "command": ["jev", "mcp"], "environment": {"JEV_BACKEND_URL": "http://127.0.0.1:8089"}}}}
```

Then tell the agent, in its instructions file, when to reach for it:

> Use the `judge` tool for any classification, routing, triage or yes/no
> decision about text or tool output. Put the facts in `state` and ask one
> judgment per question. Trust answers with confidence ≥ 0.8; otherwise
> decide yourself.

**TypeSafe SDK users.** `jev serve` speaks the same wire format as
`api.typesafe.ai`, so the official Python/JS SDKs, the
[`jev`](https://crates.io/crates/jev) crate and any Jev integration run
against a local model unchanged:

```sh
jev --backend http://127.0.0.1:8089 serve --bind 127.0.0.1:8090
export TYPESAFE_BASE_URL=http://127.0.0.1:8090 TYPESAFE_API_KEY=local
```

## Commands

| command | does |
|---|---|
| `jev mcp` | MCP server over stdio, tool `judge` |
| `jev serve` | HTTP server: `/v1/systemone`, `/v1/models`, `/health`; `--api-keys` for bearer auth |
| `jev ask` | one request from flags, `--file`, or stdin; `--compare` also hits the hosted API |
| `jev eval cases.jsonl` | accuracy, Brier, top-label ECE, coverage at ≤5 % error, latency, token cost |
| `jev calibrate cases.jsonl` | fit per-bucket temperatures, write `calibration.json` (load with `--calibration`) |

Case-file line: `{"state": ..., "questions": {...}, "gold": {"id": "key-or-level"}}`.

## How it works

1. The state is rendered once into a shared prefix; each question is a
   suffix ending in `Answer:` with options labelled `A`, `B`, `C`… so the
   backend's prompt cache prefills the state once per request.
2. The backend returns raw next-token log-probabilities for the labels.
   Nothing is decoded; `usage.output_tokens` is always 0.
3. Restricted softmax over the labels, divided by a fitted temperature per
   `(question type, option count)` bucket. `confidence` is TypeSafe's
   documented formula; `score` is the probability-weighted level.
4. `--permutations k` averages `k` rotations of the option order to control
   position bias.

## Measured, not claimed

Apple M1 Ultra, `llama-server` b10360, Qwen3-4B Q4_K_M, zero-shot.
`examples/dev_tasks.jsonl`: 25 shell commands × 3 questions (command class
8-way, safe to re-run, output volume), hand-labelled.

| metric | value |
|---|---|
| decisions | 75 |
| accuracy: choice / noul / score | 0.92 / 0.64 / 0.56 |
| ECE before → after `jev calibrate` | 0.244 → 0.164 |
| Brier before → after | 0.512 → 0.424 |
| latency p50 / p95 per question | 75 ms / 99 ms |

Four questions on one state: one 167-token prefill plus three 33–62-token
suffixes, 394 ms end to end. Hosted Jev has been independently measured at
236–276 ms p50 per request
([jev-benchmarks](https://github.com/AbdelStark/jev-benchmarks),
[decision-model-benchmark](https://github.com/nibzard/decision-model-benchmark)).
A 4B instruct model is over-confident out of the box (fitted temperatures
3–6); calibrate on your own cases before trusting the probabilities.

## Why another one

Open Jev replacements appeared within a week of the launch
([Laya](https://huggingface.co/convaiinnovations/laya),
[jeff](https://github.com/lodos3/jeff),
[jev-bridge](https://github.com/TOSUKUi/jev-bridge)). jev-rs is built to
be the judgment engine inside two Rust systems —
[PRECC](https://github.com/peri-a-i/precc-cc), a Claude Code hook that
saves tokens, and [ds4-rs-metal](https://github.com/yijunyu/ds4-rs-metal) /
[Local Mind](https://yijunyu.github.io/local-mind/), an on-device
DeepSeek-V4 engine — where an in-process,
KV-forking scorer is the point.

## Limits (today)

- At most 26 options per question (single-letter labels); Jev accepts 255.
- One backend, `llama-server`. In-process llama.cpp and ds4-rs backends are next.
- Zero-shot only; no RLCD-style training.

## License

MIT or Apache-2.0, at your option.
