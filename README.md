# jev-rs

**System One judgments from any LLM, in one prefill.** A Rust engine that
answers typed questions about a piece of state — `noul` (yes/no), `choice`
(one of N), `score` (ordered scale) — with probabilities, never generated
text. Wire-compatible with TypeSafe's Jev
[`POST /v1/systemone`](https://docs.typesafe.ai/api), and exposed to coding
agents as an MCP tool.

Among the open Jev reimplementations, jev-rs is the only one that is not
itself a model or a single technique: it is the harness around the whole
ecosystem — a logprob scorer that reads *any* decoder, adapters for the
trained specialists, the TypeSafe wire server, and one `eval`/`calibrate`
metric stack that scores all of them side by side (tables below).

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

**Hosted or OpenAI-compatible backends.** `--backend-kind openai` scores
through `chat/completions` with `logprobs` (`max_tokens: 1`,
`top_logprobs: 20`), so the DeepSeek API, vLLM, SGLang or llama-server's
own `/v1` endpoint work without raw prompt access:

```sh
export JEV_API_KEY=$DEEPSEEK_API_KEY
jev --backend https://api.deepseek.com/v1 --backend-kind openai --model deepseek-flash \
    eval examples/dev_tasks.jsonl
```

The server applies its own chat template, so answers depend on the model
emitting the option letter as its first token; the raw `llamacpp` path is
exact and preferred when you control the server.

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

Apple M1 Ultra 128 GB, `llama-server` via Homebrew, zero-shot, same
prompts. `examples/dev_tasks.jsonl`: 25 shell commands × 3 questions
(command class 8-way, safe to re-run, output volume 3-level), hand-labelled,
75 decisions, 0 failed requests for either model.

| metric | Qwen3-4B Q4_K_M | Qwen3.8-Flash-Next Q2_K_XL (73 GB) | DeepSeek-V4-Flash MXFP4 (156 GB, SSD-streamed) |
|---|---|---|---|
| backend | llama-server, raw logprobs | llama-server, raw logprobs | ds4-rs-metal, chat logprobs, thinking off |
| accuracy overall | 0.71 | **0.84** | 0.76 |
| accuracy: choice / noul / score | 0.92 / 0.64 / 0.56 | 0.92 / 0.76 / **0.84** | 0.88 / **0.92** / 0.48 |
| ECE raw | 0.244 | 0.108 | 0.112 |
| Brier raw | 0.512 | 0.276 | 0.371 |
| coverage at ≤5 % error | 0.31 | **0.59** | 0.56 |
| latency p50 per question (warm) | **75 ms** | 700 ms | 41 s |

What changed with the larger models: the 8-way classification was already
saturated at 4B; the gains are on the yes/no and ordinal questions and on
probability quality. Both large models are close to calibrated out of the
box (fitted temperatures near 1), so their confidence can gate about twice
as many decisions at a 5 % error budget. The 4B model is far faster and
its fitted temperatures of 3–6 say its raw probabilities should not be
trusted without `jev calibrate`.

DeepSeek V4 Flash gives the best yes/no judgment of the three (0.92 on
"safe to re-run") and the worst ordinal one: of its 13 wrong `score`
answers, 12 guessed low and 11 of those by exactly one level, a systematic
bias a per-model calibration or a coarser scale would absorb. Its latency is an artefact of the run,
not the model: 156 GB of tensors on a 128 GB machine means experts stream
from SSD on every request. With the model resident (a 192 GB or 256 GB
Mac) the same engine prefills at hundreds of tokens per second.

Prompt caching depends on the architecture: Qwen3-4B re-evaluates only the
question suffix after the first question (33–62 tokens), while the hybrid
Qwen3.8-Next re-evaluates the full prompt for every question in
`llama-server`, so its four-question example costs 3.0 s against 0.4 s.

Two further experiments against real Claude Code session logs, predicting
output floods before a command runs and triaging tool output after it, are
in [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md); one is a clear win for the
judge, the other a clear loss to a blind rule.

Hosted Jev has been independently measured at 236–276 ms p50 per request
([jev-benchmarks](https://github.com/AbdelStark/jev-benchmarks),
[decision-model-benchmark](https://github.com/nibzard/decision-model-benchmark)).

### Against the open alternatives

Same `examples/dev_tasks.jsonl` (75 decisions, 0 failed), same machine,
through `jev eval` so the metrics line up with the table above. The two
open alternatives are wired as first-class backends:
[Laya](https://huggingface.co/convaiinnovations/laya)
(`--backend-kind laya`, `typed-decisions`) and
[AgentJev](https://github.com/malevrigns/agent-jev)
(`--backend-kind agentjev`, 0.6B). Both answer a whole request in one
call; probabilities are converted to log-probs and scored with the same
harness as the logprob path.

| metric | jev-rs (Qwen3-4B raw logprobs) | [Laya](https://huggingface.co/convaiinnovations/laya) typed-decisions | [AgentJev](https://github.com/malevrigns/agent-jev)-0.6B |
|---|---|---|---|
| accuracy overall | **0.71** | 0.49 | 0.49 |
| accuracy: choice / noul / score | **0.92 / 0.64 / 0.56** | 0.68 / 0.40 / 0.40 | 0.64 / 0.40 / 0.44 |
| Brier raw | **0.510** | 0.586 | 0.754 |
| Brier after `jev calibrate` | — | **0.510** | 0.571 |
| ECE raw | 0.244 | **0.102** | 0.285 |
| ECE after fit | — | **0.098** | **0.095** |
| latency p50 per request (warm) | ~90 ms | **~45 ms** | ~2 360 ms |

Neither specialist beats the logprob path on accuracy here: both sit near
chance on `noul` and `score`. Laya is about 2× faster and, after a
per-bucket temperature fit, has the best ECE of the three — useful if you
only care about calibrated confidence on its own traffic. AgentJev is
~25× slower on this CPU host and worst on raw Brier; its fitted ECE is
fine, but accuracy stays at chance. Keep the default `llamacpp`/`openai`
path for judgments; use `--backend-kind laya` or `--backend-kind agentjev`
when you specifically want those models.

### Reciprocal: Typed Decisions test split

The same three backends, scored on *their* home bench —
[Typed Decisions](https://huggingface.co/datasets/LocalLLaMA/typed-decisions)
`test` (400 cases / 2 000 decisions, revision `ea930645…`) — converted to
jev-rs case JSONL and run through `jev eval`. Datasets differ from the table
above; each bench favors the model trained on it.

| metric | jev-rs (Qwen3-4B raw logprobs) | Laya typed-decisions | AgentJev-0.6B |
|---|---|---|---|
| accuracy overall | 0.539 | 0.766 | **0.796** |
| accuracy: choice / noul / score | 0.58 / 0.56 / 0.50 | 0.73 / 0.86 / 0.72 | **0.76 / 0.88 / 0.76** |
| Brier raw | 0.878 | 0.400 | **0.321** |
| Brier after `jev calibrate`† | — | 0.314 | **0.287** |
| ECE raw | 0.425 | 0.213 | **0.134** |
| ECE after fit† | — | 0.024 | **0.018** |
| latency p50 (warm) | ~99 ms/question | ~237 ms/case | ~2 687 ms/case |

† Fit is done on the same rows `jev eval` reports (optimistic); treat as a
ceiling, not a held-out calibration number. Laya and AgentJev answers are
whole-case wall time (5 decisions per case); the llamacpp column is
per-question next-token scoring.

The two tables are one conclusion, not a contradiction. Laya and AgentJev
are **specialists**: they only know the 20 fixed Typed Decisions schemas
they were trained on. On their home bench they win (0.77 / 0.80 vs 0.54);
on `dev_tasks.jsonl` — shell-command questions they have never seen — they
fall to chance (0.49 each). Qwen3-4B through raw logprobs is a
**generalist**: zero-shot on any schema. It wins `dev_tasks` (0.71) and
sits near the Typed Decisions prior baseline (0.54) when cold against a
model fitted to that exact data. This is the specialist-vs-generalist
split Typed Decisions itself documents — each path wins where it was built
to win. Pick by traffic:

- fixed, known schema with a trained checkpoint → `--backend-kind laya`
  or `--backend-kind agentjev`
- open-ended or changing questions → the default `llamacpp` / `openai`
  logprob path

## The ecosystem, and where jev-rs sits

Open Jev solutions cluster into three groups. Wire compatibility matters
here: everything that speaks `POST /v1/systemone` (or a thin variant) can be
swapped in and out, so the groups are not walled gardens.

**1. Servers that speak the wire format.** Hosted Jev is the reference;
[jeff](https://github.com/lodos3/jeff) (GLiFormer 400M),
[SemIf](https://github.com/TheoLeeCJ/SemIf) (formerly OpenJev; frozen
Qwen3.5-4B option-logit readout, MIT),
[jev-bridge](https://github.com/TOSUKUi/jev-bridge) (Rust, OpenAI-API →
letter-slot logprob readout),
[djev-spark](https://djev.dev/) (DiffusionGemma, multimodal), and jev-rs
itself all serve the same request shape. So do
[System One Lite](https://github.com/snellingio/system-one) (MLX, Apple
silicon) and [coreai-kit](https://github.com/john-rocky/coreai-kit/blob/main/docs/SYSTEM_ONE.md)
(`decide-cli serve`, Apple Core AI); the vendor-neutral
[system-one](https://github.com/asynq-io/system-one) SDK reaches any of them
through `HTTPConfig(base_url=…)`. SemIf is the strongest open row on
the published JevBench composite (#2, 0.7 behind Jev); jeff and jeff-class
rebuilds trail further.

**2. Trained specialist models.** [Laya](https://huggingface.co/convaiinnovations/laya)
(ModernBERT, Apache-2.0),
[AgentJev](https://github.com/malevrigns/agent-jev) (0.6B) and
[NanoJev](https://huggingface.co/C-Tianyu/NanoJev) (Qwen3-0.6B + decision
heads, 2–255 candidates) return full distributions over their trained
schemas. jev-rs wires the first two as first-class backends
(`--backend-kind laya`, `--backend-kind agentjev`); NanoJev is not wired yet.

**3. Client-side adapters and alternate techniques.**
`system-one-adapter` (TypeSafe's own LLM-backed adapter),
[poorjev](https://pypi.org/project/poorjev/) (local NLI + temperature
scaling), AnyJev (Nokia), Jevlike, LocalJev, Nimble — these reach the same
interface from a different mechanism or sit in front of it rather than
behind it.

**What is unique about jev-rs.** Every other open project is one model
(Laya, AgentJev, NanoJev, jeff), one readout technique (SemIf, jev-bridge,
poorjev), or hosted (Jev, djev). jev-rs is the only project that is all of:

- **a generalizing scorer** — the same prefill + label-logprob readout works
  against *any* GGUF through llama-server or any OpenAI-compatible `/v1`
  host, zero-shot on schemas it was never trained on;
- **a hub for the specialists** — the full-request backends (Laya,
  System One, AgentJev) plug into the same `FullBackend` trait, so their
  probabilities are scored by the identical harness;
- **the wire server** — `jev serve` is a drop-in for `api.typesafe.ai`, so
  the official SDKs and every `/v1/systemone` integration run against any of
  the above;
- **one measurement stack** — `jev eval` / `jev calibrate` report accuracy,
  Brier, top-label ECE, coverage and latency for *every* backend, which is
  how the two comparison tables above were produced, and an MCP server so
  coding agents get the same judgments in-session.

In short: the ecosystem offers interchangeable parts; jev-rs is the
chassis that mounts them and the dyno that measures them.

It is also built to be the judgment engine inside two Rust systems —
[PRECC](https://github.com/peri-a-i/precc-cc), a Claude Code hook that
saves tokens, and [ds4-rs-metal](https://github.com/yijunyu/ds4-rs-metal) /
[Local Mind](https://yijunyu.github.io/local-mind/), an on-device
DeepSeek-V4 engine — where an in-process,
KV-forking scorer is the point.

## Limits (today)

- At most 26 options per question (single-letter labels); Jev accepts 255.
- Scorer backends: `llama-server` (raw logprobs) and any OpenAI-compatible
  `/v1` host. Full-request backends: Laya/`systemone`, AgentJev.
  In-process llama.cpp and ds4-rs scorers are next.
- Zero-shot only; no RLCD-style training.

## License

MIT or Apache-2.0, at your option.
