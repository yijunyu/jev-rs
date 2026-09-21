# jev-rs

**System One judgments from any LLM, in one prefill.** A Rust engine and
server that answers typed questions about a piece of state — `noul`
(yes/no), `choice` (one of N), `score` (ordered scale) — with probabilities,
never generated text. Wire-compatible with TypeSafe's
[`POST /v1/systemone`](https://docs.typesafe.ai/api), so the official SDKs
and the [`jev`](https://crates.io/crates/jev) crate work unchanged with
`TYPESAFE_BASE_URL` pointed at it.

```
client ──POST /v1/systemone──▶ jev-rs ──/completion (n_predict=1, n_probs)──▶ llama-server (any GGUF)
         (Jev wire format)               planned: in-process ds4-rs-metal · llama.cpp/ggml-tilers
```

## How it works

1. The **state** is rendered once into a shared prefix; each **question** is a
   suffix that ends in `Answer:` with the options labelled `A`, `B`, `C`…
   Backends with a prompt cache prefill the state once per request.
2. The backend returns the raw next-token log-probabilities for the option
   labels. Nothing is decoded: `usage.output_tokens` is always 0.
3. A restricted softmax over the labels, divided by a fitted temperature per
   `(question type, option count)` bucket, gives the probabilities.
   `confidence` follows TypeSafe's documented formula, `score` is the
   probability-weighted level index.
4. Optional position-bias control: `--permutations k` scores `k` cyclic
   rotations of the option order and averages per key.

## Quick start

```sh
llama-server -m Qwen3-4B-Q4_K_M.gguf --port 8089 -np 2 -c 8192   # any GGUF
cargo build --release
./target/release/jev --backend http://127.0.0.1:8089 ask --file examples/support_ticket.json
./target/release/jev --backend http://127.0.0.1:8089 serve --bind 127.0.0.1:8090
curl -s localhost:8090/v1/systemone -H 'content-type: application/json' -d @examples/support_ticket.json
```

Use `--template gemma|llama3|raw` for non-ChatML models. With
`TYPESAFE_API_KEY` set, `jev ask --compare` sends the same request to the
hosted model for a side-by-side.

## Measured, not claimed

Apple M1 Ultra, `llama-server` b10360, Qwen3-4B Q4_K_M, zero-shot, no
fine-tuning. `examples/dev_tasks.jsonl`: 25 shell commands × 3 questions
(command class 8-way, safe-to-rerun, output volume) with hand labels.

| metric | value |
|---|---|
| decisions | 75 |
| accuracy: choice / noul / score | 0.92 / 0.64 / 0.56 |
| ECE before → after `jev calibrate` | 0.244 → 0.164 |
| Brier before → after | 0.512 → 0.424 |
| latency p50 / p95 per question | 75 ms / 99 ms |
| prompt tokens evaluated / served from cache | 5 938 / 9 901 |

The 4-question support-ticket example costs one 167-token prefill plus three
33–62-token suffixes, 394 ms end to end. Hosted Jev has been independently
measured at 236–276 ms p50 for one question
([jev-benchmarks](https://github.com/AbdelStark/jev-benchmarks),
[decision-model-benchmark](https://github.com/nibzard/decision-model-benchmark)).

A 4B instruct model is badly over-confident out of the box (fitted
temperatures of 3–6). Run `jev calibrate your_cases.jsonl` and pass the
result with `--calibration` before trusting the probabilities.

## Commands

| command | does |
|---|---|
| `jev serve` | TypeSafe-compatible HTTP server (`/v1/systemone`, `/v1/models`, `/health`), optional bearer keys |
| `jev ask` | one request from `--file`, stdin, or `--state/--noul/--choice/--score` flags |
| `jev eval cases.jsonl` | accuracy, Brier, top-label ECE, coverage at ≤5 % error, latency, token cost |
| `jev calibrate cases.jsonl` | fit per-bucket temperatures by NLL grid search, write `calibration.json` |

Case-file line: `{"state": ..., "questions": {...}, "gold": {"id": "key-or-level"}}`.

## Why another one

Several open Jev replacements appeared in the week after the launch
(encoder models such as [Laya](https://huggingface.co/convaiinnovations/laya),
[jeff](https://github.com/lodos3/jeff); logprob bridges such as
[jev-bridge](https://github.com/TOSUKUi/jev-bridge)). jev-rs exists for a
narrower purpose: to be the judgment engine inside two Rust systems —
[PRECC](https://github.com/peri-a-i/precc-cc), a Claude Code hook that saves
tokens, and [ds4-rs-metal](https://github.com/yijunyu/ds4-rs-metal) /
Local Mind, an on-device DeepSeek-V4 engine — where an in-process,
KV-forking scorer rather than an HTTP bridge is the point. See
[`docs/DESIGN.md`](docs/DESIGN.md).

## Limits (today)

- At most 26 options per question (single-letter labels). Jev accepts 255.
- One backend: `llama-server`. The in-process backends are the next milestone.
- Zero-shot only; no RLCD-style training. The recipe for that is in the design doc.
- Confidence is a formula over the distribution, not a learned abstention.

## License

MIT or Apache-2.0, at your option.
