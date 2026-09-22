# Experiments against real Claude Code sessions

Both experiments read a local corpus of Claude Code session logs
(`~/.claude/projects/*/*.jsonl`, about 47 000 distinct Bash commands and
their recorded results) and ask a judge to make the decision a token-saving
hook would make. Ground truth is what actually happened next in the
session. Scripts: `scripts/predict_flood.py`, `scripts/triage_output.py`.
Hardware: Apple M1 Ultra 128 GB, `llama-server`, zero-shot, no fine-tuning.
Run 2026-09-22.

## 1. Before the command runs: will its output flood the context?

This is the decision PRECC's adaptive compressor needs at `PreToolUse`
time. Input: the command string only. Flood = the recorded result was
≥ 500 tokens (about 11 % of commands). Baseline: the leave-one-out mean
result size of the same first-two-words command class, which is how PRECC
classifies today. 200 random commands.

| predictor | precision | recall | flagged |
|---|---|---|---|
| class baseline (two words) | 0.80 | 0.21 | 5 |
| Qwen3-4B, `noul` "more than a screenful?" ≥ 0.5 | 0.10 | 0.23 | 49 |
| Qwen3.8-Flash-Next, same `noul` ≥ 0.5 | 0.53 | 0.53 | 19 |
| Qwen3.8, `noul` ≥ 0.8 | 0.58 | 0.37 | 12 |
| **union: class baseline OR Qwen3.8 ≥ 0.5** | **0.58** | **0.74** | 24 |
| flag everything | 0.10 | 1.00 | 200 |

Reading: the 4B model cannot do this from the command alone; it is at the
base rate. The 73 GB model is five times better than chance and
complementary to the class baseline, which is precise but fires only on
classes with enough history. The floods only the judge caught were
line-range dumps (`sed -n '1,120p' …`, `awk 'NR>=920 && NR<=1015' …`,
multi-command `&&` chains ending in a `head`) that a two-word class cannot
see. Cost: about 2 s per command on this machine for the large model, so
this belongs in PRECC's daemon (label once per new command, ship the
verdict in the snapshot the hook reads), never on the hook's 5 ms path.

## 2. After the command runs: how much of the output does the agent need?

Input: the command and its output (capped at 6 000 characters, middle
omitted for the judge). Policy: `score` of 0 keeps 5+5 lines, 1 keeps
30+30, 2 keeps all; a confident `noul` "the middle is routine" (≥ 0.7)
downgrades a 2 to a 1. Harm proxy: the agent's next turn reuses a
24-character shingle, or a distinctive identifier that occurs only in the
elided middle. 150 random outputs of ≥ 40 lines, 124 000 tokens in total.

| policy | tokens saved | truncated | harmed (proxy) |
|---|---|---|---|
| Qwen3-4B judge | 5.0 % | 25 / 150 | 1 |
| Qwen3.8-Flash-Next judge | 3.2 % | 35 / 150 | 1 |
| blind head 30 + tail 30 on everything | **24.2 %** | 138 / 150 | 1 |

Reading: both judges default to "keep all of it" (115–125 of 150), so they
save little, and each still harmed one case. Under this proxy a blind
head+tail rule saves five to eight times more tokens at the same harm
count. The judge does not pay for itself here as a
truncation decider. Two caveats keep this from being final: the proxy only
detects verbatim reuse, so an agent that read the middle and reasoned
without quoting it is invisible; and the sample is one user's sessions,
heavy in build and profiling output.

## What this means for the PRECC integration

- Use the judge where it beats the string rules: semantic command class
  and flood prediction for commands with no class history, computed offline
  by the daemon with the large model.
- Do not use it as the truncation decider for tool results; a blind
  head+tail policy with an error-line guard is the cheaper baseline. The
  judge's `noul` "contains an error the agent must see" was collected but
  not scored against ground truth here; if it holds up, it is the guard
  that stops truncation, not the gate that decides it.
- Every claim above should be re-measured with PRECC's own ground-truth
  savings log before it is turned into a default.
