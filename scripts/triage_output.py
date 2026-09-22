#!/usr/bin/env python3
"""After a Bash tool call returns, can a judge that SEES the output decide how
much of it the agent needs, and how many tokens would that save?

Ground truth is a proxy from the session logs: the agent's next turn (its
text and its next tool call) is scanned for 24-character shingles copied from
the part of the output the judge would have elided. If any appear, the
truncation would have hidden something the agent went on to use.

    jev --backend http://127.0.0.1:8089 serve --bind 127.0.0.1:8090 &
    python3 scripts/triage_output.py --logs ~/.claude/projects --n 150

Policy under test, by the judge's `keep` level:
  0  keep first 5 + last 5 lines          (summary/exit status is enough)
  1  keep first 30 + last 30 lines        (head and tail)
  2  keep everything
Only outputs of >= --min-lines lines are considered (short ones are kept as is).
"""
import argparse, glob, json, os, random, statistics, sys, time, urllib.request

QUESTIONS = {
    "keep": {
        "type": "score",
        "instructions": "The agent that ran this command will read the output to decide its next step. "
                        "How much of the output does it need?",
        "criteria": [
            "only a summary: the exit status and the first and last few lines suffice",
            "the head and the tail: the middle is repetitive or routine",
            "all of it: details in the middle matter (specific errors, values, file lists)",
        ],
    },
    "routine": {
        "type": "noul",
        "instructions": "Is the MIDDLE of this output routine or repetitive (progress lines, compile/download "
                        "logs, long file or test listings, dumps) so that the first and last 30 lines would tell "
                        "the agent everything it needs?",
    },
    "error": {
        "type": "noul",
        "instructions": "Does the output contain an error, failure or warning the agent must see?",
    },
}
HEAD, TAIL = {0: (5, 5), 1: (30, 30)}, None


def iter_calls(logs_dir):
    """Yield (command, output_text, next_turn_text) for Bash calls with a following assistant turn."""
    for path in glob.glob(os.path.join(logs_dir, "*", "*.jsonl")):
        try:
            msgs = []
            with open(path, errors="replace") as f:
                for line in f:
                    try:
                        m = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if m.get("type") in ("assistant", "user"):
                        msgs.append(m)
        except OSError:
            continue
        uses = {}
        for i, m in enumerate(msgs):
            content = (m.get("message") or {}).get("content")
            if not isinstance(content, list):
                continue
            for b in content:
                if not isinstance(b, dict):
                    continue
                if b.get("type") == "tool_use" and b.get("name") == "Bash":
                    uses[b.get("id")] = (b.get("input") or {}).get("command", "")
                elif b.get("type") == "tool_result" and b.get("tool_use_id") in uses:
                    c = b.get("content")
                    text = "".join(x.get("text", "") for x in c if isinstance(x, dict)) if isinstance(c, list) else (c or "")
                    nxt = ""
                    for m2 in msgs[i + 1:i + 3]:
                        if m2.get("type") != "assistant":
                            continue
                        c2 = (m2.get("message") or {}).get("content")
                        if isinstance(c2, list):
                            for b2 in c2:
                                if isinstance(b2, dict):
                                    nxt += b2.get("text", "") + json.dumps(b2.get("input", ""))
                        break
                    if nxt:
                        yield uses[b["tool_use_id"]], text, nxt


def shingles(text, k=24):
    text = " ".join(text.split())
    return {text[i:i + k] for i in range(0, max(0, len(text) - k), 8)}


import re
_ID = re.compile(r"[A-Za-z_][A-Za-z0-9_./:-]{5,}|\b\d{3,}(?:\.\d+)?\b")
_COMMON = set("target release debug build error warning src main test tests cargo python".split())


def idents(text):
    """Distinctive tokens: identifiers/paths/numbers the agent could only know from the text."""
    return {t for t in _ID.findall(text) if t.lower() not in _COMMON}


def used_from(elided, kept, nxt):
    """True if the next turn reuses a shingle, or a distinctive identifier that
    occurs ONLY in the elided text (not also in the kept head/tail), so the
    agent could not have read it from what was kept."""
    if not elided:
        return False
    if shingles(elided) & shingles(nxt):
        return True
    return bool((idents(elided) - idents(kept)) & idents(nxt))


def render(cmd, out, cap=6000):
    if len(out) > cap:
        out = out[: cap // 2] + "\n... [middle omitted for the judge] ...\n" + out[-cap // 2:]
    return f"$ {cmd}\n\n--- output ({out.count(chr(10)) + 1} lines) ---\n{out}"


def ask(url, state):
    body = json.dumps({"model": "jev-latest", "state": state, "questions": QUESTIONS}).encode()
    req = urllib.request.Request(url, body, {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.load(r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", default=os.path.expanduser("~/.claude/projects"))
    ap.add_argument("--jev", default="http://127.0.0.1:8090/v1/systemone")
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--min-lines", type=int, default=40)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default=None)
    ap.add_argument("--routine", type=float, default=0.7, help="P(routine middle) at which head+tail is applied")
    a = ap.parse_args()

    pool = [(c, o, n) for c, o, n in iter_calls(a.logs) if o.count("\n") + 1 >= a.min_lines and len(c) < 400]
    random.Random(a.seed).shuffle(pool)
    sample = pool[: a.n]
    print(f"corpus: {len(pool)} long outputs, sampled {len(sample)}", file=sys.stderr)

    rows = []
    t_all = time.time()
    for i, (cmd, out, nxt) in enumerate(sample):
        t0 = time.time()
        try:
            ev = ask(a.jev, render(cmd, out))
        except Exception as e:
            print(f"[{i}] jev error: {e}", file=sys.stderr)
            continue
        ms = (time.time() - t0) * 1e3
        score = ev["answers"]["keep"]["score"]
        routine_p = ev["answers"]["routine"]["noul"]
        level = 0 if score < 0.5 else 1 if score < 1.5 else 2
        # A confident "the middle is routine" overrides a timid "keep all".
        if level == 2 and routine_p >= a.routine:
            level = 1
        lines = out.split("\n")
        if level == 2 or len(lines) <= sum(HEAD[level]):
            elided, kept = "", out
        else:
            h, t = HEAD[level]
            kept = "\n".join(lines[:h] + ["[... elided ...]"] + lines[-t:])
            elided = "\n".join(lines[h:-t])
        used = used_from(elided, kept, nxt)
        rows.append({
            "command": cmd, "lines": len(lines), "tokens": len(out) / 4, "kept_tokens": len(kept) / 4,
            "level": level, "score": score, "routine_p": routine_p, "error_p": ev["answers"]["error"]["noul"],
            "elided_used": bool(used), "latency_ms": ms,
        })
        if (i + 1) % 25 == 0:
            print(f"  {i+1}/{len(sample)}", file=sys.stderr)
    if a.out:
        with open(a.out, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")

    tot = sum(r["tokens"] for r in rows)
    kept = sum(r["kept_tokens"] for r in rows)
    n_trunc = sum(1 for r in rows if r["level"] < 2)
    harmed = sum(1 for r in rows if r["elided_used"])
    by = {lv: [r for r in rows if r["level"] == lv] for lv in (0, 1, 2)}
    print(f"\n{len(rows)} outputs, {tot:.0f} tokens; judge kept {kept:.0f} ({100*(1-kept/tot):.1f}% saved)")
    print(f"truncated {n_trunc}/{len(rows)}; next turn used elided text in {harmed} cases "
          f"({100*harmed/max(1,n_trunc):.1f}% of truncations)")
    for lv in (0, 1, 2):
        rs = by[lv]
        if rs:
            print(f"  level {lv}: {len(rs):3d} outputs, mean {statistics.mean(r['tokens'] for r in rs):6.0f} tokens, "
                  f"elided-used {sum(r['elided_used'] for r in rs)}")
    print(f"jev latency p50 {statistics.median(r['latency_ms'] for r in rows):.0f} ms, wall {time.time()-t_all:.0f} s")
    # Reference: a blind head+tail 30/30 on everything, same proxy.
    blind_used = 0
    blind_kept = 0
    for (cmd, out, nxt), r in zip(sample, rows):
        lines = out.split("\n")
        if len(lines) <= 60:
            blind_kept += len(out) / 4
            continue
        blind_kept += len("\n".join(lines[:30] + lines[-30:])) / 4
        blind_used += used_from("\n".join(lines[30:-30]), "\n".join(lines[:30] + lines[-30:]), nxt)
    print(f"blind head30+tail30 on all: {100*(1-blind_kept/tot):.1f}% saved, elided text used in {blind_used} cases")


if __name__ == "__main__":
    main()
