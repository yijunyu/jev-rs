#!/usr/bin/env python3
"""Can a judge predict, from a Bash command alone, that its output will flood
the agent's context? Ground truth comes from Claude Code session logs, which
record every Bash tool call and the size of the result the model then read.

    jev --backend http://127.0.0.1:8089 serve --bind 127.0.0.1:8090 &
    python3 scripts/predict_flood.py --logs ~/.claude/projects --n 200

Reports precision/recall of "will exceed --flood tokens" against a
leave-one-out baseline that mirrors precc's first-two-words command class,
plus the tokens that a policy of wrapping flagged commands would have kept
out of the context. Standard library only; no session content leaves the
machine except to the local jev server.
"""
import argparse, glob, json, os, random, re, statistics, sys, time, urllib.request

QUESTIONS = {
    "verbosity": {
        "type": "score",
        "instructions": "How much terminal output will this command produce?",
        "criteria": ["a few lines", "about a screenful", "hundreds of lines or more"],
    },
    "flood": {
        "type": "noul",
        "instructions": "Will this command print more than a screenful (over ~40 lines) of output?",
    },
}


def iter_commands(logs_dir):
    """Yield (command, output_chars) for every Bash tool call with a result."""
    pending = {}
    for path in glob.glob(os.path.join(logs_dir, "*", "*.jsonl")):
        try:
            with open(path, errors="replace") as f:
                for line in f:
                    try:
                        m = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    msg = m.get("message") or {}
                    content = msg.get("content")
                    if not isinstance(content, list):
                        continue
                    for block in content:
                        if not isinstance(block, dict):
                            continue
                        if block.get("type") == "tool_use" and block.get("name") == "Bash":
                            cmd = (block.get("input") or {}).get("command")
                            if isinstance(cmd, str):
                                pending[block.get("id")] = cmd
                        elif block.get("type") == "tool_result" and block.get("tool_use_id") in pending:
                            c = block.get("content")
                            if isinstance(c, list):
                                text = "".join(x.get("text", "") for x in c if isinstance(x, dict))
                            else:
                                text = c if isinstance(c, str) else ""
                            yield pending.pop(block["tool_use_id"]), len(text)
        except OSError:
            continue


def two_words(cmd):
    """precc's class: strip cd/env/sudo prefixes, first two words."""
    c = re.sub(r"^\s*cd\s+\S+\s*&&\s*", "", cmd.strip())
    c = re.sub(r"^(?:\w+=\S+\s+)+", "", c)
    c = re.sub(r"^(?:sudo|nohup|time|exec)\s+", "", c)
    return " ".join(c.split()[:2])


def ask(url, state):
    body = json.dumps({"model": "jev-latest", "state": state, "questions": QUESTIONS}).encode()
    req = urllib.request.Request(url, body, {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.load(r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", default=os.path.expanduser("~/.claude/projects"))
    ap.add_argument("--jev", default="http://127.0.0.1:8090/v1/systemone")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--flood", type=int, default=500, help="tokens (chars/4) that count as a flood")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default=None, help="write per-command rows as JSONL")
    a = ap.parse_args()

    rows = {}
    for cmd, n in iter_commands(a.logs):
        if len(cmd) > 400:
            continue
        rows.setdefault(cmd, []).append(n)
    items = [(c, statistics.median(v)) for c, v in rows.items()]
    random.Random(a.seed).shuffle(items)
    sample = items[: a.n]
    print(f"corpus: {len(items)} distinct commands, sampled {len(sample)}", file=sys.stderr)

    # Baseline: leave-one-out mean output size of the same two-word class.
    cls_sizes = {}
    for c, n in items:
        cls_sizes.setdefault(two_words(c), []).append(n)

    out = []
    t_all = time.time()
    for i, (cmd, chars) in enumerate(sample):
        state = "$ " + cmd
        t0 = time.time()
        try:
            ev = ask(a.jev, state)
        except Exception as e:
            print(f"[{i}] jev error: {e}", file=sys.stderr)
            continue
        ms = (time.time() - t0) * 1e3
        ans = ev["answers"]
        k = two_words(cmd)
        peers = [n for n in cls_sizes.get(k, []) if n != chars] or [n for n in cls_sizes.get(k, [])]
        baseline = statistics.mean(peers) / 4 if peers else 0
        out.append({
            "command": cmd, "actual_tokens": chars / 4, "class": k,
            "score": ans["verbosity"]["score"], "flood_p": ans["flood"]["noul"],
            "baseline_tokens": baseline, "latency_ms": ms,
        })
        if (i + 1) % 25 == 0:
            print(f"  {i+1}/{len(sample)}", file=sys.stderr)
    total_s = time.time() - t_all
    if a.out:
        with open(a.out, "w") as f:
            for r in out:
                f.write(json.dumps(r) + "\n")

    def report(name, flag):
        tp = sum(1 for r in out if flag(r) and r["actual_tokens"] >= a.flood)
        fp = sum(1 for r in out if flag(r) and r["actual_tokens"] < a.flood)
        fn = sum(1 for r in out if not flag(r) and r["actual_tokens"] >= a.flood)
        prec = tp / (tp + fp) if tp + fp else 0
        rec = tp / (tp + fn) if tp + fn else 0
        kept_out = sum(r["actual_tokens"] for r in out if flag(r))
        flood_tokens = sum(r["actual_tokens"] for r in out if r["actual_tokens"] >= a.flood)
        print(f"{name:32s} precision {prec:.2f} recall {rec:.2f} flagged {tp+fp:3d}  "
              f"tokens in flagged outputs {kept_out:8.0f} (of {flood_tokens:.0f} in floods)")

    floods = sum(1 for r in out if r["actual_tokens"] >= a.flood)
    total_tokens = sum(r["actual_tokens"] for r in out)
    print(f"\n{len(out)} commands, {floods} floods (>= {a.flood} tokens), {total_tokens:.0f} output tokens total")
    print(f"jev latency p50 {statistics.median(r['latency_ms'] for r in out):.0f} ms, wall {total_s:.0f} s\n")
    report("jev score >= 1.5 (hundreds+)", lambda r: r["score"] >= 1.5)
    report("jev score >= 1.0 (screenful+)", lambda r: r["score"] >= 1.0)
    report("jev flood noul >= 0.5", lambda r: r["flood_p"] >= 0.5)
    report("jev flood noul >= 0.8", lambda r: r["flood_p"] >= 0.8)
    report("baseline 2-word class mean", lambda r: r["baseline_tokens"] >= a.flood)
    report("flag everything", lambda r: True)


if __name__ == "__main__":
    main()
