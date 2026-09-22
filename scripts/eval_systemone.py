#!/usr/bin/env python3
"""Evaluate a JSONL case file against ANY `/v1/systemone` endpoint — hosted
Jev, `jev serve`, or an engine with the route built in (ds4-rs-metal). Same
case format and the same accuracy / Brier / ECE / latency numbers as
`jev eval`, so a server-side implementation can be compared with the CLI.

    python3 scripts/eval_systemone.py examples/dev_tasks.jsonl --url http://127.0.0.1:8002/v1/systemone
"""
import argparse, json, statistics, sys, time, urllib.request


def post(url, body, key=None):
    req = urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json"})
    if key:
        req.add_header("Authorization", f"Bearer {key}")
    with urllib.request.urlopen(req, timeout=3600) as r:
        return json.load(r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("file")
    ap.add_argument("--url", default="http://127.0.0.1:8090/v1/systemone")
    ap.add_argument("--key", default=None)
    a = ap.parse_args()
    cases = [json.loads(l) for l in open(a.file) if l.strip() and not l.lstrip().startswith("#")]

    rows, failed, lat = [], 0, []
    t_all = time.time()
    model = "?"
    for i, c in enumerate(cases):
        t0 = time.time()
        try:
            ev = post(a.url, {"model": "jev-latest", "state": c["state"], "questions": c["questions"]}, a.key)
        except Exception as e:
            print(f"case {i}: {e}", file=sys.stderr)
            failed += 1
            continue
        lat.append((time.time() - t0) * 1e3)
        model = ev.get("model", model)
        for qid, gold in c["gold"].items():
            ans = ev["answers"].get(qid)
            if not ans:
                continue
            q = c["questions"][qid]
            if ans["type"] == "noul":
                probs = [ans["noul"], 1 - ans["noul"]]
                keys = ["yes", "no"]
            elif ans["type"] == "choice":
                keys = list(q["criteria"].keys())
                probs = [ans["probabilities"].get(k, 0.0) for k in keys]
            else:
                keys = [str(j) for j in range(len(q["criteria"]))]
                probs = [ans["probabilities"].get(k, 0.0) for k in keys]
            g = keys.index(str(gold)) if str(gold) in keys else (int(gold) if isinstance(gold, int) else None)
            if g is None:
                continue
            pred = max(range(len(probs)), key=lambda j: probs[j])
            rows.append((ans["type"], probs, g, pred))
    n = len(rows)
    if not n:
        print("no scored rows"); return
    acc = sum(p == g for _, _, g, p in rows) / n
    brier = sum(sum((pi - (1.0 if j == g else 0.0)) ** 2 for j, pi in enumerate(pr)) for _, pr, g, _ in rows) / n
    bins = [[0, 0, 0.0] for _ in range(10)]
    for _, pr, g, p in rows:
        b = min(9, int(pr[p] * 10)); bins[b][0] += 1; bins[b][1] += (p == g); bins[b][2] += pr[p]
    ece = sum(c / n * abs(k / c - s / c) for c, k, s in bins if c)
    by = {}
    for t, _, g, p in rows:
        by.setdefault(t, [0, 0]); by[t][0] += 1; by[t][1] += (p == g)
    print(json.dumps({
        "model": model, "questions": n, "failed_cases": failed, "accuracy": round(acc, 3),
        "brier": round(brier, 3), "ece": round(ece, 3),
        "latency_p50_ms_per_case": round(statistics.median(lat), 1) if lat else None,
        "wall_s": round(time.time() - t_all, 1),
        "by_kind": {k: round(v[1] / v[0], 3) for k, v in by.items()},
    }, indent=1))


if __name__ == "__main__":
    main()
