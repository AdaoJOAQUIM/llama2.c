#!/usr/bin/env python3
"""report.py -- final deliverable: grid, table, Pareto front, atypicality, verdicts."""
import json, math, os, sys, statistics
import lab

def runs():
    """Later entries win, so a `remeasure` record supersedes the original `run`."""
    rows = {}
    for line in open(lab.JOURNAL_JL):
        d = json.loads(line)
        if d.get("kind") in ("run", "remeasure") and "val_bpb" in (d.get("m") or {}):
            m = dict(d["m"])
            prev = rows.get(m["id"], {})
            m["_hyp"] = d.get("hypothesis", "") or prev.get("_hyp", "")
            m["_verdict"] = d.get("verdict", "") or prev.get("_verdict", "")
            m["_wave"] = d.get("wave", prev.get("_wave"))
            rows[m["id"]] = m
    return rows

def baseline_stats(rows):
    b = [r for r in rows.values() if r["arch"] == "llama" and r["id"].startswith("base")]
    v = [r["val_bpb"] for r in b]
    if not v: return None
    return dict(n=len(v), mean=statistics.mean(v),
                sd=statistics.stdev(v) if len(v) > 1 else 0.0,
                lo=min(v), hi=max(v), vals=sorted(v))

# atypicality: normalised distance from the baseline in the MEASURED behaviour space.
# log axes because these span orders of magnitude; a proxy for design-space distance.
AX = ["params", "wbytes_per_tok", "sbytes_per_tok", "macs_per_tok", "wtraffic_per_tok"]
def atypicality(rows, base):
    out = []
    logs = {a: [math.log(max(1.0, r.get(a, 1))) for r in rows.values()] for a in AX}
    sd = {a: (statistics.stdev(logs[a]) if len(logs[a]) > 1 else 1.0) or 1.0 for a in AX}
    for r in rows.values():
        d = 0.0; parts = []
        for a in AX:
            z = (math.log(max(1.0, r.get(a, 1))) - math.log(max(1.0, base.get(a, 1)))) / sd[a]
            d += z * z; parts.append((a, z))
        out.append((math.sqrt(d), r, parts))
    return sorted(out, key=lambda t: -t[0])

def main():
    R = runs()
    if not R:
        print("no runs recorded yet"); return
    bs = baseline_stats(R)
    base = R.get("base_s1") or next((r for r in R.values() if r["arch"] == "llama"), None)

    print("=" * 100)
    print("MAP-ELITES ARCHIVE")
    print("=" * 100)
    print(lab.grid_str())
    print()
    print(lab.occupants())

    print()
    print("=" * 100)
    print("ALL RUNS  (val_bpb | params | unique weight bytes/token | state bytes/token | MACs/token | tok/s | peak RSS)")
    print("=" * 100)
    hdr = "%-18s %-11s %8s %10s %11s %11s %9s %10s %8s %7s %5s" % (
        "id", "arch", "bpb", "params", "uB/tok", "ws16", "sB/tok", "MACs/tok",
        "tok/s", "rssKB", "cons")
    print(hdr); print("-" * len(hdr))
    for r in sorted(R.values(), key=lambda x: x["val_bpb"]):
        print("%-18s %-11s %8.4f %10d %11d %11s %9d %10d %8.0f %7d %5s" % (
            r["id"], r["arch"], r["val_bpb"], r["params"], r["wbytes_per_tok"],
            (str(r["working_set_16tok"]) if "working_set_16tok" in r else "-"),
            r.get("sbytes_per_tok", 0), r.get("macs_per_tok", 0), r["tok_per_sec"],
            r["peak_rss_kb"], "ok" if r["consistency"] < 1e-3 else "BAD"))

    if bs:
        print()
        print("=" * 100)
        print("NOISE FLOOR  (baseline llama, %d seeds, identical protocol)" % bs["n"])
        print("=" * 100)
        print("  val_bpb: " + ", ".join("%.4f" % v for v in bs["vals"]))
        print("  mean %.4f   sd %.4f   range %.4f" % (bs["mean"], bs["sd"], bs["hi"] - bs["lo"]))
        print("  2-sigma = %.4f  -> a variant must beat %.4f to have beaten the baseline at all"
              % (2 * bs["sd"], bs["mean"] - 2 * bs["sd"]))
        best_any = min([r["val_bpb"] for r in R.values() if r["arch"] == "llama"] + [9])
        print("  best baseline observed at ANY learning rate: %.4f (variants are compared against this)" % best_any)

    print()
    print("=" * 100)
    print("PARETO FRONT over (val_bpb v, unique bytes/tok v, peak RSS v, params v, tok/s ^)")
    print("=" * 100)
    for r in sorted(lab.pareto(list(R.values())), key=lambda x: x["val_bpb"]):
        print("  %-18s bpb=%.4f  uB/tok=%-9d params=%-9d rss=%-7d tok/s=%.0f"
              % (r["id"], r["val_bpb"], r["wbytes_per_tok"], r["params"], r["peak_rss_kb"], r["tok_per_sec"]))

    if base:
        print()
        print("=" * 100)
        print("MOST ATYPICAL INDIVIDUALS  (furthest from baseline in behaviour space, NOT the best)")
        print("=" * 100)
        for d, r, parts in atypicality(R, base)[:5]:
            if r["arch"] == "llama": continue
            print("  %-18s distance %.2f   bpb=%.4f" % (r["id"], d, r["val_bpb"]))
            print("      " + "  ".join("%s %+.1f sd" % (a.replace("_per_tok", "/tok").replace("_", ""), z)
                                        for a, z in parts))

if __name__ == "__main__":
    main()
