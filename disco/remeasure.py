#!/usr/bin/env python3
"""remeasure.py -- re-run every variant's inference pass with the FINAL binary,
serially, so that peak RSS, tok/s, bytes/token and the windowed working set are
all measured under identical conditions and with identical instrumentation.

Waves 1-4 were benchmarked before two evaluator changes (peak RSS is now read
before the consistency check, and a 16-token working-set measurement was added).
Rather than compare numbers produced by two different instruments, everything is
re-measured here.  Training is untouched: the checkpoints are reused verbatim,
so val_bpb does not change.

Append-only discipline is preserved: the original entries stay in the journal and
the re-measurements are appended as their own records.
"""
import json, os, sys, time
import lab

def load_specs():
    """The wave*.json files are the authoritative record of how each variant was
       built.  The journal stores measured metrics, not shape flags like
       hidden=153, and rebuilding with the wrong shape makes ckpt_load fail."""
    import glob
    specs = {}
    for f in sorted(glob.glob(os.path.join(os.path.dirname(os.path.abspath(__file__)), "wave*.json"))):
        for sp in json.load(open(f)).get("specs", []):
            specs[sp["id"]] = sp        # later waves win (e.g. multitok8 was re-run)
    return specs

def main():
    specs = load_specs()
    order, seen = [], set()
    for line in open(lab.JOURNAL_JL):
        d = json.loads(line)
        if d.get("kind") != "run" or "val_bpb" not in (d.get("m") or {}):
            continue
        if d["id"] not in seen:
            seen.add(d["id"]); order.append(d["id"])

    print("re-measuring %d variants with the final binary (serial)" % len(order))
    out = {}
    for vid in order:
        spec = specs.get(vid)
        ck = os.path.join(lab.RUNS, vid + ".bin")
        if spec is None or not os.path.exists(ck):
            print("  %-20s SKIP (%s)" % (vid, "no spec" if spec is None else "no checkpoint"))
            continue
        t0 = time.time()
        mi = lab.infer_one(spec)
        if "error" in mi:
            print("  %-20s ERROR %s" % (vid, mi["error"])); continue
        # keep the training-side numbers, replace every inference-side measurement
        prev = {}
        for line in open(lab.JOURNAL_JL):
            d = json.loads(line)
            if d.get("kind") == "run" and d.get("id") == vid and "val_bpb" in (d.get("m") or {}):
                prev = d["m"]
        m = dict(prev); m.update(mi); m["id"] = vid
        out[vid] = m
        lab.journal({"kind": "remeasure", "id": vid, "arch": m["arch"], "m": m})
        print("  %-20s bpb=%.4f uB/tok=%-9d ws16=%-9d rss=%-7d tok/s=%-7.0f cons=%.1e (%.0fs)"
              % (vid, m["val_bpb"], m["wbytes_per_tok"], m.get("working_set_16tok", -1),
                 m["peak_rss_kb"], m["tok_per_sec"], m["consistency"], time.time() - t0))

    json.dump(out, open(os.path.join(lab.ARCH, "final_metrics.json"), "w"), indent=1)
    arc = {}
    for m in out.values():
        lab.try_insert(m, arc)
    lab.save_archive(arc)
    print("\narchive rebuilt from re-measured metrics")
    print(lab.grid_str(arc))

if __name__ == "__main__":
    main()
