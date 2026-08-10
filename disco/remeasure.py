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

def main():
    seen, order = {}, []
    for line in open(lab.JOURNAL_JL):
        d = json.loads(line)
        if d.get("kind") != "run" or "val_bpb" not in (d.get("m") or {}):
            continue
        if d["id"] not in seen: order.append(d["id"])
        seen[d["id"]] = d

    print("re-measuring %d variants with the final binary (serial)" % len(order))
    out = {}
    for vid in order:
        d = seen[vid]
        ck = os.path.join(lab.RUNS, vid + ".bin")
        if not os.path.exists(ck):
            print("  %-18s SKIP (no checkpoint)" % vid); continue
        spec = {"id": vid, "arch": d["arch"], "knobs": d["m"].get("knobs") or {},
                "args": {"lr": d["m"].get("lr", 0.012)}, "seed": d["m"].get("seed", 1337)}
        if d["arch"] == "llama" and d["m"]["params"] not in (247552,):
            # width/depth controls need their shape flags back or the ckpt will not load
            p = d["m"]["params"]
            if p == 754432: spec["args"]["hidden"] = 704
            elif p == 62720: spec["args"]["layers"] = 1
        t0 = time.time()
        mi = lab.infer_one(spec)
        if "error" in mi:
            print("  %-18s ERROR %s" % (vid, mi["error"])); continue
        m = dict(d["m"]); m.update(mi)
        out[vid] = m
        lab.journal({"kind": "remeasure", "id": vid, "arch": d["arch"], "m": m})
        print("  %-18s bpb=%.4f uB/tok=%-9d win16=%-9d rss=%-7d tok/s=%-7.0f cons=%.1e (%.0fs)"
              % (vid, m["val_bpb"], m["wbytes_per_tok"], m.get("wbytes_window16", -1),
                 m["peak_rss_kb"], m["tok_per_sec"], m["consistency"], time.time() - t0))

    json.dump(out, open(os.path.join(lab.ARCH, "final_metrics.json"), "w"), indent=1)

    # rebuild the archive from the re-measured numbers
    arc = {}
    for m in out.values():
        lab.try_insert(m, arc)
    lab.save_archive(arc)
    print("\narchive rebuilt from re-measured metrics")
    print(lab.grid_str(arc))

if __name__ == "__main__":
    main()
