#!/usr/bin/env python3
"""wave.py -- run one wave of variants, journal them, update the archive.

usage: wave.py <wavefile.json>
wavefile: {"wave": N, "specs": [ {id, arch, knobs, args, note, hypothesis,
                                  prediction: {val_bpb:[lo,hi], params:P, wbytes:B}} ]}
"""
import json, sys, os, time
import lab

def fmt(m):
    return ("bpb=%.4f  params=%d  uB/tok=%d  traffic/tok=%d  rss=%dKB  tok/s=%.0f  "
            "macs/tok=%d  consist=%.1e" %
            (m["val_bpb"], m["params"], m["wbytes_per_tok"], m.get("wtraffic_per_tok", -1),
             m["peak_rss_kb"], m["tok_per_sec"], m["macs_per_tok"], m["consistency"]))

def verdict(spec, m):
    p = spec.get("prediction", {})
    lines = []
    ok_cell = True
    for key, pk in (("params", "params"), ("wbytes_per_tok", "wbytes")):
        if pk in p:
            pred, got = p[pk], m[key]
            rel = abs(got - pred) / max(1.0, pred)
            lines.append("  %-16s predicted %-10s measured %-10s %s"
                         % (pk, f"{pred:,}", f"{got:,}", "ok" if rel < 0.02 else "MISS"))
            if rel >= 0.02: ok_cell = False
    if "val_bpb" in p:
        lo, hi = p["val_bpb"]; got = m["val_bpb"]
        inside = lo <= got <= hi
        lines.append("  %-16s predicted [%.2f, %.2f]  measured %.4f  -> %s"
                     % ("val_bpb", lo, hi, got, "CONFIRMED" if inside else
                        ("REFUTED (better)" if got < lo else "REFUTED (worse)")))
    return "\n".join(lines), ok_cell

def main(path):
    W = json.load(open(path))
    n = W["wave"]; specs = W["specs"]
    print(f"=== wave {n}: {len(specs)} variants ===", flush=True)
    t0 = time.time()
    res = lab.run_wave(specs)
    arc = lab.load_archive()
    md = [f"## Wave {n}", ""]
    for spec, m in zip(specs, res):
        if "error" in m:
            print(f"[{spec['id']}] ERROR {m['error']}")
            md.append(f"### {spec['id']} — FAILED: {m['error']}")
            lab.journal({"kind": "run", "wave": n, "id": spec["id"], "error": m["error"]})
            continue
        ins, ck, prev = lab.try_insert(m, arc)
        vtxt, _ = verdict(spec, m)
        flags = m.get("flags") or []
        print(f"[{m['id']}] {fmt(m)}")
        if flags: print("   FLAGS:", flags)
        print(vtxt)
        md += [f"### {spec['id']} — `{spec['arch']}`", "",
               "**Hypothesis.** " + spec.get("hypothesis", ""), "",
               "**Prediction (recorded before the run).** " + spec.get("pred_text", ""), "",
               "```", fmt(m), "```", "",
               "**Verdict.**", "```", vtxt, "```", ""]
        if flags:
            md += ["> **FLAGGED — result not trusted:** " + ", ".join(flags), ""]
        cellmsg = ("new cell %s" % ck) if prev is None else (
            "improves cell %s (%.4f -> %.4f)" % (ck, prev["val_bpb"], m["val_bpb"]) if ins
            else "cell %s already held by %s at %.4f" % (ck, prev["id"], prev["valued"] if False else prev["val_bpb"]))
        md += ["Archive: " + cellmsg, ""]
        if m.get("sample"):
            md += ["<details><summary>sample</summary>", "", "```",
                   m["sample"][:280].replace("`", "'"), "```", "", "</details>", ""]
        print("   archive:", cellmsg)
        lab.journal({"kind": "run", "wave": n, "id": spec["id"], "arch": spec["arch"],
                     "hypothesis": spec.get("hypothesis", ""),
                     "prediction": spec.get("prediction", {}),
                     "verdict": vtxt, "cell": ck, "inserted": ins, "m": m})
    lab.save_archive(arc)
    lab.journal_md("\n".join(md))
    print("\nwave %d done in %.1f min" % (n, (time.time() - t0) / 60))
    print(lab.grid_str(arc))

if __name__ == "__main__":
    main(sys.argv[1])
