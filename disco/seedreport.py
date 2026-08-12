#!/usr/bin/env python3
"""seedreport.py -- consolidate the compression-ratio experiment.

Everything printed here is read from files written by executed runs: model.lab's
append-only RESULT lines, the stress_*.json produced by stress.py, and
archive/ceiling.json.  Nothing is interpolated between measured points.
"""
import json, math, os, re, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
LAB = os.path.join(HERE, "archive", "model.lab")
DENSE_BYTES = 990404          # the dense llama .bin, measured
HDR = 192
NTOT = 247552

# (id, trainable fraction, human ratio label)
POINTS = [("seed_r2", 0.499903), ("seed_r4", 0.249855), ("seed_r10", 0.0998255),
          ("seed_r20", 0.0498158), ("seed_r100", 0.00980806),
          ("seed_r200", 0.00480707), ("seed_r1000", 0.000803873)]
CONTROLS = ["seed_r200_f9", "seed_r200_lr20", "seed_r200_lr5", "seed_r100_lr5"]


def lab_rows(kind):
    out = []
    for ln in open(LAB):
        if ln.startswith(kind + "|"):
            out.append(ln.rstrip("\n").split("|")[1:])
    return out


def results():
    d = {}
    for r in lab_rows("RESULT"):
        d.setdefault(r[0], []).append(float(r[2]))
    return d


def preds():
    return {r[0]: (float(r[1]), float(r[2])) for r in lab_rows("PRED")}


def ratio_of(frac):
    k = int(frac * NTOT + 0.5)
    return DENSE_BYTES / (HDR + 4 * k), k, HDR + 4 * k


def dense_anchor():
    vs = []
    for s in (1337, 2):
        p = os.path.join(HERE, "runs", "seed", "dense_s%d.json" % s)
        if os.path.exists(p):
            vs.append(json.load(open(p))["val_bpb"])
    return vs


def main():
    R, P = results(), preds()
    C = {}
    cp = os.path.join(HERE, "archive", "ceiling.json")
    if os.path.exists(cp):
        C = json.load(open(cp))
    anchor = dense_anchor()

    print("=" * 78)
    print("SEED FORMAT -- compression ratio vs quality, on the 247,552-parameter archive")
    print("=" * 78)
    if anchor:
        print("dense 1x anchor (this binary, threads=1, seeds %s): %s  mean %.4f"
              % ("/".join("1337 2".split()), " ".join("%.4f" % v for v in anchor),
                 sum(anchor) / len(anchor)))
    print()
    print("%-14s %8s %9s %8s   %-14s %9s  %s"
          % ("id", "ratio", "K train", "bytes", "predicted", "measured", "verdict"))
    print("-" * 78)
    for pid, frac in POINTS:
        r, k, b = ratio_of(frac)
        vs = R.get(pid, [])
        m = sum(vs) / len(vs) if vs else float("nan")
        lo, hi = P.get(pid, (float("nan"),) * 2)
        ok = "in" if vs and lo <= m <= hi else ("OUT" if vs else "-")
        print("%-14s %7.1fx %9d %8d   [%.2f,%.2f]  %9s  %s"
              % (pid, r, k, b, lo, hi,
                 ("%.4f" % m) if vs else "pending", ok))
    print()
    print("fairness controls at a fixed ratio (different mask / different LR)")
    print("-" * 78)
    for pid in CONTROLS:
        vs = R.get(pid, [])
        m = sum(vs) / len(vs) if vs else float("nan")
        lo, hi = P.get(pid, (float("nan"),) * 2)
        print("%-14s %38s [%.2f,%.2f]  %9s"
              % (pid, "", lo, hi, ("%.4f" % m) if vs else "pending"))

    if C:
        print()
        print("zero-capacity reference points, computed on the SAME scored tokens")
        print("-" * 78)
        for k, lbl in (("uniform_bpb", "uniform over 257 symbols"),
                       ("order0_bpb", "unigram table fitted on train"),
                       ("order1_bpb", "bigram  table fitted on train"),
                       ("order2_bpb", "trigram table fitted on train")):
            print("  %-32s %.4f bpb" % (lbl, C[k]))
        print("  a compressed model at or above the trigram line contains less usable")
        print("  structure than counting triples of bytes.")

    # stress battery
    st = []
    for f in sorted(os.listdir(os.path.join(HERE, "archive"))):
        if f.startswith("stress_") and f.endswith(".json"):
            st.append(json.load(open(os.path.join(HERE, "archive", f))))
    if st:
        print()
        print("=" * 78)
        print("DYNAMIC STRESS BATTERY")
        print("=" * 78)
        print("%-14s %8s %9s %8s %8s %9s %7s"
              % ("label", "ratio", "lossless", "load_s", "RSS_MB", "tok/s", "uniqB"))
        print("-" * 78)
        for s in st:
            L, rt = s["lossless"], s["runtime"]
            print("%-14s %7.1fx %9s %8.3f %8.1f %9.1f %7d"
                  % (s["label"], L["ratio"], L["lossless"], rt["load_sec"],
                     rt["peak_rss_kb"] / 1024.0, rt["tok_per_sec"], rt["wbytes_per_tok"]))
        print()
        print("out-of-distribution bpb (ood_id is the untransformed control)")
        print("-" * 78)
        keys = ["ood_id", "ood_rot13", "ood_case", "ood_shuf", "ood_synth"]
        print("%-14s %9s %s" % ("label", "val", " ".join("%9s" % k[4:] for k in keys)))
        for s in st:
            print("%-14s %9.4f %s"
                  % (s["label"], s["val_bpb"],
                     " ".join("%9.4f" % s["ood"][k] for k in keys)))
        print()
        print("long-horizon decode (2048 tokens) and payload fault injection")
        print("-" * 78)
        print("%-14s %8s %9s %10s %11s %11s"
              % ("label", "distinct4", "maxrepeat", "payloadbits", "flip16 med", "BER1e-4 med"))
        for s in st:
            h, f = s["horizon"], s["faults"]
            print("%-14s %8.3f %9d %10d %11s %11s"
                  % (s["label"], h["distinct4"], h["longest_period_repeat"],
                     f["payload_bits"],
                     ("%.3f" % f["flip16"]["median"]) if f["flip16"]["median"] else "NaN",
                     ("%.3f" % f["rate1e4"]["median"]) if f["rate1e4"]["median"] else "NaN"))
        if any("bypos" in s for s in st):
            print()
            print("bpb by position in the 256-token window (16 buckets of 16)")
            print("-" * 78)
            for s in st:
                if "bypos" in s:
                    print("%-14s %s" % (s["label"],
                                        " ".join("%.2f" % v for v in s["bypos"])))


if __name__ == "__main__":
    main()
