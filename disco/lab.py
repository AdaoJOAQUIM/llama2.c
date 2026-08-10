#!/usr/bin/env python3
"""lab.py -- MAP-Elites archive, append-only journal, and wave runner.

Python stdlib only.  Holds no opinions about architectures; it only records
what the C evaluator measured and where that lands in the behaviour grid.
"""
import json, os, subprocess, sys, time, math, statistics
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, "runs")
ARCH = os.path.join(HERE, "archive")
JOURNAL_MD = os.path.join(HERE, "JOURNAL.md")
JOURNAL_JL = os.path.join(HERE, "journal.jsonl")
ARCHIVE_JSON = os.path.join(ARCH, "archive.json")

# ---------------- standard protocol (frozen; changing it invalidates comparisons) ----------
PROTO = dict(steps=800, bs=16, seq=256, valb=24, gen=256, threads=1, arena=700, lr=None)
BASE_SEED = 1337

# ---------------- MAP-Elites grid ----------------
# axis 1: total parameters.  axis 2: weight BYTES READ per emitted token.
P_EDGES = [0, 48e3, 96e3, 160e3, 224e3, 288e3, 384e3, 512e3, 768e3, float("inf")]
B_EDGES = [0, 32e3, 64e3, 128e3, 256e3, 512e3, 768e3, 1.05e6, 1.6e6, float("inf")]
P_LBL = ["<48K","48-96K","96-160K","160-224K","224-288K","288-384K","384-512K","512-768K",">768K"]
B_LBL = ["<32K","32-64K","64-128K","128-256K","256-512K","512-768K","768K-1.05M","1.05-1.6M",">1.6M"]

def binof(v, edges):
    for i in range(len(edges)-1):
        if edges[i] <= v < edges[i+1]:
            return i
    return len(edges)-2

def cell_of(m):
    return (binof(m["params"], P_EDGES), binof(m["wbytes_per_tok"], B_EDGES))

# ---------------- journal (append-only) ----------------
def journal(entry):
    entry = dict(entry)
    entry["ts"] = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(JOURNAL_JL, "a") as f:
        f.write(json.dumps(entry) + "\n")
    return entry

def journal_md(text):
    with open(JOURNAL_MD, "a") as f:
        f.write(text.rstrip() + "\n\n")

# ---------------- runner ----------------
def _cmd(mode, vid, arch, knobs, args):
    ck = os.path.join(RUNS, vid + ".bin")
    js = os.path.join(RUNS, vid + f".{mode}.json")
    c = [os.path.join(HERE, "disco"), mode, "--arch", arch, "--ckpt", ck, "--json", js]
    a = dict(PROTO); a.update(args or {})
    for k, v in a.items():
        if v is None: continue
        if mode == "infer" and k in ("steps","lr","valb","bs","arena"): continue
        if mode == "rescore" and k in ("gen","threads","repeats"): continue
        c += ["--" + k, str(v)]
    if mode == "infer": c += ["--arena", "16"]
    if mode == "rescore": c += ["--threads", "1"]
    for k, v in (knobs or {}).items():
        c += ["--knob", f"{k}={v}"]
    return c, js

def _spec_args(spec):
    args = dict(spec.get("args") or {})
    args["seed"] = spec.get("seed", BASE_SEED)
    return args

def train_one(spec):
    """If the spec reuses another variant's checkpoint, skip training entirely and
       just re-score it under this spec's emission rule (tau)."""
    vid = spec["id"]
    t0 = time.time()
    if spec.get("reuse_ckpt"):
        src = os.path.join(RUNS, spec["reuse_ckpt"] + ".bin")
        dst = os.path.join(RUNS, vid + ".bin")
        if os.path.abspath(src) != os.path.abspath(dst):
            import shutil; shutil.copyfile(src, dst)
        c, js = _cmd("rescore", vid, spec["arch"], spec.get("knobs"), _spec_args(spec))
        with open(os.path.join(RUNS, vid + ".log"), "w") as lg:
            r = subprocess.run(c, stdout=lg, stderr=subprocess.STDOUT, cwd=HERE)
        if r.returncode != 0:
            return {"id": vid, "error": "rescore failed", "cmd": " ".join(c)}
        m = json.load(open(js)); m["id"] = vid
        m["train_wall"] = round(time.time() - t0, 1); m["reused"] = spec["reuse_ckpt"]
        return m
    ctrain, jtrain = _cmd("train", vid, spec["arch"], spec.get("knobs"), _spec_args(spec))
    with open(os.path.join(RUNS, vid + ".log"), "w") as lg:
        r = subprocess.run(ctrain, stdout=lg, stderr=subprocess.STDOUT, cwd=HERE)
    if r.returncode != 0:
        return {"id": vid, "error": "train failed", "cmd": " ".join(ctrain)}
    m = json.load(open(jtrain))
    m["id"] = vid; m["train_wall"] = round(time.time() - t0, 1)
    return m

def infer_one(spec):
    """MUST run serialized: tok/s is wall-clock and concurrent load skews it 40%+."""
    vid = spec["id"]
    cinf, jinf = _cmd("infer", vid, spec["arch"], spec.get("knobs"), _spec_args(spec))
    with open(os.path.join(RUNS, vid + ".log"), "a") as lg:
        r = subprocess.run(cinf, stdout=lg, stderr=subprocess.STDOUT, cwd=HERE)
    if r.returncode != 0:
        return {"id": vid, "error": "infer failed", "cmd": " ".join(cinf)}
    return json.load(open(jinf))

def run_wave(specs, workers=4):
    """Train concurrently (CPU-bound, scales), then benchmark one at a time (timing-sensitive)."""
    with ThreadPoolExecutor(max_workers=workers) as ex:
        trained = list(ex.map(train_one, specs))
    out = []
    for spec, m in zip(specs, trained):
        if "error" in m: out.append(m); continue
        mi = infer_one(spec)
        if "error" in mi: out.append({**m, **mi}); continue
        m.update(mi)
        m["knobs"] = spec.get("knobs") or {}
        m["seed"] = _spec_args(spec)["seed"]
        m["note"] = spec.get("note", "")
        sp = os.path.join(RUNS, spec["id"] + ".bin.sample.txt")
        m["sample"] = open(sp).read()[:300] if os.path.exists(sp) else ""
        # --- evaluator self-checks: refuse to record a run that cannot be trusted ---
        flags = []
        if m.get("consistency", 0) > 1e-3:
            flags.append("DECODE_MISMATCH=%.2e" % m["consistency"])
        if m["wbytes_per_tok"] > m["stored_bytes"] + 8:
            flags.append("UNIQUE>STORED")
        if m["val_bpb"] != m["val_bpb"] or m["val_bpb"] > 8.1:
            flags.append("DIVERGED")
        m["flags"] = flags
        out.append(m)
    return out

# ---------------- Pareto front over the full metric vector ----------------
# MAP-Elites ranks only within a cell; this ranks globally on all five axes.
VEC = [("val_bpb", -1), ("wbytes_per_tok", -1), ("peak_rss_kb", -1),
       ("params", -1), ("tok_per_sec", +1)]
def dominates(a, b):
    ge = all((a[k] - b[k]) * s >= 0 for k, s in VEC)
    gt = any((a[k] - b[k]) * s > 0 for k, s in VEC)
    return ge and gt
def pareto(rows):
    return [r for r in rows if not any(dominates(o, r) for o in rows if o["id"] != r["id"])]

# ---------------- archive ----------------
def load_archive():
    if os.path.exists(ARCHIVE_JSON):
        return json.load(open(ARCHIVE_JSON))
    return {}

def save_archive(a):
    os.makedirs(ARCH, exist_ok=True)
    json.dump(a, open(ARCHIVE_JSON, "w"), indent=1)

def try_insert(m, arc=None):
    """Insert if it is the best val_bpb in its cell.  Returns (inserted, cellkey, prev)."""
    own = arc is None
    if own: arc = load_archive()
    ck = "%d,%d" % cell_of(m)
    prev = arc.get(ck)
    ins = prev is None or m["val_bpb"] < prev["val_bpb"]
    if ins: arc[ck] = m
    if own: save_archive(arc)
    return ins, ck, prev

def grid_str(arc=None):
    if arc is None: arc = load_archive()
    w = 13
    out = ["", "MAP-Elites: rows = params, cols = weight bytes read / token",
           "cell value = best val_bpb (bits per byte), lower is better", ""]
    hdr = " " * 10 + "".join(l.center(w) for l in B_LBL)
    out.append(hdr); out.append(" " * 10 + "-" * (w * len(B_LBL)))
    for i, pl in enumerate(P_LBL):
        row = pl.rjust(9) + "|"
        for j in range(len(B_LBL)):
            e = arc.get("%d,%d" % (i, j))
            row += (("%.3f" % e["val_bpb"]).center(w) if e else " . ".center(w))
        out.append(row)
    filled = len(arc); total = len(P_LBL) * len(B_LBL)
    out.append("")
    out.append("filled %d/%d cells" % (filled, total))
    return "\n".join(out)

def occupants(arc=None):
    if arc is None: arc = load_archive()
    rows = []
    for ck, e in sorted(arc.items(), key=lambda kv: kv[1]["val_bpb"]):
        i, j = map(int, ck.split(","))
        rows.append("%-22s %-9s bpb=%.4f params=%-8d wb/tok=%-9d tok/s=%-8.0f rss=%dKB  [%s | %s]"
                    % (e["id"], e["arch"], e["val_bpb"], e["params"], e["wbytes_per_tok"],
                       e["tok_per_sec"], e["peak_rss_kb"], P_LBL[i], B_LBL[j]))
    return "\n".join(rows)

def all_runs():
    rows = []
    if os.path.exists(JOURNAL_JL):
        for line in open(JOURNAL_JL):
            d = json.loads(line)
            if d.get("kind") == "run" and "val_bpb" in d.get("m", {}):
                rows.append(d["m"])
    return rows

if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "grid"
    if cmd == "grid":
        print(grid_str()); print(); print(occupants())
    elif cmd == "pareto":
        rows = {r["id"]: r for r in all_runs()}
        pf = pareto(list(rows.values()))
        print("Pareto front over (val_bpb, unique bytes/tok, peak RSS, params, tok/s):")
        for r in sorted(pf, key=lambda x: x["val_bpb"]):
            print("  %-24s bpb=%.4f  uB/tok=%-9d rss=%-7d params=%-8d tok/s=%.0f"
                  % (r["id"], r["val_bpb"], r["wbytes_per_tok"], r["peak_rss_kb"],
                     r["params"], r["tok_per_sec"]))
