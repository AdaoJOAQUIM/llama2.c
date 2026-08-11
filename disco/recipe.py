#!/usr/bin/env python3
"""recipe.py -- DR1, a representation format for trained checkpoints.

The idea is not compression in the Shannon sense and this file does not pretend
otherwise.  A trained checkpoint is a DETERMINISTIC function of

    (source code, corpus, architecture, hyper-parameters, seed, thread count)

so the checkpoint need not be stored at all: store the arguments and recompute.
That is only a legitimate *representation* if reproduction is exact, so every
recipe carries the SHA-256 of the checkpoint it claims to generate, and `verify`
regenerates and compares. Anything that does not reproduce bit-for-bit is
reported as a failure, not rounded away.

Measured prerequisites (see JOURNAL.md, "Wave 12"):
  - same thread count  -> bit-identical checkpoints
  - different thread count -> different checkpoints
    (per-thread accumulator tiles in the linear backward change summation order)
  so `threads` is part of the recipe, not an execution detail.

DR1 wire format, one line, '|' separated:

  DR1|arch|dim,layers,heads,kvheads,hidden,vocab,tie|knobs|steps,bs,seq,threads
     |lr,wd,warmup,seed|corpus16|src16|ckpt64

`corpus16` and `src16` are truncated hashes of the data and of the C sources:
they do not reconstruct those inputs, they only detect that you are trying to
replay a recipe against the wrong ones.
"""
import hashlib, json, os, subprocess, sys, glob

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, "runs")
SRC = ["nn.c", "nn.h", "archs.c", "main.c", "mem.c", "mem.h", "model.h", "Makefile"]
DEFAULTS = dict(dim=64, layers=5, heads=8, kvheads=4, hidden=176, vocab=257, tie=1,
                steps=800, bs=16, seq=256, threads=1, lr=0.012, wd=0.1, warmup=100, seed=1337)

def sha(path, n=None):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    d = h.hexdigest()
    return d[:n] if n else d

def src_hash(n=16):
    h = hashlib.sha256()
    for s in SRC:
        p = os.path.join(HERE, s)
        if os.path.exists(p):
            h.update(open(p, "rb").read())
    return h.hexdigest()[:n]

def emit(spec, ckpt_path):
    """spec: {arch, knobs, args{...}, seed} -> one DR1 line."""
    a = dict(DEFAULTS); a.update(spec.get("args") or {})
    a["seed"] = spec.get("seed", a["seed"])
    cfg = "%d,%d,%d,%d,%d,%d,%d" % (a["dim"], a["layers"], a["heads"], a["kvheads"],
                                    a["hidden"], a["vocab"], a["tie"])
    kn = ",".join("%s:%g" % (k, v) for k, v in sorted((spec.get("knobs") or {}).items()))
    tr = "%d,%d,%d,%d" % (a["steps"], a["bs"], a["seq"], a["threads"])
    hp = "%g,%g,%d,%d" % (a["lr"], a["wd"], a["warmup"], a["seed"])
    return "|".join(["DR1", spec["arch"], cfg, kn, tr, hp,
                     sha(os.path.join(HERE, "data/corpus.bin"), 16), src_hash(),
                     sha(ckpt_path)])

def parse(line):
    p = line.strip().split("|")
    assert p[0] == "DR1", "not a DR1 recipe"
    dim, layers, heads, kvheads, hidden, vocab, tie = map(int, p[2].split(","))
    steps, bs, seq, threads = map(int, p[4].split(","))
    lr, wd, warmup, seed = p[5].split(",")
    knobs = {}
    if p[3]:
        for kv in p[3].split(","):
            k, v = kv.split(":"); knobs[k] = v
    return dict(arch=p[1], dim=dim, layers=layers, heads=heads, kvheads=kvheads,
                hidden=hidden, vocab=vocab, tie=tie, steps=steps, bs=bs, seq=seq,
                threads=threads, lr=lr, wd=wd, warmup=int(warmup), seed=int(seed),
                knobs=knobs, corpus16=p[6], src16=p[7], ckpt=p[8])

def verify(line, workdir="/tmp"):
    """Regenerate from the recipe and compare SHA-256. Returns (ok, got, expected)."""
    r = parse(line)
    if sha(os.path.join(HERE, "data/corpus.bin"), 16) != r["corpus16"]:
        return (False, "CORPUS MISMATCH", r["ckpt"])
    if src_hash() != r["src16"]:
        return (False, "SOURCE MISMATCH", r["ckpt"])
    out = os.path.join(workdir, "dr1_replay.bin")
    cmd = [os.path.join(HERE, "disco"), "train", "--arch", r["arch"],
           "--dim", str(r["dim"]), "--layers", str(r["layers"]), "--heads", str(r["heads"]),
           "--kvheads", str(r["kvheads"]), "--hidden", str(r["hidden"]), "--tie", str(r["tie"]),
           "--steps", str(r["steps"]), "--bs", str(r["bs"]), "--seq", str(r["seq"]),
           "--threads", str(r["threads"]), "--lr", r["lr"], "--wd", r["wd"],
           "--seed", str(r["seed"]), "--ckpt", out, "--arena", "1200"]
    for k, v in r["knobs"].items():
        cmd += ["--knob", "%s=%s" % (k, v)]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, cwd=HERE, check=True)
    got = sha(out)
    return (got == r["ckpt"], got, r["ckpt"])

def load_specs():
    specs = {}
    for f in sorted(glob.glob(os.path.join(HERE, "wave*.json"))):
        for sp in json.load(open(f)).get("specs", []):
            specs[sp["id"]] = sp
    # the wave-11 runs were driven directly rather than through a wave file
    for v, mem in (("mem0", 0), ("mem1", 1)):
        for suf, seed in (("", 1337), ("_s2", 2)):
            specs["w11_ortho_%s%s" % (v, suf)] = dict(
                id="w11_ortho_%s%s" % (v, suf), arch="orthostack",
                knobs={"mem": mem, "experts": 4, "convw": 4},
                args={"lr": 0.012, "threads": 2, "arena": 900}, seed=seed)
    return specs

def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "stat"
    specs = load_specs()
    if cmd == "emit":
        lines, tot_ck = [], 0
        for vid, sp in sorted(specs.items()):
            ck = os.path.join(RUNS, vid + ".bin")
            if not os.path.exists(ck) or sp.get("reuse_ckpt"): continue
            lines.append(emit(sp, ck)); tot_ck += os.path.getsize(ck)
        book = os.path.join(HERE, "archive", "recipes.dr1")
        open(book, "w").write("\n".join(lines) + "\n")
        rb = os.path.getsize(book)
        corpus = os.path.getsize(os.path.join(HERE, "data/corpus.bin"))
        srcb = sum(os.path.getsize(os.path.join(HERE, s)) for s in SRC if os.path.exists(os.path.join(HERE, s)))
        print("recipes written: %d  -> %s" % (len(lines), book))
        print()
        print("  checkpoints represented      %13s bytes" % f"{tot_ck:,}")
        print("  recipe book                  %13s bytes" % f"{rb:,}")
        print("  mean recipe                  %13s bytes" % f"{rb//max(1,len(lines)):,}")
        print()
        print("  REFERENTIAL ratio (corpus+source already present, the re-run case)")
        print("     %.0fx   overall        %.0fx   per checkpoint"
              % (tot_ck / rb, (tot_ck / len(lines)) / (rb / len(lines))))
        print()
        print("  SELF-CONTAINED ratio (ship corpus + C source + recipes, nothing else)")
        sc = rb + corpus + srcb
        print("     payload = recipes %s + corpus %s + source %s = %s bytes"
              % (f"{rb:,}", f"{corpus:,}", f"{srcb:,}", f"{sc:,}"))
        print("     %.2fx   (this is the honest number for a cold recipient)" % (tot_ck / sc))
        print("     break-even at %.0f checkpoints; archive currently holds %d"
              % ((corpus + srcb) / max(1e-9, (tot_ck / len(lines)) - (rb / len(lines))), len(lines)))
    elif cmd == "verify":
        vid = sys.argv[2]
        book = os.path.join(HERE, "archive", "recipes.dr1")
        specs_line = None
        for ln in open(book):
            sp = specs.get(vid)
            if sp and parse(ln)["arch"] == sp["arch"] and parse(ln)["ckpt"] == sha(os.path.join(RUNS, vid + ".bin")):
                specs_line = ln; break
        if not specs_line: print("no recipe matching %s" % vid); return
        print("replaying %s ..." % vid)
        ok, got, exp = verify(specs_line)
        print("  expected %s" % exp)
        print("  got      %s" % got)
        print("  -> %s" % ("BIT-EXACT REPRODUCTION" if ok else "MISMATCH"))
    else:
        print(__doc__)

if __name__ == "__main__":
    main()
