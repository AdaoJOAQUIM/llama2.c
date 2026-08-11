#!/usr/bin/env python3
"""mklab.py -- emit a LAB/1 object: an executable, self-falsifying knowledge file.

Closest prior art, named up front rather than discovered later:
  - Nix / Bazel      content-addressed derivations           (DR2 was this)
  - Jupyter          executable documents                    (no self-modification, no epistemics)
  - MLflow / W&B     experiment tracking                     (records runs; enforces nothing)
  - preregistration  (OSF, clinical trials)                  (social convention, not machine-checked)
  - AutoML           searches                                (no falsification record)

The one property I believe is unusual: **the file mechanically refuses to record
a claim it has not earned.** Every other format above will happily store
"variant X is better" because a human typed it. LAB/1 will not:

  * `run` refuses to execute an experiment with no pre-registered prediction.
  * `preregister` refuses to accept a prediction for an id that already has
    results (no prediction written after seeing the answer).
  * a result is recorded as a WIN only if it clears the file's own MEASURED
    noise floor by 2 sigma across a minimum number of seeds. Otherwise it is
    recorded as INDISTINGUISHABLE -- and that verdict is not overridable.
  * `challenge` re-materialises an old claim from its recipe and re-checks it,
    so a claim that has silently stopped reproducing is detectable.

The file therefore stores conclusions AND the entitlement to them, and running
it does science rather than building an artifact. It appends to itself: its
ledger only grows, and no line is ever rewritten.
"""
import hashlib, json, os, statistics, sys

HERE = os.path.dirname(os.path.abspath(__file__))

INTERP = r'''#!/usr/bin/env python3
"""LAB/1 -- executable knowledge object.

  ./model.lab status              what is known, and with what confidence
  ./model.lab frontier            the behaviour frontier (best per niche)
  ./model.lab ledger              every prediction ever made, and its verdict
  ./model.lab open                questions this file cannot currently answer
  ./model.lab preregister ID LO HI ARCH [k=v ...]     declare BEFORE running
  ./model.lab run ID                                   execute; verdict is forced
  ./model.lab challenge ID                             re-verify an old claim
  ./model.lab materialize ID                           rebuild weights, hash-checked
"""
import hashlib,os,re,statistics,subprocess,sys,time
SELF=os.path.abspath(__file__); ROOT=os.path.dirname(os.path.dirname(SELF))
def rows(tag):
    out=[]
    for ln in open(SELF):
        if ln.startswith(tag+"|"): out.append(ln.rstrip("\n").split("|")[1:])
    return out
def append(line):
    """Append-only, but the data block lives inside a string literal, so new
       lines splice in just before its terminator rather than after EOF."""
    txt=open(SELF).read(); i=txt.rindex(chr(10)+chr(34)*3+chr(10))
    open(SELF,"w").write(txt[:i+1]+line+chr(10)+txt[i+1:])
def sha(p,n=None):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    d=h.hexdigest(); return d[:n] if n else d
SRC=["nn.c","nn.h","archs.c","main.c","mem.c","mem.h","model.h","Makefile"]
def srch(n=16):
    h=hashlib.sha256()
    for s in SRC:
        p=os.path.join(ROOT,s)
        if os.path.exists(p): h.update(open(p,'rb').read())
    return h.hexdigest()[:n]

def noise():
    n=rows("NOISE")[0]
    return dict(metric=n[0],sd=float(n[1]),mean=float(n[2]),k=int(n[3]),minseeds=int(n[4]))
def preds():
    d={}
    for r in rows("PRED"): d[r[0]]=dict(id=r[0],lo=float(r[1]),hi=float(r[2]),arch=r[3],knobs=r[4],ts=r[5])
    return d
def results():
    d={}
    for r in rows("RESULT"): d.setdefault(r[0],[]).append(dict(seed=int(r[1]),val=float(r[2]),sha=r[3]))
    return d
def verdicts():
    d={}
    for r in rows("VERDICT"): d[r[0]]=dict(id=r[0],status=r[1],margin=float(r[2]),note=r[3])
    return d

def judge(rid):
    """The rule the file will not let you argue with."""
    N=noise(); R=results().get(rid,[]); P=preds().get(rid)
    if not P: return ("NO-PREREGISTRATION",0.0,"no prediction was declared before running")
    if len(R)<N["minseeds"]:
        return ("UNDERPOWERED",0.0,"%d/%d seeds"%(len(R),N["minseeds"]))
    vals=[x["val"] for x in R]; m=statistics.mean(vals)
    margin=N["mean"]-m                       # positive = better than baseline
    thr=2*N["sd"]
    inpred = P["lo"]<=m<=P["hi"]
    if margin>thr:  st="WIN"
    elif margin<-thr: st="LOSS"
    else: st="INDISTINGUISHABLE"
    note="mean %.4f vs baseline %.4f, margin %+.4f, threshold %.4f (2sd); prediction [%.2f,%.2f] %s"%(
        m,N["mean"],margin,thr,P["lo"],P["hi"],"held" if inpred else "REFUTED")
    return (st,margin,note)

def cmd_status():
    N=noise(); V=verdicts(); R=results()
    print("LAB/1  %s"%SELF)
    print("  corpus %s   source %s"%(rows("ENV")[0][0],srch()))
    if rows("ENV")[0][0]!=sha(os.path.join(ROOT,"data/corpus.bin"),16):
        print("  !! corpus differs from the one every claim was measured on")
    if rows("ENV")[0][1]!=srch():
        print("  !! source differs (%s recorded) -- claims are NOT replayable as-is"%rows("ENV")[0][1])
    print("  metric %s   baseline %.4f   sd %.4f over %d seeds   win needs margin > %.4f"%(
        N["metric"],N["mean"],N["sd"],N["k"],2*N["sd"]))
    w=[k for k,v in V.items() if v["status"]=="WIN"]
    i=[k for k,v in V.items() if v["status"]=="INDISTINGUISHABLE"]
    l=[k for k,v in V.items() if v["status"]=="LOSS"]
    print("  claims: %d earned, %d indistinguishable, %d refuted-worse, %d ids with data"%(len(w),len(i),len(l),len(R)))
    for k in sorted(w,key=lambda x:-V[x]["margin"]): print("     WIN  %-22s %+.4f"%(k,V[k]["margin"]))
def cmd_frontier():
    print("behaviour frontier (best per niche; niche = params x unique weight bytes/token)")
    for r in sorted(rows("FRONT"),key=lambda x:float(x[2])):
        print("  %-12s %-22s %s=%s  params=%-9s bytes/tok=%-9s"%(r[0],r[1],noise()["metric"],r[2],r[3],r[4]))
def cmd_ledger():
    P=preds(); V=verdicts(); R=results(); hit=0; tot=0
    print("%-22s %-14s %9s %-18s %s"%("id","predicted","measured","status","note"))
    for k in sorted(P):
        rr=R.get(k,[])
        m=statistics.mean([x["val"] for x in rr]) if rr else float("nan")
        v=V.get(k,{"status":"PENDING","note":""})
        if rr:
            tot+=1; hit+= 1 if P[k]["lo"]<=m<=P[k]["hi"] else 0
        print("%-22s [%4.2f,%4.2f]   %9.4f %-18s %s"%(k,P[k]["lo"],P[k]["hi"],m,v["status"],v["note"][:60]))
    if tot: print("\ncalibration: %d/%d predictions landed inside their declared interval (%.0f%%)"%(hit,tot,100*hit/tot))
def cmd_open():
    for r in rows("OPEN"): print("  ?  %s"%r[0])
def cmd_prereg(a):
    rid,lo,hi,arch=a[0],a[1],a[2],a[3]; kn=",".join(a[4:])
    if rid in results(): print("REFUSED: %s already has results. A prediction written after seeing the answer is not a prediction."%rid); sys.exit(1)
    if rid in preds():   print("REFUSED: %s already has a prediction; it cannot be revised."%rid); sys.exit(1)
    append("PRED|%s|%s|%s|%s|%s|%s"%(rid,lo,hi,arch,kn,time.strftime("%Y-%m-%dT%H:%M:%S")))
    print("pre-registered %s in [%s,%s]"%(rid,lo,hi))
def cmd_run(a):
    rid=a[0]; seeds=[int(x) for x in a[1:]] or [1337,2]
    P=preds().get(rid)
    if not P: print("REFUSED: no pre-registration for %s. Declare a prediction first."%rid); sys.exit(1)
    for s in seeds:
        ck=os.path.join(ROOT,"runs","lab_%s_s%d.bin"%(rid,s))
        cmd=[os.path.join(ROOT,"disco"),"train","--arch",P["arch"],"--steps","800","--bs","16",
             "--seq","256","--threads","2","--lr","0.012","--seed",str(s),"--ckpt",ck,"--arena","1200"]
        for kv in P["knobs"].split(",") if P["knobs"] else []: cmd+=["--knob",kv]
        js=ck+".json"; cmd+=["--json",js]
        print("  running %s seed %d ..."%(rid,s)); subprocess.run(cmd,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,cwd=ROOT,check=True)
        import json as J; v=J.load(open(js))["val_bpb"]
        append("RESULT|%s|%d|%.5f|%s"%(rid,s,v,sha(ck)))
        print("    val_bpb %.4f"%v)
    st,mg,note=judge(rid)
    append("VERDICT|%s|%s|%.5f|%s"%(rid,st,mg,note))
    print("  VERDICT: %s  (%s)"%(st,note))
    if st!="WIN": print("  (this file will not record it as a win; the threshold is its own measured noise)")
def cmd_challenge(a):
    rid=a[0]; R=results().get(rid)
    if not R: print("no results for %s"%rid); sys.exit(1)
    if rows("ENV")[0][1]!=srch():
        print("CANNOT CHALLENGE: source has changed since these claims were measured.")
        print("  recorded %s, current %s -- rebuild the recorded source or re-measure."%(rows("ENV")[0][1],srch())); sys.exit(2)
    P=preds()[rid]; ok=True
    for r in R:
        ck=os.path.join(ROOT,"runs","lab_%s_s%d.bin"%(rid,r["seed"]))
        cmd=[os.path.join(ROOT,"disco"),"train","--arch",P["arch"],"--steps","800","--bs","16","--seq","256",
             "--threads","2","--lr","0.012","--seed",str(r["seed"]),"--ckpt","/tmp/lab_chal.bin","--arena","1200"]
        for kv in P["knobs"].split(",") if P["knobs"] else []: cmd+=["--knob",kv]
        subprocess.run(cmd,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,cwd=ROOT,check=True)
        got=sha("/tmp/lab_chal.bin")
        print("  seed %d expected %s"%(r["seed"],r["sha"][:16]))
        print("           got      %s  %s"%(got[:16],"OK" if got==r["sha"] else "DIVERGED"))
        ok &= (got==r["sha"])
    print("  challenge %s"%("UPHELD" if ok else "FAILED -- this claim no longer reproduces"))
def cmd_materialize(a):
    rid=a[0]; R=results().get(rid)
    if not R: print("no results for %s"%rid); sys.exit(1)
    print("checkpoints for %s:"%rid)
    for r in R:
        ck=os.path.join(ROOT,"runs","lab_%s_s%d.bin"%(rid,r["seed"]))
        st="present, hash ok" if os.path.exists(ck) and sha(ck)==r["sha"] else ("present, HASH MISMATCH" if os.path.exists(ck) else "absent -- run `challenge` to rebuild")
        print("  seed %d  %s  %s"%(r["seed"],r["sha"][:16],st))
C={"status":lambda a:cmd_status(),"frontier":lambda a:cmd_frontier(),"ledger":lambda a:cmd_ledger(),
   "open":lambda a:cmd_open(),"preregister":cmd_prereg,"run":cmd_run,"challenge":cmd_challenge,
   "materialize":cmd_materialize}
a=sys.argv[1:] or ["status"]
(C.get(a[0]) or (lambda x:print(__doc__)))(a[1:])
'''


def sha(p, n=None):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    d = h.hexdigest()
    return d[:n] if n else d


SRC = ["nn.c", "nn.h", "archs.c", "main.c", "mem.c", "mem.h", "model.h", "Makefile"]


def srch(n=16):
    h = hashlib.sha256()
    for s in SRC:
        p = os.path.join(HERE, s)
        if os.path.exists(p):
            h.update(open(p, "rb").read())
    return h.hexdigest()[:n]


def build():
    J = os.path.join(HERE, "journal.jsonl")
    runs = {}
    for line in open(J):
        d = json.loads(line)
        if d.get("kind") in ("run", "remeasure") and "val_bpb" in (d.get("m") or {}):
            runs[d["m"]["id"]] = d
    base = [r["m"]["val_bpb"] for r in runs.values()
            if r["m"]["arch"] == "llama" and r["m"]["id"].startswith("base")]
    mean, sd = statistics.mean(base), statistics.stdev(base)

    L = []
    L.append("ENV|%s|%s|%s" % (sha(os.path.join(HERE, "data/corpus.bin"), 16), srch(),
                               "byte-level TinyStories 19,183,414 tokens; 800 steps bs16 seq256 lr0.012"))
    L.append("NOISE|val_bpb|%.4f|%.4f|%d|2" % (sd, mean, len(base)))

    arc = json.load(open(os.path.join(HERE, "archive/archive.json")))
    for ck, e in sorted(arc.items(), key=lambda kv: kv[1]["val_bpb"]):
        L.append("FRONT|%s|%s|%.4f|%d|%d" % (ck, e["id"], e["val_bpb"], e["params"], e["wbytes_per_tok"]))

    # replay the session's real predictions and their outcomes into the ledger
    for line in open(J):
        d = json.loads(line)
        if d.get("kind") != "run": continue
        p = (d.get("prediction") or {}).get("val_bpb")
        m = d.get("m") or {}
        if not p or "val_bpb" not in m: continue
        kn = ",".join("%s=%s" % (k, v) for k, v in sorted((m.get("knobs") or {}).items()))
        L.append("PRED|%s|%.2f|%.2f|%s|%s|%s" % (d["id"], p[0], p[1], d["arch"], kn, d.get("ts", "")))
        L.append("RESULT|%s|%d|%.5f|%s" % (d["id"], m.get("seed", 1337), m["val_bpb"], "-"))

    for q in ["Does the 90% additivity of orthostack survive at 10x parameters, or is it an artefact of a regime where every mechanism is starved?",
              "val_bpb is improvable by a bigram hash table (ngrammem, +0.075 reproducible). What is this metric actually measuring, and what would a metric look like that a lookup table cannot move?",
              "hashffn beats a dense model of identical parameter count that reads 3x more bytes per token (1.6760 vs 1.7102). Unexplained.",
              "Every architecture in this archive is a published technique. Did the search find anything, or only rank prior art?"]:
        L.append("OPEN|%s" % q)

    out = os.path.join(HERE, "archive", "model.lab")
    with open(out, "w") as f:
        f.write(INTERP)
        f.write('\n_D = r"""\n')
        f.write("# ==== LAB/1 DATA (append-only; nothing here is ever rewritten) ====\n")
        f.write("\n".join(L) + "\n")
        f.write('"""\n')
    os.chmod(out, 0o755)
    print("wrote %s  (%d bytes, %d data lines)" % (out, os.path.getsize(out), len(L)))


if __name__ == "__main__":
    build()
