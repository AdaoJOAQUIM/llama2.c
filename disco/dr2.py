#!/usr/bin/env python3
"""dr2.py -- build an executable DR2 book: a derivation DAG for checkpoints.

DR1 was a flat list: every artifact rebuilt from scratch, no relationships.
DR2 records how artifacts DERIVE from one another, which makes the book a
schedulable graph rather than a catalogue. Three node kinds:

  T  train from scratch
  C  continue training from a parent checkpoint (needs exact resume; verified)
  R  re-score a parent with different decode-time knobs (no training at all)

The emitted book is itself executable: it carries a small interpreter, so
`./model.dr2 plan` / `make <id>` / `verify <id>` work with no other tooling.

What makes it more than a shell script:
  - PLANS before acting: topologically orders the DAG, reports CPU cost, and
    reports the cost avoided by shared prefixes.
  - MEMOISES: a node whose artifact already exists with the right SHA-256 is
    skipped, so re-running a plan converges instead of redoing work.
  - SELF-VERIFIES: every materialised node is hashed against its declared
    digest; a mismatch is a hard failure, never a warning.
  - REFUSES stale replays: corpus and source digests are checked first, so a
    book cannot silently reproduce different weights against a changed binary.

The honesty constraint that shapes all of this: a derivation graph is only
sound if resume is bit-exact. That was measured before this file was written
(JOURNAL.md, wave 13) -- train(300) and train(150)+resume(150) produce an
identical SHA-256. Without that, sharing prefixes would produce artifacts that
do not match their own declared hashes.
"""
import hashlib, json, os, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))

INTERP = r'''#!/usr/bin/env python3
"""DR2 executable checkpoint book. Commands: plan | make <id> | make-all | verify <id> | ls | why <id>"""
import hashlib,os,subprocess,sys,time
HERE=os.path.dirname(os.path.abspath(__file__)); ROOT=os.path.dirname(HERE)
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
def nodes():
    out={}
    for ln in open(os.path.abspath(__file__)):
        if not ln.startswith("DR2|"): continue
        p=ln.rstrip("\n").split("|")
        out[p[2]]=dict(kind=p[1],id=p[2],parent=p[3],arch=p[4],cfg=p[5],knobs=p[6],
                       sched=p[7],opt=p[8],corpus16=p[9],src16=p[10],sha=p[11],cost=float(p[12]))
    return out
def ckpt(i): return os.path.join(ROOT,"runs",i+".bin")
def order(N,tgt):
    seen,seq=set(),[]
    def go(i):
        if i in seen: return
        seen.add(i)
        p=N[i]["parent"]
        if p: go(p)
        seq.append(i)
    for t in tgt: go(t)
    return seq
def fresh(N,i):
    c=ckpt(i)
    return os.path.exists(c) and sha(c)==N[i]["sha"]
def build(N,i,verbose=True):
    n=N[i]
    if fresh(N,i):
        if verbose: print("  [cached] %s"%i); return 0.0
    dim,L,H,KV,hid,V,tie=n["cfg"].split(",")
    steps,bs,seq,thr,stopat=n["sched"].split(",")
    lr,wd,warm,seed=n["opt"].split(",")
    cmd=[os.path.join(ROOT,"disco"),"train","--arch",n["arch"],"--dim",dim,"--layers",L,
         "--heads",H,"--kvheads",KV,"--hidden",hid,"--tie",tie,"--steps",steps,"--bs",bs,
         "--seq",seq,"--threads",thr,"--lr",lr,"--wd",wd,"--seed",seed,
         "--ckpt",ckpt(i),"--arena","1200","--savestate","1"]
    if stopat!="0": cmd+=["--stopat",stopat]
    if n["kind"]=="C":
        pc=ckpt(n["parent"])
        if not os.path.exists(pc+".state"):
            print("  !! parent %s has no optimiser state; cannot continue"%n["parent"]); sys.exit(1)
        # resume mutates in place from a copy of the parent
        import shutil; shutil.copyfile(pc,ckpt(i)); shutil.copyfile(pc+".state",ckpt(i)+".state")
        cmd+=["--resume",ckpt(i)]
    for kv in n["knobs"].split(",") if n["knobs"] else []:
        cmd+=["--knob",kv.replace(":","=")]
    t0=time.time()
    subprocess.run(cmd,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,cwd=ROOT,check=True)
    el=time.time()-t0
    got=sha(ckpt(i))
    if got!=n["sha"]:
        print("  !! %s HASH MISMATCH\n     expected %s\n     got      %s"%(i,n["sha"],got)); sys.exit(2)
    if verbose: print("  [built ] %s  %.1fs  sha ok"%(i,el))
    return el
def guard(N):
    cp=sha(os.path.join(ROOT,"data/corpus.bin"),16); sh=srch()
    for n in N.values():
        if n["corpus16"]!=cp: print("REFUSING: corpus digest differs (%s vs %s)"%(cp,n["corpus16"])); sys.exit(3)
        if n["src16"]!=sh:    print("REFUSING: source digest differs (%s vs %s)"%(sh,n["src16"]));   sys.exit(3)
def main():
    N=nodes(); cmd=sys.argv[1] if len(sys.argv)>1 else "plan"
    if cmd=="ls":
        for i,n in N.items(): print("%-24s %s parent=%-20s %.0fs %s"%(i,n["kind"],n["parent"] or "-",n["cost"],"cached" if fresh(N,i) else ""))
    elif cmd=="why":
        i=sys.argv[2]; ch=[]
        while i: ch.append(i); i=N[i]["parent"]
        print(" <- ".join(ch))
    elif cmd=="verify":
        guard(N); i=sys.argv[2]
        if os.path.exists(ckpt(i)): os.remove(ckpt(i))
        for j in order(N,[i]): build(N,j)
        print("verified %s"%i)
    elif cmd in ("plan","make","make-all"):
        tgt=[sys.argv[2]] if cmd=="make" else list(N)
        seq=order(N,tgt)
        todo=[i for i in seq if not fresh(N,i)]
        scratch=sum(N[i]["cost"] for i in seq if N[i]["kind"]!="R")
        dag=sum(N[i]["cost"] for i in todo)
        print("plan: %d nodes, %d to build, %d cached"%(len(seq),len(todo),len(seq)-len(todo)))
        for i in seq: print("   %s %-24s %s"%(N[i]["kind"],i,"cached" if fresh(N,i) else "BUILD %.0fs"%N[i]["cost"]))
        print("cost if each node were trained from scratch : %6.0f s"%scratch)
        print("cost following the derivation graph         : %6.0f s"%dag)
        if scratch>0: print("shared-work saving                          : %6.1f%%"%(100*(1-dag/max(scratch,1e-9))))
        if cmd!="plan":
            guard(N); tot=0.0
            for i in seq: tot+=build(N,i)
            print("materialised in %.1fs"%tot)
    else: print(__doc__)
main()
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


def node(kind, nid, parent, arch, cfg, knobs, sched, opt, cost):
    return "|".join(["DR2", kind, nid, parent, arch, cfg, knobs, sched, opt,
                     sha(os.path.join(HERE, "data/corpus.bin"), 16), srch(),
                     sha(os.path.join(HERE, "runs", nid + ".bin")), "%.1f" % cost])


def write_book(path, lines):
    with open(path, "w") as f:
        f.write(INTERP)
        f.write("\n# ---- DR2 nodes ----\n")
        f.write("\n".join(lines) + "\n")
    os.chmod(path, 0o755)


if __name__ == "__main__":
    print(__doc__)
