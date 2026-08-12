#!/usr/bin/env python3
"""stress.py -- dynamic stress battery for the SEED compression format.

A static val_bpb on the tuning split is the weakest possible evidence about a
compressed model.  This runs the tests that a claimed 200x format has to survive
if the number is to mean anything.  Everything here is measured by executing the
artifact; nothing is extrapolated.

  lossless    rebuild from (seed, K values) and diff against the dense
              checkpoint coordinate by coordinate.  Must be exactly 0.
  runtime     load time, peak RSS, tok/s and unique weight-bytes per token, all
              read from a process that opens ONLY the .seed file.  This is where
              a storage-compression claim meets the deployment footprint.
  ood         bpb on five transformed corpora (see mkood.py), against the
              identity control, so the shift is isolated from the segment.
  horizon     2048-token free generation: repetition rate and distinct-4, which
              catch the collapse that a 256-token sample hides.
  faults      random bit flips in the stored payload, at a fixed count and at a
              fixed bit-error rate.  A 200x artifact has 200x fewer bits, and
              each of them carries 200x more of the model.

Usage: stress.py <label> <ckpt-path> <frozen_frac> [seed] [arch] [knob ...]
"""
import json, os, random, shutil, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
DISCO = os.path.join(HERE, "disco")
SCRATCH = os.path.join(HERE, "runs", "seed")
OODS = ["ood_id", "ood_rot13", "ood_case", "ood_shuf", "ood_synth"]
HDR = 192          # SeedHdr bytes; flips are confined to the payload


def run(args, **kw):
    return subprocess.run(args, cwd=HERE, capture_output=True, text=True, **kw)


def artifact_args(sd, ck, frac):
    return ["--loadseed", sd] if frac < 1.0 else ["--ckpt", ck]


def eval_on(common, art, data, valb=24):
    r = run([DISCO, "eval"] + common + art + ["--data", data, "--valb", str(valb),
                                              "--bs", "16", "--seq", "256"])
    for tok in r.stdout.split():
        pass
    try:
        return float(r.stdout.strip().split()[-1])
    except Exception:
        return float("nan")


def main():
    label, ck, frac = sys.argv[1], os.path.abspath(sys.argv[2]), float(sys.argv[3])
    seed = int(sys.argv[4]) if len(sys.argv) > 4 else 1337
    arch = sys.argv[5] if len(sys.argv) > 5 else "llama"
    sd = ck + ".seed"
    common = ["--arch", arch, "--seed", str(seed), "--arena", "1200"]
    for kv in sys.argv[6:]:
        common += ["--knob", kv]
    if frac < 1.0:
        common += ["--frozen", "%.9g" % frac, "--fseed", "7"]
    art = artifact_args(sd, ck, frac)
    out = dict(label=label, arch=arch, frozen_frac=frac, ckpt=ck)

    # ---- 1. losslessness + true ratio ----
    if frac < 1.0:
        r = run([DISCO, "seedcheck"] + common + ["--ckpt", ck, "--loadseed", sd])
        out["lossless"] = json.loads(r.stdout.strip())
    else:
        out["lossless"] = dict(max_abs_diff=0.0, dense_bytes=os.path.getsize(ck),
                               seed_bytes=os.path.getsize(ck), ratio=1.0, lossless=True)

    # ---- 2. runtime footprint, measured from the artifact alone ----
    js = os.path.join(SCRATCH, "%s.stress.infer.json" % label)
    r = run([DISCO, "infer"] + common + art + ["--gen", "256", "--repeats", "3",
                                               "--ckpt", ck, "--json", js])
    out["runtime"] = json.load(open(js))

    # ---- 3. out-of-distribution ----
    ood = {}
    for name in OODS:
        ood[name] = eval_on(common, art, "data/%s.bin" % name)
    out["ood"] = ood
    out["val_bpb"] = eval_on(common, art, "data/corpus.bin")

    # ---- 3b. does compression cost more at long range than at short? ----
    r = run([DISCO, "eval"] + common + art + ["--data", "data/corpus.bin", "--valb", "24",
                                              "--bs", "16", "--seq", "256", "--knob", "bypos=1"])
    for ln in r.stdout.splitlines():
        if ln.startswith("bypos"):
            out["bypos"] = [float(v) for v in ln.split()[1:]]

    # ---- 4. long-horizon generation ----
    jl = os.path.join(SCRATCH, "%s.stress.long.json" % label)
    run([DISCO, "infer"] + common + art + ["--gen", "2048", "--repeats", "1",
                                           "--ckpt", ck, "--json", jl])
    txt = open(ck + ".sample.txt", "rb").read().decode("utf-8", "replace")
    out["horizon"] = degeneracy(txt)

    # ---- 5. fault injection ----
    out["faults"] = faults(label, common, art, frac, ck, sd)

    p = os.path.join(HERE, "archive", "stress_%s.json" % label)
    json.dump(out, open(p, "w"), indent=1)
    print(json.dumps(out, indent=1))


def degeneracy(txt):
    n = len(txt)
    g4 = [txt[i:i + 4] for i in range(max(0, n - 3))]
    d4 = len(set(g4)) / max(1, len(g4))
    # longest immediately-repeating substring, the signature of a decode loop
    best = 0
    for p in range(1, 129):
        run_len = 0
        for i in range(n - p):
            if txt[i] == txt[i + p]:
                run_len += 1
                best = max(best, run_len)
            else:
                run_len = 0
    return dict(chars=n, distinct4=d4,
                distinct_chars=len(set(txt)) if n else 0,
                longest_period_repeat=best)


def faults(label, common, art, frac, ck, sd):
    src = sd if frac < 1.0 else ck
    nbytes = os.path.getsize(src)
    payload_off = HDR if frac < 1.0 else 8
    nbits = (nbytes - payload_off) * 8
    tmp = os.path.join(SCRATCH, "%s.fault" % label)
    res = {"payload_bits": nbits}
    rnd = random.Random(20260812)

    def trial(k):
        data = bytearray(open(src, "rb").read())
        for _ in range(k):
            b = rnd.randrange(payload_off, nbytes)
            data[b] ^= 1 << rnd.randrange(8)
        open(tmp, "wb").write(data)
        a = ["--loadseed", tmp] if frac < 1.0 else ["--ckpt", tmp]
        r = run([DISCO, "eval"] + common + a + ["--data", "data/corpus.bin",
                                                "--valb", "8", "--bs", "16", "--seq", "256"])
        try:
            v = float(r.stdout.strip().split()[-1])
        except Exception:
            v = float("nan")
        return v

    # (a) fixed COUNT: 16 flipped bits, wherever they land
    vs = [trial(16) for _ in range(8)]
    res["flip16"] = summarize(vs)
    # (b) fixed RATE: 1e-4 of the payload bits, i.e. proportional damage
    k = max(1, int(nbits * 1e-4))
    res["rate_1e4_bits"] = k
    vs = [trial(k) for _ in range(8)]
    res["rate1e4"] = summarize(vs)
    if os.path.exists(tmp):
        os.remove(tmp)
    return res


def summarize(vs):
    ok = [v for v in vs if v == v and v < 1e3]
    ok.sort()
    return dict(n=len(vs), n_finite=len(ok),
                median=ok[len(ok) // 2] if ok else None,
                worst=ok[-1] if ok else None,
                best=ok[0] if ok else None)


if __name__ == "__main__":
    main()
