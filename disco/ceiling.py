#!/usr/bin/env python3
"""ceiling.py -- the reference points a compressed model must beat to mean anything.

A compression ratio is meaningless without knowing what score a model of ZERO
capacity gets.  This computes, on EXACTLY the tokens `eval_val` scores (same
deterministic offsets, same batch layout, same T), the bpb of:

  order 0   the unigram distribution of the TRAIN split
  order 1   bigram, Laplace-smoothed, fitted on TRAIN
  order 2   trigram, Laplace-smoothed, fitted on TRAIN
  uniform   log2(257), a model that knows nothing at all

Any "compressed model" scoring at or above the order-0 line has been compressed
past the point of containing a model.  Order 1 and 2 matter for a different
reason: `ngrammem` -- a bigram table -- already scores near the archive's best,
which is the strongest evidence that this metric can be gamed by lookup.
"""
import json, math, os, sys
from collections import Counter
import array

HERE = os.path.dirname(os.path.abspath(__file__))
M64 = (1 << 64) - 1


def val_offsets(ntok, ntrain, nb, span_t):
    """bit-exact port of val_offsets() in main.c (called with T = T*B+2)."""
    s = 0xC0FFEE
    span = ntok - ntrain - span_t - 2
    out = []
    for _ in range(nb):
        s ^= s >> 12
        s = (s ^ (s << 25)) & M64
        s ^= s >> 27
        out.append(ntrain + (((s * 0x2545F4914F6CDD1D) & M64) >> 32) % span)
    return out


def main():
    B, T, nb = 16, 256, 24
    data = array.array("H")
    p = os.path.join(HERE, "data/corpus.bin")
    with open(p, "rb") as f:
        data.frombytes(f.read())
    ntok = len(data)
    ntrain = int(ntok * 0.95)
    V = 257

    # the scored positions, exactly as fill_batch lays them out
    offs = val_offsets(ntok, ntrain, nb, T * B + 2)
    pairs = []          # (context tuple, target)
    for o in offs:
        for b in range(B):
            base = o + b * T
            for t in range(T):
                pairs.append((base + t, data[base + t + 1]))

    # counts fitted on TRAIN only
    c0 = Counter()
    c1 = Counter()
    c2 = Counter()
    for i in range(ntrain - 1):
        a = data[i]
        c0[data[i + 1]] += 1
        c1[(a, data[i + 1])] += 1
    ctx1 = Counter()
    for (a, _), n in c1.items():
        ctx1[a] += n
    for i in range(ntrain - 2):
        c2[(data[i], data[i + 1], data[i + 2])] += 1
    ctx2 = Counter()
    for (a, b, _), n in c2.items():
        ctx2[(a, b)] += n

    n0 = sum(c0.values())
    L2 = math.log(2.0)
    s0 = s1 = s2 = 0.0
    for idx, y in pairs:
        s0 += -math.log((c0[y] + 1.0) / (n0 + V)) / L2
        a = data[idx]
        s1 += -math.log((c1[(a, y)] + 1.0) / (ctx1[a] + V)) / L2
        aa = data[idx - 1] if idx > 0 else 256
        s2 += -math.log((c2[(aa, a, y)] + 1.0) / (ctx2[(aa, a)] + V)) / L2
    n = len(pairs)

    out = dict(n_scored=n,
               uniform_bpb=math.log(V, 2),
               order0_bpb=s0 / n,
               order1_bpb=s1 / n,
               order2_bpb=s2 / n)
    print(json.dumps(out, indent=1))
    with open(os.path.join(HERE, "archive/ceiling.json"), "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
