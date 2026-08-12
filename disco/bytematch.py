#!/usr/bin/env python3
"""bytematch.py -- the byte-matched baseline the SEED format actually has to beat.

Comparing a 4,952-byte artifact against a 990,404-byte model tells you the ratio
and nothing else.  The question that decides whether the compression preserved a
MODEL is: at that same byte budget, what does the dumbest possible thing get?

So for each budget we build the best truncated n-gram model that FITS, and score
it on exactly the tokens `eval_val` scores.

Encoding, stated explicitly because the byte count depends on it:
  unigram   257 symbols x 2 bytes  = 514 B   (16-bit quantised log2-prob)
  bigram    key 2 B + logprob 2 B  = 4 B / entry
  trigram   key 3 B + logprob 2 B  = 5 B / entry
Quantisation is APPLIED, not assumed: probabilities are round-tripped through
the 16-bit grid before scoring, so its cost is inside the reported bpb.
Entries are chosen by descending count, and anything absent backs off to the
next shorter order.  This is a generous baseline -- it is allowed to see the
whole training split to pick its entries.
"""
import array, json, math, os
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
M64 = (1 << 64) - 1
V = 257
QLO, QN = -20.0, 65535     # log2-prob grid


def q(lp):
    """round-trip a log2 probability through the 16-bit grid"""
    if lp < QLO:
        lp = QLO
    i = int((lp - QLO) / (-QLO) * QN + 0.5)
    return QLO + (-QLO) * i / QN


def val_offsets(ntok, ntrain, nb, span_t):
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
    d = array.array("H")
    d.frombytes(open(os.path.join(HERE, "data/corpus.bin"), "rb").read())
    ntok = len(d)
    ntrain = int(ntok * 0.95)

    c0, c1, c2 = Counter(), Counter(), Counter()
    for i in range(ntrain - 1):
        c0[d[i + 1]] += 1
        c1[(d[i], d[i + 1])] += 1
    for i in range(ntrain - 2):
        c2[(d[i], d[i + 1], d[i + 2])] += 1
    n0 = sum(c0.values())
    ctx1 = Counter()
    for (a, _), n in c1.items():
        ctx1[a] += n
    ctx2 = Counter()
    for (a, b, _), n in c2.items():
        ctx2[(a, b)] += n

    uni = {y: q(math.log2((c0[y] + 1.0) / (n0 + V))) for y in range(V)}
    bi_sorted = sorted(c1.items(), key=lambda kv: -kv[1])
    tri_sorted = sorted(c2.items(), key=lambda kv: -kv[1])

    offs = val_offsets(ntok, ntrain, nb, T * B + 2)
    pos = []
    for o in offs:
        for b in range(B):
            base = o + b * T
            for t in range(T):
                pos.append(base + t)

    def score(nbi, ntri):
        bi = {}
        for (a, y), n in bi_sorted[:nbi]:
            bi[(a, y)] = q(math.log2((n + 1.0) / (ctx1[a] + V)))
        tri = {}
        for (a, b, y), n in tri_sorted[:ntri]:
            tri[(a, b, y)] = q(math.log2((n + 1.0) / (ctx2[(a, b)] + V)))
        s = 0.0
        for i in pos:
            y = d[i + 1]
            a = d[i]
            aa = d[i - 1] if i > 0 else 256
            lp = tri.get((aa, a, y))
            if lp is None:
                lp = bi.get((a, y))
            if lp is None:
                lp = uni[y]
            s -= lp
        return s / len(pos)

    budgets = [4952, 9904, 49520, 99040, 247601, 495202, 990404, 988]
    out = {}
    for Bb in sorted(budgets):
        rest = Bb - 514
        if rest < 0:
            out[str(Bb)] = dict(bpb=None, note="does not even fit the unigram table")
            continue
        best = None
        # (a) spend everything on bigram entries
        cand = [("bigram", min(len(bi_sorted), rest // 4), 0)]
        # (b) full bigram, then trigram entries
        if rest >= 4 * len(bi_sorted):
            cand.append(("bi+tri", len(bi_sorted),
                         min(len(tri_sorted), (rest - 4 * len(bi_sorted)) // 5)))
        # (c) trigram only
        cand.append(("trigram", 0, min(len(tri_sorted), rest // 5)))
        for name, nbi, ntri in cand:
            v = score(nbi, ntri)
            used = 514 + 4 * nbi + 5 * ntri
            if best is None or v < best["bpb"]:
                best = dict(bpb=v, mix=name, n_bigram=nbi, n_trigram=ntri, bytes_used=used)
        out[str(Bb)] = best
        print("%8d B  ->  %.4f bpb   (%s: %d bigram, %d trigram entries, %d B used)"
              % (Bb, best["bpb"], best["mix"], best["n_bigram"], best["n_trigram"],
                 best["bytes_used"]))
    json.dump(out, open(os.path.join(HERE, "archive/bytematch.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
