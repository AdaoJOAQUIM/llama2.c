#!/usr/bin/env python3
"""mkood.py -- build the out-of-distribution stress corpora.

The point of a DYNAMIC stress test is that a compressed model can score fine on
the held-out split it was tuned against and still be hollow.  Each file below is
the SAME source segment under a different transformation, so the only thing that
changes between them is the transformation -- and `ood_id`, the untransformed
control, absorbs any effect of using a segment rather than the usual val split.

  ood_id     identity.  control.
  ood_rot13  letters rotated 13.  Every structural regularity (word lengths,
             spacing, punctuation, syntax) survives; only the symbol identities
             move.  Separates "knows English structure" from "memorised strings".
  ood_case   case flipped.  A mild, realistic domain shift.
  ood_shuf   words shuffled inside 64-word blocks.  Local spelling intact,
             long-range order destroyed.  Isolates what the mixer contributes.
  ood_synth  bytes drawn i.i.d. from the segment's own unigram distribution.
             A calibrated model should land near order0_bpb here; scoring far
             worse means the model is confidently wrong off-distribution.
"""
import array, os, random

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "data")
SEG = 900_000          # tokens per stress corpus


def load():
    a = array.array("H")
    with open(os.path.join(HERE, "data/corpus.bin"), "rb") as f:
        a.frombytes(f.read())
    return a


def save(name, toks):
    a = array.array("H", toks)
    with open(os.path.join(OUT, name), "wb") as f:
        a.tofile(f)
    print("%-20s %8d tokens  %8d bytes" % (name, len(toks), 2 * len(toks)))


def main():
    d = load()
    ntok = len(d)
    ntrain = int(ntok * 0.95)
    # a segment fully inside the held-out region, disjoint from nothing in
    # particular -- every variant uses these exact source tokens.
    seg = list(d[ntrain: ntrain + SEG])
    save("ood_id.bin", seg)

    def rot13(t):
        if 65 <= t <= 90:
            return (t - 65 + 13) % 26 + 65
        if 97 <= t <= 122:
            return (t - 97 + 13) % 26 + 97
        return t
    save("ood_rot13.bin", [rot13(t) for t in seg])

    def flip(t):
        if 65 <= t <= 90:
            return t + 32
        if 97 <= t <= 122:
            return t - 32
        return t
    save("ood_case.bin", [flip(t) for t in seg])

    # word-level shuffle inside 64-word blocks
    rnd = random.Random(4242)
    words, cur = [], []
    for t in seg:
        cur.append(t)
        if t == 32:
            words.append(cur)
            cur = []
    if cur:
        words.append(cur)
    out = []
    for i in range(0, len(words), 64):
        blk = words[i:i + 64]
        rnd.shuffle(blk)
        for w in blk:
            out.extend(w)
    save("ood_shuf.bin", out[:SEG])

    # i.i.d. resample from the segment's own unigram distribution
    from collections import Counter
    c = Counter(seg)
    syms = list(c)
    wts = [c[s] for s in syms]
    rnd2 = random.Random(99)
    save("ood_synth.bin", rnd2.choices(syms, weights=wts, k=SEG))


if __name__ == "__main__":
    main()
