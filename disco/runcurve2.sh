#!/bin/sh
# Second batch: the readout-limit hypothesis.
#
# The first batch showed 200x and 100x collapsing far past the pre-registered
# intervals.  The structural explanation on offer is that the binding constraint
# is not the trunk but the READOUT: with a tied embedding, 257 output symbols
# need 257x64 = 16,448 numbers before the logits stop being a random projection,
# and 16,448 fp32 values plus the header is 65,984 bytes -- a hard ceiling of
# 15.01x on this architecture, which no allocation strategy can move.
#
# Two runs at exactly that ratio decide it: the same budget spent at random
# (fmode=0) versus spent entirely on the largest tensor, i.e. the readout
# (fmode=2).  A large gap in favour of fmode=2 confirms the explanation; no gap
# refutes it.  A third run puts the same allocation at 200x, where the budget is
# far too small to fill the readout, and should therefore NOT rescue it.
#
# Requires a rebuild for fmode=2, so this must not start before batch one ends.
cd "$(dirname "$0")" || exit 1
# The two extra mask draws test a separate question raised by seed_r200_f9:
# changing WHICH coordinates are trainable moved the score by 1.19 bpb, fourteen
# times the spread from changing the training seed at a fixed mask.  If the draw
# matters that much, the "indices are free" trick is exactly what caps the
# format -- and an encode-time search over fseed is a legitimate exploit, since
# the winning fseed costs 4 bytes.  Four draws are enough to say how wide the
# distribution is and whether its tail could plausibly reach the n-gram bar.
for id in seed_r15_emb seed_r15_rand seed_r200_emb seed_r200_struct seed_r1000_struct \
          seed_r200_f11 seed_r200_f13; do
  echo "=== $id ==="
  ./archive/model.lab run "$id" 1337 2>&1
done
echo "=== batch two complete ==="
