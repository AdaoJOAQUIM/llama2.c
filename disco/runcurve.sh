#!/bin/sh
# Drive the pre-registered compression-ratio curve through model.lab, serially.
# Serial, not parallel: model.lab is an append-only knowledge object and two
# concurrent writers could interleave inside a single splice.  Wall clock is
# worth less than the integrity of the file that forces the verdicts.
# Most decisive ids first, so a truncated run still answers the question asked.
cd "$(dirname "$0")" || exit 1
for id in seed_r200 seed_r100 seed_r1000 seed_r20 seed_r10 seed_r4 seed_r2 \
          seed_r200_lr20 seed_r200_lr5 seed_r200_f9 seed_r100_lr5; do
  case "$id" in
    seed_r200|seed_r100) seeds="1337 2" ;;
    *)                   seeds="1337" ;;
  esac
  echo "=== $id (seeds $seeds) ==="
  ./archive/model.lab run "$id" $seeds 2>&1
done
echo "=== curve complete ==="
