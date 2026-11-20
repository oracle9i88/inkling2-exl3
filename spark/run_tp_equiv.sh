#!/usr/bin/env bash
# Stage 1 on one Spark (inside the dev container): verify the copied weights, then the KV-group split equivalence test
# (unsplit vs 3 ranks on one GPU, first $LAYERS layers). Log: /work/logs/tp-equiv.log, results /work/tmp/tp*/compare.json
set -u
M=${MODEL:-/models/Inkling-2-Small-EXL3-3.48bpw}
LAYERS=${LAYERS:-12}
export PYTHONPATH=/work/build/exllamav3
cd /work/engine
say() { echo "$(date -u +%FT%TZ) $*"; }

if [ "${SKIP_VERIFY:-0}" != 1 ]; then
  say "verify $M against its manifest"
  python /work/tools/verify_manifest.py "$M" "$M.expected.json" 4 || { say "VERIFY FAILED"; exit 1; }
fi

for kv in "16 16" "4 8"; do
  set -- $kv
  OUT=/work/tmp/tp-L$LAYERS-kv$1-$2
  rm -rf "$OUT"
  say "unsplit run, $LAYERS layers, KV bits global $1 local $2"
  python spark/test_tp_equiv.py --model "$M" --layers $LAYERS --bits-g $1 --bits-l $2 --out "$OUT" || { say "FAILED (world 1)"; exit 1; }
  say "3 ranks on one GPU (gloo)"
  torchrun --standalone --nproc_per_node 3 spark/test_tp_equiv.py --model "$M" --layers $LAYERS --bits-g $1 --bits-l $2 --out "$OUT" \
    || { say "FAILED (world 3)"; exit 1; }
  python spark/test_tp_equiv.py --compare --out "$OUT" || { say "FAILED (compare)"; exit 1; }
done
say "TP-EQUIV-DONE"
