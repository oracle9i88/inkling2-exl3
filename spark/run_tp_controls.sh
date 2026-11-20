#!/usr/bin/env bash
# Stage 1 controls for the KV-group split: how far do *any* two runs differ? (inside the dev container, one GPU)
#   - unsplit vs unsplit again           -> run-to-run noise of the kernels (atomics, split-K)
#   - unsplit vs 2 / 4 ranks (even split, the original layout) and vs 3 ranks (KV-group split)
#   - unsplit vs 3 ranks at depth 1, 3, 6 -> does the gap grow smoothly with depth (noise) or jump (a bug)?
# bf16 KV, fp32 all-reduce. Log: /work/logs/tp-controls.log, results /work/tmp/tpc-L*/compare_*.json
set -u
M=${MODEL:-/models/Inkling-2-Small-EXL3-3.48bpw}
export PYTHONPATH=/work/build/exllamav3
cd /work/engine
say() { echo "$(date -u +%FT%TZ) $*"; }
T=spark/test_tp_equiv.py
run() {  # run <out> <layers> <world> [tag]
  if [ "$3" = 1 ]; then python $T --model "$M" --layers $2 --out "$1" --tag "${4:-}"
  else torchrun --standalone --nproc_per_node $3 $T --model "$M" --layers $2 --out "$1" --tag "${4:-}"; fi
}
cmp() { python $T --compare --out "$1" --ref "$2" --cmp "$3"; }

O=/work/tmp/tpc-L12; rm -rf $O
for w in 1 2 3 4; do say "L12 world $w"; run $O 12 $w || { say "FAILED L12 world $w"; exit 1; }; done
say "L12 world 1 again"; run $O 12 1 b || { say "FAILED L12 world1b"; exit 1; }
for c in world1b world2 world3 world4; do cmp $O world1 $c; done

for L in 1 3 6; do
  O=/work/tmp/tpc-L$L; rm -rf $O
  say "L$L world 1 / 1 again / 3"
  run $O $L 1 && run $O $L 1 b && run $O $L 3 || { say "FAILED L$L"; exit 1; }
  cmp $O world1 world1b; cmp $O world1 world3
done
say "TP-CONTROLS-DONE"
