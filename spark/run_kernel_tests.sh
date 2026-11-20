#!/usr/bin/env bash
# exllamav3's own self-contained kernel/module tests (synthetic weights, no checkpoints) on this GPU.
# Usage (in the dev container): bash /work/engine/spark/run_kernel_tests.sh > /work/logs/kernel-tests.log 2>&1
cd /work/build/exllamav3
export PYTHONPATH=/work/build/exllamav3
date -u +%FT%TZ
python -m pytest --version
for d in tests/kernels/gemm tests/kernels/quant tests/kernels/hadamard tests/kernels/activation tests/kernels/moe \
         tests/kernels/norm tests/kernels/routing tests/kernels/attention tests/kernels/tp tests/modules/linear tests/modules/moe; do
  echo "=== $d"
  timeout 3600 python -m pytest "$d" -q -m "not model" -p no:cacheprovider 2>&1 | tail -n 25
done
date -u +%FT%TZ
echo KERNEL-TESTS-DONE
