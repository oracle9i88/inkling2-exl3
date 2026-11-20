#!/usr/bin/env bash
# Start / stop the Inkling-2 server on DGX Spark nodes (one GB10 each), from a controller that can ssh to all of them.
# Rank 0 serves the OpenAI-compatible API (host network, port $PORT) and the vision tower; the other ranks follow its
# command stream. Tensor-parallel all-reduces go over the nodes' 200G RoCE links, the NCCL/torch bootstrap over the LAN.
# This script runs NODES3 (three nodes); spark/launch_2spark.sh runs the two-Spark build on NODES2 through it.
#
#   spark/launch_3spark.sh start [server args...]    e.g. --slots 4 --max-model-len 131072
#   spark/launch_3spark.sh stop | status | logs [rank]
#   SCRIPT=spark/bench_collectives.py NAME=i2bench MASTER_PORT=29630 spark/launch_3spark.sh start   (any torchrun script)
#
# Cluster settings come from spark/cluster.env (not in the repository; start from spark/cluster.env.example) or the
# environment. NODES: "ssh-target:home:lan-ip" per rank, rank 0 first ("local" = this machine). Each home holds
# inkling2-spark/ (engine/ = this repository, build/exllamav3 = the patched exllamav3 with its extension built, wheels/)
# and models/$MODEL_NAME. Every container also sees the node's models/ read-only at /models.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
[ -f "$HERE/cluster.env" ] && . "$HERE/cluster.env"
NODES=(${NODES:-${NODES3:-}})
[ ${#NODES[@]} -ge 2 ] || { echo "no nodes: set NODES3 / NODES2 in spark/cluster.env (see spark/cluster.env.example)" >&2; exit 1; }
IMAGE=${IMAGE:-nvcr.io/nvidia/pytorch:26.07-py3}  # the image the extension was built in
BOOTSTRAP_IF=${BOOTSTRAP_IF:?set BOOTSTRAP_IF in spark/cluster.env: the LAN interface for the NCCL/gloo bootstrap}
NCCL_DIR=${NCCL_DIR:-}                            # optional NCCL build to use instead of the image's, mounted at
                                                  # /opt/nccl-patched (three nodes wired as a triangle need it);
                                                  # a leading ~/ is each node's own home
MODEL_NAME=${MODEL_NAME:-${MODEL3:-Inkling-2-Small-EXL3-3.48bpw}}
PORT=${PORT:-18100}
MASTER_PORT=${MASTER_PORT:-29620}
NAME=${NAME:-i2spark}
RESERVE_GB=${RESERVE_GB:-10}          # unified memory: keep room for the OS (GLM's TensorFold reserved 10.5 GiB)
# glibc: blocks of 256 KiB and more are mapped on their own and returned when freed (with the server's malloc_trim,
# this keeps rank 0's memory flat under long agent prompts; without them it grew ~100 MB per request until OOM).
# Defaults: one-exchange all-reduce for decode-sized calls (I2_P2P_AR=1). Faster, slightly lossy: EXTRA_ENV="-e I2_ATTN_FP8=1"

field() { echo "$1" | cut -d: -f"$2"; }
on() {  # on <node> <command>   (DRY=1 prints instead of running)
  local t; t=$(field "$1" 1)
  if [ "${DRY:-0}" = 1 ]; then printf '[%s] %s\n\n' "$t" "$2"; return; fi
  if [ "$t" = local ]; then bash -c "$2"; else ssh -n -o BatchMode=yes "$t" "$2"; fi
}
MASTER=$(field "${NODES[0]}" 3)
N=${#NODES[@]}

NCCL_ENV="-e NCCL_SOCKET_IFNAME=$BOOTSTRAP_IF -e GLOO_SOCKET_IFNAME=$BOOTSTRAP_IF \
 -e NCCL_IB_HCA=rocep1s0f0,roceP2p1s0f0,rocep1s0f1,roceP2p1s0f1 -e NCCL_IB_SUBNET_PREFIX_LEN=30 \
 -e NCCL_IB_SUBNET_AWARE_ROUTING=1 -e NCCL_CROSS_NIC=1 -e NCCL_IB_MERGE_NICS=0 -e NCCL_NET_PLUGIN=none \
 -e NCCL_CUMEM_ENABLE=0 -e NCCL_MIN_NCHANNELS=4 -e NCCL_MAX_NCHANNELS=4 -e NCCL_P2P_DISABLE=1 -e NCCL_SHM_DISABLE=1 \
 -e TORCH_NCCL_USE_COMM_NONBLOCKING=0 -e NCCL_DEBUG=${NCCL_DEBUG:-WARN} -e I2_P2P_AR=${I2_P2P_AR:-1} \
 -e MALLOC_MMAP_THRESHOLD_=262144 -e MALLOC_TRIM_THRESHOLD_=33554432 ${EXTRA_ENV:-}"     # EXTRA_ENV="-e I2_PROFILE_AT=60 ..."
FLAGS="--gpus all --network host --ipc=host --shm-size 16g --device /dev/infiniband --cap-add IPC_LOCK \
 --ulimit memlock=-1 --ulimit stack=67108864"

nccl_dir() { local d=${NCCL_DIR/#\~/$(field "$1" 2)}; echo "$d"; }   # NCCL_DIR on <node>, ~/ = its home

start() {
  local args="$*" r node home cmd nccl mount
  for node in "${NODES[@]}"; do              # a missing library would be skipped by LD_PRELOAD without an error
    nccl=$(nccl_dir "$node")
    [ -z "$nccl" ] || on "$node" "test -f $nccl/libnccl.so.2" || { echo "no $nccl/libnccl.so.2 on $(field "$node" 1)" >&2; exit 1; }
  done
  for r in $(seq $((N - 1)) -1 0); do       # followers first; rank 0 (the store) last is fine: they wait for it
    node=${NODES[$r]}; home=$(field "$node" 2); nccl=$(nccl_dir "$node")
    mount=${nccl:+-v $nccl:/opt/nccl-patched:ro -e LD_PRELOAD=/opt/nccl-patched/libnccl.so.2}
    cmd="pip install -q --no-index --find-links /work/wheels transformers==5.23.0 fastapi==0.163.0 uvicorn==0.54.0 marisa-trie==1.4.1 2>/dev/null; \
cd /work/engine && PYTHONPATH=/work/build/exllamav3 exec torchrun --nnodes $N --nproc_per_node 1 --node_rank $r \
--master_addr $MASTER --master_port $MASTER_PORT ${SCRIPT:-server/i2_server.py} --model /model --port $PORT --reserve-gb $RESERVE_GB $args"
    on "$node" "docker rm -f $NAME-r$r >/dev/null 2>&1; docker run -d --name $NAME-r$r $FLAGS $NCCL_ENV \
 -v $home/inkling2-spark:/work -v $home/models/$MODEL_NAME:/model:ro -v $home/models:/models:ro $mount \
 --entrypoint bash $IMAGE -c '$cmd' >/dev/null && echo 'rank $r started on '\$(hostname)"
  done
}

case "${1:-status}" in
  start) shift; start "$@" ;;
  stop)  for r in $(seq 0 $((N - 1))); do on "${NODES[$r]}" "docker rm -f $NAME-r$r >/dev/null 2>&1; echo 'rank $r stopped on '\$(hostname)"; done ;;
  status) for r in $(seq 0 $((N - 1))); do on "${NODES[$r]}" "echo \$(hostname): \$(docker ps --filter name=$NAME-r$r --format '{{.Status}}') \$(grep MemAvailable /proc/meminfo | awk '{printf \"%.1f GiB avail\", \$2/1048576}')"; done
          curl -s -m 5 "http://$MASTER:$PORT/health" || echo "rank 0 API not answering" ;;
  logs)  r=${2:-0}; on "${NODES[$r]}" "docker logs --tail ${TAIL:-40} $NAME-r$r 2>&1" ;;
  *) echo "usage: $0 start [server args] | stop | status | logs [rank]"; exit 1 ;;
esac
