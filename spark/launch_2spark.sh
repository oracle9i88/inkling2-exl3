#!/usr/bin/env bash
# The two-Spark build on NODES2 (spark/cluster.env): rank 0 serves the API on port 18100 and the vision tower, as with
# three. It goes through launch_3spark.sh with the same container names, so stop a running three-node server first:
#   spark/launch_3spark.sh stop
#   spark/launch_2spark.sh start --slots 4 --max-model-len 262144 --prefix-snapshots 24 --prefix-every 16384 ...
#   SCRIPT=spark/eval/heldout_check.py spark/launch_2spark.sh start --kv 16,8     (held-out check instead of the server)
HERE=$(cd "$(dirname "$0")" && pwd)
[ -f "$HERE/cluster.env" ] && . "$HERE/cluster.env"
export NODES=${NODES:-${NODES2:-}}
export MODEL_NAME=${MODEL_NAME:-${MODEL2:-Inkling-2-Small-EXL3-2.55bpw-2Spark}}
# Two nodes hold ~100 GiB of weights each and leave ~15 GiB free after loading: a 10 GiB reserve would leave the KV
# cache 16 pages. 6 GiB keeps ~4 GiB free once the decode graphs are captured (peak extra use ~0.6 GiB). Fewer prefix
# snapshots (~84 MiB each here) for the same reason: start with --prefix-snapshots 24 --prefix-every 16384.
export RESERVE_GB=${RESERVE_GB:-6}
exec "$HERE/launch_3spark.sh" "$@"
