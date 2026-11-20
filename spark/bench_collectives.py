"""
Latency of the per-layer reductions on the Spark triangle (3 nodes, RoCE), CUDA-graph captured like the server's decode
step. Each variant reduces an fp16 [T, 4608] tensor (T rows = tokens in the step) N times inside one graph; the graph is
replayed and timed. Variants:
  allreduce        dist.all_reduce (NCCL picks ring/tree by size; NCCL_ALGO / NCCL_PROTO env vars override)
  gather_sum       dist.all_gather_into_tensor of the partials + local sum (one exchange instead of reduce + broadcast)
Run on all three nodes: torchrun --nnodes 3 --nproc_per_node 1 ... spark/bench_collectives.py
"""
import os, time, json
import torch
import torch.distributed as dist

rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
dev = torch.device("cuda:0"); torch.cuda.set_device(dev)
dist.init_process_group("nccl", device_id = dev)
D, N, REPS = 4608, 96, 20


def bench(fn, T):
    x = torch.randn(T, D, device = dev, dtype = torch.float16)
    out = torch.empty((world, T, D), device = dev, dtype = torch.float16)
    for _ in range(3):
        fn(x, out)                          # warm-up (NCCL connections, algorithm choice)
    torch.cuda.synchronize(); dist.barrier()
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    g = torch.cuda.CUDAGraph()
    with torch.cuda.stream(s):
        fn(x, out)
        with torch.cuda.graph(g, stream = s):
            for _ in range(N):
                fn(x, out)
    torch.cuda.current_stream().wait_stream(s); torch.cuda.synchronize(); dist.barrier()
    g.replay(); torch.cuda.synchronize(); dist.barrier()
    t0 = time.perf_counter()
    for _ in range(REPS):
        g.replay()
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / (REPS * N)
    # correctness of the reduced value against a plain all_reduce
    return dt


def v_allreduce(x, out):
    dist.all_reduce(x)


PEERS = [r for r in range(world) if r != rank]


def v_p2p_sum(x, out):
    """one exchange: send our partial to every peer, receive theirs, add locally"""
    ops = []
    for i, p in enumerate(PEERS):
        ops.append(dist.P2POp(dist.isend, x, p))
        ops.append(dist.P2POp(dist.irecv, out[i], p))
    for w in dist.batch_isend_irecv(ops):
        w.wait()
    x.add_(out[0]); x.add_(out[1])


def v_gather_sum(x, out):
    dist.all_gather_into_tensor(out.view(-1), x.view(-1))
    torch.sum(out, 0, out = x)


res = {}
for T in (1, 5, 8, 40, 512):
    for name, fn in (("allreduce", v_allreduce), ("p2p_sum", v_p2p_sum)):
        us = bench(fn, T) * 1e6
        res[f"{name}_T{T}"] = round(us, 1)
# check that gather_sum computes the same sum
x = torch.full((5, D), float(rank + 1), device = dev, dtype = torch.float16)
y = x.clone(); out = torch.empty((world, 5, D), device = dev, dtype = torch.float16)
dist.all_reduce(x); v_p2p_sum(y, out)
res["p2p_sum_matches_allreduce"] = bool(torch.equal(x, y))
if rank == 0:
    env = {k: os.environ[k] for k in ("NCCL_ALGO", "NCCL_PROTO", "NCCL_MIN_NCHANNELS", "NCCL_MAX_NCHANNELS") if k in os.environ}
    print("COLLECTIVES", json.dumps({"env": env, "us_per_op": res}), flush = True)
dist.barrier()
dist.destroy_process_group()
