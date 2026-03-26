#!/bin/bash
# Sprint 13: Sweep NCCL configurations for AllReduce tail latency
set -e

SCRIPT="benchmark_allreduce_p99_v2.py"
ITERS=10000
PORT_BASE=29510

run_bench() {
    local label="$1"
    shift
    local port=$((PORT_BASE++))
    echo ""
    echo "================================================================"
    echo "CONFIG: $label"
    echo "================================================================"
    env "$@" python3 -m torch.distributed.launch --nproc_per_node=4 --master_port=$port --use_env \
        $SCRIPT --iters $ITERS --sizes "8192,16384" --label "$label" 2>&1 | \
        grep -E "p50=|Spikes|p99/p50|Top 20|Size:|NCCL env|^="
}

# 1. Test NCCL algorithms
run_bench "ALGO=Ring" NCCL_ALGO=Ring
run_bench "ALGO=Tree" NCCL_ALGO=Tree

# 2. Test NCCL protocols
run_bench "PROTO=Simple" NCCL_PROTO=Simple
run_bench "PROTO=LL" NCCL_PROTO=LL
run_bench "PROTO=LL128" NCCL_PROTO=LL128

# 3. Test buffer/thread tuning
run_bench "BUFFSIZE=4MB" NCCL_BUFFSIZE=4194304
run_bench "NTHREADS=64" NCCL_NTHREADS=64
run_bench "NTHREADS=256" NCCL_NTHREADS=256

# 4. Test channel count
run_bench "CHANNELS=1" NCCL_MIN_NCHANNELS=1 NCCL_MAX_NCHANNELS=1
run_bench "CHANNELS=4" NCCL_MIN_NCHANNELS=4 NCCL_MAX_NCHANNELS=4
run_bench "CHANNELS=8" NCCL_MIN_NCHANNELS=8 NCCL_MAX_NCHANNELS=8

# 5. Test P2P/SHM paths
run_bench "P2P_DISABLE" NCCL_P2P_DISABLE=1
run_bench "SHM_DISABLE" NCCL_SHM_DISABLE=1

# 6. Combined best guess: LL protocol + fewer channels
run_bench "LL+CH1" NCCL_PROTO=LL NCCL_MIN_NCHANNELS=1 NCCL_MAX_NCHANNELS=1

echo ""
echo "================================================================"
echo "SWEEP COMPLETE"
echo "================================================================"
