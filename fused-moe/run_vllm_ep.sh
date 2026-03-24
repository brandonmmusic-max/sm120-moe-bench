#!/bin/bash
# =============================================================================
# vLLM EP Mode — Expert Parallel for Qwen3.5-397B
# Hybrid: TP=4 for attention, EP=4 for MoE experts
# =============================================================================
set -euo pipefail

MODEL_DIR="/home/brandonmusic/models/lukealonso-qwen35-nvfp4"
CONTAINER_NAME="vllm-ep"
IMAGE="vllm-qwen35-k64:latest"
PORT=9200
PATCH_DIR="/home/brandonmusic/sm120-moe-bench/fused-moe/patches"

# Pre-flight
if [ ! -d "$MODEL_DIR" ]; then
    echo "[FAIL] Model not found at $MODEL_DIR"
    exit 1
fi

# Cap GPUs 1 and 3 to 300W
sudo nvidia-smi -i 1 -pl 300 2>/dev/null || true
sudo nvidia-smi -i 3 -pl 300 2>/dev/null || true

# Stop existing containers
for name in vllm-ep vllm-qwen35 vllm-research; do
    docker rm -f "$name" 2>/dev/null || true
done

echo "============================================================"
echo "vLLM EP Mode — Expert Parallel (4 GPUs × 128 experts each)"
echo "============================================================"
echo "  Model:     lukealonso/Qwen3.5-397B-A17B-NVFP4"
echo "  Image:     $IMAGE"
echo "  Mode:      TP=4 attention + EP=4 MoE"
echo "  Endpoint:  http://localhost:${PORT}/v1"
echo "============================================================"

docker run -d \
  --name "$CONTAINER_NAME" \
  --gpus all --ipc host --shm-size 32g \
  --entrypoint bash \
  -p ${PORT}:8000 \
  -v "${MODEL_DIR}":/model:ro \
  -v "${PATCH_DIR}":/patches:ro \
  -v "/home/brandonmusic/klc-linux/cache/vllm":/root/.cache/vllm \
  -v "/home/brandonmusic/klc-linux/cache/torch_extensions":/root/.cache/torch_extensions \
  -v "/home/brandonmusic/klc-linux/cache/triton":/root/.triton \
  -v "/home/brandonmusic/klc-linux/cache/inductor":/root/.cache/torch/inductor \
  -v "/home/brandonmusic/klc-linux/cache/flashinfer":/cache/jit/flashinfer/.cache/flashinfer \
  -e VLLM_LOG_STATS_INTERVAL=1 \
  -e NCCL_SHM_DISABLE=0 \
  -e SAFETENSORS_FAST_GPU=1 \
  -e OMP_NUM_THREADS=24 \
  -e CUDA_DEVICE_ORDER=PCI_BUS_ID \
  -e CUDA_DEVICE_MAX_CONNECTIONS=32 \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
  -e VLLM_SLEEP_WHEN_IDLE=1 \
  -e VLLM_USE_FLASHINFER_SAMPLER=1 \
  -e VLLM_USE_FUSED_MOE_GROUPED_TOPK=1 \
  -e VLLM_USE_FLASHINFER_MOE_MXFP4_MXFP8=1 \
  "$IMAGE" \
  -c '
# Apply EP patch to CutlassExpertsFp4
echo "Applying EP patch..."
python3 /patches/ep_cutlass_fp4_v2.py

# Start vLLM with Expert Parallel enabled
exec python3 -m vllm.entrypoints.openai.api_server \
  --model /model \
  --served-model-name qwen3.5-397b-nvfp4 \
  --host 0.0.0.0 --port 8000 \
  --trust-remote-code --tensor-parallel-size 4 \
  --enable-expert-parallel \
  --gpu-memory-utilization 0.90 \
  --kv-cache-dtype fp8_e4m3 \
  --max-num-batched-tokens 8192 --max-num-seqs 96 \
  --max-model-len 262144 \
  --enable-prefix-caching --reasoning-parser qwen3 \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder \
  --mm-encoder-tp-mode data --mm-processor-cache-type shm \
  --speculative-config '"'"'{"method":"mtp","num_speculative_tokens":3}'"'"'
'

echo ""
echo "[INFO] Container started. Watching for startup..."
echo "[INFO] Logs: docker logs -f $CONTAINER_NAME"
echo ""

# Wait for health
MAX_WAIT=900
ELAPSED=0
while [ $ELAPSED -lt $MAX_WAIT ]; do
    sleep 10
    ELAPSED=$((ELAPSED + 10))

    if ! docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
        echo "[FAIL] Container exited. Last logs:"
        docker logs --tail 50 "$CONTAINER_NAME" 2>&1
        exit 1
    fi

    RESPONSE=$(curl -s -o /dev/null -w "%{http_code}" "http://localhost:${PORT}/v1/models" 2>/dev/null || true)
    if [ "$RESPONSE" = "200" ]; then
        echo ""
        echo "============================================================"
        echo "SERVER READY — EP MODE"
        echo "============================================================"
        echo "  Endpoint: http://localhost:${PORT}/v1"
        echo "  Mode:     TP=4 attention + EP=4 experts"
        echo "============================================================"
        exit 0
    fi

    echo "  ... waiting (${ELAPSED}s / ${MAX_WAIT}s)"
done

echo "[FAIL] Server did not start within ${MAX_WAIT}s"
docker logs --tail 50 "$CONTAINER_NAME" 2>&1
exit 1
