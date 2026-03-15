#!/bin/bash
# Tuned TP=4 configuration — Qwen3.5-397B-NVFP4 on 4x RTX PRO 6000 Blackwell
# Changes from original: gpu_mem 0.85→0.90, batched_tokens 8192→16384, MTP=3, P2P enabled, FlashInfer MoE env
# Requires: iommu=pt in kernel params for P2P support

MODEL_PATH="/path/to/sehyo-qwen35-nvfp4"  # Change this
CACHE_DIR="/path/to/cache"                 # Change this — persistent cache avoids 20min JIT on cold start

docker run -d \
  --name vllm-qwen35 \
  --gpus all --ipc host --shm-size 32g \
  --restart on-failure:3 \
  --entrypoint bash \
  -p 9200:8000 \
  -v "${MODEL_PATH}":/model:ro \
  -v "${CACHE_DIR}/vllm":/root/.cache/vllm \
  -v "${CACHE_DIR}/torch_extensions":/root/.cache/torch_extensions \
  -v "${CACHE_DIR}/triton":/root/.triton \
  -v "${CACHE_DIR}/inductor":/root/.cache/torch/inductor \
  -e VLLM_LOG_STATS_INTERVAL=1 \
  -e SAFETENSORS_FAST_GPU=1 \
  -e OMP_NUM_THREADS=6 \
  -e CUDA_DEVICE_ORDER=PCI_BUS_ID \
  -e CUDA_DEVICE_MAX_CONNECTIONS=32 \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
  -e VLLM_SLEEP_WHEN_IDLE=1 \
  -e VLLM_USE_FLASHINFER_MOE_MXFP4_MXFP8=1 \
  verdictai/vllm-blackwell-k64:latest \
  -c "exec python3 -m vllm.entrypoints.openai.api_server \
  --model /model \
  --served-model-name qwen3.5-397b-nvfp4 \
  --host 0.0.0.0 --port 8000 \
  --trust-remote-code --tensor-parallel-size 4 \
  --gpu-memory-utilization 0.90 \
  --max-num-batched-tokens 16384 --max-num-seqs 128 \
  --max-model-len 262144 \
  --enable-prefix-caching --reasoning-parser qwen3 \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder \
  --mm-encoder-tp-mode data --mm-processor-cache-type shm \
  --speculative-config '{\"method\":\"mtp\",\"num_speculative_tokens\":3}'"
