#!/bin/bash
# TP=4 configuration — recommended for Qwen3.5-397B-NVFP4 on 4x RTX PRO 6000
# Requires: iommu=pt in kernel params for P2P support

MODEL_PATH="/path/to/sehyo-qwen35-nvfp4"  # Change this

docker run -d \
  --name vllm-qwen35 \
  --gpus all --ipc host --shm-size 32g \
  --restart on-failure:3 \
  --entrypoint bash \
  -p 9200:8000 \
  -v "${MODEL_PATH}":/model:ro \
  -e VLLM_LOG_STATS_INTERVAL=1 \
  -e SAFETENSORS_FAST_GPU=1 \
  -e OMP_NUM_THREADS=6 \
  -e CUDA_DEVICE_ORDER=PCI_BUS_ID \
  -e CUDA_DEVICE_MAX_CONNECTIONS=32 \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
  verdictai/vllm-blackwell-k64:latest \
  -c "exec python3 -m vllm.entrypoints.openai.api_server \
  --model /model \
  --served-model-name qwen3.5-397b-nvfp4 \
  --host 0.0.0.0 --port 8000 \
  --trust-remote-code --tensor-parallel-size 4 \
  --gpu-memory-utilization 0.85 \
  --max-num-batched-tokens 8192 --max-num-seqs 128 \
  --max-model-len 262144 \
  --enable-prefix-caching --reasoning-parser qwen3 \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder \
  --speculative-config '{\"method\":\"mtp\",\"num_speculative_tokens\":5}'"
