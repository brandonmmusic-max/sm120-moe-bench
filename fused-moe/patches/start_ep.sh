#!/bin/bash
# Apply EP patch and start vLLM
set -e
python3 /patches/ep_cutlass_fp4_v2.py
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
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}'
