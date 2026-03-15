#!/bin/bash
# MTP sweep benchmark commands — reproduces the MTP=1/2/3 comparison
# Requires: vLLM server running at localhost:9200 with `pip install vllm[bench]`
# Tokenizer must be accessible locally (download from HF or point to model dir)

TOKENIZER="/path/to/sehyo-qwen35-nvfp4"  # Change this
BASE_URL="http://localhost:9200"
MODEL="qwen3.5-397b-nvfp4"

# Single-user decode (concurrency=1, 50 prompts, 128 input / 256 output tokens)
vllm bench serve \
  --backend vllm \
  --base-url "${BASE_URL}" \
  --model "${MODEL}" \
  --tokenizer "${TOKENIZER}" \
  --endpoint /v1/completions \
  --dataset-name random \
  --random-input-len 128 \
  --random-output-len 256 \
  --num-prompts 50 \
  --max-concurrency 1 \
  --request-rate inf \
  --num-warmups 5 \
  --temperature 0

# 8-user concurrent throughput (concurrency=8, 100 prompts)
vllm bench serve \
  --backend vllm \
  --base-url "${BASE_URL}" \
  --model "${MODEL}" \
  --tokenizer "${TOKENIZER}" \
  --endpoint /v1/completions \
  --dataset-name random \
  --random-input-len 128 \
  --random-output-len 256 \
  --num-prompts 100 \
  --max-concurrency 8 \
  --request-rate inf \
  --num-warmups 5 \
  --temperature 0
