# Start vLLM in EP Mode

## Quick Start

First, clean up any old containers:
```bash
docker rm -f vllm-ep vllm-qwen35 vllm-research vllm-extract 2>/dev/null
```

Then start EP mode:
```bash
docker run -d \
  --name vllm-ep \
  --gpus all --ipc host --shm-size 32g \
  --entrypoint bash \
  -p 9200:8000 \
  -v /home/brandonmusic/models/lukealonso-qwen35-nvfp4:/model:ro \
  -v /home/brandonmusic/sm120-moe-bench/fused-moe/patches:/patches:ro \
  -v /home/brandonmusic/klc-linux/cache/vllm:/root/.cache/vllm \
  -v /home/brandonmusic/klc-linux/cache/torch_extensions:/root/.cache/torch_extensions \
  -v /home/brandonmusic/klc-linux/cache/triton:/root/.triton \
  -v /home/brandonmusic/klc-linux/cache/inductor:/root/.cache/torch/inductor \
  -v /home/brandonmusic/klc-linux/cache/flashinfer:/cache/jit/flashinfer/.cache/flashinfer \
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
  vllm-qwen35-k64:latest \
  -c 'python3 /patches/ep_cutlass_fp4_v2.py && exec python3 -m vllm.entrypoints.openai.api_server \
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
  --speculative-config '"'"'{"method":"mtp","num_speculative_tokens":3}'"'"''
```

## Monitor startup
```bash
docker logs -f vllm-ep
```

## Benchmark
```bash
python3 ~/sm120-moe-bench/fused-moe/bench_decode.py http://localhost:9200
```

## What the EP patch does
- `CutlassExpertsFp4._supports_parallel_config()`: Removes `ep_size == 1` restriction
- `CutlassExpertsFp4.supports_expert_map()`: Returns `True`
- `CutlassExpertsFp4.apply()`: Before calling `run_cutlass_moe_fp4`:
  - Maps global expert IDs (0-511) to local IDs (0-127) via `expert_map`
  - Clamps non-local experts (-1) to 0
  - Zeros routing weights for non-local experts

## Architecture in EP mode
- Attention: TP=4 (tensor parallel, same as before)
- MoE: EP=4, TP=1 (each GPU owns 128 full experts)
- Communication: NaiveAll2AllManager (broadcast + reduce) - to be replaced with P2P
