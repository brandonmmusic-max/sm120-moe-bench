# DFlash Drafter Training on RunPod

## Quick Start

### 1. Rent a Pod
- Go to [runpod.io](https://runpod.io) → GPU Cloud → Deploy
- **Pick**: 4x A100 80GB SXM (or 4x H100 80GB SXM)
  - Search for "4x A100" in the GPU filter
  - Cost: ~$8-12/hr (A100) or ~$12-16/hr (H100)
- **Template**: RunPod PyTorch 2.5 (or any CUDA 12.x template)
- **Disk**: 600GB container disk (needs space for 397B model ~234GB + training data)
- **Cloud type**: Spot (cheaper) or On-Demand (reliable)

### 2. SSH Into the Pod
Once the pod is running, click "Connect" → "Start Web Terminal" or SSH.

### 3. Run This Setup Script
Copy-paste this entire block:

```bash
# === SETUP (run once) ===
pip install torch transformers accelerate datasets huggingface_hub flash-attn --no-build-isolation
pip install bitsandbytes sentencepiece

# Download the training script
wget https://raw.githubusercontent.com/brandonmmusic-max/sm120-moe-bench/master/fused-moe/train_dflash_397b.py

# Login to HuggingFace (for model download + upload)
huggingface-cli login
# Paste your HF token when prompted

# === RUN TRAINING ===
# This will:
# 1. Download Qwen3.5-397B from HuggingFace (~20-30 min on RunPod's network)
# 2. Extract hidden states from 5000 samples (~1-2 hrs)
# 3. Train the DFlash drafter (~30-60 min)
# Total: ~2-3 hours

python train_dflash_397b.py \
    --target-model Qwen/Qwen3.5-397B-A17B \
    --draft-model z-lab/Qwen3.5-9B-DFlash \
    --num-samples 5000 \
    --max-seq-len 512 \
    --block-size 16 \
    --batch-size 2 \
    --epochs 3 \
    --lr 1e-4 \
    --target-layer-ids 2,14,26,38,50 \
    --output-dir ./dflash-397b-klc
```

### 4. Upload Trained Model
```bash
# Upload to your HuggingFace account
huggingface-cli upload <your-username>/Qwen3.5-397B-DFlash-KLC ./dflash-397b-klc

# Or just download the output directory (~2GB)
# Use runpodctl or scp to get it to your PC
```

### 5. Deploy on Your PC
```bash
# Copy the trained model to your models dir
# Then launch with:
docker run ... \
    -v /path/to/dflash-397b-klc:/draft-model:ro \
    ... \
    --speculative-config '{"method":"dflash","model":"/draft-model","num_speculative_tokens":16}'
```

## Using Kentucky Legal Data (Optional)

If you want domain-specific training:

1. Upload your KLC conversation JSONs to the pod:
```bash
# From your PC:
scp /media/brandonmusic/nvme0n1p3/Users/brand/Downloads/kentucky_legal_counsel\ update/kentucky_legal_counsel/artifacts/conversations/*.json runpod:/workspace/klc_data/
```

2. Create a JSONL from them:
```python
import json, glob
with open("klc_training.jsonl", "w") as out:
    for f in glob.glob("klc_data/*.json"):
        conv = json.load(open(f))
        for msg in conv.get("messages", []):
            if msg.get("role") == "assistant":
                out.write(json.dumps({"text": msg["content"]}) + "\n")
```

3. Use as training data:
```bash
python train_dflash_397b.py --dataset klc_training.jsonl ...
```

## Expected Results

| Metric | Pre-trained (9B→9B) | Custom (397B-specific) |
|--------|---------------------|----------------------|
| Acceptance rate | ~10-20% (mismatched) | ~50-75% (matched) |
| Tokens/round (B=16) | ~2-3 | ~8-12 |
| Single-user tok/s | ~100-140 | **~250-350** |

## Cost Estimate

| Item | Cost |
|------|------|
| 4x A100 80GB × 3 hours | ~$25-35 |
| Storage (600GB) | ~$1-2 |
| **Total** | **~$27-37** |

## Troubleshooting

- **OOM during extraction**: Reduce `--num-samples` or `--max-seq-len`
- **Slow download**: RunPod pods in US regions download from HF fastest
- **Flash attention error**: `pip install flash-attn --no-build-isolation`
- **Model too large**: The 397B loads with `device_map="auto"` across all 4 GPUs
