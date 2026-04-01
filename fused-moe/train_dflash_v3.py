#!/usr/bin/env python3
"""
DFlash v3: Paper-faithful DFlash drafter training for Qwen3.5-397B-A17B

Key differences from v2:
- Custom 5-layer DFlash drafter built FROM SCRATCH (no z-lab pretrained weights)
- KV injection of target features into every draft layer (paper Section 4.1)
- Nemotron Qwen3 data + KLC conversation data (target-aligned)
- Frozen 397B embedding + LM head extracted from safetensors
- GPTQ-Int4 model for extraction (fits 4xH200)
- Random anchor sampling, block_size=16, γ=7

Usage:
    pip install torch transformers accelerate datasets safetensors auto-gptq
    python train_dflash_v3.py --output-dir ./dflash-v3 --num-samples 50000 --epochs 3
"""

import argparse
import json
import math
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast


# =============================================================================
# DFlash Draft Model Architecture (from scratch, paper-faithful)
# =============================================================================

class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x):
        variance = x.float().pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return (self.weight * x).to(x.dtype)


class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position=262144, base=10000000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, x, position_ids):
        # position_ids: [B, S]
        inv_freq = self.inv_freq[None, :, None].float()  # [1, D/2, 1]
        pos = position_ids[:, None, :].float()  # [B, 1, S]
        freqs = (inv_freq @ pos).transpose(1, 2)  # [B, S, D/2]
        emb = torch.cat([freqs, freqs], dim=-1)  # [B, S, D]
        return emb.cos(), emb.sin()


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary(q, k, cos, sin):
    # q: [B, H, Sq, D], k: [B, H, Sk, D]
    # cos/sin: [B, S, D] where S >= max(Sq, Sk)
    cos_q = cos[:, None, -q.size(2):, :]  # [B, 1, Sq, D]
    sin_q = sin[:, None, -q.size(2):, :]
    q = (q * cos_q) + (rotate_half(q) * sin_q)
    cos_k = cos[:, None, :k.size(2), :]  # [B, 1, Sk, D]
    sin_k = sin[:, None, :k.size(2), :]
    k = (k * cos_k) + (rotate_half(k) * sin_k)
    return q, k


class DFlashAttention(nn.Module):
    """DFlash attention with KV injection from target features.

    Q comes from draft hidden_states.
    K,V come from concat(target_context, hidden_states) — KV injection.
    """
    def __init__(self, hidden_size, num_heads, num_kv_heads, head_dim):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.q_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=False)
        self.q_norm = RMSNorm(head_dim)
        self.k_norm = RMSNorm(head_dim)

    def forward(self, hidden_states, target_context, cos, sin):
        B, S, _ = hidden_states.shape
        ctx_len = target_context.size(1)

        # Q from draft hidden states
        q = self.q_proj(hidden_states).view(B, S, self.num_heads, self.head_dim)
        q = self.q_norm(q).transpose(1, 2)  # [B, H, S, D]

        # K,V from concat(target_context, hidden_states) — KV injection
        kv_input = torch.cat([target_context, hidden_states], dim=1)  # [B, ctx+S, H_dim]
        k = self.k_proj(kv_input).view(B, ctx_len + S, self.num_kv_heads, self.head_dim)
        v = self.v_proj(kv_input).view(B, ctx_len + S, self.num_kv_heads, self.head_dim)
        k = self.k_norm(k).transpose(1, 2)  # [B, KH, ctx+S, D]
        v = v.transpose(1, 2)

        # Rotary embeddings
        q, k = apply_rotary(q, k, cos, sin)

        # GQA: expand kv heads
        if self.num_kv_heads < self.num_heads:
            rep = self.num_heads // self.num_kv_heads
            k = k.repeat_interleave(rep, dim=1)
            v = v.repeat_interleave(rep, dim=1)

        # Scaled dot-product attention (no causal mask — bidirectional within block)
        attn = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        attn = attn.transpose(1, 2).reshape(B, S, -1)
        return self.o_proj(attn)


class DFlashMLP(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class DFlashLayer(nn.Module):
    def __init__(self, hidden_size, num_heads, num_kv_heads, head_dim, intermediate_size):
        super().__init__()
        self.self_attn = DFlashAttention(hidden_size, num_heads, num_kv_heads, head_dim)
        self.mlp = DFlashMLP(hidden_size, intermediate_size)
        self.input_layernorm = RMSNorm(hidden_size)
        self.post_attention_layernorm = RMSNorm(hidden_size)

    def forward(self, hidden_states, target_context, cos, sin):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, target_context, cos, sin)
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class DFlashDrafter(nn.Module):
    """Paper-faithful DFlash draft model.

    5 transformer layers with KV injection of target features.
    fc projects concatenated target hidden states -> hidden_size.
    Frozen embedding + LM head from target model.
    """
    def __init__(self, hidden_size=4096, num_layers=5, num_heads=32,
                 num_kv_heads=8, head_dim=128, intermediate_size=12288,
                 num_target_layers=5, vocab_size=248320):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_target_layers = num_target_layers

        # Project concatenated target hidden states
        self.fc = nn.Linear(num_target_layers * hidden_size, hidden_size, bias=False)
        self.hidden_norm = RMSNorm(hidden_size)

        # Transformer layers
        self.layers = nn.ModuleList([
            DFlashLayer(hidden_size, num_heads, num_kv_heads, head_dim, intermediate_size)
            for _ in range(num_layers)
        ])
        self.norm = RMSNorm(hidden_size)
        self.rotary_emb = RotaryEmbedding(head_dim)

    def forward(self, noise_embedding, target_hidden, position_ids):
        """
        noise_embedding: [B, S, hidden_size] — masked token embeddings
        target_hidden: [B, S, num_target_layers * hidden_size] — concat hidden states
        position_ids: [B, 2*S] — covers both target context and draft tokens
        """
        # Project and normalize target features
        target_context = self.hidden_norm(self.fc(target_hidden))  # [B, S, H]

        hidden_states = noise_embedding
        cos, sin = self.rotary_emb(hidden_states, position_ids)

        for layer in self.layers:
            hidden_states = layer(hidden_states, target_context, cos, sin)

        return self.norm(hidden_states)


# =============================================================================
# Data loading
# =============================================================================

def load_nemotron_qwen3(num_samples, cache_dir="./data_cache"):
    """Load Qwen3 samples from Nemotron-Post-Training-Dataset-v2."""
    from datasets import load_dataset
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f"nemotron_qwen3_{num_samples}.jsonl")

    if os.path.exists(cache_file):
        texts = []
        with open(cache_file) as f:
            for line in f:
                texts.append(json.loads(line)["text"])
        print(f"  Loaded {len(texts)} cached Nemotron samples")
        return texts

    print("  Downloading Nemotron-Post-Training-Dataset-v2 (Qwen3 subset)...")
    texts = []

    # Try math, code, chat, stem subsets
    subsets = ["math", "code", "chat", "stem"]
    per_subset = num_samples // len(subsets) + 1

    for subset in subsets:
        try:
            ds = load_dataset(
                "nvidia/Nemotron-Post-Training-Dataset-v2",
                subset,
                split="train",
                streaming=True,
            )
            count = 0
            for item in ds:
                # Filter for Qwen3 model outputs
                model = item.get("model", "") or ""
                if "qwen3" not in model.lower() and "Qwen3" not in model:
                    continue

                # Extract prompt + response
                messages = item.get("messages", []) or item.get("conversations", [])
                if not messages:
                    # Try input/output format
                    inp = item.get("input", "") or item.get("prompt", "")
                    out = item.get("output", "") or item.get("response", "")
                    if inp and out:
                        texts.append(f"{inp}\n{out}")
                        count += 1
                else:
                    parts = []
                    for m in messages:
                        role = m.get("role", "")
                        content = m.get("content", "")
                        if content:
                            parts.append(content)
                    if parts:
                        texts.append("\n".join(parts))
                        count += 1

                if count >= per_subset:
                    break

            print(f"    {subset}: {count} Qwen3 samples")
        except Exception as e:
            print(f"    {subset}: failed ({e})")

    # Cache for reuse
    with open(cache_file, "w") as f:
        for t in texts[:num_samples]:
            json.dump({"text": t}, f)
            f.write("\n")

    print(f"  Total Nemotron Qwen3 samples: {len(texts)}")
    return texts[:num_samples]


def load_klc_conversations(klc_jsonl_path=None):
    """Load KLC conversation pairs."""
    # Try to find KLC conversations
    if klc_jsonl_path and os.path.exists(klc_jsonl_path):
        texts = []
        with open(klc_jsonl_path) as f:
            for line in f:
                texts.append(json.loads(line)["text"])
        print(f"  Loaded {len(texts)} KLC samples from {klc_jsonl_path}")
        return texts

    # Try default location
    conv_dir = None
    for base in ["/media/brandonmusic/nvme0n1p3", "/media/brandonmusic/nvme1n1p3"]:
        p = os.path.join(base, "Users/brand/Downloads/kentucky_legal_counsel_local/artifacts/conversations")
        if os.path.isdir(p):
            conv_dir = p
            break

    if not conv_dir:
        print("  KLC conversations not found")
        return []

    import glob
    texts = []
    for f in glob.glob(os.path.join(conv_dir, "*.json")):
        try:
            with open(f) as fh:
                conv = json.load(fh)
            msgs = conv.get("messages", [])
            for i in range(len(msgs) - 1):
                if msgs[i].get("role") == "user" and msgs[i + 1].get("role") == "assistant":
                    content = msgs[i + 1].get("content", "")
                    if len(content) > 100:
                        prompt = msgs[i]["content"]
                        texts.append(f"{prompt}\n{content}")
        except Exception:
            pass

    print(f"  Loaded {len(texts)} KLC conversation pairs")
    return texts


# =============================================================================
# Phase 1: Extract hidden states
# =============================================================================

def extract_hidden_states(args):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from safetensors import safe_open
    from huggingface_hub import snapshot_download

    print("\n" + "=" * 60)
    print("Phase 1: Extract hidden states from target model")
    print("=" * 60)

    cache_dir = Path(args.hidden_states_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    manifest = cache_dir / "manifest.json"

    if manifest.exists() and args.skip_extraction:
        with open(manifest) as f:
            meta = json.load(f)
        print(f"Using cached hidden states ({meta['num_samples']} samples)")
        return

    target_layers = [int(x) for x in args.target_layer_ids.split(",")]

    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.target_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load training data
    print("Loading training data...")
    texts = []

    # KLC data first (highest quality — actual 397B outputs)
    klc_texts = load_klc_conversations(args.klc_data)
    texts.extend(klc_texts)

    # Nemotron Qwen3 data
    remaining = args.num_samples - len(texts)
    if remaining > 0:
        nemotron_texts = load_nemotron_qwen3(remaining)
        texts.extend(nemotron_texts)

    # Fallback to general datasets if needed
    remaining = args.num_samples - len(texts)
    if remaining > 0:
        print(f"  Need {remaining} more samples, loading fallback datasets...")
        from datasets import load_dataset
        fallbacks = [
            ("openai/gsm8k", "main", "train", lambda x: x["question"] + "\n" + x["answer"]),
            ("tatsu-lab/alpaca", None, "train", lambda x: x.get("instruction", "") + "\n" + x.get("output", "")),
        ]
        for ds_name, ds_config, ds_split, text_fn in fallbacks:
            if remaining <= 0:
                break
            try:
                if ds_config:
                    ds = load_dataset(ds_name, ds_config, split=ds_split)
                else:
                    ds = load_dataset(ds_name, split=ds_split)
                new_texts = [text_fn(item) for item in ds][:remaining]
                texts.extend(new_texts)
                remaining -= len(new_texts)
                print(f"    {ds_name}: {len(new_texts)} samples")
            except Exception as e:
                print(f"    {ds_name}: failed ({e})")

    texts = texts[:args.num_samples]
    print(f"  Total training samples: {len(texts)}")

    # Tokenize
    print("Tokenizing...")
    all_token_ids = []
    for i, text in enumerate(texts):
        ids = tokenizer.encode(text, max_length=args.max_seq_len, truncation=True)
        if len(ids) >= 4:
            all_token_ids.append(ids)
        if (i + 1) % 10000 == 0:
            print(f"  Tokenized {i + 1}/{len(texts)}")
    print(f"  {len(all_token_ids)} valid samples")

    # Load model (GPTQ-Int4 preferred, BF16 fallback)
    num_gpus = torch.cuda.device_count()
    print(f"Loading target model on {num_gpus} GPUs...")
    t0 = time.time()

    model = AutoModelForCausalLM.from_pretrained(
        args.target_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    print(f"  Model loaded in {time.time() - t0:.0f}s")

    # Extract frozen embeddings from safetensors
    embed_weights = None
    lm_head_weights = None
    try:
        model_path = snapshot_download(args.target_model, local_files_only=True)
        index_path = os.path.join(model_path, "model.safetensors.index.json")
        if os.path.exists(index_path):
            with open(index_path) as f:
                index = json.load(f)
            weight_map = index["weight_map"]
            for key in weight_map:
                if "embed_tokens.weight" in key and embed_weights is None:
                    shard = os.path.join(model_path, weight_map[key])
                    with safe_open(shard, framework="pt", device="cpu") as f:
                        embed_weights = f.get_tensor(key).clone()
                    print(f"  Extracted embed_tokens: {embed_weights.shape}")
                if "lm_head.weight" in key and lm_head_weights is None:
                    shard = os.path.join(model_path, weight_map[key])
                    with safe_open(shard, framework="pt", device="cpu") as f:
                        lm_head_weights = f.get_tensor(key).clone()
                    print(f"  Extracted lm_head: {lm_head_weights.shape}")
            if lm_head_weights is None and embed_weights is not None:
                lm_head_weights = embed_weights.clone()
                print("  Using tied weights for lm_head")
    except Exception as e:
        print(f"  WARNING: Safetensors extraction failed: {e}")

    # Save embeddings
    torch.save({
        "embed_weights": embed_weights,
        "lm_head_weights": lm_head_weights,
    }, cache_dir / "target_embeddings.pt")

    # Extract hidden states
    print(f"Extracting hidden states (batch_size={args.extract_batch_size})...")
    first_device = next(model.parameters()).device
    batch_num = 0
    batch_hidden = []
    batch_ids = []
    t_extract = time.time()

    # Resume support
    existing = sorted(cache_dir.glob("batch_*.pt"))
    if existing:
        batch_num = len(existing)
        samples_done = batch_num * args.save_every
        print(f"  Resuming from batch {batch_num} ({samples_done} samples)")
        all_token_ids = all_token_ids[samples_done:]

    for i in range(0, len(all_token_ids), args.extract_batch_size):
        mini_batch = all_token_ids[i:i + args.extract_batch_size]
        try:
            max_len = max(len(ids) for ids in mini_batch)
            padded = [ids + [tokenizer.pad_token_id] * (max_len - len(ids)) for ids in mini_batch]
            input_ids = torch.tensor(padded, device=first_device)
            attention_mask = torch.tensor(
                [[1] * len(ids) + [0] * (max_len - len(ids)) for ids in mini_batch],
                device=first_device,
            )
            with torch.no_grad():
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                    return_dict=True,
                )
            for j, ids in enumerate(mini_batch):
                seq_len = len(ids)
                layer_h = []
                for lid in target_layers:
                    if lid < len(outputs.hidden_states):
                        layer_h.append(outputs.hidden_states[lid][j, :seq_len].cpu().half())
                if len(layer_h) == len(target_layers):
                    concat_h = torch.cat(layer_h, dim=-1).unsqueeze(0)
                    batch_hidden.append(concat_h)
                    batch_ids.append(torch.tensor(ids).unsqueeze(0))
            del outputs
            torch.cuda.empty_cache()
        except Exception as e:
            if i < 5 * args.extract_batch_size:
                print(f"    Warn batch {i}: {e}")
            for j, ids in enumerate(mini_batch):
                try:
                    single = torch.tensor([ids], device=first_device)
                    with torch.no_grad():
                        out = model(input_ids=single, output_hidden_states=True, return_dict=True)
                    layer_h = []
                    for lid in target_layers:
                        if lid < len(out.hidden_states):
                            layer_h.append(out.hidden_states[lid].cpu().half())
                    if len(layer_h) == len(target_layers):
                        batch_hidden.append(torch.cat(layer_h, dim=-1))
                        batch_ids.append(single.cpu())
                    del out
                except Exception:
                    continue

        if len(batch_hidden) >= args.save_every:
            torch.save({"hidden_states": batch_hidden, "input_ids": batch_ids},
                       cache_dir / f"batch_{batch_num}.pt")
            elapsed = time.time() - t_extract
            total_done = (batch_num + 1) * args.save_every
            rate = total_done / elapsed
            remaining_s = (len(all_token_ids) - (i + 1)) / max(rate, 0.01)
            print(f"    Saved batch {batch_num}: {total_done} samples, {rate:.1f}/s, ETA: {remaining_s / 3600:.1f}h")
            batch_num += 1
            batch_hidden = []
            batch_ids = []

    if batch_hidden:
        torch.save({"hidden_states": batch_hidden, "input_ids": batch_ids},
                   cache_dir / f"batch_{batch_num}.pt")
        batch_num += 1

    # Manifest
    total_samples = 0
    for b in range(batch_num):
        bp = cache_dir / f"batch_{b}.pt"
        if bp.exists():
            total_samples += len(torch.load(bp, weights_only=False)["hidden_states"])

    with open(manifest, "w") as f:
        json.dump({
            "num_samples": total_samples,
            "num_batches": batch_num,
            "target_layers": target_layers,
            "max_seq_len": args.max_seq_len,
            "model_id": args.target_model,
        }, f, indent=2)

    print(f"  Extraction complete: {total_samples} samples in {(time.time() - t_extract) / 3600:.1f}h")
    del model
    torch.cuda.empty_cache()


# =============================================================================
# Dataset with anchor-based masking
# =============================================================================

class DFlashDataset(Dataset):
    def __init__(self, cache_dir, block_size=16, num_anchors=512):
        self.block_size = block_size
        self.num_anchors = num_anchors
        cache_dir = Path(cache_dir)

        with open(cache_dir / "manifest.json") as f:
            meta = json.load(f)

        self.all_hidden = []
        self.all_ids = []
        for b in range(meta["num_batches"]):
            data = torch.load(cache_dir / f"batch_{b}.pt", weights_only=False)
            self.all_hidden.extend(data["hidden_states"])
            self.all_ids.extend(data["input_ids"])
        print(f"  Loaded {len(self.all_hidden)} samples from {meta['num_batches']} batches")

    def __len__(self):
        return len(self.all_hidden)

    def __getitem__(self, idx):
        hidden = self.all_hidden[idx].squeeze(0).float()
        tokens = self.all_ids[idx].squeeze(0)
        seq_len = tokens.shape[0]

        num_possible = max(1, seq_len - self.block_size)
        num_anchors = min(self.num_anchors, num_possible)
        anchors = torch.randperm(num_possible)[:num_anchors].sort().values

        mask = torch.zeros(seq_len, dtype=torch.bool)
        block_positions = torch.zeros(seq_len, dtype=torch.long)
        for anchor in anchors:
            for k in range(1, self.block_size):
                pos = anchor.item() + k
                if pos < seq_len:
                    mask[pos] = True
                    block_positions[pos] = k

        return {
            "hidden_states": hidden,
            "input_ids": tokens,
            "mask": mask,
            "block_positions": block_positions,
        }


def collate_fn(batch):
    max_len = max(item["input_ids"].shape[0] for item in batch)
    hidden_dim = batch[0]["hidden_states"].shape[-1]
    B = len(batch)

    result = {
        "hidden_states": torch.zeros(B, max_len, hidden_dim),
        "input_ids": torch.zeros(B, max_len, dtype=torch.long),
        "mask": torch.zeros(B, max_len, dtype=torch.bool),
        "block_positions": torch.zeros(B, max_len, dtype=torch.long),
    }
    for i, item in enumerate(batch):
        sl = item["input_ids"].shape[0]
        result["hidden_states"][i, :sl] = item["hidden_states"]
        result["input_ids"][i, :sl] = item["input_ids"]
        result["mask"][i, :sl] = item["mask"]
        result["block_positions"][i, :sl] = item["block_positions"]
    return result


# =============================================================================
# Phase 2: Train DFlash drafter
# =============================================================================

def train_dflash(args):
    from transformers import AutoTokenizer

    print("\n" + "=" * 60)
    print("Phase 2: Training DFlash v3 drafter FROM SCRATCH")
    print("=" * 60)

    device = torch.device("cuda:0")
    cache_dir = Path(args.hidden_states_dir)

    # Tokenizer from target
    tokenizer = AutoTokenizer.from_pretrained(args.target_model, trust_remote_code=True)
    vocab_size = len(tokenizer)
    print(f"  Tokenizer vocab_size: {vocab_size}")

    # Build DFlash drafter FROM SCRATCH
    print("Building custom DFlash drafter (random init)...")
    target_layers = [int(x) for x in args.target_layer_ids.split(",")]
    draft = DFlashDrafter(
        hidden_size=4096,
        num_layers=5,
        num_heads=32,
        num_kv_heads=8,
        head_dim=128,
        intermediate_size=12288,
        num_target_layers=len(target_layers),
        vocab_size=vocab_size,
    ).to(device=device, dtype=torch.bfloat16)

    num_draft_params = sum(p.numel() for p in draft.parameters())
    print(f"  Draft model: {num_draft_params / 1e6:.0f}M params (random init)")

    # Frozen embedding + LM head from target
    embed = nn.Embedding(vocab_size, 4096).to(device=device, dtype=torch.bfloat16)
    lm_head = nn.Linear(4096, vocab_size, bias=False).to(device=device, dtype=torch.bfloat16)

    # Load from safetensors cache
    emb_path = cache_dir / "target_embeddings.pt"
    embed_loaded = False
    head_loaded = False

    if emb_path.exists():
        target_emb = torch.load(emb_path, weights_only=False)
        if target_emb.get("embed_weights") is not None:
            embed.weight.data.copy_(target_emb["embed_weights"][:vocab_size].to(torch.bfloat16))
            embed.weight.requires_grad = False
            embed_loaded = True
            print("  Loaded + froze target embedding weights")
        if target_emb.get("lm_head_weights") is not None:
            lm_head.weight.data.copy_(target_emb["lm_head_weights"][:vocab_size].to(torch.bfloat16))
            lm_head.weight.requires_grad = False
            head_loaded = True
            print("  Loaded + froze target LM head weights")

    if not embed_loaded or not head_loaded:
        print("  Extracting embeddings from safetensors...")
        try:
            from safetensors import safe_open
            from huggingface_hub import snapshot_download
            model_path = snapshot_download(args.target_model, local_files_only=True)
            index_path = os.path.join(model_path, "model.safetensors.index.json")
            if os.path.exists(index_path):
                with open(index_path) as f:
                    index = json.load(f)
                weight_map = index["weight_map"]
                if not embed_loaded:
                    for key in weight_map:
                        if "embed_tokens.weight" in key:
                            shard = os.path.join(model_path, weight_map[key])
                            with safe_open(shard, framework="pt", device="cpu") as f:
                                w = f.get_tensor(key)
                            embed.weight.data.copy_(w[:vocab_size].to(torch.bfloat16))
                            embed.weight.requires_grad = False
                            embed_loaded = True
                            print(f"  Extracted + froze embed_tokens: {w.shape}")
                            break
                if not head_loaded:
                    for key in weight_map:
                        if "lm_head.weight" in key:
                            shard = os.path.join(model_path, weight_map[key])
                            with safe_open(shard, framework="pt", device="cpu") as f:
                                w = f.get_tensor(key)
                            lm_head.weight.data.copy_(w[:vocab_size].to(torch.bfloat16))
                            lm_head.weight.requires_grad = False
                            head_loaded = True
                            print(f"  Extracted + froze lm_head: {w.shape}")
                            break
                if not head_loaded and embed_loaded:
                    lm_head.weight.data.copy_(embed.weight.data)
                    lm_head.weight.requires_grad = False
                    head_loaded = True
                    print("  Using embed as lm_head (tied)")
                # Update cache
                torch.save({
                    "embed_weights": embed.weight.data.cpu() if embed_loaded else None,
                    "lm_head_weights": lm_head.weight.data.cpu() if head_loaded else None,
                }, emb_path)
        except Exception as e:
            print(f"  WARNING: {e}")

    if not embed_loaded:
        print("  *** WARNING: embed weights are RANDOM ***")
    if not head_loaded:
        print("  *** WARNING: lm_head weights are RANDOM ***")

    # Trainable params = draft model only
    trainable_params = list(draft.parameters())
    total_params = num_draft_params + embed.weight.numel() + lm_head.weight.numel()
    print(f"  Total: {total_params / 1e6:.0f}M, Trainable: {num_draft_params / 1e6:.0f}M (frozen embed+head)")

    draft.train()

    # Dataset
    print("Loading training dataset...")
    dataset = DFlashDataset(args.hidden_states_dir, block_size=args.block_size, num_anchors=args.num_anchors)
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=4, pin_memory=True,
    )
    print(f"  {len(dataset)} samples, {len(dataloader)} batches/epoch")

    # Loss weights
    gamma = args.gamma
    loss_weights = torch.tensor(
        [math.exp(-(k - 1) / gamma) for k in range(1, args.block_size + 1)],
        dtype=torch.float32, device=device,
    )
    print(f"  Loss weights (γ={gamma}): {[f'{w:.3f}' for w in loss_weights[:5].tolist()]}...")

    # Optimizer
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=0.01, betas=(0.9, 0.95))
    total_steps = len(dataloader) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    mask_token_id = min(248070, vocab_size - 1)

    # Training loop
    print(f"\nStarting training ({args.epochs} epochs, {total_steps} steps)...")
    global_step = 0
    best_loss = float("inf")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):
        epoch_loss = 0
        epoch_correct = 0
        epoch_total = 0
        t_epoch = time.time()

        for batch_idx, batch in enumerate(dataloader):
            hidden = batch["hidden_states"].to(device)
            tokens = batch["input_ids"].to(device).clamp(0, vocab_size - 1)
            mask = batch["mask"].to(device)
            block_pos = batch["block_positions"].to(device)

            optimizer.zero_grad()

            with autocast("cuda", dtype=torch.bfloat16):
                # Embed tokens (frozen)
                with torch.no_grad():
                    noise_emb = embed(tokens)

                # Replace masked positions with mask token embedding
                mask_emb = embed(torch.tensor([mask_token_id], device=device))
                noise_emb = torch.where(
                    mask.unsqueeze(-1).expand_as(noise_emb),
                    mask_emb.expand_as(noise_emb),
                    noise_emb,
                )

                # Position IDs (2*S for KV injection: target context + draft tokens)
                B, S = tokens.shape
                pos_ids = torch.arange(2 * S, device=device).unsqueeze(0).expand(B, -1)

                # Forward
                output = draft(noise_emb, hidden, pos_ids)

                # LM head (frozen, but grad flows through output)
                logits = F.linear(output, lm_head.weight)

                # Position-weighted cross-entropy loss
                if mask.any():
                    masked_logits = logits[mask]
                    masked_targets = tokens[mask]
                    masked_block_pos = block_pos[mask]

                    per_token_loss = F.cross_entropy(
                        masked_logits.view(-1, vocab_size),
                        masked_targets.view(-1),
                        reduction="none",
                    )

                    weights = torch.zeros_like(per_token_loss)
                    for k in range(1, args.block_size + 1):
                        k_mask = masked_block_pos == k
                        if k_mask.any():
                            weights[k_mask] = loss_weights[k - 1]

                    loss = (per_token_loss * weights).sum() / weights.sum().clamp(min=1e-8)
                else:
                    loss = torch.tensor(0.0, device=device)

            # Debug first few steps
            if batch_idx < 3 and epoch == 0:
                print(f"  DEBUG step {batch_idx}: loss={loss.item():.4f}, mask_sum={mask.sum().item()}, "
                      f"logits shape={logits.shape}, output range=[{output.min().item():.2f}, {output.max().item():.2f}]")

            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            global_step += 1

            with torch.no_grad():
                if mask.any():
                    preds = logits[mask].argmax(-1)
                    epoch_correct += (preds == tokens[mask]).sum().item()
                    epoch_total += mask.sum().item()

            if global_step % 200 == 0:
                n = batch_idx + 1
                avg_loss = epoch_loss / n
                acc = epoch_correct / max(epoch_total, 1)
                lr = scheduler.get_last_lr()[0]
                elapsed = time.time() - t_epoch
                eta = elapsed / n * (len(dataloader) - n)
                print(f"  E{epoch + 1} Step {global_step} | Loss: {avg_loss:.4f} | Acc: {acc:.3f} | "
                      f"LR: {lr:.2e} | ETA: {eta / 60:.0f}min")

        # Epoch summary
        avg_loss = epoch_loss / max(len(dataloader), 1)
        acc = epoch_correct / max(epoch_total, 1)
        elapsed = time.time() - t_epoch
        print(f"\n  Epoch {epoch + 1}/{args.epochs}: loss={avg_loss:.4f} acc={acc:.3f} time={elapsed / 60:.1f}min")

        # Save best
        if avg_loss < best_loss:
            best_loss = avg_loss
            # Save draft model
            torch.save(draft.state_dict(), output_dir / "draft_model.pt")
            # Save extra weights
            torch.save({
                "lm_head": lm_head.state_dict(),
                "embed": embed.state_dict(),
            }, output_dir / "extra_weights.pt")
            # Save config for vLLM
            config = {
                "architectures": ["DFlashDraftModel"],
                "hidden_size": 4096,
                "num_hidden_layers": 5,
                "num_attention_heads": 32,
                "num_key_value_heads": 8,
                "head_dim": 128,
                "intermediate_size": 12288,
                "vocab_size": vocab_size,
                "block_size": args.block_size,
                "dflash_config": {
                    "target_layer_ids": [int(x) for x in args.target_layer_ids.split(",")],
                    "mask_token_id": mask_token_id,
                    "block_size": args.block_size,
                },
                "eagle_aux_hidden_state_layer_ids": [int(x) for x in args.target_layer_ids.split(",")],
                "trained_on": args.target_model,
                "training_samples": len(dataset),
                "epochs_completed": epoch + 1,
                "best_loss": best_loss,
                "accuracy": acc,
                "gamma": args.gamma,
                "model_type": "dflash_v3",
            }
            with open(output_dir / "config.json", "w") as f:
                json.dump(config, f, indent=2)
            tokenizer.save_pretrained(output_dir)
            print(f"  Saved best model -> {output_dir}")

    print(f"\nTraining complete. Best loss: {best_loss:.4f}")
    print(f"Model saved to: {args.output_dir}")


# =============================================================================
# Main
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(description="DFlash v3: Paper-faithful drafter training")
    p.add_argument("--target-model", default="Qwen/Qwen3.5-397B-A17B")
    p.add_argument("--output-dir", default="./dflash-v3")
    p.add_argument("--num-samples", type=int, default=50000)
    p.add_argument("--max-seq-len", type=int, default=3072)
    p.add_argument("--block-size", type=int, default=16)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--extract-batch-size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=6e-4)
    p.add_argument("--warmup-ratio", type=float, default=0.04)
    p.add_argument("--gamma", type=float, default=7.0)
    p.add_argument("--target-layer-ids", default="2,14,26,38,50")
    p.add_argument("--num-anchors", type=int, default=512)
    p.add_argument("--skip-extraction", action="store_true")
    p.add_argument("--hidden-states-dir", default="./hidden_states_cache")
    p.add_argument("--save-every", type=int, default=500)
    p.add_argument("--klc-data", default=None, help="Path to KLC JSONL file")
    return p.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("DFlash v3: Paper-faithful Drafter Training")
    print(f"  Target: {args.target_model}")
    print(f"  Drafter: Custom 5-layer DFlash (FROM SCRATCH)")
    print(f"  Samples: {args.num_samples}, Epochs: {args.epochs}")
    print(f"  Block size: {args.block_size}, γ={args.gamma}")
    print(f"  Batch size: {args.batch_size} (extraction: {args.extract_batch_size})")
    print(f"  Output: {args.output_dir}")
    print("=" * 60)

    if not args.skip_extraction:
        extract_hidden_states(args)

    train_dflash(args)

    print("\n" + "=" * 60)
    print("DONE!")
    print(f"  Model at: {args.output_dir}")
    print("  Next: download and deploy with vLLM")
    print("=" * 60)


if __name__ == "__main__":
    main()
