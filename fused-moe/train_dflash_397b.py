#!/usr/bin/env python3
"""
DFlash Drafter Training for Qwen3.5-397B-A17B
Based on arXiv:2602.06036 — Block Diffusion for Flash Speculative Decoding

Key training details from the paper:
- Loss: cross-entropy with exponential decay weighting w_k = exp(-(k-1)/γ), γ=7 for B=16
- Masking: sample 512 anchor positions per sequence, mask next block_size-1 after each
- LR: 6e-4, cosine schedule, 0.04 warmup ratio
- Epochs: 6
- Embeddings: frozen, shared from target model (token embedding + LM head)
- Hidden states: extracted from 5 uniformly-sampled target layers, projected via fc
- Single forward pass denoising (NOT iterative diffusion)

Usage:
    python train_dflash_397b.py --target-model Qwen/Qwen3.5-397B-A17B --num-samples 289000 --epochs 6
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
from torch.cuda.amp import autocast, GradScaler


def parse_args():
    p = argparse.ArgumentParser(description="Train DFlash drafter for Qwen3.5-397B")
    p.add_argument("--target-model", default="Qwen/Qwen3.5-397B-A17B")
    p.add_argument("--draft-model", default="z-lab/Qwen3.5-9B-DFlash")
    p.add_argument("--output-dir", default="./dflash-397b-trained")
    p.add_argument("--num-samples", type=int, default=289000)
    p.add_argument("--max-seq-len", type=int, default=3072,
                    help="Paper uses 3072 (4096 for coder)")
    p.add_argument("--block-size", type=int, default=16)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--epochs", type=int, default=6, help="Paper uses 6")
    p.add_argument("--lr", type=float, default=6e-4, help="Paper uses 6e-4")
    p.add_argument("--warmup-ratio", type=float, default=0.04)
    p.add_argument("--gamma", type=float, default=7.0,
                    help="Loss decay param. Paper: 7 for B=16, 5 for B=10, 4 for B=8")
    p.add_argument("--target-layer-ids", default="2,14,26,38,50")
    p.add_argument("--num-anchors", type=int, default=512,
                    help="Number of anchor positions per sequence (paper: 512)")
    p.add_argument("--skip-extraction", action="store_true")
    p.add_argument("--hidden-states-dir", default="./hidden_states_cache")
    p.add_argument("--save-every", type=int, default=500,
                    help="Save checkpoint every N extraction samples")
    return p.parse_args()


# =============================================================================
# Phase 1: Extract hidden states from the target model
# =============================================================================

def extract_hidden_states(args):
    """Extract hidden states from target model at specified layers."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from datasets import load_dataset

    print(f"\n{'='*60}")
    print(f"Phase 1: Extracting hidden states from {args.target_model}")
    print(f"  Layers: {args.target_layer_ids}")
    print(f"  Samples: {args.num_samples}")
    print(f"  Max seq len: {args.max_seq_len}")
    print(f"{'='*60}\n")

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

    # Load target model
    print(f"Loading target model {args.target_model}...")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        args.target_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    print(f"  Model loaded in {time.time()-t0:.0f}s")

    # Also extract embeddings to freeze in draft model
    embed_weights = None
    lm_head_weights = None
    for name, param in model.named_parameters():
        if "embed_tokens" in name and embed_weights is None:
            embed_weights = param.data.cpu().clone()
            print(f"  Extracted embedding weights: {embed_weights.shape}")
        if "lm_head" in name and lm_head_weights is None:
            lm_head_weights = param.data.cpu().clone()
            print(f"  Extracted LM head weights: {lm_head_weights.shape}")

    # Load diverse training data
    print("Loading training datasets...")
    texts = []

    datasets_to_load = [
        ("openai/gsm8k", "main", "train", lambda x: x["question"] + "\n" + x["answer"], 50000),
        ("tatsu-lab/alpaca", None, "train", lambda x: x.get("instruction","") + "\n" + x.get("output",""), 80000),
        ("TIGER-Lab/MMLU-Pro", None, "test", lambda x: x.get("question","") + "\n" + str(x.get("answer","")), 20000),
    ]

    for ds_name, ds_config, ds_split, text_fn, max_n in datasets_to_load:
        try:
            if ds_config:
                ds = load_dataset(ds_name, ds_config, split=ds_split)
            else:
                ds = load_dataset(ds_name, split=ds_split)
            new_texts = [text_fn(item) for item in ds][:max_n]
            texts.extend(new_texts)
            print(f"  {ds_name}: {len(new_texts)} samples")
        except Exception as e:
            print(f"  {ds_name}: failed ({e})")

    # Fill remaining with OpenOrca (streaming for large dataset)
    remaining = args.num_samples - len(texts)
    if remaining > 0:
        try:
            ds = load_dataset("Open-Orca/OpenOrca", split="train", streaming=True)
            orca = []
            for i, item in enumerate(ds):
                if i >= remaining:
                    break
                orca.append(item.get("question", "") + "\n" + item.get("response", ""))
            texts.extend(orca)
            print(f"  OpenOrca: {len(orca)} samples")
        except Exception as e:
            print(f"  OpenOrca: failed ({e})")

    texts = texts[:args.num_samples]
    print(f"  Total: {len(texts)} training samples")

    # Extract hidden states in batches
    print("Extracting hidden states...")
    batch_num = 0
    batch_hidden = []
    batch_ids = []

    for i, text in enumerate(texts):
        try:
            inputs = tokenizer(
                text, max_length=args.max_seq_len, truncation=True, return_tensors="pt"
            )
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True, return_dict=True)

            layer_h = []
            for lid in target_layers:
                if lid < len(outputs.hidden_states):
                    layer_h.append(outputs.hidden_states[lid].cpu().half())

            if len(layer_h) == len(target_layers):
                concat_h = torch.cat(layer_h, dim=-1)
                batch_hidden.append(concat_h)
                batch_ids.append(inputs["input_ids"].cpu())
        except Exception as e:
            if i < 5:
                print(f"    Warn: sample {i} failed: {e}")
            continue

        # Save periodically
        if (i + 1) % args.save_every == 0:
            torch.save({
                "hidden_states": batch_hidden,
                "input_ids": batch_ids,
            }, cache_dir / f"batch_{batch_num}.pt")
            print(f"    {i+1}/{len(texts)} extracted (saved batch {batch_num}, {len(batch_hidden)} samples)")
            batch_num += 1
            batch_hidden = []
            batch_ids = []

        if (i + 1) % 1000 == 0 and len(batch_hidden) > 0:
            print(f"    {i+1}/{len(texts)} extracted")

    # Save final batch
    if batch_hidden:
        torch.save({
            "hidden_states": batch_hidden,
            "input_ids": batch_ids,
        }, cache_dir / f"batch_{batch_num}.pt")
        batch_num += 1

    # Save embeddings
    torch.save({
        "embed_weights": embed_weights,
        "lm_head_weights": lm_head_weights,
    }, cache_dir / "target_embeddings.pt")

    # Save manifest
    total_samples = sum(
        len(torch.load(cache_dir / f"batch_{b}.pt")["hidden_states"])
        for b in range(batch_num)
    )
    meta = {
        "num_samples": total_samples,
        "num_batches": batch_num,
        "target_layers": target_layers,
        "max_seq_len": args.max_seq_len,
        "model_id": args.target_model,
    }
    with open(manifest, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"  Extracted {total_samples} samples in {batch_num} batches → {cache_dir}")

    del model
    torch.cuda.empty_cache()


# =============================================================================
# Phase 2: Training dataset with DFlash anchor-based masking
# =============================================================================

class DFlashDataset(Dataset):
    """DFlash training dataset with anchor-based block masking.

    Per the paper: randomly sample anchor positions, mask next block_size-1
    tokens after each anchor. The anchor token is the "verified" token,
    and the masked tokens are what the draft model learns to predict.
    """

    def __init__(self, cache_dir, block_size=16, num_anchors=512):
        self.cache_dir = Path(cache_dir)
        self.block_size = block_size
        self.num_anchors = num_anchors

        with open(self.cache_dir / "manifest.json") as f:
            meta = json.load(f)
        self.num_batches = meta["num_batches"]

        # Load all batches into memory
        self.all_hidden = []
        self.all_ids = []
        for b in range(self.num_batches):
            data = torch.load(self.cache_dir / f"batch_{b}.pt")
            self.all_hidden.extend(data["hidden_states"])
            self.all_ids.extend(data["input_ids"])

        print(f"  Loaded {len(self.all_hidden)} samples from {self.num_batches} batches")

    def __len__(self):
        return len(self.all_hidden)

    def __getitem__(self, idx):
        hidden = self.all_hidden[idx].squeeze(0)  # [seq, concat_H]
        tokens = self.all_ids[idx].squeeze(0)      # [seq]
        seq_len = tokens.shape[0]

        # Anchor-based masking (paper Section 3.2):
        # Sample anchor positions, mask next block_size-1 tokens after each
        num_possible_anchors = max(1, seq_len - self.block_size)
        num_anchors = min(self.num_anchors, num_possible_anchors)
        anchor_positions = torch.randperm(num_possible_anchors)[:num_anchors].sort().values

        # Create mask: True = masked (need to predict)
        mask = torch.zeros(seq_len, dtype=torch.bool)
        # Create position-in-block tensor for loss weighting
        block_positions = torch.zeros(seq_len, dtype=torch.long)

        for anchor in anchor_positions:
            for k in range(1, self.block_size):
                pos = anchor.item() + k
                if pos < seq_len:
                    mask[pos] = True
                    block_positions[pos] = k  # position within block (1-indexed)

        return {
            "hidden_states": hidden.float(),
            "input_ids": tokens,
            "mask": mask,
            "block_positions": block_positions,
        }


def collate_fn(batch):
    max_len = max(item["input_ids"].shape[0] for item in batch)
    hidden_dim = batch[0]["hidden_states"].shape[-1]

    result = {
        "hidden_states": torch.zeros(len(batch), max_len, hidden_dim),
        "input_ids": torch.zeros(len(batch), max_len, dtype=torch.long),
        "mask": torch.zeros(len(batch), max_len, dtype=torch.bool),
        "block_positions": torch.zeros(len(batch), max_len, dtype=torch.long),
        "lengths": torch.tensor([item["input_ids"].shape[0] for item in batch]),
    }

    for i, item in enumerate(batch):
        sl = item["input_ids"].shape[0]
        result["hidden_states"][i, :sl] = item["hidden_states"]
        result["input_ids"][i, :sl] = item["input_ids"]
        result["mask"][i, :sl] = item["mask"]
        result["block_positions"][i, :sl] = item["block_positions"]

    return result


# =============================================================================
# Phase 3: Train with position-weighted cross-entropy loss
# =============================================================================

def train_dflash(args):
    from transformers import AutoTokenizer, AutoModel

    print(f"\n{'='*60}")
    print(f"Phase 2: Training DFlash draft model")
    print(f"  Base: {args.draft_model}")
    print(f"  Block size: {args.block_size}, γ={args.gamma}")
    print(f"  LR: {args.lr}, Epochs: {args.epochs}")
    print(f"  Anchor masking: {args.num_anchors} anchors/seq")
    print(f"{'='*60}\n")

    device = torch.device("cuda:0")
    cache_dir = Path(args.hidden_states_dir)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.draft_model, trust_remote_code=True)
    vocab_size = len(tokenizer)

    # Load draft model
    print("Loading base DFlash model...")
    draft = AutoModel.from_pretrained(
        args.draft_model, torch_dtype=torch.bfloat16, trust_remote_code=True
    ).to(device)

    # Create LM head and embedding (frozen, from target model)
    hidden_size = draft.config.hidden_size
    lm_head = nn.Linear(hidden_size, vocab_size, bias=False).to(device=device, dtype=torch.bfloat16)
    embed = nn.Embedding(vocab_size, hidden_size).to(device=device, dtype=torch.bfloat16)

    # Load frozen weights from target model
    target_emb_path = cache_dir / "target_embeddings.pt"
    if target_emb_path.exists():
        target_emb = torch.load(target_emb_path)
        if target_emb["embed_weights"] is not None:
            embed.weight.data.copy_(target_emb["embed_weights"].to(torch.bfloat16))
            embed.weight.requires_grad = False  # FREEZE per paper
            print("  Loaded + froze target embedding weights")
        if target_emb["lm_head_weights"] is not None:
            lm_head.weight.data.copy_(target_emb["lm_head_weights"].to(torch.bfloat16))
            lm_head.weight.requires_grad = False  # FREEZE per paper
            print("  Loaded + froze target LM head weights")
    else:
        print("  WARNING: No target embeddings found, training from scratch")

    # Only train draft model parameters (not embed/lm_head)
    trainable_params = [p for p in draft.parameters() if p.requires_grad]
    num_total = sum(p.numel() for p in draft.parameters()) + embed.weight.numel() + lm_head.weight.numel()
    num_trainable = sum(p.numel() for p in trainable_params)
    print(f"  Parameters: {num_total/1e6:.0f}M total, {num_trainable/1e6:.0f}M trainable (frozen embed+head)")

    draft.train()

    # Dataset
    print("Loading training dataset...")
    dataset = DFlashDataset(
        args.hidden_states_dir,
        block_size=args.block_size,
        num_anchors=args.num_anchors,
    )
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=4, pin_memory=True,
    )
    print(f"  {len(dataset)} samples, {len(dataloader)} batches/epoch")

    # Precompute loss weights: w_k = exp(-(k-1)/gamma) for k=1..block_size
    gamma = args.gamma
    loss_weights = torch.tensor(
        [math.exp(-(k - 1) / gamma) for k in range(1, args.block_size + 1)],
        dtype=torch.float32, device=device,
    )
    print(f"  Loss weights (γ={gamma}): {[f'{w:.3f}' for w in loss_weights[:5].tolist()]}...")

    # Optimizer (paper: AdamW, lr=6e-4, cosine, 0.04 warmup)
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=0.01, betas=(0.9, 0.95))

    total_steps = len(dataloader) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = GradScaler()

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
            hidden = batch["hidden_states"].to(device)        # [B, S, concat_H]
            tokens = batch["input_ids"].to(device)             # [B, S]
            mask = batch["mask"].to(device)                    # [B, S]
            block_pos = batch["block_positions"].to(device)    # [B, S]

            optimizer.zero_grad()

            with autocast(dtype=torch.bfloat16):
                # Embed tokens (frozen embeddings)
                with torch.no_grad():
                    noise_emb = embed(tokens)

                # Block diffusion noise: replace masked positions with [MASK] token embedding
                # Standard discrete diffusion — NO Gaussian noise, just mask token substitution
                # (Arriola et al. 2025, DFlash paper Section 3.2)
                # DFlash uses token 248070 (<|audio_start|>) as mask — a special token
                # outside normal vocab that never appears in real text
                mask_token_id = 248070  # From z-lab DFlash config (dflash_config.mask_token_id)
                mask_embedding = embed(torch.tensor([mask_token_id], device=device))  # [1, H]
                noise_emb = torch.where(
                    mask.unsqueeze(-1).expand_as(noise_emb),
                    mask_embedding.expand_as(noise_emb),  # Replace with mask token embedding
                    noise_emb,
                )

                # Position IDs
                B, S = tokens.shape
                pos_ids = torch.arange(S, device=device).unsqueeze(0).expand(B, -1)

                # Forward through draft model
                output = draft(
                    position_ids=pos_ids,
                    noise_embedding=noise_emb,
                    target_hidden=hidden,
                )

                # LM head (frozen)
                with torch.no_grad():
                    logits = lm_head(output.float()).to(output.dtype)

                # Wait — LM head needs gradients to flow through output
                # Actually: the output of draft model needs gradients, lm_head is linear
                # so grad flows through even with frozen weights
                logits = F.linear(output, lm_head.weight)  # explicit to allow grad flow

                # Position-weighted cross-entropy loss on masked positions
                if mask.any():
                    masked_logits = logits[mask]  # [N_masked, vocab]
                    masked_targets = tokens[mask]  # [N_masked]
                    masked_block_pos = block_pos[mask]  # [N_masked]

                    # Per-token loss
                    per_token_loss = F.cross_entropy(
                        masked_logits.view(-1, vocab_size),
                        masked_targets.view(-1),
                        reduction="none",
                    )

                    # Apply position-dependent weights: w_k = exp(-(k-1)/gamma)
                    weights = torch.zeros_like(per_token_loss)
                    for k in range(1, args.block_size + 1):
                        k_mask = masked_block_pos == k
                        if k_mask.any():
                            weights[k_mask] = loss_weights[k - 1]

                    loss = (per_token_loss * weights).sum() / weights.sum().clamp(min=1e-8)
                else:
                    loss = torch.tensor(0.0, device=device)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            epoch_loss += loss.item()
            global_step += 1

            with torch.no_grad():
                if mask.any():
                    preds = logits[mask].argmax(-1)
                    epoch_correct += (preds == tokens[mask]).sum().item()
                    epoch_total += mask.sum().item()

            if global_step % 200 == 0:
                n_batches = batch_idx + 1
                avg_loss = epoch_loss / n_batches
                acc = epoch_correct / max(epoch_total, 1)
                lr = scheduler.get_last_lr()[0]
                elapsed = time.time() - t_epoch
                eta = elapsed / n_batches * (len(dataloader) - n_batches)
                print(f"  E{epoch+1} Step {global_step} | "
                      f"Loss: {avg_loss:.4f} | Acc: {acc:.3f} | "
                      f"LR: {lr:.2e} | ETA: {eta/60:.0f}min")

        # Epoch summary
        n_batches = len(dataloader)
        avg_loss = epoch_loss / max(n_batches, 1)
        acc = epoch_correct / max(epoch_total, 1)
        elapsed = time.time() - t_epoch
        print(f"\n  Epoch {epoch+1}/{args.epochs}: loss={avg_loss:.4f} acc={acc:.3f} time={elapsed/60:.1f}min")

        # Save
        if avg_loss < best_loss:
            best_loss = avg_loss
            draft.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
            # Save extra weights
            torch.save({
                "lm_head": lm_head.state_dict(),
                "embed": embed.state_dict(),
            }, output_dir / "extra_weights.pt")

            config = {
                "dflash_config": {
                    "target_layer_ids": [int(x) for x in args.target_layer_ids.split(",")],
                    "mask_token_id": 248070,
                    "block_size": args.block_size,
                },
                "trained_on": args.target_model,
                "training_samples": len(dataset),
                "epochs_completed": epoch + 1,
                "best_loss": best_loss,
                "accuracy": acc,
                "gamma": args.gamma,
            }
            with open(output_dir / "training_config.json", "w") as f:
                json.dump(config, f, indent=2)
            print(f"  Saved best model → {output_dir}")

    print(f"\nTraining complete. Best loss: {best_loss:.4f}")
    print(f"Model saved to: {args.output_dir}")


# =============================================================================
# Main
# =============================================================================

def main():
    args = parse_args()

    print("=" * 60)
    print("DFlash Drafter Training for Qwen3.5-397B")
    print(f"  Target: {args.target_model}")
    print(f"  Draft base: {args.draft_model}")
    print(f"  Samples: {args.num_samples}")
    print(f"  Epochs: {args.epochs}, LR: {args.lr}, γ={args.gamma}")
    print(f"  Block size: {args.block_size}, Anchors: {args.num_anchors}")
    print(f"  Output: {args.output_dir}")
    print("=" * 60)

    if not args.skip_extraction:
        extract_hidden_states(args)

    train_dflash(args)

    print("\n" + "=" * 60)
    print("DONE! Next steps:")
    print(f"  1. Download: scp -r <runpod>:{args.output_dir} ~/models/dflash-397b-trained/")
    print(f"  2. Deploy with vLLM:")
    print(f'     --speculative-config \'{{"method":"dflash","model":"<path>","num_speculative_tokens":16}}\'')
    print("=" * 60)


if __name__ == "__main__":
    main()
