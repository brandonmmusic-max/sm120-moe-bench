#!/usr/bin/env python3
"""
DFlash Drafter Training for Qwen3.5-397B-A17B
Reverse-engineered from z-lab/dflash model architecture + paper (arXiv:2602.06036)

Run on RunPod (4x A100 80GB) or locally.

Usage:
    # On RunPod:
    pip install torch transformers accelerate datasets huggingface_hub bitsandbytes
    pip install flash-attn --no-build-isolation
    python train_dflash_397b.py --num-samples 5000 --epochs 3 --output-dir ./dflash-397b-trained

    # Upload result:
    huggingface-cli upload <your-username>/Qwen3.5-397B-DFlash ./dflash-397b-trained

Architecture (from dflash.py):
    - 5 DFlash decoder layers with cross-attention to target hidden states
    - fc: projects concatenated hidden states from 5 target layers → hidden_size
    - hidden_norm: RMSNorm on projected hidden states
    - Each layer: self-attn(Q=noise, K/V=concat(target_hidden, noise)) + MLP
    - Output: denoised hidden states → LM head for token prediction

Training objective (block diffusion):
    - Given target hidden states from layers [2,14,26,38,50] of the 397B
    - Mask random tokens in the target sequence
    - Train draft model to predict masked tokens from noised embeddings + hidden states
    - This teaches the model to "denoise" multiple positions in parallel
"""

import argparse
import json
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
    p.add_argument("--target-model", default="Qwen/Qwen3.5-397B-A17B",
                    help="Target model (HF ID or local path)")
    p.add_argument("--draft-model", default="z-lab/Qwen3.5-9B-DFlash",
                    help="Base DFlash architecture to initialize from")
    p.add_argument("--output-dir", default="./dflash-397b-trained",
                    help="Output directory for trained model")
    p.add_argument("--num-samples", type=int, default=5000,
                    help="Number of training samples to generate")
    p.add_argument("--max-seq-len", type=int, default=512,
                    help="Max sequence length for training")
    p.add_argument("--block-size", type=int, default=16,
                    help="DFlash block size (draft tokens per round)")
    p.add_argument("--batch-size", type=int, default=2,
                    help="Training batch size (per GPU)")
    p.add_argument("--epochs", type=int, default=3,
                    help="Training epochs")
    p.add_argument("--lr", type=float, default=1e-4,
                    help="Learning rate")
    p.add_argument("--target-layer-ids", default="2,14,26,38,50",
                    help="Comma-separated target layer IDs in 397B")
    p.add_argument("--mask-ratio", type=float, default=0.5,
                    help="Fraction of tokens to mask during training")
    p.add_argument("--skip-extraction", action="store_true",
                    help="Skip hidden state extraction (use cached)")
    p.add_argument("--hidden-states-dir", default="./hidden_states_cache",
                    help="Directory for cached hidden states")
    p.add_argument("--dataset", default="openai/gsm8k",
                    help="HuggingFace dataset for training prompts")
    p.add_argument("--wandb", action="store_true", help="Enable wandb logging")
    return p.parse_args()


# =============================================================================
# Phase 1: Extract hidden states from the 397B target model
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

    # Check if already cached
    manifest = cache_dir / "manifest.json"
    if manifest.exists() and args.skip_extraction:
        print("Using cached hidden states")
        return

    target_layers = [int(x) for x in args.target_layer_ids.split(",")]

    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.target_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load target model with 4-bit quantization to fit in memory
    print(f"Loading target model {args.target_model}...")
    print("  (This downloads ~234GB on first run, ~20 min on RunPod)")
    t0 = time.time()

    model = AutoModelForCausalLM.from_pretrained(
        args.target_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
    )
    model.eval()
    print(f"  Model loaded in {time.time()-t0:.0f}s")

    # Load dataset
    print(f"Loading dataset {args.dataset}...")
    if args.dataset == "openai/gsm8k":
        ds = load_dataset("openai/gsm8k", "main", split="train")
        texts = [item["question"] + "\n" + item["answer"] for item in ds]
    elif args.dataset.endswith(".jsonl"):
        texts = []
        with open(args.dataset) as f:
            for line in f:
                item = json.loads(line)
                texts.append(item.get("prompt", "") + "\n" + item.get("completion", ""))
    else:
        ds = load_dataset(args.dataset, split="train")
        texts = [item.get("text", item.get("content", str(item))) for item in ds]

    texts = texts[:args.num_samples]
    print(f"  {len(texts)} samples loaded")

    # Extract hidden states
    print("Extracting hidden states...")
    all_hidden_states = []
    all_input_ids = []

    for i, text in enumerate(texts):
        inputs = tokenizer(
            text,
            max_length=args.max_seq_len,
            truncation=True,
            return_tensors="pt",
        ).to(model.device)

        with torch.no_grad():
            outputs = model(
                **inputs,
                output_hidden_states=True,
                return_dict=True,
            )

        # Extract hidden states at target layers
        # outputs.hidden_states is tuple of (num_layers+1) tensors
        layer_hidden = []
        for lid in target_layers:
            if lid < len(outputs.hidden_states):
                h = outputs.hidden_states[lid].cpu().to(torch.float16)
                layer_hidden.append(h)

        if len(layer_hidden) == len(target_layers):
            # Concatenate along hidden dim: [batch, seq, num_layers * hidden_size]
            concat_hidden = torch.cat(layer_hidden, dim=-1)
            all_hidden_states.append(concat_hidden)
            all_input_ids.append(inputs["input_ids"].cpu())

        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(texts)} samples processed")

        # Save periodically
        if (i + 1) % 500 == 0:
            torch.save({
                "hidden_states": all_hidden_states,
                "input_ids": all_input_ids,
            }, cache_dir / f"batch_{i//500}.pt")
            print(f"  Saved batch {i//500}")

    # Save final
    torch.save({
        "hidden_states": all_hidden_states,
        "input_ids": all_input_ids,
        "target_layers": target_layers,
        "model_id": args.target_model,
    }, cache_dir / "all_hidden_states.pt")

    meta = {
        "num_samples": len(all_hidden_states),
        "target_layers": target_layers,
        "max_seq_len": args.max_seq_len,
        "model_id": args.target_model,
    }
    with open(manifest, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"  Extracted {len(all_hidden_states)} samples → {cache_dir}")

    # Free target model memory
    del model
    torch.cuda.empty_cache()


# =============================================================================
# Phase 2: Training dataset
# =============================================================================

class DFlashTrainingDataset(Dataset):
    """Dataset for DFlash training with block diffusion masking."""

    def __init__(self, hidden_states_dir, block_size=16, mask_ratio=0.5):
        data = torch.load(Path(hidden_states_dir) / "all_hidden_states.pt")
        self.hidden_states = data["hidden_states"]  # list of [1, seq, concat_hidden]
        self.input_ids = data["input_ids"]           # list of [1, seq]
        self.block_size = block_size
        self.mask_ratio = mask_ratio

    def __len__(self):
        return len(self.hidden_states)

    def __getitem__(self, idx):
        hidden = self.hidden_states[idx].squeeze(0)  # [seq, concat_hidden]
        tokens = self.input_ids[idx].squeeze(0)       # [seq]
        seq_len = tokens.shape[0]

        # Block diffusion masking:
        # Randomly mask `mask_ratio` fraction of positions
        num_mask = max(1, int(seq_len * self.mask_ratio))
        mask_positions = torch.randperm(seq_len)[:num_mask].sort().values

        # Create masked version of tokens
        masked_tokens = tokens.clone()
        # Use a special noise embedding approach:
        # Replace masked positions with random embeddings (will be learned)
        mask = torch.zeros(seq_len, dtype=torch.bool)
        mask[mask_positions] = True

        return {
            "hidden_states": hidden.float(),     # [seq, concat_hidden]
            "input_ids": tokens,                  # [seq]
            "masked_ids": masked_tokens,          # [seq]
            "mask": mask,                         # [seq]
        }


def collate_fn(batch):
    """Pad and collate training batch."""
    max_len = max(item["input_ids"].shape[0] for item in batch)

    hidden_dim = batch[0]["hidden_states"].shape[-1]
    padded = {
        "hidden_states": torch.zeros(len(batch), max_len, hidden_dim),
        "input_ids": torch.zeros(len(batch), max_len, dtype=torch.long),
        "masked_ids": torch.zeros(len(batch), max_len, dtype=torch.long),
        "mask": torch.zeros(len(batch), max_len, dtype=torch.bool),
        "lengths": torch.tensor([item["input_ids"].shape[0] for item in batch]),
    }

    for i, item in enumerate(batch):
        sl = item["input_ids"].shape[0]
        padded["hidden_states"][i, :sl] = item["hidden_states"]
        padded["input_ids"][i, :sl] = item["input_ids"]
        padded["masked_ids"][i, :sl] = item["masked_ids"]
        padded["mask"][i, :sl] = item["mask"]

    return padded


# =============================================================================
# Phase 3: Train the DFlash draft model
# =============================================================================

def train_dflash(args):
    """Train the DFlash draft model using block diffusion objective."""
    from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

    print(f"\n{'='*60}")
    print(f"Phase 2: Training DFlash draft model")
    print(f"  Base: {args.draft_model}")
    print(f"  Block size: {args.block_size}")
    print(f"  Epochs: {args.epochs}, LR: {args.lr}")
    print(f"{'='*60}\n")

    device = torch.device("cuda:0")

    # Load tokenizer and draft model config
    tokenizer = AutoTokenizer.from_pretrained(args.draft_model, trust_remote_code=True)
    vocab_size = tokenizer.vocab_size

    # Load the DFlash draft model (small, ~1-2B params)
    print("Loading base DFlash model...")
    draft_model = AutoModelForCausalLM.from_pretrained(
        args.draft_model,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    ).to(device)

    # Add LM head if not present (for token prediction)
    if not hasattr(draft_model, "lm_head"):
        config = draft_model.config
        draft_model.lm_head = nn.Linear(
            config.hidden_size, vocab_size, bias=False
        ).to(device=device, dtype=torch.bfloat16)

    # Embedding layer for noise input
    if not hasattr(draft_model, "embed_tokens"):
        config = draft_model.config
        draft_model.embed_tokens = nn.Embedding(
            vocab_size, config.hidden_size
        ).to(device=device, dtype=torch.bfloat16)

    draft_model.train()
    num_params = sum(p.numel() for p in draft_model.parameters())
    num_trainable = sum(p.numel() for p in draft_model.parameters() if p.requires_grad)
    print(f"  Parameters: {num_params/1e6:.0f}M total, {num_trainable/1e6:.0f}M trainable")

    # Dataset
    print("Loading training dataset...")
    dataset = DFlashTrainingDataset(
        args.hidden_states_dir,
        block_size=args.block_size,
        mask_ratio=args.mask_ratio,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=True,
    )
    print(f"  {len(dataset)} samples, {len(dataloader)} batches/epoch")

    # Optimizer
    optimizer = torch.optim.AdamW(
        draft_model.parameters(),
        lr=args.lr,
        weight_decay=0.01,
        betas=(0.9, 0.95),
    )

    # LR scheduler
    total_steps = len(dataloader) * args.epochs
    warmup_steps = min(500, total_steps // 10)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.lr,
        total_steps=total_steps,
        pct_start=warmup_steps / total_steps,
    )

    scaler = GradScaler()

    # Training loop
    print(f"\nStarting training ({args.epochs} epochs, {total_steps} steps)...")
    global_step = 0
    best_loss = float("inf")

    for epoch in range(args.epochs):
        epoch_loss = 0
        epoch_correct = 0
        epoch_total = 0
        t_epoch = time.time()

        for batch_idx, batch in enumerate(dataloader):
            hidden_states = batch["hidden_states"].to(device)  # [B, S, concat_H]
            input_ids = batch["input_ids"].to(device)          # [B, S]
            mask = batch["mask"].to(device)                    # [B, S]
            lengths = batch["lengths"]

            optimizer.zero_grad()

            with autocast(dtype=torch.bfloat16):
                # Create noise embeddings for masked positions
                noise_emb = draft_model.embed_tokens(input_ids)

                # Add noise to masked positions (diffusion noise)
                # Simple approach: replace masked embeddings with random noise
                noise = torch.randn_like(noise_emb) * 0.1
                noise_emb = torch.where(
                    mask.unsqueeze(-1).expand_as(noise_emb),
                    noise_emb + noise,  # Add noise to masked positions
                    noise_emb,          # Keep unmasked as-is
                )

                # Position IDs
                B, S = input_ids.shape
                position_ids = torch.arange(S, device=device).unsqueeze(0).expand(B, -1)

                # Forward through draft model
                output = draft_model(
                    position_ids=position_ids,
                    noise_embedding=noise_emb,
                    target_hidden=hidden_states,
                )

                # LM head prediction
                logits = draft_model.lm_head(output)  # [B, S, vocab]

                # Loss: only on masked positions
                loss = F.cross_entropy(
                    logits[mask].view(-1, vocab_size),
                    input_ids[mask].view(-1),
                    reduction="mean",
                )

                # Accuracy on masked positions
                with torch.no_grad():
                    preds = logits[mask].argmax(dim=-1)
                    correct = (preds == input_ids[mask]).sum().item()
                    total = mask.sum().item()

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(draft_model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            epoch_loss += loss.item()
            epoch_correct += correct
            epoch_total += total
            global_step += 1

            if (batch_idx + 1) % 50 == 0:
                avg_loss = epoch_loss / (batch_idx + 1)
                accuracy = epoch_correct / max(epoch_total, 1)
                lr = scheduler.get_last_lr()[0]
                print(f"  Epoch {epoch+1}/{args.epochs} | "
                      f"Step {batch_idx+1}/{len(dataloader)} | "
                      f"Loss: {avg_loss:.4f} | "
                      f"Acc: {accuracy:.3f} | "
                      f"LR: {lr:.2e}")

        avg_loss = epoch_loss / len(dataloader)
        accuracy = epoch_correct / max(epoch_total, 1)
        elapsed = time.time() - t_epoch
        print(f"\n  Epoch {epoch+1} complete: loss={avg_loss:.4f}, "
              f"acc={accuracy:.3f}, time={elapsed:.0f}s")

        # Save best
        if avg_loss < best_loss:
            best_loss = avg_loss
            save_path = Path(args.output_dir)
            save_path.mkdir(parents=True, exist_ok=True)
            draft_model.save_pretrained(save_path)
            tokenizer.save_pretrained(save_path)

            # Save config with target layer info
            config_extra = {
                "dflash_config": {
                    "target_layer_ids": [int(x) for x in args.target_layer_ids.split(",")],
                    "mask_token_id": tokenizer.convert_tokens_to_ids("<|mask|>")
                        if "<|mask|>" in tokenizer.get_vocab()
                        else tokenizer.eos_token_id,
                    "block_size": args.block_size,
                },
                "trained_on": args.target_model,
                "training_samples": len(dataset),
                "best_loss": best_loss,
                "best_accuracy": accuracy,
            }
            with open(save_path / "training_config.json", "w") as f:
                json.dump(config_extra, f, indent=2)

            print(f"  Saved best model → {save_path}")

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
    print(f"  Output: {args.output_dir}")
    print("=" * 60)

    # Phase 1: Extract hidden states
    if not args.skip_extraction:
        extract_hidden_states(args)

    # Phase 2: Train
    train_dflash(args)

    print("\n" + "=" * 60)
    print("DONE! Next steps:")
    print(f"  1. Upload: huggingface-cli upload <username>/Qwen3.5-397B-DFlash {args.output_dir}")
    print(f"  2. Download to your PC")
    print(f"  3. Launch with vLLM:")
    print(f'     --speculative-config \'{{"method":"dflash","model":"<path>","num_speculative_tokens":16}}\'')
    print("=" * 60)


if __name__ == "__main__":
    main()
