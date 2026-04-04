#!/usr/bin/env python3
"""
DFlash v6 Training — Paper-faithful implementation.
Key fixes vs v5:
1. Block-diagonal attention mask (Figure 4) — blocks can't see each other
2. Correct position IDs — each block uses its actual sequence positions
3. Proper block construction — anchors + masked positions per the paper
"""
import os, json, math, time, shutil
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast
from pathlib import Path

CACHE_DIR = Path("/root/extract_cache/shuffled")
DRAFT_DIR = "/root/draft_model"
MODEL_DIR = "/root/model"
OUTPUT_DIR = Path("/root/dflash_trained")
TARGET_LAYERS = [2, 14, 26, 38, 50]
BLOCK_SIZE = 16
NUM_ANCHORS = 512
GAMMA = 7.0
LR = 6e-4
EPOCHS = 6
BATCH_SIZE = 2  # small batch because each sample has many blocks with full mask

# =============================================================================
# Dataset — returns per-sample blocks with proper structure
# =============================================================================
class DFlashBlockDataset(Dataset):
    """Each sample returns:
    - hidden_states: [seq_len, 20480] context features from target model
    - input_ids: [seq_len] original token IDs
    - blocks: list of (anchor_pos, block_token_ids, block_mask, block_positions)
    """
    def __init__(self, cache_dir):
        cache_dir = Path(cache_dir)
        meta = json.load(open(cache_dir / "manifest.json"))
        self.files = [cache_dir / f"batch_{b}.pt" for b in range(meta["num_batches"])
                      if (cache_dir / f"batch_{b}.pt").exists()]
        self.total = meta["num_samples"]
        self._d = None; self._o = 0; self._l = 0; self._i = 0
        print(f"  {self.total} samples, {len(self.files)} batches")

    def __len__(self): return self.total

    def __getitem__(self, idx):
        if self._d is None or self._o >= self._l:
            if self._i >= len(self.files): self._i = 0
            self._d = torch.load(self.files[self._i], weights_only=False)
            self._l = len(self._d["hidden_states"]); self._o = 0; self._i += 1

        hidden = self._d["hidden_states"][self._o].squeeze(0)  # [seq_len, 20480]
        tokens = self._d["input_ids"][self._o].squeeze(0)       # [seq_len]
        self._o += 1
        seq_len = tokens.shape[0]

        # Sample anchor positions (paper: 512 random anchors per sequence)
        num_possible = max(1, seq_len - BLOCK_SIZE)
        num_anchors = min(NUM_ANCHORS, num_possible)
        anchor_positions = torch.randperm(num_possible)[:num_anchors].sort().values

        return {
            "hidden_states": hidden.float(),
            "input_ids": tokens,
            "anchor_positions": anchor_positions,
            "seq_len": seq_len,
        }


def train_step(draft, embed, lm_head, loss_weights, mask_token_id, vocab_size,
               hidden_states, input_ids, anchor_positions, seq_len, device):
    """
    Paper-faithful training step with block-diagonal attention mask.

    For each sample:
    1. Extract context features (target hidden states at all positions)
    2. For each anchor, create a block of BLOCK_SIZE tokens
    3. Replace masked positions with mask token embedding
    4. Concatenate all blocks into one sequence
    5. Build block-diagonal attention mask
    6. Forward through draft model with mask
    7. Compute position-weighted loss on masked positions
    """
    B = hidden_states.shape[0]  # batch size
    total_loss = torch.tensor(0.0, device=device)
    total_correct = 0
    total_masked = 0

    for b in range(B):
        h = hidden_states[b][:seq_len[b]]      # [S, 20480] context features
        tok = input_ids[b][:seq_len[b]]          # [S] tokens
        anchors = anchor_positions[b]
        anchors = anchors[anchors < seq_len[b] - BLOCK_SIZE + 1]  # valid anchors only

        if len(anchors) == 0:
            continue

        # Limit anchors to keep memory manageable
        max_anchors = min(len(anchors), 64)  # cap at 64 blocks per sample
        anchors = anchors[:max_anchors]
        num_blocks = len(anchors)

        # Build block tokens and masks
        # Each block: [anchor_token, mask, mask, ..., mask] (BLOCK_SIZE tokens)
        block_tokens = []       # [num_blocks * BLOCK_SIZE]
        block_targets = []      # what the model should predict
        block_mask = []         # True = masked (needs prediction)
        block_positions = []    # actual sequence positions
        block_pos_in_block = [] # position within block (for loss weighting)
        block_ids = []          # which block each token belongs to

        for bi, anchor in enumerate(anchors):
            a = anchor.item()
            for k in range(BLOCK_SIZE):
                pos = a + k
                if pos >= seq_len[b]:
                    break
                block_positions.append(pos)
                block_targets.append(tok[pos].item())
                block_ids.append(bi)
                if k == 0:
                    # Anchor position — use real token
                    block_tokens.append(tok[pos].item())
                    block_mask.append(False)
                    block_pos_in_block.append(0)
                else:
                    # Masked position — use mask token
                    block_tokens.append(mask_token_id)
                    block_mask.append(True)
                    block_pos_in_block.append(k)

        if not any(block_mask):
            continue

        # Convert to tensors
        block_tok_t = torch.tensor(block_tokens, device=device, dtype=torch.long)
        block_tgt_t = torch.tensor(block_targets, device=device, dtype=torch.long)
        block_mask_t = torch.tensor(block_mask, device=device, dtype=torch.bool)
        block_pos_t = torch.tensor(block_positions, device=device, dtype=torch.long)
        block_bpos_t = torch.tensor(block_pos_in_block, device=device, dtype=torch.long)
        block_ids_t = torch.tensor(block_ids, device=device, dtype=torch.long)

        draft_len = len(block_tokens)  # total draft tokens (all blocks concatenated)
        ctx_len = seq_len[b].item()     # context = all positions in original sequence

        # Embed draft tokens (replace masks with mask embedding)
        with torch.no_grad():
            draft_embeds = embed(block_tok_t)  # [draft_len, hidden]

        # Context features: target hidden states for the full sequence
        ctx_hidden = h.unsqueeze(0)  # [1, ctx_len, 20480]

        # Position IDs: [context_positions, draft_positions]
        # Context positions: 0..ctx_len-1
        # Draft positions: actual positions from original sequence
        ctx_pos = torch.arange(ctx_len, device=device)
        all_positions = torch.cat([ctx_pos, block_pos_t])  # [ctx_len + draft_len]
        all_positions = all_positions.unsqueeze(0)  # [1, ctx_len + draft_len]

        # Build block-diagonal attention mask for SDPA
        # Shape: [draft_len, ctx_len + draft_len] (additive mask, -inf = blocked)
        total_kv = ctx_len + draft_len
        attn_mask = torch.full((draft_len, total_kv), float('-inf'),
                               device=device, dtype=torch.bfloat16)

        # All draft tokens attend to all context features
        attn_mask[:, :ctx_len] = 0.0

        # Block-diagonal: each block attends only to itself
        for bi in range(num_blocks):
            members = (block_ids_t == bi).nonzero(as_tuple=True)[0]
            if len(members) == 0:
                continue
            row_start = members[0].item()
            row_end = members[-1].item() + 1
            col_start = ctx_len + row_start
            col_end = ctx_len + row_end
            attn_mask[row_start:row_end, col_start:col_end] = 0.0

        # Unsqueeze for SDPA: [1, 1, draft_len, ctx_len + draft_len]
        attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)

        # Forward through draft model
        draft_embeds = draft_embeds.unsqueeze(0)  # [1, draft_len, hidden]
        ctx_hidden_single = ctx_hidden             # [1, ctx_len, 20480]

        out = draft(
            position_ids=all_positions,
            noise_embedding=draft_embeds,
            target_hidden=ctx_hidden_single,
            attention_mask=attn_mask,
        )  # [1, draft_len, hidden]

        # Compute logits
        logits = F.linear(out.squeeze(0), lm_head.weight)  # [draft_len, vocab]

        # Loss on masked positions only
        if block_mask_t.any():
            masked_logits = logits[block_mask_t]       # [N_masked, vocab]
            masked_targets = block_tgt_t[block_mask_t]  # [N_masked]
            masked_bpos = block_bpos_t[block_mask_t]    # [N_masked]

            per_token_loss = F.cross_entropy(
                masked_logits, masked_targets, reduction="none"
            )

            # Position-weighted loss: w_k = exp(-(k-1)/gamma)
            weights = torch.zeros_like(per_token_loss)
            for k in range(1, BLOCK_SIZE + 1):
                km = masked_bpos == k
                if km.any():
                    weights[km] = loss_weights[k - 1]

            sample_loss = (per_token_loss * weights).sum() / weights.sum().clamp(min=1e-8)
            total_loss = total_loss + sample_loss

            with torch.no_grad():
                preds = masked_logits.argmax(-1)
                total_correct += (preds == masked_targets).sum().item()
                total_masked += block_mask_t.sum().item()

    # Average loss across batch
    total_loss = total_loss / max(B, 1)
    return total_loss, total_correct, total_masked


# =============================================================================
# Main training loop
# =============================================================================
def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0")

    from transformers import AutoTokenizer, AutoModel
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
    vocab_size = len(tokenizer)
    print(f"vocab={vocab_size}")

    print("Loading draft model...")
    draft = AutoModel.from_pretrained(DRAFT_DIR, torch_dtype=torch.bfloat16,
                                       trust_remote_code=True).to(device)
    print(f"  params={sum(p.numel() for p in draft.parameters())/1e6:.0f}M")

    lm_head = nn.Linear(draft.config.hidden_size, vocab_size, bias=False).to(device, dtype=torch.bfloat16)
    embed = nn.Embedding(vocab_size, draft.config.hidden_size).to(device, dtype=torch.bfloat16)

    ep = CACHE_DIR / "target_embeddings.pt"
    if ep.exists():
        tw = torch.load(ep, weights_only=False)
        if tw.get("embed_weights") is not None:
            w = tw["embed_weights"].to(torch.bfloat16)
            embed.weight.data[:min(w.shape[0], embed.weight.shape[0])] = w[:embed.weight.shape[0]]
            embed.weight.requires_grad = False
            print(f"  embed: {w.shape}")
        if tw.get("lm_head_weights") is not None:
            w = tw["lm_head_weights"].to(torch.bfloat16)
            lm_head.weight.data[:min(w.shape[0], lm_head.weight.shape[0])] = w[:lm_head.weight.shape[0]]
            lm_head.weight.requires_grad = False
            print(f"  lm_head: {w.shape}")

    trainable = [p for p in draft.parameters() if p.requires_grad]
    print(f"  trainable: {sum(p.numel() for p in trainable)/1e6:.0f}M")
    draft.train()

    loss_weights = torch.tensor([math.exp(-(k-1)/GAMMA) for k in range(1, BLOCK_SIZE+1)],
                                dtype=torch.float32, device=device)
    mask_token_id = min(248070, vocab_size - 1)

    dataset = DFlashBlockDataset(str(CACHE_DIR))

    # Custom collate — variable-length anchor lists
    def collate(batch):
        max_seq = max(b["seq_len"] for b in batch)
        max_anchors = max(len(b["anchor_positions"]) for b in batch)
        hd = batch[0]["hidden_states"].shape[-1]
        B = len(batch)

        hs = torch.zeros(B, max_seq, hd)
        ids = torch.zeros(B, max_seq, dtype=torch.long)
        anchors = torch.zeros(B, max_anchors, dtype=torch.long)
        seq_lens = torch.tensor([b["seq_len"] for b in batch])

        for i, b in enumerate(batch):
            sl = b["seq_len"]
            hs[i, :sl] = b["hidden_states"]
            ids[i, :sl] = b["input_ids"]
            na = len(b["anchor_positions"])
            anchors[i, :na] = b["anchor_positions"]

        return {"hidden_states": hs, "input_ids": ids,
                "anchor_positions": anchors, "seq_len": seq_lens}

    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False,
                       collate_fn=collate, num_workers=0, pin_memory=True)

    optimizer = torch.optim.AdamW(trainable, lr=LR, weight_decay=0.01, betas=(0.9, 0.95))
    total_steps = len(loader) * EPOCHS
    warmup = int(total_steps * 0.04)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
        lambda s: s/max(warmup,1) if s < warmup else 0.5*(1+math.cos(math.pi*(s-warmup)/max(total_steps-warmup,1))))

    print(f"\nTraining: {EPOCHS} epochs, {len(loader)} batches/epoch, {total_steps} steps")
    print(f"  batch_size={BATCH_SIZE}, block_size={BLOCK_SIZE}, anchors={NUM_ANCHORS}")
    print(f"  Block-diagonal attention mask: ENABLED", flush=True)

    step = 0
    for epoch in range(EPOCHS):
        el, ec, et = 0, 0, 0
        t0 = time.time()
        for bi, batch in enumerate(loader):
            h = batch["hidden_states"].to(device)
            tok = batch["input_ids"].to(device).clamp(0, vocab_size - 1)
            anchors = batch["anchor_positions"].to(device)
            seq_lens = batch["seq_len"].to(device)

            optimizer.zero_grad()
            with autocast(dtype=torch.bfloat16):
                loss, correct, masked = train_step(
                    draft, embed, lm_head, loss_weights, mask_token_id, vocab_size,
                    h, tok, anchors, seq_lens, device
                )

            if loss.isnan():
                print(f"  NaN step {step+1}, skip", flush=True)
                optimizer.zero_grad()
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            scheduler.step()
            el += loss.item()
            step += 1
            ec += correct
            et += masked

            if step % 100 == 0:
                acc = ec / max(et, 1)
                eta = (time.time() - t0) / (bi + 1) * (len(loader) - bi - 1)
                print(f"  E{epoch+1} Step {step} | Loss: {el/(bi+1):.4f} | Acc: {acc:.3f} | "
                      f"LR: {scheduler.get_last_lr()[0]:.2e} | ETA: {eta/60:.0f}min", flush=True)

                # Save checkpoint (NaN-safe)
                for name, p in draft.named_parameters():
                    if p.isnan().any():
                        print(f"  NaN in {name}, skip save!", flush=True)
                        break
                else:
                    draft.save_pretrained(str(OUTPUT_DIR))
                    torch.save({"embed": {"weight": embed.weight.data.cpu()},
                                "lm_head": {"weight": lm_head.weight.data.cpu()}},
                               OUTPUT_DIR / "extra_weights.pt")
                    cfg = json.loads(draft.config.to_json_string())
                    cfg["eagle_aux_hidden_state_layer_ids"] = TARGET_LAYERS
                    cfg["dflash_config"] = {"mask_token_id": 248070, "target_layer_ids": TARGET_LAYERS}
                    json.dump(cfg, open(OUTPUT_DIR / "config.json", "w"), indent=2)

        acc = ec / max(et, 1)
        print(f"\n  Epoch {epoch+1}: Loss={el/max(bi+1,1):.4f} Acc={acc:.3f} ({time.time()-t0:.0f}s)", flush=True)
        draft.save_pretrained(str(OUTPUT_DIR))
        torch.save({"embed": {"weight": embed.weight.data.cpu()},
                    "lm_head": {"weight": lm_head.weight.data.cpu()}},
                   OUTPUT_DIR / "extra_weights.pt")
        cfg = json.loads(draft.config.to_json_string())
        cfg["eagle_aux_hidden_state_layer_ids"] = TARGET_LAYERS
        cfg["dflash_config"] = {"mask_token_id": 248070, "target_layer_ids": TARGET_LAYERS}
        json.dump(cfg, open(OUTPUT_DIR / "config.json", "w"), indent=2)
        print(f"  Saved", flush=True)

    for fn in ["tokenizer.json", "tokenizer_config.json", "dflash.py"]:
        for src in [DRAFT_DIR, MODEL_DIR]:
            s = os.path.join(src, fn)
            if os.path.exists(s) and not os.path.exists(str(OUTPUT_DIR / fn)):
                shutil.copy2(s, str(OUTPUT_DIR / fn)); break

    print(f"\n=== TRAINING COMPLETE ===")


if __name__ == "__main__":
    main()
