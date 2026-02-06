"""
Autoregressive GRU baseline for the 2x2 Rubik's Cube.

Encodes the cube state into a GRU initial hidden state, then decodes one
move at a time.  Training uses teacher forcing on a randomly sampled optimal
solution; evaluation uses greedy autoregressive decoding and checks against
all optimal solutions.

Usage:
    python pretrain_fc.py                       # defaults
    python pretrain_fc.py --max_solution_len 4  # curriculum: short scrambles only
"""

import os
import json
import argparse
import time
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# ─── Hyperparameters / CLI ────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="FC baseline – 2x2 Rubik's Cube")
    p.add_argument("--data_path", type=str, default="data/cube-2-by-2-all-solutions")
    p.add_argument("--max_solution_len", type=int, default=None,
                   help="Only train on puzzles whose *shortest* solution ≤ this length (curriculum).")

    p.add_argument("--epochs",       type=int,   default=100)
    p.add_argument("--batch_size",   type=int,   default=512)
    p.add_argument("--lr",           type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--warmup_steps", type=int,   default=500)
    p.add_argument("--hidden_dim",   type=int,   default=512)
    p.add_argument("--num_layers",   type=int,   default=2,
                   help="Number of GRU layers.")
    p.add_argument("--dropout",      type=float, default=0.0)

    p.add_argument("--eval_interval", type=int, default=5)
    p.add_argument("--checkpoint_dir", type=str, default="checkpoints/2x2-rnn-baseline")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


# ─── Dataset ──────────────────────────────────────────────────────────────────

class CubeDataset(Dataset):
    """
    Loads the numpy arrays produced by ``dataset/build_2x2.py``.

    Each sample returns:
        inputs  : (24,)                  int  – sticker values 1-6
        labels  : (num_solutions, 11)    int  – all optimal solutions (0 = pad)
        mask    : (num_solutions,)       bool – which solution slots are valid
    """

    def __init__(self, data_path: str, split: str, max_solution_len: int | None = None):
        super().__init__()
        meta_path = os.path.join(data_path, split, "dataset.json")
        with open(meta_path) as f:
            meta = json.load(f)

        set_name = meta["sets"][0]
        base = os.path.join(data_path, split)

        self.inputs = np.load(os.path.join(base, f"{set_name}__inputs.npy"))                     # (N, 24)
        self.labels = np.load(os.path.join(base, f"{set_name}__labels.npy"), mmap_mode="r")     # (N, S, 11)
        self.group_indices = np.load(os.path.join(base, f"{set_name}__group_indices.npy"))        # (G+1,)

        self.seq_len     = meta["seq_len"]       # 11
        self.vocab_size  = meta["vocab_size"]     # 19

        # Precompute valid-solution mask (solution row has at least one nonzero)
        self.sol_mask = self.labels.sum(axis=-1) > 0  # (N, S)

        # Optional curriculum filter: keep only puzzles whose *shortest*
        # solution is ≤ max_solution_len moves.
        if max_solution_len is not None:
            keep = self._filter_by_length(max_solution_len)
            self.inputs  = self.inputs[keep]
            self.labels  = self.labels[keep]
            self.sol_mask = self.sol_mask[keep]
            print(f"[{split}] max_solution_len={max_solution_len}: "
                  f"kept {keep.sum()}/{len(keep)} puzzles")

    # ------------------------------------------------------------------
    def _filter_by_length(self, max_len: int) -> np.ndarray:
        """Return a boolean mask over puzzles whose shortest solution ≤ max_len."""
        # For each puzzle, get the length of each valid solution
        # Length = number of non-zero tokens in the solution
        lengths = (self.labels != 0).sum(axis=-1)  # (N, S)
        # Replace invalid (all-zero) solutions with a large number
        lengths = np.where(self.sol_mask, lengths, 9999)
        shortest = lengths.min(axis=-1)  # (N,)
        return shortest <= max_len

    # ------------------------------------------------------------------
    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.inputs[idx].astype(np.int64)),   # (24,)
            torch.from_numpy(self.labels[idx].astype(np.int64)),   # (S, 11)
            torch.from_numpy(self.sol_mask[idx]),                   # (S,)
        )


# ─── Model ────────────────────────────────────────────────────────────────────

class CubeRNN(nn.Module):
    """
    Autoregressive GRU that encodes the 24 sticker state into an initial
    hidden state, then decodes one move token at a time.

    Training  : teacher-forced  (``forward``)
    Inference : greedy decoding (``generate``)
    """

    def __init__(self, num_stickers: int = 24, num_colors: int = 6,
                 seq_len: int = 11, vocab_size: int = 19,
                 hidden_dim: int = 512, num_layers: int = 2,
                 dropout: float = 0.0):
        super().__init__()
        self.num_stickers    = num_stickers
        self.num_colors      = num_colors
        self.seq_len         = seq_len
        self.vocab_size      = vocab_size
        self.hidden_dim      = hidden_dim
        self.num_gru_layers  = num_layers

        input_dim = num_stickers * num_colors  # 144

        # ── Encoder: cube state → GRU initial hidden ──────────────────────
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim * num_layers),
        )

        # ── Decoder ───────────────────────────────────────────────────────
        self.tok_embed   = nn.Embedding(vocab_size, hidden_dim, padding_idx=0)
        self.gru         = nn.GRU(
            hidden_dim, hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.out_norm    = nn.LayerNorm(hidden_dim)
        self.output_head = nn.Linear(hidden_dim, vocab_size)

    # ------------------------------------------------------------------
    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode sticker state → GRU h0  (num_layers, B, hidden_dim)."""
        x_oh   = F.one_hot(x - 1, num_classes=self.num_colors).float()  # (B, 24, 6)
        x_flat = x_oh.reshape(x.size(0), -1)                            # (B, 144)
        h = self.encoder(x_flat)                                         # (B, L*H)
        h = h.view(x.size(0), self.num_gru_layers, self.hidden_dim)      # (B, L, H)
        return h.permute(1, 0, 2).contiguous()                           # (L, B, H)

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Teacher-forced forward pass.

        x       : (B, 24)  sticker values 1-6
        targets : (B, T)   move tokens 1-18, pad=0
        returns : (B, T, vocab_size)  logits
        """
        h = self._encode(x)  # (L, B, H)

        # Shift targets right: [BOS=0, tok0, tok1, …, tok_{T-2}]
        bos = torch.zeros(x.size(0), 1, dtype=targets.dtype, device=targets.device)
        decoder_input = torch.cat([bos, targets[:, :-1]], dim=1)  # (B, T)

        embedded    = self.tok_embed(decoder_input)                # (B, T, H)
        output, _   = self.gru(embedded, h)                        # (B, T, H)
        logits      = self.output_head(self.out_norm(output))      # (B, T, V)
        return logits

    # ------------------------------------------------------------------
    @torch.no_grad()
    def generate(self, x: torch.Tensor, max_steps: int | None = None) -> torch.Tensor:
        """
        Greedy autoregressive decoding.

        returns : (B, T, vocab_size)  logits at each step
        """
        if max_steps is None:
            max_steps = self.seq_len
        B = x.size(0)
        h = self._encode(x)  # (L, B, H)

        token = torch.zeros(B, 1, dtype=torch.long, device=x.device)  # BOS
        all_logits = []

        for _ in range(max_steps):
            embedded        = self.tok_embed(token)                  # (B, 1, H)
            output, h       = self.gru(embedded, h)                  # (B, 1, H)
            step_logits     = self.output_head(self.out_norm(output)) # (B, 1, V)
            all_logits.append(step_logits)
            token = step_logits.argmax(dim=-1)                       # greedy

        return torch.cat(all_logits, dim=1)  # (B, T, V)


# ─── Loss (min over all optimal solutions) ────────────────────────────────────

def min_solution_loss(logits: torch.Tensor,
                      all_labels: torch.Tensor,
                      sol_mask: torch.Tensor) -> torch.Tensor:
    """
    Compute cross-entropy against every valid optimal solution and
    keep only the minimum per example.

    logits     : (B, 11, 19)
    all_labels : (B, S, 11)    0 = pad  (S = max solutions dim)
    sol_mask   : (B, S)        True for valid solutions

    Returns scalar loss (mean over batch).
    """
    B, S, T = all_labels.shape
    V = logits.shape[-1]

    # Expand logits to match solutions dim  →  (B, S, T, V)
    logits_exp = logits.unsqueeze(1).expand(-1, S, -1, -1)

    # Flatten for F.cross_entropy:  (B*S*T, V)  vs  (B*S*T,)
    logits_flat = logits_exp.reshape(-1, V)
    labels_flat = all_labels.reshape(-1)

    # Per-token CE, ignoring pad positions (label == 0)
    per_token = F.cross_entropy(logits_flat, labels_flat,
                                ignore_index=0, reduction="none")  # (B*S*T,)
    per_token = per_token.view(B, S, T)

    # Mean CE per solution (only over non-pad positions)
    token_mask = (all_labels != 0).float()           # (B, S, T)
    token_counts = token_mask.sum(dim=-1).clamp(min=1)  # (B, S)
    per_solution = (per_token * token_mask).sum(dim=-1) / token_counts  # (B, S)

    # Mask out invalid solution slots with +inf so they never win the min
    per_solution = per_solution.masked_fill(~sol_mask, float("inf"))

    # Best (lowest) loss per example
    best_loss = per_solution.min(dim=1).values  # (B,)
    return best_loss.mean()


# ─── Accuracy helpers ─────────────────────────────────────────────────────────

@torch.no_grad()
def compute_metrics(logits: torch.Tensor,
                    all_labels: torch.Tensor,
                    sol_mask: torch.Tensor):
    """
    Returns a dict with:
      - loss           : min-solution cross-entropy
      - seq_accuracy   : fraction of examples where the argmax prediction
                         exactly matches at least one optimal solution
      - token_accuracy : average per-token accuracy (against best solution)
    """
    B, S, T = all_labels.shape
    preds = logits.argmax(dim=-1)  # (B, 11)

    # Check per-solution exact match
    preds_exp = preds.unsqueeze(1).expand(-1, S, -1)       # (B, S, 11)
    token_match = (preds_exp == all_labels)                  # (B, S, 11)

    # Only compare non-pad positions
    non_pad = all_labels != 0                                # (B, S, 11)
    # A solution matches if every non-pad token matches
    seq_match = (token_match | ~non_pad).all(dim=-1)         # (B, S)
    seq_match = seq_match & sol_mask                          # ignore invalid slots
    any_match = seq_match.any(dim=1).float()                  # (B,)

    # Token accuracy against the best-matching solution
    # Pick the solution with most matching tokens
    matching_tokens = (token_match & non_pad).sum(dim=-1)     # (B, S)
    matching_tokens = matching_tokens.masked_fill(~sol_mask, -1)
    best_sol_idx = matching_tokens.argmax(dim=1)              # (B,)
    best_labels = all_labels[torch.arange(B, device=all_labels.device), best_sol_idx]  # (B, 11)
    valid = best_labels != 0
    tok_acc = ((preds == best_labels) & valid).float().sum() / valid.float().sum().clamp(min=1)

    loss = min_solution_loss(logits, all_labels, sol_mask)

    return {
        "loss":           loss.item(),
        "seq_accuracy":   any_match.mean().item(),
        "token_accuracy": tok_acc.item(),
    }


# ─── Training loop ────────────────────────────────────────────────────────────

def train(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Data ──────────────────────────────────────────────────────────────
    train_ds = CubeDataset(args.data_path, "train", max_solution_len=args.max_solution_len)
    val_ds   = CubeDataset(args.data_path, "val",   max_solution_len=None)  # always evaluate on all

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=0, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              num_workers=0, pin_memory=True)

    print(f"Train: {len(train_ds):,} puzzles  |  Val: {len(val_ds):,} puzzles")
    print(f"Solutions dim (S): {train_ds.labels.shape[1]}")

    # ── Model ─────────────────────────────────────────────────────────────
    model = CubeRNN(
        seq_len=train_ds.seq_len,
        vocab_size=train_ds.vocab_size,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)

    # Warmup + cosine decay schedule
    steps_per_epoch = len(train_loader)
    total_steps = args.epochs * steps_per_epoch
    warmup_steps = args.warmup_steps

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    print(f"Steps/epoch: {steps_per_epoch}  |  Total steps: {total_steps}  |  Warmup: {warmup_steps}")

    # ── Checkpoint dir ────────────────────────────────────────────────────
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    best_val_loss = float("inf")
    global_step = 0

    # ── Loop ──────────────────────────────────────────────────────────────
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        running_loss = 0.0
        num_batches  = 0

        for inputs, labels, mask in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            mask   = mask.to(device)

            B, S, T = labels.shape

            # Sample one valid solution per example for teacher forcing
            probs = mask.float()
            probs = probs / probs.sum(dim=1, keepdim=True).clamp(min=1e-8)
            chosen = torch.multinomial(probs, 1).squeeze(1)       # (B,)
            targets = labels[torch.arange(B, device=device), chosen]  # (B, T)

            logits = model(inputs, targets)                        # (B, T, V)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                ignore_index=0,
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            running_loss += loss.item()
            num_batches  += 1
            global_step  += 1

        train_loss = running_loss / max(num_batches, 1)
        epoch_time = time.time() - t0

        # ── Eval ──────────────────────────────────────────────────────────
        if epoch % args.eval_interval == 0 or epoch == 1:
            model.eval()
            all_metrics = {"loss": 0.0, "seq_accuracy": 0.0, "token_accuracy": 0.0}
            val_batches = 0

            for inputs, labels, mask in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                mask   = mask.to(device)

                logits = model.generate(inputs)
                m = compute_metrics(logits, labels, mask)
                for k in all_metrics:
                    all_metrics[k] += m[k]
                val_batches += 1

            for k in all_metrics:
                all_metrics[k] /= max(val_batches, 1)

            tag = ""
            if all_metrics["loss"] < best_val_loss:
                best_val_loss = all_metrics["loss"]
                torch.save(model.state_dict(),
                           os.path.join(args.checkpoint_dir, "best.pt"))
                tag = " ★"

            print(f"Epoch {epoch:>4d}/{args.epochs} | "
                  f"train_loss {train_loss:.4f} | "
                  f"val_loss {all_metrics['loss']:.4f} | "
                  f"seq_acc {all_metrics['seq_accuracy']*100:.2f}% | "
                  f"tok_acc {all_metrics['token_accuracy']*100:.2f}% | "
                  f"lr {optimizer.param_groups[0]['lr']:.2e} | "
                  f"{epoch_time:.1f}s{tag}")
        else:
            print(f"Epoch {epoch:>4d}/{args.epochs} | "
                  f"train_loss {train_loss:.4f} | "
                  f"{epoch_time:.1f}s")

    # Save final
    torch.save(model.state_dict(),
               os.path.join(args.checkpoint_dir, "final.pt"))
    print(f"\nDone. Best val loss: {best_val_loss:.4f}")
    print(f"Checkpoints saved to {args.checkpoint_dir}/")


if __name__ == "__main__":
    train(parse_args())
