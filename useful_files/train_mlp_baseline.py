"""
Train Simple MLP baseline for 2x2 solution sequence prediction.

This trains a per-position classifier that predicts the move token at each
sequence step. It is a simple baseline (no recurrence/attention).
"""
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from pathlib import Path
import json
import time
from tqdm import tqdm
import os
import matplotlib.pyplot as plt


class SimpleMLP(nn.Module):
    """Simple 4-layer MLP with one-hot encoding for sequence prediction."""
    def __init__(self, input_dim=144, hidden_dim=256, seq_len=15, vocab_size=10, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, seq_len * vocab_size)
        self.len_head = nn.Linear(hidden_dim, seq_len + 1)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        
    def forward(self, x):
        # x: (batch, 24) state indices (values 1..6)
        # Convert to one-hot
        x = (x.long() - 1).clamp(min=0, max=5)
        x_onehot = torch.nn.functional.one_hot(x, num_classes=6).float()
        x_onehot = x_onehot.reshape(x_onehot.shape[0], -1)  # (batch, 144)
        
        x = torch.nn.functional.relu(self.fc1(x_onehot))
        if self.dropout:
            x = self.dropout(x)
        
        x = torch.nn.functional.relu(self.fc2(x))
        if self.dropout:
            x = self.dropout(x)
            
        x = torch.nn.functional.relu(self.fc3(x))
        if self.dropout:
            x = self.dropout(x)
            
        seq_logits = self.fc4(x)
        len_logits = self.len_head(x)
        return seq_logits.view(x.shape[0], self.seq_len, self.vocab_size), len_logits


def load_dataset_split(data_dir: Path, split: str):
    """Load a dataset split"""
    split_dir = data_dir / split
    if not split_dir.exists():
        raise FileNotFoundError(f"Split {split} not found in {data_dir}")

    with open(split_dir / "dataset.json", "r") as f:
        metadata = json.load(f)

    inputs = np.load(split_dir / "all__inputs.npy")
    labels = np.load(split_dir / "all__labels.npy")

    return {
        'metadata': metadata,
        'inputs': inputs,
        'labels': labels,
    }


def train_model(
    model,
    train_data,
    test_data,
    epochs=25,
    lr=1e-3,
    weight_decay=1e-5,
    batch_size=128,
    device='cpu',
    early_stopping_patience=5,
    curriculum=True,
    curriculum_start_len=3,
    curriculum_end_len=None,
    curriculum_epochs=10,
    length_loss_weight=0.5,
):
    """Train MLP with CrossEntropyLoss, early stopping, and return history."""
    def _format_seconds(seconds: float) -> str:
        seconds = max(0, int(seconds))
        m, s = divmod(seconds, 60)
        h, m = divmod(m, 60)
        if h > 0:
            return f"{h}h {m}m {s}s"
        return f"{m}m {s}s"

    def _curriculum_max_len(epoch_idx: int, seq_len: int) -> int:
        if not curriculum:
            return seq_len
        end_len = curriculum_end_len if curriculum_end_len is not None else seq_len
        end_len = max(curriculum_start_len, min(end_len, seq_len))
        if curriculum_epochs <= 1:
            return end_len
        progress = min(1.0, epoch_idx / float(curriculum_epochs - 1))
        return int(round(curriculum_start_len + progress * (end_len - curriculum_start_len)))
    
    train_inputs = torch.from_numpy(train_data['inputs'])
    train_labels = torch.from_numpy(train_data['labels'].astype(np.int64))
    train_lengths = (train_labels != 0).sum(dim=1)

    test_inputs = torch.from_numpy(test_data['inputs'])
    test_labels = torch.from_numpy(test_data['labels'].astype(np.int64))
    test_lengths = (test_labels != 0).sum(dim=1)

    train_dataset = TensorDataset(train_inputs, train_labels, train_lengths)
    test_dataset = TensorDataset(test_inputs, test_labels, test_lengths)
    
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # PAD token = 0
    len_criterion = nn.CrossEntropyLoss()
    
    history = {
        'train_loss': [],
        'test_loss': [],
        'train_token_acc': [],
        'test_token_acc': [],
        'train_seq_acc': [],
        'test_seq_acc': [],
        'train_len_acc': [],
        'test_len_acc': []
    }
    
    start_time = time.time()
    epoch_times = []
    
    best_test_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    for epoch in range(epochs):
        epoch_start = time.time()
        
        # Train
        max_len = _curriculum_max_len(epoch, train_labels.shape[1])
        if curriculum:
            mask = train_lengths <= max_len
            train_indices = torch.where(mask)[0]
            train_subset = torch.utils.data.Subset(train_dataset, train_indices)
        else:
            train_subset = train_dataset

        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)

        model.train()
        train_loss = 0
        train_token_correct = 0
        train_token_total = 0
        train_seq_correct = 0
        train_len_correct = 0
        train_count = 0
        
        for inputs, targets, lengths in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False):
            inputs, targets, lengths = inputs.to(device), targets.to(device), lengths.to(device)
            
            optimizer.zero_grad()
            seq_logits, len_logits = model(inputs)
            seq_loss = criterion(seq_logits.view(-1, seq_logits.shape[-1]), targets.view(-1))
            len_loss = len_criterion(len_logits, lengths)
            loss = seq_loss + length_loss_weight * len_loss
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)
            
            with torch.no_grad():
                preds = torch.argmax(seq_logits, dim=-1)
                mask = targets != 0
                train_token_correct += (preds.eq(targets) & mask).sum().item()
                train_token_total += mask.sum().item()
                train_seq_correct += (preds.eq(targets) | ~mask).all(dim=1).sum().item()
                train_len_correct += (len_logits.argmax(dim=1) == lengths).sum().item()
                train_count += inputs.size(0)
        
        train_loss /= train_count
        train_token_acc = train_token_correct / max(1, train_token_total)
        train_seq_acc = train_seq_correct / train_count
        train_len_acc = train_len_correct / train_count
        
        # Evaluate
        model.eval()
        test_loss = 0
        test_token_correct = 0
        test_token_total = 0
        test_seq_correct = 0
        test_len_correct = 0
        test_count = 0
        
        with torch.no_grad():
            for inputs, targets, lengths in test_loader:
                inputs, targets, lengths = inputs.to(device), targets.to(device), lengths.to(device)
                seq_logits, len_logits = model(inputs)
                seq_loss = criterion(seq_logits.view(-1, seq_logits.shape[-1]), targets.view(-1))
                len_loss = len_criterion(len_logits, lengths)
                loss = seq_loss + length_loss_weight * len_loss
                test_loss += loss.item() * inputs.size(0)
                
                preds = torch.argmax(seq_logits, dim=-1)
                mask = targets != 0
                test_token_correct += (preds.eq(targets) & mask).sum().item()
                test_token_total += mask.sum().item()
                test_seq_correct += (preds.eq(targets) | ~mask).all(dim=1).sum().item()
                test_len_correct += (len_logits.argmax(dim=1) == lengths).sum().item()
                test_count += inputs.size(0)
        
        test_loss /= test_count
        test_token_acc = test_token_correct / max(1, test_token_total)
        test_seq_acc = test_seq_correct / test_count
        test_len_acc = test_len_correct / test_count
        
        history['train_loss'].append(train_loss)
        history['test_loss'].append(test_loss)
        history['train_token_acc'].append(train_token_acc)
        history['test_token_acc'].append(test_token_acc)
        history['train_seq_acc'].append(train_seq_acc)
        history['test_seq_acc'].append(test_seq_acc)
        history['train_len_acc'].append(train_len_acc)
        history['test_len_acc'].append(test_len_acc)
        
        epoch_time = time.time() - epoch_start
        epoch_times.append(epoch_time)
        avg_epoch_time = sum(epoch_times) / len(epoch_times)
        remaining_time = avg_epoch_time * (epochs - epoch - 1)

        # Early stopping check
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            patience_counter = 0
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            status = "✓"
        else:
            patience_counter += 1
            status = f"({patience_counter}/{early_stopping_patience})"

        print(f"Epoch {epoch+1}/{epochs} (max_len={max_len}): "
              f"Train Loss={train_loss:.4f}, Test Loss={test_loss:.4f}, "
              f"Train Token Acc={train_token_acc:.4f}, Test Token Acc={test_token_acc:.4f}, "
              f"Test Seq Acc={test_seq_acc:.4f}, Test Len Acc={test_len_acc:.4f}, "
              f"Epoch Time={_format_seconds(epoch_time)}, "
              f"ETA={_format_seconds(remaining_time)} {status}")
        
        # Early stopping
        if patience_counter >= early_stopping_patience:
            print(f"\nEarly stopping triggered at epoch {epoch+1}")
            break
    
    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"Restored best model (test loss: {best_test_loss:.4f})")
    
    total_time = time.time() - start_time
    
    return history, total_time


if __name__ == "__main__":
    print("="*70)
    print("Training Simple MLP Baseline (Sequence Prediction)")
    print("Configuration: CrossEntropyLoss (ignore PAD), 25 epochs, early stopping, curriculum")
    print("="*70)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")
    
    data_dir = Path("data/cube-2-by-2-solution")
    
    # Load data
    print("\nLoading dataset...")
    train_data = load_dataset_split(data_dir, "train")
    test_data = load_dataset_split(data_dir, "test")
    print(f"Train: {len(train_data['inputs'])}, Test: {len(test_data['inputs'])}")
    
    seq_len = train_data['labels'].shape[1]
    vocab_size = train_data['metadata']['vocab_size']
    print(f"\nCreating Simple MLP (4 layers, hidden_dim=256, seq_len={seq_len}, vocab_size={vocab_size})...")
    model = SimpleMLP(hidden_dim=256, seq_len=seq_len, vocab_size=vocab_size, dropout=0.1)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")
    
    # Train
    print("\nTraining...")
    history, total_time = train_model(
        model,
        train_data,
        test_data,
        epochs=25,
        lr=1e-3,
        weight_decay=1e-5,
        batch_size=128,
        device=device,
        early_stopping_patience=5,
        curriculum=True,
        curriculum_start_len=3,
        curriculum_end_len=11,
        curriculum_epochs=10,
        length_loss_weight=0.5
    )
    
    # Save results plot
    output_dir = Path(r"C:\personal\Uni\5\ML\Project\HRM\2x2_solution_results")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "mlp_baseline_sequence_results.png"

    epochs_range = np.arange(1, len(history['train_loss']) + 1)
    fig, axes = plt.subplots(2, 2, figsize=(18, 10))

    # Loss plot
    axes[0, 0].plot(epochs_range, history['train_loss'], label="Train Loss", marker='o', markersize=3)
    axes[0, 0].plot(epochs_range, history['test_loss'], label="Test Loss", marker='s', markersize=3)
    axes[0, 0].set_title("Cross-Entropy Loss")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Token accuracy plot
    axes[0, 1].plot(epochs_range, history['train_token_acc'], label="Train Token Acc", marker='o', markersize=3)
    axes[0, 1].plot(epochs_range, history['test_token_acc'], label="Test Token Acc", marker='s', markersize=3)
    axes[0, 1].set_title("Token Accuracy (non-PAD)")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Accuracy")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Sequence accuracy plot
    axes[1, 0].plot(epochs_range, history['train_seq_acc'], label="Train Seq Acc", marker='o', markersize=3)
    axes[1, 0].plot(epochs_range, history['test_seq_acc'], label="Test Seq Acc", marker='s', markersize=3)
    axes[1, 0].set_title("Full Sequence Accuracy")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("Accuracy")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Length accuracy plot
    axes[1, 1].plot(epochs_range, history['train_len_acc'], label="Train Len Acc", marker='o', markersize=3)
    axes[1, 1].plot(epochs_range, history['test_len_acc'], label="Test Len Acc", marker='s', markersize=3)
    axes[1, 1].set_title("Solution Length Accuracy")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("Accuracy")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    fig.suptitle(
        f"MLP 2x2 Solution Sequence | params={num_params:,} | time={total_time:.1f}s | "
        f"lr=1e-3 wd=1e-5 batch=128 | Curriculum + Len Head",
        fontsize=12
    )
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(output_file, dpi=150)
    plt.close(fig)
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE")
    print("="*70)
    print(f"\nFinal Results:")
    print(f"  Test Loss: {history['test_loss'][-1]:.4f}")
    print(f"  Test Token Accuracy: {history['test_token_acc'][-1]:.4f}")
    print(f"  Test Sequence Accuracy: {history['test_seq_acc'][-1]:.4f}")
    print(f"  Train Token Accuracy: {history['train_token_acc'][-1]:.4f}")
    print(f"  Overfitting Gap (Test-Train Loss): {history['test_loss'][-1] - history['train_loss'][-1]:.4f}")
    print(f"  Total Training Time: {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"  Parameters: {num_params:,}")
    print(f"\n✅ Results plot saved to: {output_file}")
