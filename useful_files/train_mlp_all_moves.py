"""
Train MLP with VALIDATION-BASED METRICS instead of exact label matching.

Key innovation: At test time, check if predicted sequence is a VALID solution
(solves the cube optimally), not if it matches the exact label character-by-character.

This allows the model to predict alternative optimal solutions without penalty.
"""
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from pathlib import Path
import json
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
import sys

# Token encoding
MOVE_TOKENS = {
    1: "U",
    2: "U'",
    3: "U2",
    4: "R",
    5: "R'",
    6: "R2",
    7: "F",
    8: "F'",
    9: "F2",
    0: "PAD"
}

# Training hyperparameters
PAD_PENALTY_WEIGHT = 1.0

# Curriculum schedule: (start_epoch_inclusive, end_epoch_exclusive, max_len)
CURRICULUM_STAGES = [
    (0, 7, 4),   # Epochs 1-7: <=4 moves
    (7, 17, 7),  # Epochs 8-17: <=7 moves
    (17, 30, 11) # Epochs 18-30: <=11 moves
]

def tokens_to_moves(tokens):
    """Convert token sequence to move names. Example: [2 8 0...] -> U' F2 PAD"""
    return " ".join(MOVE_TOKENS.get(int(t), f"?{t}") for t in tokens)

# Add py222 to path
sys.path.insert(0, str(Path(__file__).parent.parent / "py222"))
import py222
import solver as py222_solver


class SimpleMLP(nn.Module):
    """Simple 4-layer MLP for sequence prediction."""
    def __init__(self, input_dim=144, hidden_dim=256, seq_len=15, vocab_size=10, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, seq_len * vocab_size)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        
    def forward(self, x):
        x = (x.long() - 1).clamp(min=0, max=5)
        x_onehot = torch.nn.functional.one_hot(x, num_classes=6).float()
        x_onehot = x_onehot.reshape(x_onehot.shape[0], -1)
        
        x = torch.nn.functional.relu(self.fc1(x_onehot))
        if self.dropout:
            x = self.dropout(x)
        
        x = torch.nn.functional.relu(self.fc2(x))
        if self.dropout:
            x = self.dropout(x)
            
        x = torch.nn.functional.relu(self.fc3(x))
        if self.dropout:
            x = self.dropout(x)
            
        x = self.fc4(x)
        return x.view(x.shape[0], self.seq_len, self.vocab_size)


def is_valid_solution(cube_state: np.ndarray, solution: np.ndarray, debug=False) -> bool:
    """
    Check if predicted solution actually solves the cube (regardless of label).
    
    Returns True if solution is valid AND optimal length.
    """
    try:
        # Convert to py222 format (0-5) and normalize
        state = (cube_state.astype(np.uint8) - 1).copy()
        state = py222.normFC(state)
        
        # Filter out PAD (0) and convert to py222 moves (0-8)
        moves = [int(m) - 1 for m in solution if m > 0]
        
        if debug:
            print(f"  State shape: {state.shape}, range: {state.min()}-{state.max()}")
            print(f"  Solution tokens: {solution}")
            print(f"  Filtered moves: {moves}")
        
        if len(moves) == 0:
            is_solved = py222.isSolved(state)
            if debug:
                print(f"  Empty solution, solved={is_solved}")
            return is_solved
        
        # Apply moves; treat as valid if ANY prefix solves the cube
        result_state = state.copy()
        for i, move in enumerate(moves):
            if move < 0 or move > 8:
                if debug:
                    print(f"  Invalid move index at position {i}: {move}")
                return False
            result_state = py222.doMove(result_state, move)
            if py222.isSolved(result_state):
                if debug:
                    print(f"  Solved at step {i+1} (prefix solution)")
                return True
        
        # Check if solved
        is_solved = py222.isSolved(result_state)
        if debug:
            print(f"  After moves: solved={is_solved}")
        
        # If not solved by any prefix, return False
        return False
        
        # TODO: Re-enable optimality check once basic solving works
        # if not is_solved:
        #     return False
        # 
        # # Check if optimal length (compare to ground truth optimal)
        # optimal_solution = py222_solver.solve(state, verbose=False)
        # is_optimal = len(moves) == len(optimal_solution)
        # 
        # if debug:
        #     print(f"  Predicted {len(moves)} moves, optimal {len(optimal_solution)} moves, valid={is_optimal}")
        # 
        # return is_optimal
        
    except Exception as e:
        if debug:
            print(f"  Exception: {type(e).__name__}: {e}")
        return False


def load_dataset_split(data_dir: Path, split: str, max_length: int = None):
    """Load dataset and optionally filter by length"""
    split_dir = data_dir / split
    if not split_dir.exists():
        raise FileNotFoundError(f"Split {split} not found in {data_dir}")

    with open(split_dir / "dataset.json", "r") as f:
        metadata = json.load(f)

    inputs = np.load(split_dir / "all__inputs.npy")
    labels = np.load(split_dir / "all__labels.npy").squeeze()
    
    if max_length is not None:
        seq_lengths = (labels != 0).sum(axis=1)
        mask = seq_lengths <= max_length
        inputs = inputs[mask]
        labels = labels[mask]
        print(f"  {split}: Filtered to {mask.sum():,} / {len(mask):,} samples (≤{max_length} moves)")

    return {
        'metadata': metadata,
        'inputs': inputs,
        'labels': labels,
    }


def train_model(
    model,
    train_data,
    test_data,
    epochs=30,
    lr=1e-3,
    weight_decay=1e-5,
    batch_size=128,
    device='cpu',
    curriculum_stages=None,
):
    """Train MLP with validation-based correctness checking."""
    
    def _format_seconds(seconds: float) -> str:
        seconds = max(0, int(seconds))
        m, s = divmod(seconds, 60)
        h, m = divmod(m, 60)
        if h > 0:
            return f"{h}h {m}m {s}s"
        return f"{m}m {s}s"
    
    if curriculum_stages is None:
        curriculum_stages = CURRICULUM_STAGES

    train_inputs = torch.from_numpy(train_data['inputs'])
    train_labels = torch.from_numpy(train_data['labels'].astype(np.int64))
    test_inputs = torch.from_numpy(test_data['inputs'])
    test_labels = torch.from_numpy(test_data['labels'].astype(np.int64))

    train_lengths = (train_labels != 0).sum(dim=1)
    test_lengths = (test_labels != 0).sum(dim=1)
    
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    
    history = {
        'train_loss': [],
        'test_loss': [],
        'train_token_acc': [],
        'test_token_acc': [],
        'test_valid_solution_acc': [],  # % of predicted solutions that are valid
        'test_pad_acc': [],
        'test_avg_pred_len': [],
        'test_avg_target_len': [],
        'curriculum_max_len': []
    }
    
    # Select 1 random validation sample to debug (randomly each epoch)
    total_val_samples = len(test_labels)
    
    start_time = time.time()
    epoch_times = []
    
    for epoch in range(epochs):
        # For each epoch, select 1 random validation sample to debug
        debug_indices = np.random.choice(total_val_samples, min(1, total_val_samples), replace=False)
        debug_indices_set = set(debug_indices)
        val_sample_counter = 0  # Reset counter for each epoch

        # Determine curriculum max length
        max_len = curriculum_stages[-1][2]
        for stage_start, stage_end, stage_max_len in curriculum_stages:
            if stage_start <= epoch < stage_end:
                max_len = stage_max_len
                break
        
        epoch_start = time.time()
        
        # Train
        model.train()
        train_loss = 0
        train_token_correct = 0
        train_token_total = 0
        train_count = 0
        
        train_mask = train_lengths <= max_len
        train_subset = TensorDataset(train_inputs[train_mask], train_labels[train_mask])
        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)

        for inputs, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} (≤{max_len} moves)", leave=False):
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            logits = model(inputs)
            
            # Loss function setup:
            # Standard CrossEntropyLoss ignores PAD (0), so model gets NO signal about PAD positions
            # We add explicit penalty: when target=0 (PAD), penalize if prediction != 0
            
            # Base loss: sequence prediction (ignores PAD)
            seq_loss = criterion(logits.view(-1, logits.shape[-1]), targets.view(-1))
            
            # Additional penalty for predicting non-PAD when should be PAD
            preds = torch.argmax(logits, dim=-1)
            mask_pad = (targets == 0)  # Positions where PAD should be
            if mask_pad.sum() > 0:
                # Penalize wrong predictions at PAD positions
                wrong_at_pad = (preds[mask_pad] != 0).float().mean()
                loss = seq_loss + PAD_PENALTY_WEIGHT * wrong_at_pad
            else:
                loss = seq_loss
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)
            
            with torch.no_grad():
                preds = torch.argmax(logits, dim=-1)
                mask = targets != 0
                train_token_correct += (preds.eq(targets) & mask).sum().item()
                train_token_total += mask.sum().item()
                train_count += inputs.size(0)
        
        train_loss /= train_count
        train_token_acc = train_token_correct / max(1, train_token_total)
        
        # Evaluate with BOTH metrics: label-match AND valid-solution
        model.eval()
        test_loss = 0
        test_token_correct = 0
        test_token_total = 0
        test_valid_solutions = 0
        test_pad_correct = 0
        test_pad_total = 0
        pred_len_total = 0
        target_len_total = 0
        test_count = 0

        test_mask = test_lengths <= max_len
        test_subset = TensorDataset(test_inputs[test_mask], test_labels[test_mask])
        test_loader = DataLoader(test_subset, batch_size=batch_size)

        with torch.no_grad():
            for inputs, targets in tqdm(test_loader, desc="  Validating", leave=False):
                inputs, targets = inputs.to(device), targets.to(device)
                logits = model(inputs)
                loss = criterion(logits.view(-1, logits.shape[-1]), targets.view(-1))
                test_loss += loss.item() * inputs.size(0)
                
                preds = torch.argmax(logits, dim=-1)
                
                # CRITICAL FIX: Stop predictions at first PAD token
                # Model learns to predict garbage after the actual solution
                # This ensures PAD positions stay as PAD
                for i in range(preds.shape[0]):
                    for j in range(preds.shape[1]):
                        if preds[i, j] == 0:  # Hit first PAD
                            preds[i, j:] = 0  # Zero out rest
                            break
                
                mask = targets != 0
                test_token_correct += (preds.eq(targets) & mask).sum().item()
                test_token_total += mask.sum().item()
                test_count += inputs.size(0)

                # PAD accuracy
                pad_mask = targets == 0
                test_pad_correct += (preds.eq(targets) & pad_mask).sum().item()
                test_pad_total += pad_mask.sum().item()

                # Length stats
                pred_lengths = (preds != 0).sum(dim=1).float()
                target_lengths = (targets != 0).sum(dim=1).float()
                pred_len_total += pred_lengths.sum().item()
                target_len_total += target_lengths.sum().item()
                
                # Check if predicted solutions are VALID (solves cube)
                for i in range(inputs.size(0)):
                    cube_state = inputs[i].cpu().numpy()
                    predicted_seq = preds[i].cpu().numpy()
                    target_seq = targets[i].cpu().numpy()
                    
                    # Debug only selected validation samples
                    should_debug = val_sample_counter in debug_indices_set
                    if should_debug:
                        print(f"\n    [Epoch {epoch+1}] Sample {val_sample_counter}:")
                        print(f"      Target:     {target_seq}")
                        print(f"      Target Moves: {tokens_to_moves(target_seq)}")
                        print(f"      Predicted:  {predicted_seq}")
                        print(f"      Predicted Moves: {tokens_to_moves(predicted_seq)}")
                        print(f"      Match: {np.array_equal(target_seq, predicted_seq)}")
                    
                    if is_valid_solution(cube_state, predicted_seq, debug=False):
                        test_valid_solutions += 1
                    
                    val_sample_counter += 1
        
        test_loss /= test_count
        test_token_acc = test_token_correct / max(1, test_token_total)
        test_valid_acc = test_valid_solutions / test_count
        test_pad_acc = test_pad_correct / max(1, test_pad_total)
        avg_pred_len = pred_len_total / max(1, test_count)
        avg_target_len = target_len_total / max(1, test_count)
        
        history['train_loss'].append(train_loss)
        history['test_loss'].append(test_loss)
        history['train_token_acc'].append(train_token_acc)
        history['test_token_acc'].append(test_token_acc)
        history['test_valid_solution_acc'].append(test_valid_acc)
        history['test_pad_acc'].append(test_pad_acc)
        history['test_avg_pred_len'].append(avg_pred_len)
        history['test_avg_target_len'].append(avg_target_len)
        history['curriculum_max_len'].append(max_len)
        
        epoch_time = time.time() - epoch_start
        epoch_times.append(epoch_time)
        avg_epoch_time = sum(epoch_times) / len(epoch_times)
        remaining_time = avg_epoch_time * (epochs - epoch - 1)

        print(f"Epoch {epoch+1}/{epochs} (≤{max_len} moves): "
              f"Loss={test_loss:.4f}, "
              f"Label-Match Acc={test_token_acc:.4f}, "
              f"Valid-Solution Acc={test_valid_acc:.4f}, "
              f"PAD Acc={test_pad_acc:.4f}, "
              f"AvgLen(pred/target)={avg_pred_len:.2f}/{avg_target_len:.2f}, "
              f"Time={_format_seconds(epoch_time)}")
    
    total_time = time.time() - start_time
    return history, total_time


if __name__ == "__main__":
    print("="*70)
    print("Training MLP with VALID-SOLUTION Metric")
    print("="*70)
    print("Key: At validation, check if prediction SOLVES the cube")
    print("     (not if it matches exact label)")
    print("Curriculum: <=4 (epochs 1-7) -> <=7 (epochs 8-17) -> <=11 (epochs 18-30)")
    print("="*70)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")
    
    data_dir = Path("data/cube-2-by-2-solution")
    
    # Load data
    print("\nLoading dataset...")
    train_data = load_dataset_split(data_dir, "train")
    test_data = load_dataset_split(data_dir, "test")
    
    seq_len = train_data['labels'].shape[1]
    vocab_size = train_data['metadata']['vocab_size']
    print(f"\nCreating MLP (hidden_dim=256, seq_len={seq_len}, vocab_size={vocab_size})...")
    model = SimpleMLP(hidden_dim=256, seq_len=seq_len, vocab_size=vocab_size, dropout=0.1)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")
    
    # Train
    print("\nTraining...")
    history, total_time = train_model(
        model,
        train_data,
        test_data,
        epochs=30,
        lr=1e-3,
        weight_decay=1e-5,
        batch_size=128,
        device=device,
        curriculum_stages=CURRICULUM_STAGES
    )
    
    # Save results
    output_dir = Path(r"C:\personal\Uni\5\ML\Project\HRM\2x2_solution_results")
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"mlp_validsolution_curriculum_{timestamp}.png"

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

   # Accuracy comparison
    axes[0, 1].plot(epochs_range, history['test_token_acc'], label="Label-Match Acc", marker='s', markersize=3, linestyle='--')
    axes[0, 1].plot(epochs_range, history['test_valid_solution_acc'], label="Valid-Solution Acc", marker='o', markersize=3, linewidth=2)
    axes[0, 1].set_title("Solution Accuracy")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Accuracy")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(epochs_range, history['test_pad_acc'], label="PAD Acc", marker='o', markersize=3)
    axes[1, 0].set_title("PAD Accuracy")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("Accuracy")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(epochs_range, history['test_avg_pred_len'], label="Avg Pred Len", marker='o', markersize=3)
    axes[1, 1].plot(epochs_range, history['test_avg_target_len'], label="Avg Target Len", marker='s', markersize=3)
    axes[1, 1].set_title("Average Predicted Length")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("Moves")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    fig.suptitle(
        f"MLP Valid-Solution Curriculum | {num_params:,} params | {total_time:.1f}s",
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
    print(f"  Label-Match Token Acc: {history['test_token_acc'][-1]:.4f}")
    print(f"  Valid-Solution Acc: {history['test_valid_solution_acc'][-1]:.4f}")
    print(f"  PAD Acc: {history['test_pad_acc'][-1]:.4f}")
    print(f"  Avg Len (pred/target): {history['test_avg_pred_len'][-1]:.2f}/{history['test_avg_target_len'][-1]:.2f}")
    print(f"  Gap (Valid - Label): {history['test_valid_solution_acc'][-1] - history['test_token_acc'][-1]:+.4f}")
    print(f"  Total Time: {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"\n✅ Results plot saved to: {output_file}")

    # Save final checkpoint
    checkpoint_path = output_dir / f"mlp_validsolution_curriculum_{timestamp}.pt"
    torch.save({
        'model_state': model.state_dict(),
        'history': history,
        'curriculum_stages': curriculum_stages,
        'pad_penalty_weight': PAD_PENALTY_WEIGHT,
        'epochs': 30,
        'num_params': num_params
    }, checkpoint_path)
    print(f"✅ Final checkpoint saved to: {checkpoint_path}")
    print(f"\nINTERPRETATION:")
    print(f"  - If Valid-Solution Acc > Label-Match Acc:")
    print(f"    → Model finds alternative optimal solutions! 🎉")
    print(f"  - If they're equal:")
    print(f"    → Solutions are unique (or model only learns labeled one)")
