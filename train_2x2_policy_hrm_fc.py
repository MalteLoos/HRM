"""
Training script for 2x2 Cube Policy prediction using HRM FC architecture.
Uses hierarchical reasoning structure with fully connected layers.
One-hot encoding for cube states (8 pieces * 3 orientations = 24 dims per position).
"""

import os
import json
import numpy as np
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import wandb
from tqdm import tqdm

# Import the HRM FC Policy model
from models.hrm.hrm_fc_policy import HRM_FC_Policy, create_hrm_fc_policy

# Import cube simulator for solution verification
import py222


# ============================================================================
# Configuration
# ============================================================================

class Config:
    # Data
    data_path = "data/cube-2-by-2"
    max_train_samples = None  # None = use all
    max_val_samples = None
    max_test_samples = None
    
    # Model architecture (HRM FC)
    seq_len = 24  # 24 sticker positions
    one_hot_dim = 6  # 6 colors per position (simple color-based encoding)
    hidden_size = 512
    H_layers = 4  # Reduced from 8 for faster convergence
    L_layers = 4  # Reduced from 8 for faster convergence
    H_cycles = 2
    L_cycles = 2
    expansion = 2.0
    
    num_actions = 9  # U, U', U2, R, R', R2, F, F', F2
    max_solution_len = 11  # Full sequence prediction (God's number for 2x2)
    
    # ACT config (set to 1 for no adaptive computation)
    halt_max_steps = 11  # Disable ACT for now (just single forward pass)
    halt_exploration_prob = 0.1
    
    # Training
    batch_size = 256
    epochs = 50
    learning_rate = 3e-3  # Increased from 1e-3
    weight_decay = 1e-4
    
    # Logging and checkpoints
    project_name = "2x2-cube-policy-hrm-fc"
    checkpoint_dir = "checkpoints/2x2-policy-hrm-fc"
    checkpoint_every = 1
    log_gradients = True
    log_gradients_freq = 100
    
    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Random seed
    seed = 42


# ============================================================================
# Dataset
# ============================================================================

class CubePolicyDataset(Dataset):
    """Dataset for 2x2 cube policy prediction with simple color-based one-hot encoding."""
    
    def __init__(self, data_path: str, split: str = "train", max_samples: int = None):
        self.data_path = Path(data_path) / split
        
        # Load data
        self.inputs = np.load(self.data_path / "all__inputs.npy")
        self.labels = np.load(self.data_path / "all__labels.npy")
        
        # Subsample if requested
        total_samples = len(self.inputs)
        if max_samples is not None and max_samples < total_samples:
            rng = np.random.RandomState(42)
            indices = rng.choice(total_samples, size=max_samples, replace=False)
            indices.sort()
            self.inputs = self.inputs[indices]
            self.labels = self.labels[indices]
            print(f"Subsampled {split} set: {total_samples} -> {max_samples} examples")
        
        # Load metadata
        with open(self.data_path / "dataset.json", "r") as f:
            self.metadata = json.load(f)
        
        print(f"Loaded {split} set: {len(self.inputs)} examples")
        print(f"  Input shape: {self.inputs.shape}")
        print(f"  Label shape: {self.labels.shape}")
    
    def __len__(self):
        return len(self.inputs)
    
    def _encode_cube_state(self, state: np.ndarray) -> np.ndarray:
        """
        Encode cube state using simple color-based one-hot encoding.
        
        Each of 24 sticker positions gets a 6-dim one-hot vector for its color.
        This preserves all information and is proven to work for cube learning.
        
        Args:
            state: (24,) array with colors 1-6 (0 is pad)
        
        Returns:
            (24, 6) one-hot encoded state
        """
        # Convert from stored values (1-6) to actual colors (0-5)
        colors = state - 1
        
        # Simple one-hot encoding: 24 positions x 6 colors
        one_hot = np.zeros((24, 6), dtype=np.float32)
        for i, color in enumerate(colors):
            if 0 <= color < 6:
                one_hot[i, color] = 1.0
        
        return one_hot
    
    def __getitem__(self, idx):
        # Get cube state and encode
        state = self.inputs[idx]
        one_hot = self._encode_cube_state(state)
        
        # Get label (move sequence)
        label = self.labels[idx].astype(np.int64)
        
        # Also return raw state for solution verification (colors 0-5)
        raw_state = (state - 1).astype(np.int64)  # Convert from 1-6 to 0-5
        
        return {
            "inputs": torch.from_numpy(one_hot),
            "labels": torch.from_numpy(label),
            "raw_state": torch.from_numpy(raw_state),  # For solution verification
            "puzzle_identifiers": torch.tensor([0], dtype=torch.int64),
        }


def collate_fn(batch):
    """Custom collate function to handle dict batches."""
    return {
        "inputs": torch.stack([item["inputs"] for item in batch]),
        "labels": torch.stack([item["labels"] for item in batch]),
        "raw_state": torch.stack([item["raw_state"] for item in batch]),
        "puzzle_identifiers": torch.stack([item["puzzle_identifiers"] for item in batch]),
    }


# ============================================================================
# Cube Solution Verification
# ============================================================================

def apply_moves_and_check_solved(raw_states: np.ndarray, move_sequences: np.ndarray) -> np.ndarray:
    """
    Apply move sequences to cube states and check if they result in solved cubes.
    
    Args:
        raw_states: (batch, 24) cube states with colors 0-5
        move_sequences: (batch, seq_len) move indices 0-9 (0=pad, 1-9=moves)
    
    Returns:
        (batch,) boolean array indicating if each cube is solved after applying moves
    """
    batch_size = raw_states.shape[0]
    solved = np.zeros(batch_size, dtype=bool)
    
    for i in range(batch_size):
        state = raw_states[i].copy()
        moves = move_sequences[i]
        
        # Apply each move (skip padding = 0)
        for move_idx in moves:
            if move_idx == 0:  # Padding, stop
                break
            # Convert from 1-indexed (dataset) to 0-indexed (py222)
            # Moves 1-9 correspond to U, U', U2, R, R', R2, F, F', F2 (indices 0-8)
            actual_move = int(move_idx) - 1
            if 0 <= actual_move < 9:
                state = py222.doMove(state, actual_move)
        
        # Check if solved
        solved[i] = py222.isSolved(state)
    
    return solved


def verify_solutions_batch(raw_states: torch.Tensor, pred_moves: torch.Tensor) -> dict:
    """
    Verify if predicted move sequences actually solve the cubes.
    
    Args:
        raw_states: (batch, 24) cube states
        pred_moves: (batch, seq_len) predicted move indices
    
    Returns:
        dict with solve rate and other metrics
    """
    raw_states_np = raw_states.cpu().numpy()
    pred_moves_np = pred_moves.cpu().numpy()
    
    solved = apply_moves_and_check_solved(raw_states_np, pred_moves_np)
    
    return {
        "solve_rate": solved.mean(),
        "num_solved": solved.sum(),
        "total": len(solved),
    }


# ============================================================================
# Training Functions
# ============================================================================

def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_checkpoint(model, optimizer, epoch, step, train_loss, val_loss, path):
    """Save model checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss,
        'config': model.config.model_dump() if hasattr(model.config, 'model_dump') else dict(model.config),
    }
    torch.save(checkpoint, path)
    print(f"Checkpoint saved: {path}")


def compute_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> dict:
    """
    Compute accuracy metrics for policy prediction.
    
    Args:
        logits: (batch, seq_len, num_actions+1) predicted action logits
        labels: (batch, seq_len) ground truth actions (0 = PAD/ignore)
    
    Returns:
        dict with accuracy metrics
    """
    # Get predictions
    preds = logits.argmax(dim=-1)  # (batch, seq_len)
    
    # Mask for non-padding positions (labels > 0)
    mask = labels > 0
    
    # Token-level accuracy (only on non-pad positions)
    correct_tokens = ((preds == labels) & mask).sum()
    total_tokens = mask.sum()
    token_acc = (correct_tokens / total_tokens).item() if total_tokens > 0 else 0.0
    
    # First-token accuracy (most important for greedy decoding)
    first_token_correct = (preds[:, 0] == labels[:, 0]).sum()
    first_token_acc = (first_token_correct / labels.shape[0]).item()
    
    # Sequence-level accuracy (all tokens correct)
    # A sequence is correct if all non-pad positions match
    seq_correct = ((preds == labels) | ~mask).all(dim=1)
    seq_acc = seq_correct.float().mean().item()
    
    return {
        "token_acc": token_acc,
        "first_token_acc": first_token_acc,
        "seq_acc": seq_acc,
    }


def train_epoch(model, train_loader, optimizer, criterion, device, epoch, config):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    total_token_acc = 0.0
    total_first_token_acc = 0.0
    num_batches = 0
    global_step = epoch * len(train_loader)
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.epochs} [Train]")
    
    for batch_idx, batch in enumerate(pbar):
        # Move batch to device
        batch = {k: v.to(device) for k, v in batch.items()}
        
        # Initialize carry
        carry = model.initial_carry(batch)
        
        # Forward pass
        optimizer.zero_grad()
        carry, outputs = model(carry, batch)
        
        logits = outputs["logits"]  # (batch, max_solution_len, num_actions+1)
        labels = batch["labels"][:, :logits.shape[1]]  # Truncate labels to match output length
        
        # Compute loss (cross-entropy, ignore padding = 0)
        loss = criterion(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1))
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Compute metrics
        with torch.no_grad():
            acc_metrics = compute_accuracy(logits, labels)
        
        total_loss += loss.item()
        total_token_acc += acc_metrics["token_acc"]
        total_first_token_acc += acc_metrics["first_token_acc"]
        num_batches += 1
        global_step += 1
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'tok_acc': f'{acc_metrics["token_acc"]:.4f}',
            'first_acc': f'{acc_metrics["first_token_acc"]:.4f}'
        })
        
        # Log to wandb
        wandb.log({
            'train/batch_loss': loss.item(),
            'train/batch_token_acc': acc_metrics["token_acc"],
            'train/batch_first_token_acc': acc_metrics["first_token_acc"],
            'train/step': global_step,
        }, step=global_step)
        
        # Log gradients periodically
        if config.log_gradients and batch_idx % config.log_gradients_freq == 0:
            grad_norms = {}
            for name, param in model.named_parameters():
                if param.grad is not None:
                    grad_norms[f'gradients/{name}'] = param.grad.norm().item()
            wandb.log(grad_norms, step=global_step)
    
    avg_loss = total_loss / num_batches
    avg_token_acc = total_token_acc / num_batches
    avg_first_token_acc = total_first_token_acc / num_batches
    
    return avg_loss, avg_token_acc, avg_first_token_acc, global_step


def validate(model, val_loader, criterion, device, config, verify_solutions=True):
    """Validate the model with optional solution verification."""
    model.eval()
    total_loss = 0.0
    total_token_acc = 0.0
    total_first_token_acc = 0.0
    total_seq_acc = 0.0
    total_solved = 0
    total_samples = 0
    num_batches = 0
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc="Validating")
        
        for batch in pbar:
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # Initialize carry
            carry = model.initial_carry(batch)
            
            # Forward pass
            carry, outputs = model(carry, batch)
            
            logits = outputs["logits"]
            labels = batch["labels"][:, :logits.shape[1]]
            
            # Compute loss
            loss = criterion(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1))
            
            # Compute metrics
            acc_metrics = compute_accuracy(logits, labels)
            
            # Verify solutions (check if predicted moves actually solve the cube)
            if verify_solutions and "raw_state" in batch:
                preds = logits.argmax(dim=-1)
                solve_metrics = verify_solutions_batch(batch["raw_state"], preds)
                total_solved += solve_metrics["num_solved"]
                total_samples += solve_metrics["total"]
            
            total_loss += loss.item()
            total_token_acc += acc_metrics["token_acc"]
            total_first_token_acc += acc_metrics["first_token_acc"]
            total_seq_acc += acc_metrics["seq_acc"]
            num_batches += 1
            
            solve_rate = total_solved / total_samples if total_samples > 0 else 0.0
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'tok_acc': f'{acc_metrics["token_acc"]:.4f}',
                'solve': f'{solve_rate:.4f}'
            })
    
    result = {
        'loss': total_loss / num_batches,
        'token_acc': total_token_acc / num_batches,
        'first_token_acc': total_first_token_acc / num_batches,
        'seq_acc': total_seq_acc / num_batches,
    }
    
    if total_samples > 0:
        result['solve_rate'] = total_solved / total_samples
    
    return result


def main():
    """Main training function."""
    config = Config()
    
    # Set seed
    set_seed(config.seed)
    
    # Create checkpoint directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_checkpoint_dir = Path(config.checkpoint_dir) / timestamp
    run_checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize wandb
    run = wandb.init(
        project=config.project_name,
        config={
            'hidden_size': config.hidden_size,
            'H_layers': config.H_layers,
            'L_layers': config.L_layers,
            'H_cycles': config.H_cycles,
            'L_cycles': config.L_cycles,
            'expansion': config.expansion,
            'num_actions': config.num_actions,
            'max_solution_len': config.max_solution_len,
            'halt_max_steps': config.halt_max_steps,
            'batch_size': config.batch_size,
            'epochs': config.epochs,
            'learning_rate': config.learning_rate,
            'weight_decay': config.weight_decay,
            'seed': config.seed,
        }
    )
    
    print(f"Wandb run: {run.name}")
    print(f"Device: {config.device}")
    print(f"Checkpoints will be saved to: {run_checkpoint_dir}")
    
    # Create datasets
    print("\nLoading datasets...")
    train_dataset = CubePolicyDataset(config.data_path, split="train", max_samples=config.max_train_samples)
    val_dataset = CubePolicyDataset(config.data_path, split="val", max_samples=config.max_val_samples)
    test_dataset = CubePolicyDataset(config.data_path, split="test", max_samples=config.max_test_samples)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    
    # Create model
    print("\nCreating model...")
    model = create_hrm_fc_policy(
        batch_size=config.batch_size,
        seq_len=config.seq_len,
        one_hot_dim=config.one_hot_dim,
        hidden_size=config.hidden_size,
        H_layers=config.H_layers,
        L_layers=config.L_layers,
        H_cycles=config.H_cycles,
        L_cycles=config.L_cycles,
        expansion=config.expansion,
        num_actions=config.num_actions,
        max_solution_len=config.max_solution_len,
        halt_max_steps=config.halt_max_steps,
        halt_exploration_prob=config.halt_exploration_prob,
    )
    model = model.to(config.device)
    
    # Print model summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Log model architecture to wandb
    wandb.watch(model, log='all', log_freq=config.log_gradients_freq)
    
    # Create optimizer and loss function
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    # CrossEntropy WITHOUT ignore_index - we need to train on padding positions too!
    # The model must learn to output 0 (PAD) at the end of sequences, otherwise
    # apply_moves_and_check_solved will apply extra moves and fail to solve.
    criterion = nn.CrossEntropyLoss()
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.epochs,
        eta_min=config.learning_rate * 0.01
    )
    
    # Training loop
    print("\nStarting training...")
    best_val_loss = float('inf')
    global_step = 0
    
    for epoch in range(config.epochs):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch + 1}/{config.epochs}")
        print(f"{'='*60}")
        
        # Train
        train_loss, train_token_acc, train_first_acc, global_step = train_epoch(
            model, train_loader, optimizer, criterion,
            config.device, epoch, config
        )
        
        # Validate
        val_metrics = validate(model, val_loader, criterion, config.device, config)
        
        # Update learning rate
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        
        # Log epoch metrics
        wandb.log({
            'epoch': epoch + 1,
            'train/epoch_loss': train_loss,
            'train/epoch_token_acc': train_token_acc,
            'train/epoch_first_token_acc': train_first_acc,
            'val/loss': val_metrics['loss'],
            'val/token_acc': val_metrics['token_acc'],
            'val/first_token_acc': val_metrics['first_token_acc'],
            'val/seq_acc': val_metrics['seq_acc'],
            'val/solve_rate': val_metrics.get('solve_rate', 0.0),
            'learning_rate': current_lr,
        }, step=global_step)
        
        solve_rate_str = f", Solve Rate: {val_metrics.get('solve_rate', 0.0):.4f}" if 'solve_rate' in val_metrics else ""
        print(f"\nEpoch {epoch + 1} Summary:")
        print(f"  Train Loss: {train_loss:.4f}, Token Acc: {train_token_acc:.4f}, First Acc: {train_first_acc:.4f}")
        print(f"  Val Loss: {val_metrics['loss']:.4f}, Token Acc: {val_metrics['token_acc']:.4f}")
        print(f"  Val First Token Acc: {val_metrics['first_token_acc']:.4f}, Seq Acc: {val_metrics['seq_acc']:.4f}{solve_rate_str}")
        print(f"  Learning Rate: {current_lr:.6f}")
        
        # Save best model
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            best_path = run_checkpoint_dir / "best_model.pt"
            save_checkpoint(
                model, optimizer, epoch, global_step,
                train_loss, val_metrics['loss'], best_path
            )
            wandb.log({'best_val_loss': best_val_loss}, step=global_step)
            print(f"  New best model saved! Val Loss: {best_val_loss:.4f}")
        
        # Save periodic checkpoint
        if (epoch + 1) % config.checkpoint_every == 0:
            checkpoint_path = run_checkpoint_dir / f"checkpoint_epoch_{epoch + 1}.pt"
            save_checkpoint(
                model, optimizer, epoch, global_step,
                train_loss, val_metrics['loss'], checkpoint_path
            )
    
    # Save final model
    final_path = run_checkpoint_dir / "final_model.pt"
    save_checkpoint(
        model, optimizer, config.epochs - 1, global_step,
        train_loss, val_metrics['loss'], final_path
    )
    
    # Final evaluation on test set
    print("\n" + "="*60)
    print("Final Evaluation on Test Set")
    print("="*60)
    
    test_metrics = validate(model, test_loader, criterion, config.device, config)
    
    print(f"Test Loss: {test_metrics['loss']:.4f}")
    print(f"Test Token Accuracy: {test_metrics['token_acc']:.4f}")
    print(f"Test First Token Accuracy: {test_metrics['first_token_acc']:.4f}")
    print(f"Test Sequence Accuracy: {test_metrics['seq_acc']:.4f}")
    if 'solve_rate' in test_metrics:
        print(f"Test Solve Rate: {test_metrics['solve_rate']:.4f}")
    
    # Log final test metrics
    wandb.log({
        'test/loss': test_metrics['loss'],
        'test/token_acc': test_metrics['token_acc'],
        'test/first_token_acc': test_metrics['first_token_acc'],
        'test/seq_acc': test_metrics['seq_acc'],
        'test/solve_rate': test_metrics.get('solve_rate', 0.0),
    })
    
    # Log artifact
    artifact = wandb.Artifact(
        name=f"model-{run.name}",
        type="model",
        description="Trained 2x2 cube policy HRM FC model"
    )
    artifact.add_dir(str(run_checkpoint_dir))
    run.log_artifact(artifact)
    
    wandb.finish()
    
    print("\nTraining complete!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Checkpoints saved to: {run_checkpoint_dir}")


if __name__ == "__main__":
    main()
