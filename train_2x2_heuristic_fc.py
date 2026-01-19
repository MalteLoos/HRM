"""
Training script for 2x2 Cube Heuristic prediction using a simple fully connected network.
4 layers, 512 units per layer, ReLU activation.
Uses wandb for logging metrics and gradients, and saves checkpoints.
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


# ============================================================================
# Configuration
# ============================================================================

class Config:
    # Data
    data_path = "data/cube-2-by-2-heuristic"
    max_train_samples = 1000  # None = use all, or int to subsample (e.g., 10000)
    max_val_samples = 100    # None = use all
    max_test_samples = None   # None = use all
    
    # Model architecture
    input_size = 24 * 6  # 24 positions, one-hot encoded with 6 colors
    hidden_size = 512
    num_layers = 4
    output_size = 1  # Regression: distance to solved
    
    # Training
    batch_size = 256
    epochs = 20
    learning_rate = 1e-3
    weight_decay = 1e-4
    
    # Logging and checkpoints
    project_name = "2x2-cube-heuristic-fc"
    checkpoint_dir = "checkpoints/2x2-heuristic-fc"
    checkpoint_every = 5  # Save checkpoint every N epochs
    log_gradients = True
    log_gradients_freq = 100  # Log gradients every N steps
    
    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Random seed
    seed = 42


# ============================================================================
# Dataset
# ============================================================================

class CubeHeuristicDataset(Dataset):
    """Dataset for 2x2 cube heuristic prediction."""
    
    def __init__(self, data_path: str, split: str = "train", max_samples: int = None):
        self.data_path = Path(data_path) / split
        
        # Load data
        self.inputs = np.load(self.data_path / "all__inputs.npy")
        self.labels = np.load(self.data_path / "all__labels.npy")
        
        # Subsample if requested
        total_samples = len(self.inputs)
        if max_samples is not None and max_samples < total_samples:
            # Random subsample with fixed seed for reproducibility
            rng = np.random.RandomState(42)
            indices = rng.choice(total_samples, size=max_samples, replace=False)
            indices.sort()  # Keep order for reproducibility
            self.inputs = self.inputs[indices]
            self.labels = self.labels[indices]
            print(f"Subsampled {split} set: {total_samples} -> {max_samples} examples")
        
        # Load metadata
        with open(self.data_path / "dataset.json", "r") as f:
            self.metadata = json.load(f)
        
        print(f"Loaded {split} set: {len(self.inputs)} examples")
        print(f"  Input shape: {self.inputs.shape}")
        print(f"  Label shape: {self.labels.shape}")
        print(f"  Label range: [{self.labels.min()}, {self.labels.max()}]")
    
    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, idx):
        # Get cube state (24 values, each 0-5)
        state = self.inputs[idx]
        
        # One-hot encode: 24 positions x 6 colors = 144 features
        one_hot = np.zeros((24, 6), dtype=np.float32)
        for i, color in enumerate(state):
            one_hot[i, color] = 1.0
        one_hot = one_hot.flatten()
        
        # Get label (distance to solved)
        label = self.labels[idx].astype(np.float32)
        if label.ndim > 0:
            label = label[0]  # Extract scalar if needed
        
        return torch.from_numpy(one_hot), torch.tensor(label, dtype=torch.float32)


# ============================================================================
# Model
# ============================================================================

class FCHeuristicNet(nn.Module):
    """4-layer fully connected network with ReLU activations."""
    
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, output_size: int):
        super().__init__()
        
        layers = []
        
        # Input layer
        layers.append(nn.Linear(input_size, hidden_size))
        layers.append(nn.ReLU())
        
        # Hidden layers
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.ReLU())
        
        # Output layer
        layers.append(nn.Linear(hidden_size, output_size))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x).squeeze(-1)


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
    }
    torch.save(checkpoint, path)
    print(f"Checkpoint saved: {path}")


def load_checkpoint(model, optimizer, path):
    """Load model checkpoint."""
    checkpoint = torch.load(path, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['epoch'], checkpoint['step']


def train_epoch(model, train_loader, optimizer, criterion, device, epoch, config):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    total_mae = 0.0
    num_batches = 0
    global_step = epoch * len(train_loader)
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.epochs} [Train]")
    
    for batch_idx, (inputs, labels) in enumerate(pbar):
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Compute metrics
        with torch.no_grad():
            mae = torch.abs(outputs - labels).mean()
        
        total_loss += loss.item()
        total_mae += mae.item()
        num_batches += 1
        global_step += 1
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'mae': f'{mae.item():.4f}'
        })
        
        # Log to wandb
        wandb.log({
            'train/batch_loss': loss.item(),
            'train/batch_mae': mae.item(),
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
    avg_mae = total_mae / num_batches
    
    return avg_loss, avg_mae, global_step


def validate(model, val_loader, criterion, device, config):
    """Validate the model."""
    model.eval()
    total_loss = 0.0
    total_mae = 0.0
    total_correct_floor = 0  # Predictions rounded to nearest int match label
    num_batches = 0
    num_samples = 0
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc="Validating")
        
        for inputs, labels in pbar:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            mae = torch.abs(outputs - labels).mean()
            
            # Compute accuracy (rounded prediction matches label)
            preds_rounded = torch.round(outputs)
            correct = (preds_rounded == labels).sum().item()
            
            total_loss += loss.item()
            total_mae += mae.item()
            total_correct_floor += correct
            num_batches += 1
            num_samples += labels.size(0)
            
            all_preds.extend(outputs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'mae': f'{mae.item():.4f}'
            })
    
    avg_loss = total_loss / num_batches
    avg_mae = total_mae / num_batches
    accuracy = total_correct_floor / num_samples
    
    # Compute additional metrics
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    rmse = np.sqrt(((all_preds - all_labels) ** 2).mean())
    
    return {
        'loss': avg_loss,
        'mae': avg_mae,
        'rmse': rmse,
        'accuracy': accuracy,
        'predictions': all_preds,
        'labels': all_labels,
    }


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
            'input_size': config.input_size,
            'hidden_size': config.hidden_size,
            'num_layers': config.num_layers,
            'output_size': config.output_size,
            'batch_size': config.batch_size,
            'epochs': config.epochs,
            'learning_rate': config.learning_rate,
            'weight_decay': config.weight_decay,
            'seed': config.seed,
            'max_train_samples': config.max_train_samples,
            'max_val_samples': config.max_val_samples,
            'max_test_samples': config.max_test_samples,
        }
    )
    
    print(f"Wandb run: {run.name}")
    print(f"Device: {config.device}")
    print(f"Checkpoints will be saved to: {run_checkpoint_dir}")
    
    # Create datasets
    print("\nLoading datasets...")
    train_dataset = CubeHeuristicDataset(config.data_path, split="train", max_samples=config.max_train_samples)
    val_dataset = CubeHeuristicDataset(config.data_path, split="val", max_samples=config.max_val_samples)
    test_dataset = CubeHeuristicDataset(config.data_path, split="test", max_samples=config.max_test_samples)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.batch_size, 
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config.batch_size, 
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=config.batch_size, 
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Create model
    print("\nCreating model...")
    model = FCHeuristicNet(
        input_size=config.input_size,
        hidden_size=config.hidden_size,
        num_layers=config.num_layers,
        output_size=config.output_size
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
    criterion = nn.MSELoss()
    
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
        train_loss, train_mae, global_step = train_epoch(
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
            'train/epoch_mae': train_mae,
            'val/loss': val_metrics['loss'],
            'val/mae': val_metrics['mae'],
            'val/rmse': val_metrics['rmse'],
            'val/accuracy': val_metrics['accuracy'],
            'learning_rate': current_lr,
        }, step=global_step)
        
        print(f"\nEpoch {epoch + 1} Summary:")
        print(f"  Train Loss: {train_loss:.4f}, Train MAE: {train_mae:.4f}")
        print(f"  Val Loss: {val_metrics['loss']:.4f}, Val MAE: {val_metrics['mae']:.4f}")
        print(f"  Val RMSE: {val_metrics['rmse']:.4f}, Val Accuracy: {val_metrics['accuracy']:.4f}")
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
    print(f"Test MAE: {test_metrics['mae']:.4f}")
    print(f"Test RMSE: {test_metrics['rmse']:.4f}")
    print(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
    
    # Log final test metrics
    wandb.log({
        'test/loss': test_metrics['loss'],
        'test/mae': test_metrics['mae'],
        'test/rmse': test_metrics['rmse'],
        'test/accuracy': test_metrics['accuracy'],
    })
    
    # Create a summary table of predictions vs labels
    wandb.log({
        'test/predictions_histogram': wandb.Histogram(test_metrics['predictions']),
        'test/labels_histogram': wandb.Histogram(test_metrics['labels']),
    })
    
    # Log artifact
    artifact = wandb.Artifact(
        name=f"model-{run.name}",
        type="model",
        description="Trained 2x2 cube heuristic FC model"
    )
    artifact.add_dir(str(run_checkpoint_dir))
    run.log_artifact(artifact)
    
    wandb.finish()
    
    print("\nTraining complete!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Checkpoints saved to: {run_checkpoint_dir}")


if __name__ == "__main__":
    main()
