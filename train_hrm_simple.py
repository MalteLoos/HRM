"""
Simple HRM training script for CPU.
Similar to notebook approach but using the HRM model with unbiased dataset.
"""
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from pathlib import Path
import json
from tqdm import tqdm
import sys
import matplotlib.pyplot as plt

# Add project root
sys.path.insert(0, str(Path.cwd()))
from models.hrm.hrm_act_v1 import (
    HierarchicalReasoningModel_ACTV1, 
    HierarchicalReasoningModel_ACTV1Config
)
import py222


def load_dataset_split(data_dir: Path, split: str):
    """Load a dataset split (train/test/val)"""
    split_dir = data_dir / split
    if not split_dir.exists():
        raise FileNotFoundError(f"Split {split} not found in {data_dir}")

    # Load metadata
    with open(split_dir / "dataset.json", "r") as f:
        metadata = json.load(f)

    # Load arrays
    inputs = np.load(split_dir / "all__inputs.npy")
    labels = np.load(split_dir / "all__labels.npy")
    puzzle_identifiers = np.load(split_dir / "all__puzzle_identifiers.npy")

    return {
        'metadata': metadata,
        'inputs': inputs,
        'labels': labels.squeeze(),  # Remove extra dimension
        'puzzle_identifiers': puzzle_identifiers,
    }


def create_hrm_model(config_dict: dict, device='cpu'):
    """Create HRM model from config"""
    model = HierarchicalReasoningModel_ACTV1(config_dict=config_dict)
    return model.to(device)


def train_hrm(
    model,
    train_data,
    test_data,
    epochs=20,
    lr=1e-4,
    batch_size=64,
    device='cpu',
    save_path=None,
    loss_type='mse',
    weight_decay=0.0
):
    """Train HRM model on CPU. loss_type: 'mse' or 'huber'"""
    
    # Prepare data
    train_dataset = TensorDataset(
        torch.from_numpy(train_data['inputs']),
        torch.from_numpy(train_data['puzzle_identifiers']),
        torch.from_numpy(train_data['labels'].astype(np.float32))
    )
    test_dataset = TensorDataset(
        torch.from_numpy(test_data['inputs']),
        torch.from_numpy(test_data['puzzle_identifiers']),
        torch.from_numpy(test_data['labels'].astype(np.float32))
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    model = model.to(device)
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss() if loss_type == 'mse' else nn.SmoothL1Loss()
    
    history = {'train_loss': [], 'test_loss': [], 'test_mae': []}
    
    best_test_mae = float('inf')
    
    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0
        for inputs, puzzle_ids, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]", leave=False):
            inputs = inputs.to(device)
            puzzle_ids = puzzle_ids.to(device)
            targets = targets.to(device)
            
            optimizer.zero_grad()
            # Create batch dict for HRM model
            batch = {'inputs': inputs, 'puzzle_identifiers': puzzle_ids}
            carry = model.initial_carry(batch)
            carry_out, outputs = model(carry, batch)
            logits = outputs['logits'].squeeze(-1)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Evaluate
        model.eval()
        test_loss = 0
        test_mae = 0
        with torch.no_grad():
            for inputs, puzzle_ids, targets in test_loader:
                inputs = inputs.to(device)
                puzzle_ids = puzzle_ids.to(device)
                targets = targets.to(device)
                batch = {'inputs': inputs, 'puzzle_identifiers': puzzle_ids}
                carry = model.initial_carry(batch)
                carry_out, outputs = model(carry, batch)
                logits = outputs['logits'].squeeze(-1)
                test_loss += criterion(logits, targets).item()
                test_mae += torch.abs(logits - targets).mean().item()
        
        test_loss /= len(test_loader)
        test_mae /= len(test_loader)
        
        history['train_loss'].append(train_loss)
        history['test_loss'].append(test_loss)
        history['test_mae'].append(test_mae)
        
        print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Test Loss={test_loss:.4f}, Test MAE={test_mae:.4f}")
        
        # Save best model
        if test_mae < best_test_mae:
            best_test_mae = test_mae
            if save_path:
                torch.save(model.state_dict(), save_path)
                print(f"  → Saved best model (MAE: {test_mae:.4f})")
    
    return model, history


if __name__ == "__main__":
    # Configuration
    device = 'cpu'  # Change to 'cuda' if you have GPU
    data_dir = Path("data/cube-2-by-2-heuristic")
    
    print(f"Device: {device}")
    print(f"Data directory: {data_dir}")
    
    if not data_dir.exists():
        raise FileNotFoundError(f"Data not found at {data_dir}")
    
    # Load data
    print("\nLoading dataset...")
    train_data = load_dataset_split(data_dir, "train")
    test_data = load_dataset_split(data_dir, "test")
    
    print(f"Train samples: {len(train_data['inputs'])}")
    print(f"Test samples: {len(test_data['inputs'])}")
    
    # HRM configuration (from your configs)
    hrm_config = {
        "batch_size": 64,
        "seq_len": 24,
        "puzzle_emb_ndim": 0,
        "num_puzzle_identifiers": 1,
        "vocab_size": 6,
        "H_cycles": 3,
        "L_cycles": 3,
        "H_layers": 2,
        "L_layers": 2,
        "hidden_size": 256,
        "expansion": 4,
        "num_heads": 4,
        "pos_encodings": "rope",
        "halt_max_steps": 10,
        "halt_exploration_prob": 0.1,
        "forward_dtype": "float32",  # Use float32 on CPU
    }
    
    print("\nCreating HRM model...")
    model = create_hrm_model(hrm_config, device=device)
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")
    
    # Train with winning regularization config
    print("\nStarting training with Huber Loss + L2 (weight_decay=5e-5)...")
    trained_model, history = train_hrm(
        model,
        train_data,
        test_data,
        epochs=20,
        lr=1e-4,
        batch_size=16,  # Smaller batch for CPU
        device=device,
        save_path="checkpoints/hrm_best.pt",
        loss_type='huber',  # SmoothL1Loss (Huber) - the winning config!
        weight_decay=5e-5   # Light L2 regularization
    )
    
    # Plot results
    print("\nPlotting results...")
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['test_loss'], label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('SmoothL1 Loss (Huber)')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.title('Training Progress')
    
    plt.subplot(1, 2, 2)
    plt.plot(history['test_mae'])
    plt.xlabel('Epoch')
    plt.ylabel('Mean Absolute Error')
    plt.grid(alpha=0.3)
    plt.title('Test MAE (Lower is Better)')
    
    plt.tight_layout()
    
    # Find next available number
    i = 1
    while Path(f'hrm_training_results_{i}.png').exists():
        i += 1
    filename = f'hrm_training_results_{i}.png'
    
    plt.savefig(filename, dpi=100)
    print(f"Results saved to {filename}")
    
    print(f"\n✅ Final Test MAE: {history['test_mae'][-1]:.4f} moves")
