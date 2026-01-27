"""
Test different regularization techniques one by one.
Each run saves plots with numbered filenames so you can compare.
"""
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from pathlib import Path
import json
import time
import sys
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime

sys.path.insert(0, str(Path.cwd()))


class SimpleMLP(nn.Module):
    """Simple 4-layer MLP with one-hot encoding"""
    def __init__(self, input_dim=144, hidden_dim=256, output_dim=1, dropout=0.0):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
        
    def forward(self, x):
        # x: (batch, 24) state indices
        # Convert to one-hot
        x_onehot = torch.nn.functional.one_hot(x.long(), num_classes=6).float()
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
            
        x = self.fc4(x)
        return x.squeeze(-1)


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
        'labels': labels.squeeze(),
    }


def train_model(
    model,
    model_name,
    train_data,
    test_data,
    epochs=20,
    lr=1e-4,
    weight_decay=0.0,
    batch_size=64,
    device='cpu',
    loss_type="mse",
):
    """Train a model and return history"""
    
    train_dataset = TensorDataset(
        torch.from_numpy(train_data['inputs']),
        torch.from_numpy(train_data['labels'].astype(np.float32))
    )
    test_dataset = TensorDataset(
        torch.from_numpy(test_data['inputs']),
        torch.from_numpy(test_data['labels'].astype(np.float32))
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss() if loss_type == "mse" else nn.SmoothL1Loss()
    
    history = {'train_loss': [], 'test_loss': [], 'test_mae': [], 'train_mae': []}
    
    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0
        train_mae = 0
        for inputs, targets in tqdm(train_loader, desc=f"{model_name} Epoch {epoch+1}/{epochs}", leave=False):
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_mae += torch.abs(outputs - targets).mean().item()
        
        train_loss /= len(train_loader)
        train_mae /= len(train_loader)
        
        # Evaluate
        model.eval()
        test_loss = 0
        test_mae = 0
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                test_loss += criterion(outputs, targets).item()
                test_mae += torch.abs(outputs - targets).mean().item()
        
        test_loss /= len(test_loader)
        test_mae /= len(test_loader)
        
        history['train_loss'].append(train_loss)
        history['test_loss'].append(test_loss)
        history['train_mae'].append(train_mae)
        history['test_mae'].append(test_mae)
        
        if (epoch + 1) % 5 == 0:
            gap = test_loss - train_loss
            print(f"  Epoch {epoch+1}: Train={train_loss:.4f}, Test={test_loss:.4f}, Gap={gap:.4f}")
    
    return history


def get_next_run_number():
    """Find the next available run number"""
    i = 1
    while Path(f"results_run_{i}.png").exists():
        i += 1
    return i


def save_plot(history, config_name, run_num):
    """Save plot with run number"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Loss curves
    axes[0].plot(history['train_loss'], label='Train Loss', marker='o', markersize=3)
    axes[0].plot(history['test_loss'], label='Test Loss', marker='s', markersize=3)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('MSE Loss')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    axes[0].set_title('Loss Curves')
    
    # MAE curves
    axes[1].plot(history['train_mae'], label='Train MAE', marker='o', markersize=3)
    axes[1].plot(history['test_mae'], label='Test MAE', marker='s', markersize=3)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('MAE (moves)')
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    axes[1].set_title('MAE Curves')
    
    # Gap over time
    gap = np.array(history['test_loss']) - np.array(history['train_loss'])
    axes[2].plot(gap, marker='o', markersize=3, color='red')
    axes[2].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Test Loss - Train Loss')
    axes[2].grid(alpha=0.3)
    axes[2].set_title('Overfitting Gap (Lower is Better)')
    
    plt.suptitle(f"Run #{run_num}: {config_name}", fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    filename = f"results_run_{run_num}.png"
    plt.savefig(filename, dpi=100)
    print(f"✅ Saved: {filename}")
    plt.close()


if __name__ == "__main__":
    device = 'cpu'
    data_dir = Path("data/cube-2-by-2-heuristic")
    
    # Load data
    print("Loading dataset...")
    train_data = load_dataset_split(data_dir, "train")
    test_data = load_dataset_split(data_dir, "test")
    print(f"Train: {len(train_data['inputs'])}, Test: {len(test_data['inputs'])}\n")
    
    # Define experiments
    experiments = [
        {
            "name": "Baseline (hidden=256, no reg)",
            "hidden_dim": 256,
            "dropout": 0.0,
            "weight_decay": 0.0,
            "batch_size": 64,
            "loss_type": "mse",
        },
        {
            "name": "Dropout (p=0.05)",
            "hidden_dim": 256,
            "dropout": 0.05,
            "weight_decay": 0.0,
            "batch_size": 64,
            "loss_type": "mse",
        },
        {
            "name": "Dropout (p=0.10)",
            "hidden_dim": 256,
            "dropout": 0.10,
            "weight_decay": 0.0,
            "batch_size": 64,
            "loss_type": "mse",
        },
        {
            "name": "L2 Regularization (weight_decay=5e-5)",
            "hidden_dim": 256,
            "dropout": 0.0,
            "weight_decay": 5e-5,
            "batch_size": 64,
            "loss_type": "mse",
        },
        {
            "name": "L2 Regularization (weight_decay=1e-4)",
            "hidden_dim": 256,
            "dropout": 0.0,
            "weight_decay": 1e-4,
            "batch_size": 64,
            "loss_type": "mse",
        },
        {
            "name": "L2 + Dropout (wd=5e-5, p=0.05)",
            "hidden_dim": 256,
            "dropout": 0.05,
            "weight_decay": 5e-5,
            "batch_size": 64,
            "loss_type": "mse",
        },
        {
            "name": "Smaller Model (hidden=128)",
            "hidden_dim": 128,
            "dropout": 0.0,
            "weight_decay": 0.0,
            "batch_size": 64,
            "loss_type": "mse",
        },
        {
            "name": "SmoothL1Loss + Dropout (p=0.05)",
            "hidden_dim": 256,
            "dropout": 0.05,
            "weight_decay": 0.0,
            "batch_size": 64,
            "loss_type": "huber",
        },
        {
            "name": "SmoothL1Loss + L2 (wd=5e-5)",
            "hidden_dim": 256,
            "dropout": 0.0,
            "weight_decay": 5e-5,
            "batch_size": 64,
            "loss_type": "huber",
        },
    ]
    
    # Run all experiments
    run_num = get_next_run_number()
    results_summary = []
    
    for exp in experiments:
        print("="*70)
        print(f"Run #{run_num}: {exp['name']}")
        print("="*70)
        
        # Create model
        model = SimpleMLP(
            hidden_dim=exp['hidden_dim'],
            dropout=exp['dropout']
        )
        
        # Train
        history = train_model(
            model,
            exp['name'],
            train_data,
            test_data,
            epochs=20,
            lr=1e-4,
            weight_decay=exp['weight_decay'],
            batch_size=exp['batch_size'],
            device=device,
            loss_type=exp.get('loss_type', 'mse')
        )
        
        # Calculate final metrics
        final_train_loss = history['train_loss'][-1]
        final_test_loss = history['test_loss'][-1]
        final_gap = final_test_loss - final_train_loss
        final_test_mae = history['test_mae'][-1]
        
        results_summary.append({
            'run': run_num,
            'name': exp['name'],
            'train_loss': final_train_loss,
            'test_loss': final_test_loss,
            'gap': final_gap,
            'test_mae': final_test_mae,
        })
        
        # Save plot
        save_plot(history, exp['name'], run_num)
        
        print(f"\n📊 Final Results:")
        print(f"   Train Loss: {final_train_loss:.4f}")
        print(f"   Test Loss:  {final_test_loss:.4f}")
        print(f"   Gap:        {final_gap:.4f} ({'GOOD' if final_gap < 0.2 else 'BAD OVERFITTING'})")
        print(f"   Test MAE:   {final_test_mae:.4f}\n")
        
        run_num += 1
    
    # Print summary comparison
    print("\n" + "="*70)
    print("SUMMARY: ALL EXPERIMENTS")
    print("="*70)
    print(f"{'Run':<5} {'Train Loss':<12} {'Test Loss':<12} {'Gap':<10} {'Test MAE':<10} Config")
    print("-"*70)
    for r in results_summary:
        gap_str = f"{r['gap']:.4f}"
        if r['gap'] < 0.15:
            gap_indicator = "✅"
        elif r['gap'] < 0.3:
            gap_indicator = "⚠️"
        else:
            gap_indicator = "❌"
        
        print(f"{r['run']:<5} {r['train_loss']:<12.4f} {r['test_loss']:<12.4f} {gap_str:<10} {r['test_mae']:<10.4f} {gap_indicator} {r['name']}")
    
    print("\n✅ All runs complete! Check results_run_*.png for visualizations")
    print("   Gap < 0.15: ✅ Good generalization")
    print("   Gap 0.15-0.3: ⚠️ Moderate overfitting")
    print("   Gap > 0.3: ❌ Severe overfitting")
