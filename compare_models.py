"""
Compare Simple MLP vs HRM on the same dataset.
Trains both models and compares:
- Training speed
- Accuracy (MAE)
- Memory usage
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

sys.path.insert(0, str(Path.cwd()))


class SimpleMLP(nn.Module):
    """Simple 4-layer MLP with one-hot encoding (best baseline)"""
    def __init__(self, input_dim=144, hidden_dim=256, output_dim=1):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        # x: (batch, 24) state indices
        # Convert to one-hot
        x_onehot = torch.nn.functional.one_hot(x.long(), num_classes=6).float()
        x_onehot = x_onehot.reshape(x_onehot.shape[0], -1)  # (batch, 144)
        
        x = torch.nn.functional.relu(self.fc1(x_onehot))
        x = torch.nn.functional.relu(self.fc2(x))
        x = torch.nn.functional.relu(self.fc3(x))
        x = self.fc4(x)  # Output: distance estimate
        return x.squeeze(-1)


def load_dataset_split(data_dir: Path, split: str):
    """Load a dataset split (train/test/val)"""
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
    batch_size=32,
    device='cpu',
):
    """Train a model and return history + timing info"""
    
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
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    history = {'train_loss': [], 'test_loss': [], 'test_mae': []}
    timings = []
    
    start_time = time.time()
    
    for epoch in range(epochs):
        epoch_start = time.time()
        
        # Train
        model.train()
        train_loss = 0
        for inputs, targets in tqdm(train_loader, desc=f"{model_name} Epoch {epoch+1}/{epochs}", leave=False):
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
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
        
        epoch_time = time.time() - epoch_start
        timings.append(epoch_time)
        
        history['train_loss'].append(train_loss)
        history['test_loss'].append(test_loss)
        history['test_mae'].append(test_mae)
        
        print(f"  {model_name} Epoch {epoch+1}: Train={train_loss:.4f}, Test={test_loss:.4f}, MAE={test_mae:.4f}, Time={epoch_time:.1f}s")
    
    total_time = time.time() - start_time
    
    return history, {
        'total_time': total_time,
        'avg_epoch_time': np.mean(timings),
        'num_params': sum(p.numel() for p in model.parameters()),
    }


if __name__ == "__main__":
    device = 'cpu'
    data_dir = Path("data/cube-2-by-2-heuristic")
    
    print(f"Device: {device}")
    
    # Load data
    print("\nLoading dataset...")
    train_data = load_dataset_split(data_dir, "train")
    test_data = load_dataset_split(data_dir, "test")
    
    print(f"Train: {len(train_data['inputs'])}, Test: {len(test_data['inputs'])}")
    
    # Create models
    print("\nCreating models...")
    mlp = SimpleMLP()
    
    # Train MLP
    print("\n" + "="*60)
    print("Training Simple MLP")
    print("="*60)
    mlp_history, mlp_timing = train_model(
        mlp, "SimpleMLP",
        train_data, test_data,
        epochs=20, lr=1e-4, batch_size=64, device=device
    )
    
    # Print comparison
    print("\n" + "="*60)
    print("COMPARISON RESULTS")
    print("="*60)
    print(f"\nSimple MLP:")
    print(f"  Parameters: {mlp_timing['num_params']:,}")
    print(f"  Total time: {mlp_timing['total_time']:.1f}s ({mlp_timing['total_time']/60:.1f} min)")
    print(f"  Avg epoch time: {mlp_timing['avg_epoch_time']:.1f}s")
    print(f"  Final Test MAE: {mlp_history['test_mae'][-1]:.4f} moves")
    
    # Plot comparison
    plt.figure(figsize=(14, 4))
    
    plt.subplot(1, 3, 1)
    plt.plot(mlp_history['train_loss'], label='MLP Train', marker='o')
    plt.plot(mlp_history['test_loss'], label='MLP Test', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.title('Loss Curves')
    
    plt.subplot(1, 3, 2)
    plt.plot(mlp_history['test_mae'], label='MLP', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('MAE (moves)')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.title('Test MAE')
    
    plt.subplot(1, 3, 3)
    plt.bar(['MLP'], [mlp_timing['avg_epoch_time']], alpha=0.7)
    plt.ylabel('Time (seconds)')
    plt.title('Avg Epoch Time')
    plt.grid(alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    # Find next available number
    i = 1
    while Path(f'mlp_vs_hrm_comparison_{i}.png').exists():
        i += 1
    filename = f'mlp_vs_hrm_comparison_{i}.png'
    
    plt.savefig(filename, dpi=100)
    print(f"\nComparison plot saved to {filename}")
