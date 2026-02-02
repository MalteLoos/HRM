"""
Train Simple MLP baseline with the best configuration found:
- SmoothL1Loss (Huber loss)
- L2 regularization (weight_decay=5e-5)
- 15 epochs to match HRM training
- Saves results for comparison
"""
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from pathlib import Path
import json
import time
from tqdm import tqdm


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
    train_data,
    test_data,
    epochs=15,
    lr=1e-4,
    weight_decay=5e-5,
    batch_size=64,
    device='cpu',
):
    """Train MLP with SmoothL1Loss and return history"""
    
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
    criterion = nn.SmoothL1Loss()  # Huber loss
    
    history = {
        'train_loss': [],
        'test_loss': [],
        'train_mae': [],
        'test_mae': [],
        'train_exact_acc': [],
        'test_exact_acc': []
    }
    
    start_time = time.time()
    
    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0
        train_mae = 0
        train_exact = 0
        train_count = 0
        
        for inputs, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False):
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)
            train_mae += torch.abs(outputs - targets).sum().item()
            
            # Exact accuracy: rounded prediction == target
            rounded = torch.round(outputs.clamp(0, 11))
            train_exact += (rounded == targets).sum().item()
            train_count += inputs.size(0)
        
        train_loss /= train_count
        train_mae /= train_count
        train_exact_acc = train_exact / train_count
        
        # Evaluate
        model.eval()
        test_loss = 0
        test_mae = 0
        test_exact = 0
        test_count = 0
        
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                
                loss = criterion(outputs, targets)
                test_loss += loss.item() * inputs.size(0)
                test_mae += torch.abs(outputs - targets).sum().item()
                
                # Exact accuracy
                rounded = torch.round(outputs.clamp(0, 11))
                test_exact += (rounded == targets).sum().item()
                test_count += inputs.size(0)
        
        test_loss /= test_count
        test_mae /= test_count
        test_exact_acc = test_exact / test_count
        
        history['train_loss'].append(train_loss)
        history['test_loss'].append(test_loss)
        history['train_mae'].append(train_mae)
        history['test_mae'].append(test_mae)
        history['train_exact_acc'].append(train_exact_acc)
        history['test_exact_acc'].append(test_exact_acc)
        
        print(f"Epoch {epoch+1}/{epochs}: "
              f"Train Loss={train_loss:.4f}, Test Loss={test_loss:.4f}, "
              f"Train MAE={train_mae:.4f}, Test MAE={test_mae:.4f}, "
              f"Test Acc={test_exact_acc:.4f}")
    
    total_time = time.time() - start_time
    
    return history, total_time


if __name__ == "__main__":
    print("="*70)
    print("Training Simple MLP Baseline")
    print("Configuration: SmoothL1Loss + L2 (wd=5e-5), 15 epochs")
    print("="*70)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")
    
    data_dir = Path("data/cube-2-by-2-heuristic")
    
    # Load data
    print("\nLoading dataset...")
    train_data = load_dataset_split(data_dir, "train")
    test_data = load_dataset_split(data_dir, "test")
    print(f"Train: {len(train_data['inputs'])}, Test: {len(test_data['inputs'])}")
    
    # Create model
    print("\nCreating Simple MLP (4 layers, hidden_dim=256)...")
    model = SimpleMLP(hidden_dim=256, dropout=0.0)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")
    
    # Train
    print("\nTraining...")
    history, total_time = train_model(
        model,
        train_data,
        test_data,
        epochs=15,
        lr=1e-4,
        weight_decay=5e-5,
        batch_size=64,
        device=device
    )
    
    # Save results
    results = {
        'config': {
            'model': 'SimpleMLP',
            'hidden_dim': 256,
            'dropout': 0.0,
            'loss': 'SmoothL1Loss',
            'weight_decay': 5e-5,
            'lr': 1e-4,
            'batch_size': 64,
            'epochs': 15,
            'device': device,
        },
        'history': history,
        'summary': {
            'num_params': num_params,
            'total_time': total_time,
            'final_train_loss': history['train_loss'][-1],
            'final_test_loss': history['test_loss'][-1],
            'final_train_mae': history['train_mae'][-1],
            'final_test_mae': history['test_mae'][-1],
            'final_train_acc': history['train_exact_acc'][-1],
            'final_test_acc': history['test_exact_acc'][-1],
        }
    }
    
    output_file = Path("mlp_baseline_results.json")
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE")
    print("="*70)
    print(f"\nFinal Results:")
    print(f"  Test MAE: {history['test_mae'][-1]:.4f} moves")
    print(f"  Test Loss: {history['test_loss'][-1]:.4f}")
    print(f"  Test Exact Accuracy: {history['test_exact_acc'][-1]:.4f}")
    print(f"  Train MAE: {history['train_mae'][-1]:.4f} moves")
    print(f"  Overfitting Gap (Test-Train Loss): {history['test_loss'][-1] - history['train_loss'][-1]:.4f}")
    print(f"  Total Training Time: {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"  Parameters: {num_params:,}")
    print(f"\n✅ Results saved to: {output_file}")
