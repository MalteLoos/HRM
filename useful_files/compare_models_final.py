"""
Complete MLP vs HRM Comparison Script
=====================================
This script:
1. Trains Simple MLP with optimal config (or loads existing results)
2. Loads HRM results from wandb
3. Creates comprehensive comparison visualization

Usage: python compare_models_final.py
"""
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from pathlib import Path
import json
import time
import yaml
from tqdm import tqdm
import matplotlib.pyplot as plt

from pretrain import (
    PretrainConfig,
    create_dataloader as hrm_create_dataloader,
    evaluate as hrm_evaluate,
    init_train_state as hrm_init_train_state,
)


class SimpleMLP(nn.Module):
    """Simple 4-layer MLP with one-hot encoding"""
    def __init__(self, input_dim=144, hidden_dim=256, output_dim=1):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        x_onehot = torch.nn.functional.one_hot(x.long(), num_classes=6).float()
        x_onehot = x_onehot.reshape(x_onehot.shape[0], -1)
        x = torch.nn.functional.relu(self.fc1(x_onehot))
        x = torch.nn.functional.relu(self.fc2(x))
        x = torch.nn.functional.relu(self.fc3(x))
        x = self.fc4(x)
        return x.squeeze(-1)


def load_dataset_split(data_dir: Path, split: str):
    """Load a dataset split"""
    split_dir = data_dir / split
    with open(split_dir / "dataset.json", "r") as f:
        metadata = json.load(f)
    inputs = np.load(split_dir / "all__inputs.npy")
    labels = np.load(split_dir / "all__labels.npy")
    return {'metadata': metadata, 'inputs': inputs, 'labels': labels.squeeze()}


def train_mlp(train_data, test_data, epochs=15, lr=1e-4, weight_decay=5e-5, batch_size=64, device='cpu'):
    """Train MLP with SmoothL1Loss"""
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
    
    model = SimpleMLP(hidden_dim=256).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.SmoothL1Loss()
    
    history = {'train_loss': [], 'test_loss': [], 'train_mae': [], 'test_mae': [], 'test_exact_acc': []}
    start_time = time.time()
    
    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = train_mae = train_count = 0
        for inputs, targets in tqdm(train_loader, desc=f"MLP Epoch {epoch+1}/{epochs}", leave=False):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)
            train_mae += torch.abs(outputs - targets).sum().item()
            train_count += inputs.size(0)
        
        # Evaluate
        model.eval()
        test_loss = test_mae = test_exact = test_count = 0
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                test_loss += criterion(outputs, targets).item() * inputs.size(0)
                test_mae += torch.abs(outputs - targets).sum().item()
                rounded = torch.round(outputs.clamp(0, 11))
                test_exact += (rounded == targets).sum().item()
                test_count += inputs.size(0)
        
        history['train_loss'].append(train_loss / train_count)
        history['test_loss'].append(test_loss / test_count)
        history['train_mae'].append(train_mae / train_count)
        history['test_mae'].append(test_mae / test_count)
        history['test_exact_acc'].append(test_exact / test_count)
        
        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1}: Test MAE={history['test_mae'][-1]:.4f}, Test Acc={history['test_exact_acc'][-1]:.4f}")
    
    total_time = time.time() - start_time
    num_params = sum(p.numel() for p in model.parameters())
    
    return {
        'history': history,
        'num_params': num_params,
        'total_time': total_time,
        'final_test_mae': history['test_mae'][-1],
        'final_test_loss': history['test_loss'][-1],
        'final_test_acc': history['test_exact_acc'][-1],
    }


def get_latest_hrm_checkpoint(project_dir: Path):
    """Return info for the numerically largest step_* checkpoint under the project directory."""
    if not project_dir.exists():
        return None

    latest = None
    for run_dir in project_dir.iterdir():
        if not run_dir.is_dir():
            continue
        for ckpt in run_dir.glob("step_*"):
            try:
                step = int(ckpt.name.split("_")[1])
            except (IndexError, ValueError):
                continue
            if latest is None or step > latest["step"]:
                latest = {"checkpoint": ckpt, "run_dir": run_dir, "step": step}
    return latest


def load_hrm_config(run_dir: Path):
    cfg_path = run_dir / "all_config.yaml"
    if not cfg_path.exists():
        return None
    with open(cfg_path, "r") as f:
        cfg_dict = yaml.safe_load(f)
    try:
        return PretrainConfig(**cfg_dict)
    except Exception as exc:  # noqa: BLE001
        print(f"   ⚠️  Could not parse HRM config: {exc}")
        return None


def evaluate_latest_hrm_checkpoint(device="cuda"):
    """Load the newest HRM checkpoint and evaluate it on the test split."""
    project_dir = Path("checkpoints") / "Cube-2-by-2-heuristic ACT-torch"
    checkpoint_info = get_latest_hrm_checkpoint(project_dir)
    if checkpoint_info is None:
        print("   ⚠️  No HRM checkpoints found. Train HRM first.")
        return None

    config = load_hrm_config(checkpoint_info["run_dir"])
    if config is None:
        return None

    config.eval_save_outputs = []
    config.checkpoint_path = str(checkpoint_info["run_dir"])

    # Data
    train_loader, train_metadata = hrm_create_dataloader(
        config,
        "train",
        test_set_mode=False,
        epochs_per_iter=1,
        global_batch_size=config.global_batch_size,
        rank=0,
        world_size=1,
    )
    eval_loader, eval_metadata = hrm_create_dataloader(
        config,
        "test",
        test_set_mode=True,
        epochs_per_iter=1,
        global_batch_size=config.global_batch_size,
        rank=0,
        world_size=1,
    )

    # Model
    train_state = hrm_init_train_state(config, train_metadata, world_size=1)
    state_dict = torch.load(checkpoint_info["checkpoint"], map_location=device)
    try:
        train_state.model.load_state_dict(state_dict, assign=True)
    except Exception:  # noqa: BLE001
        cleaned = {k.removeprefix("_orig_mod."): v for k, v in state_dict.items()}
        train_state.model.load_state_dict(cleaned, assign=True)
    train_state.step = checkpoint_info["step"]
    train_state.model.to(device)
    train_state.model.eval()

    metrics = hrm_evaluate(config, train_state, eval_loader, eval_metadata, rank=0, world_size=1)
    if metrics is None:
        print("   ⚠️  HRM evaluation returned no metrics.")
        return None

    # Metrics are grouped by set name; take the first (test set)
    summary_set = next(iter(metrics.values()))
    num_params = sum(p.numel() for p in train_state.model.parameters())

    return {
        "run_name": checkpoint_info["run_dir"].name,
        "checkpoint": str(checkpoint_info["checkpoint"]),
        "step": checkpoint_info["step"],
        "num_params": num_params,
        "mae": summary_set.get("mae"),
        "exact_acc": summary_set.get("exact_accuracy"),
        "loss": summary_set.get("loss"),
        "metrics": metrics,
    }


def create_comparison_plot(mlp_results, hrm_results, output_file="mlp_vs_hrm_comparison.png"):
    """Create comprehensive comparison visualization"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: MAE over epochs
    ax = axes[0, 0]
    epochs = range(1, len(mlp_results['history']['test_mae'])+1)
    ax.plot(epochs, mlp_results['history']['test_mae'], 
            label='MLP (Test)', marker='o', linewidth=2, markersize=5, color='#3498db')
    ax.plot(epochs, mlp_results['history']['train_mae'], 
            label='MLP (Train)', marker='o', linewidth=2, markersize=4, color='#85c1e9', alpha=0.7)
    if hrm_results and hrm_results.get('mae') is not None:
        ax.axhline(
            y=hrm_results['mae'],
            color='#e74c3c',
            linestyle='--',
            linewidth=2,
            label=f"HRM (step {hrm_results['step']}): {hrm_results['mae']:.3f}",
        )
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('MAE (moves)', fontsize=11)
    ax.set_title('Mean Absolute Error', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    
    # Plot 2: Loss over epochs
    ax = axes[0, 1]
    ax.plot(epochs, mlp_results['history']['test_loss'], 
            label='MLP (Test)', marker='o', linewidth=2, markersize=5, color='#3498db')
    ax.plot(epochs, mlp_results['history']['train_loss'], 
            label='MLP (Train)', marker='o', linewidth=2, markersize=4, color='#85c1e9', alpha=0.7)
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('SmoothL1 Loss', fontsize=11)
    ax.set_title('Loss Curves', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    
    # Plot 3: Exact Accuracy
    ax = axes[1, 0]
    ax.plot(epochs, mlp_results['history']['test_exact_acc'], 
            label='MLP (Test)', marker='o', linewidth=2, markersize=5, color='#2ecc71')
    if hrm_results and hrm_results.get('exact_acc') is not None:
        ax.axhline(
            y=hrm_results['exact_acc'],
            color='#e74c3c',
            linestyle='--',
            linewidth=2,
            label=f"HRM (step {hrm_results['step']}): {hrm_results['exact_acc']:.3f}",
        )
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('Exact Accuracy', fontsize=11)
    ax.set_title('Exact Prediction Accuracy', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    
    # Plot 4: Comparison bars
    ax = axes[1, 1]
    if hrm_results:
        metrics = ['Test MAE\n(moves)', 'Exact Acc', 'Params\n(×10k)']
        mlp_values = [
            mlp_results['final_test_mae'],
            mlp_results['final_test_acc'],
            mlp_results['num_params'] / 10000,
        ]
        hrm_values = [
            hrm_results.get('mae', 0),
            hrm_results.get('exact_acc', 0),
            hrm_results['num_params'] / 10000,
        ]
        
        x = np.arange(len(metrics))
        width = 0.35
        bars1 = ax.bar(x - width/2, mlp_values, width, label='MLP', color='#3498db', alpha=0.8)
        bars2 = ax.bar(x + width/2, hrm_values, width, label='HRM', color='#e74c3c', alpha=0.8)
        
        ax.set_ylabel('Value', fontsize=11)
        ax.set_title('Model Comparison', fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(metrics)
        ax.legend()
        ax.grid(alpha=0.3, axis='y')
        
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.2f}', ha='center', va='bottom', fontsize=9)
    
    plt.suptitle('Simple MLP vs HRM: 2x2 Rubik\'s Cube Heuristic\n(SmoothL1Loss, weight_decay=5e-5, 15 epochs)', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\n✅ Comparison plot saved to: {output_file}")


def main():
    print("="*70)
    print("MLP vs HRM Comparison")
    print("="*70)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")
    data_dir = Path("data/cube-2-by-2-heuristic")
    results_file = Path("mlp_baseline_results.json")
    
    # === STEP 1: Get MLP Results ===
    print("\n" + "="*70)
    print("STEP 1: MLP Training")
    print("="*70)
    
    if results_file.exists():
        print(f"\n✅ Found existing MLP results in {results_file}")
        with open(results_file) as f:
            saved = json.load(f)
            mlp_results = {
                'history': saved['history'],
                'num_params': saved['summary']['num_params'],
                'total_time': saved['summary']['total_time'],
                'final_test_mae': saved['summary']['final_test_mae'],
                'final_test_loss': saved['summary']['final_test_loss'],
                'final_test_acc': saved['summary']['final_test_acc'],
            }
        print(f"   Test MAE: {mlp_results['final_test_mae']:.4f} moves")
        
        retrain = input("\nRetrain MLP? (y/n): ").lower().strip() == 'y'
        if not retrain:
            print("   Using existing results.")
        else:
            results_file.unlink()
    else:
        retrain = True
    
    if not results_file.exists():
        print("\n📊 Training Simple MLP (15 epochs)...")
        print("   Config: SmoothL1Loss, weight_decay=5e-5, hidden_dim=256")
        
        train_data = load_dataset_split(data_dir, "train")
        test_data = load_dataset_split(data_dir, "test")
        print(f"   Train: {len(train_data['inputs'])}, Test: {len(test_data['inputs'])}")
        
        mlp_results = train_mlp(train_data, test_data, epochs=15, device=device)
        
        # Save results
        with open(results_file, 'w') as f:
            json.dump({
                'config': {'loss': 'SmoothL1Loss', 'weight_decay': 5e-5, 'hidden_dim': 256, 'epochs': 15},
                'history': mlp_results['history'],
                'summary': {
                    'num_params': mlp_results['num_params'],
                    'total_time': mlp_results['total_time'],
                    'final_test_mae': mlp_results['final_test_mae'],
                    'final_test_loss': mlp_results['final_test_loss'],
                    'final_test_acc': mlp_results['final_test_acc'],
                }
            }, f, indent=2)
        print(f"\n✅ MLP results saved to {results_file}")
    
    # === STEP 2: Load HRM Results ===
    print("\n" + "="*70)
    print("STEP 2: Loading HRM Results")
    print("="*70)
    
    hrm_results = evaluate_latest_hrm_checkpoint(device=device)
    if hrm_results:
        print(f"\n✅ Loaded HRM checkpoint {hrm_results['checkpoint']} (step {hrm_results['step']})")
        print(f"   Run dir: {hrm_results['run_name']}")
        if hrm_results.get('mae') is not None:
            print(f"   Test MAE: {hrm_results['mae']:.4f} moves")
        if hrm_results.get('exact_acc') is not None:
            print(f"   Test Exact Acc: {hrm_results['exact_acc']:.4f}")
    else:
        print("\n⚠️  No HRM checkpoints found or evaluation failed. Run pretrain.py first!")
        print("   Continuing with MLP-only visualization...")
    
    # === STEP 3: Print Comparison ===
    print("\n" + "="*70)
    print("COMPARISON SUMMARY")
    print("="*70)
    
    print(f"\n📊 Simple MLP:")
    print(f"   Parameters: {mlp_results['num_params']:,}")
    print(f"   Training Time: {mlp_results['total_time']:.1f}s ({mlp_results['total_time']/60:.1f} min)")
    print(f"   Test MAE: {mlp_results['final_test_mae']:.4f} moves")
    print(f"   Test Accuracy: {mlp_results['final_test_acc']:.4f}")
    
    if hrm_results:
        print(f"\n📊 HRM:")
        print(f"   Parameters: {hrm_results['num_params']:,}")
        if hrm_results.get('mae') is not None:
            print(f"   Test MAE: {hrm_results['mae']:.4f} moves (test set)")
        if hrm_results.get('exact_acc') is not None:
            print(f"   Test Accuracy: {hrm_results['exact_acc']:.4f}")
        
        if hrm_results.get('mae') is not None:
            print(f"\n🎯 Comparison:")
            mae_diff = ((hrm_results['mae'] - mlp_results['final_test_mae']) / mlp_results['final_test_mae']) * 100
            if mae_diff > 0:
                print(f"   ⚠️  HRM has {mae_diff:.1f}% higher MAE than MLP")
            else:
                print(f"   ✅ HRM has {abs(mae_diff):.1f}% lower MAE than MLP")
            print(f"   HRM is {hrm_results['num_params']/mlp_results['num_params']:.2f}x larger")
    
    # === STEP 4: Create Visualization ===
    print("\n" + "="*70)
    print("STEP 4: Creating Visualization")
    print("="*70)
    
    create_comparison_plot(mlp_results, hrm_results)
    
    print("\n" + "="*70)
    print("✅ DONE!")
    print("="*70)


if __name__ == "__main__":
    main()
