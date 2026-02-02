"""
Analyze the 2x2 Rubik's Cube dataset for distribution and biases
"""
import numpy as np
import json
from pathlib import Path
import matplotlib.pyplot as plt
from collections import Counter

def analyze_split(data_dir: Path, split: str):
    """Analyze a dataset split"""
    split_dir = data_dir / split
    
    with open(split_dir / "dataset.json", "r") as f:
        metadata = json.load(f)
    
    inputs = np.load(split_dir / "all__inputs.npy")
    labels = np.load(split_dir / "all__labels.npy").squeeze()
    
    return {
        'metadata': metadata,
        'inputs': inputs,
        'labels': labels,
        'size': len(labels)
    }

def main():
    data_dir = Path("data/cube-2-by-2-heuristic")
    
    print("="*70)
    print("Dataset Analysis: 2x2 Rubik's Cube Heuristic")
    print("="*70)
    
    # Load all splits
    train_data = analyze_split(data_dir, "train")
    test_data = analyze_split(data_dir, "test")
    val_data = analyze_split(data_dir, "val")
    
    print("\n📊 Dataset Sizes:")
    print(f"   Train: {train_data['size']:,}")
    print(f"   Test:  {test_data['size']:,}")
    print(f"   Val:   {val_data['size']:,}")
    print(f"   Total: {train_data['size'] + test_data['size'] + val_data['size']:,}")
    
    # Analyze label distributions
    print("\n📊 Label Distribution (Distance to Solved State):")
    print(f"\n{'Split':<10} {'Min':<6} {'Max':<6} {'Mean':<8} {'Median':<8} {'Std':<8}")
    print("-" * 60)
    
    for name, data in [("Train", train_data), ("Test", test_data), ("Val", val_data)]:
        labels = data['labels']
        print(f"{name:<10} {labels.min():<6} {labels.max():<6} {labels.mean():<8.2f} "
              f"{np.median(labels):<8.1f} {labels.std():<8.2f}")
    
    # Check distribution of each distance value
    print("\n📊 Label Value Counts (Train Set):")
    train_counts = Counter(train_data['labels'])
    test_counts = Counter(test_data['labels'])
    
    print(f"\n{'Distance':<10} {'Train Count':<15} {'Train %':<10} {'Test Count':<15} {'Test %':<10}")
    print("-" * 70)
    
    for dist in sorted(train_counts.keys()):
        train_pct = (train_counts[dist] / train_data['size']) * 100
        test_pct = (test_counts.get(dist, 0) / test_data['size']) * 100
        print(f"{dist:<10} {train_counts[dist]:<15,} {train_pct:<10.2f} "
              f"{test_counts.get(dist, 0):<15,} {test_pct:<10.2f}")
    
    # Check for bias indicators
    print("\n⚠️  Potential Bias Checks:")
    
    # 1. Class imbalance
    max_count = max(train_counts.values())
    min_count = min(train_counts.values())
    imbalance_ratio = max_count / min_count
    print(f"\n1. Class Imbalance Ratio: {imbalance_ratio:.2f}")
    if imbalance_ratio > 10:
        print(f"   ⚠️  HIGH IMBALANCE! Most common class is {imbalance_ratio:.0f}x more frequent")
    elif imbalance_ratio > 3:
        print(f"   ⚠️  Moderate imbalance detected")
    else:
        print(f"   ✅ Relatively balanced")
    
    # 2. Train/Test distribution similarity
    print(f"\n2. Train vs Test Distribution:")
    all_distances = sorted(set(list(train_counts.keys()) + list(test_counts.keys())))
    
    train_dist = np.array([train_counts.get(d, 0) / train_data['size'] for d in all_distances])
    test_dist = np.array([test_counts.get(d, 0) / test_data['size'] for d in all_distances])
    
    # KL divergence-like measure
    diff = np.abs(train_dist - test_dist).sum()
    print(f"   Distribution difference: {diff:.4f}")
    if diff < 0.1:
        print(f"   ✅ Train and test distributions are similar")
    else:
        print(f"   ⚠️  Train and test distributions differ significantly")
    
    # 3. Check for solved state bias (distance = 0)
    print(f"\n3. Solved State (distance=0) Presence:")
    if 0 in train_counts:
        print(f"   Train: {train_counts[0]:,} ({(train_counts[0]/train_data['size'])*100:.2f}%)")
    else:
        print(f"   Train: None ✅ (good - we want unsolved cubes)")
    
    if 0 in test_counts:
        print(f"   Test:  {test_counts[0]:,} ({(test_counts[0]/test_data['size'])*100:.2f}%)")
    else:
        print(f"   Test:  None ✅")
    
    # 4. Check input diversity (unique states)
    print(f"\n4. State Diversity:")
    train_unique = len(set(map(tuple, train_data['inputs'])))
    test_unique = len(set(map(tuple, test_data['inputs'])))
    print(f"   Train unique states: {train_unique:,} / {train_data['size']:,} "
          f"({(train_unique/train_data['size'])*100:.1f}%)")
    print(f"   Test unique states:  {test_unique:,} / {test_data['size']:,} "
          f"({(test_unique/test_data['size'])*100:.1f}%)")
    
    if train_unique < train_data['size']:
        duplicate_pct = ((train_data['size'] - train_unique) / train_data['size']) * 100
        print(f"   ⚠️  {duplicate_pct:.1f}% duplicate states in training set")
    else:
        print(f"   ✅ All training states are unique")
    
    # Visualization
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    # Plot 1: Distribution comparison
    ax = axes[0]
    distances = sorted(train_counts.keys())
    train_pcts = [(train_counts[d] / train_data['size']) * 100 for d in distances]
    test_pcts = [(test_counts.get(d, 0) / test_data['size']) * 100 for d in distances]
    
    x = np.arange(len(distances))
    width = 0.35
    ax.bar(x - width/2, train_pcts, width, label='Train', alpha=0.8)
    ax.bar(x + width/2, test_pcts, width, label='Test', alpha=0.8)
    ax.set_xlabel('Distance to Solved State (moves)', fontsize=11)
    ax.set_ylabel('Percentage of Dataset (%)', fontsize=11)
    ax.set_title('Label Distribution: Train vs Test', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(distances)
    ax.legend()
    ax.grid(alpha=0.3, axis='y')
    
    # Plot 2: Cumulative distribution
    ax = axes[1]
    train_sorted = np.sort(train_data['labels'])
    test_sorted = np.sort(test_data['labels'])
    
    ax.plot(train_sorted, np.linspace(0, 100, len(train_sorted)), 
            label='Train', linewidth=2)
    ax.plot(test_sorted, np.linspace(0, 100, len(test_sorted)), 
            label='Test', linewidth=2, linestyle='--')
    ax.set_xlabel('Distance to Solved State (moves)', fontsize=11)
    ax.set_ylabel('Cumulative Percentage (%)', fontsize=11)
    ax.set_title('Cumulative Distribution', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Plot 3: Box plot
    ax = axes[2]
    ax.boxplot([train_data['labels'], test_data['labels'], val_data['labels']], 
                labels=['Train', 'Test', 'Val'])
    ax.set_ylabel('Distance to Solved State (moves)', fontsize=11)
    ax.set_title('Distribution Box Plot', fontsize=12, fontweight='bold')
    ax.grid(alpha=0.3, axis='y')
    
    plt.suptitle('Dataset Analysis: 2x2 Rubik\'s Cube Heuristic', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    output_file = 'dataset_analysis.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\n✅ Visualization saved to: {output_file}")
    
    print("\n" + "="*70)
    print("Analysis Complete!")
    print("="*70)

if __name__ == "__main__":
    main()
