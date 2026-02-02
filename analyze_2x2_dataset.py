# 2x2 Cube Dataset Analyzer
# Analyzes the generated 2x2 cube dataset from build_cubes.py
# Provides statistics on scramble depths, solution lengths, state distributions, etc.

import numpy as np
import json
import os
from pathlib import Path
from collections import Counter
import matplotlib.pyplot as plt
import sys

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
import py222

# Move names for display
MOVE_NAMES = ["U", "U'", "U2", "R", "R'", "R2", "F", "F'", "F2", "D", "D'", "D2", "L", "L'", "L2", "B", "B'", "B2", "x", "x'", "x2", "y", "y'", "y2", "z", "z'", "z2"]

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
    group_indices = np.load(split_dir / "all__group_indices.npy")
    puzzle_indices = np.load(split_dir / "all__puzzle_indices.npy")
    puzzle_identifiers = np.load(split_dir / "all__puzzle_identifiers.npy")

    return {
        'metadata': metadata,
        'inputs': inputs,
        'labels': labels,
        'group_indices': group_indices,
        'puzzle_indices': puzzle_indices,
        'puzzle_identifiers': puzzle_identifiers,
    }

def decode_cube_state(encoded_state):
    """Decode the padded int array back to py222 state string"""
    # Remove padding: subtract 1
    return encoded_state - 1

def decode_moves(encoded_moves):
    """Decode the padded move sequence back to list of moves"""
    # Remove padding and filter out pads (0)
    moves = encoded_moves - 1
    return moves[moves >= 0].tolist()

def analyze_split(data, split_name):
    """Analyze a single dataset split"""
    print(f"\n=== {split_name.upper()} SPLIT ANALYSIS ===")
    print(f"Number of examples: {len(data['inputs'])}")

    # Solution lengths
    solution_lengths = []
    scramble_depths = []

    for i in range(len(data['inputs'])):
        state = decode_cube_state(data['inputs'][i])
        moves = decode_moves(data['labels'][i])

        solution_lengths.append(len(moves))

        # Scramble depth: solve the scrambled state to get original depth
        try:
            solved_moves = py222.solve(state)
            scramble_depths.append(len(solved_moves))
        except Exception as e:
            print(f"Warning: Could not solve state {i}: {e}")
            scramble_depths.append(0)  # fallback

    print(f"Solution length stats:")
    print(f"  Mean: {np.mean(solution_lengths):.2f}")
    print(f"  Median: {np.median(solution_lengths):.2f}")
    print(f"  Min: {min(solution_lengths)}")
    print(f"  Max: {max(solution_lengths)}")
    print(f"  Std: {np.std(solution_lengths):.2f}")

    print(f"Scramble depth stats:")
    print(f"  Mean: {np.mean(scramble_depths):.2f}")
    print(f"  Median: {np.median(scramble_depths):.2f}")
    print(f"  Min: {min(scramble_depths)}")
    print(f"  Max: {max(scramble_depths)}")
    print(f"  Std: {np.std(scramble_depths):.2f}")

    # Move distribution
    all_moves = []
    for moves in [decode_moves(data['labels'][i]) for i in range(len(data['inputs']))]:
        all_moves.extend(moves)

    move_counts = Counter(all_moves)
    print(f"Move distribution (top 10):")
    for move_id, count in move_counts.most_common(10):
        move_name = MOVE_NAMES[move_id] if move_id < len(MOVE_NAMES) else f"move_{move_id}"
        print(f"  {move_name}: {count} ({count/len(all_moves)*100:.1f}%)")

    # State uniqueness
    unique_states = set()
    for i in range(len(data['inputs'])):
        state = tuple(decode_cube_state(data['inputs'][i]))
        unique_states.add(state)

    print(f"Unique scrambled states: {len(unique_states)} / {len(data['inputs'])} ({len(unique_states)/len(data['inputs'])*100:.1f}%)")

    return {
        'solution_lengths': solution_lengths,
        'scramble_depths': scramble_depths,
        'move_counts': dict(move_counts),
        'unique_states_ratio': len(unique_states) / len(data['inputs'])
    }

def plot_analysis(all_stats):
    """Create plots comparing the splits"""
    splits = list(all_stats.keys())
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('2x2 Cube Dataset Analysis')

    # Solution lengths
    axes[0,0].boxplot([all_stats[s]['solution_lengths'] for s in splits], labels=splits)
    axes[0,0].set_title('Solution Length Distribution')
    axes[0,0].set_ylabel('Moves')

    # Scramble depths
    axes[0,1].boxplot([all_stats[s]['scramble_depths'] for s in splits], labels=splits)
    axes[0,1].set_title('Scramble Depth Distribution')
    axes[0,1].set_ylabel('Moves')

    # Move distributions (top 5 moves)
    all_moves = set()
    for s in splits:
        all_moves.update(all_stats[s]['move_counts'].keys())
    top_moves = sorted(all_moves, key=lambda m: sum(all_stats[s]['move_counts'].get(m, 0) for s in splits), reverse=True)[:5]

    move_names = [MOVE_NAMES[m] if m < len(MOVE_NAMES) else f"move_{m}" for m in top_moves]
    x = np.arange(len(top_moves))
    width = 0.25

    for i, split in enumerate(splits):
        counts = [all_stats[split]['move_counts'].get(m, 0) for m in top_moves]
        axes[1,0].bar(x + i*width, counts, width, label=split)

    axes[1,0].set_title('Top 5 Moves Distribution')
    axes[1,0].set_xticks(x + width)
    axes[1,0].set_xticklabels(move_names, rotation=45)
    axes[1,0].legend()

    # Unique states ratio
    ratios = [all_stats[s]['unique_states_ratio'] for s in splits]
    axes[1,1].bar(splits, ratios)
    axes[1,1].set_title('Unique States Ratio')
    axes[1,1].set_ylabel('Ratio')
    axes[1,1].set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig('outputs/2026-01-18/cube_2x2_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()

def main():
    data_dir = Path('data/cube-2-by-2')
    if not data_dir.exists():
        print(f"Data directory {data_dir} not found!")
        return

    splits = ['train', 'test', 'val']
    all_stats = {}

    for split in splits:
        try:
            data = load_dataset_split(data_dir, split)
            stats = analyze_split(data, split)
            all_stats[split] = stats
        except Exception as e:
            print(f"Error analyzing {split}: {e}")

    if all_stats:
        plot_analysis(all_stats)

        # Save summary stats
        summary = {
            'dataset_info': {
                'total_train': len(all_stats.get('train', {}).get('solution_lengths', [])),
                'total_test': len(all_stats.get('test', {}).get('solution_lengths', [])),
                'total_val': len(all_stats.get('val', {}).get('solution_lengths', [])),
            },
            'stats': all_stats
        }

        with open('outputs/2026-01-18/cube_2x2_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"\nSummary saved to outputs/2026-01-18/cube_2x2_summary.json")
        print(f"Plots saved to outputs/2026-01-18/cube_2x2_analysis.png")

if __name__ == "__main__":
    main()