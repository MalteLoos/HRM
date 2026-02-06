"""
Build 2x2 dataset with MULTIPLE OPTIMAL solutions per state.

Instead of one solution per scramble, generates ALL optimal solutions
for each state. This eliminates label ambiguity - model can predict
ANY valid optimal sequence without penalty.

Output format:
  - inputs: (N, 24) cube states
  - labels_multi: List[List[int]] where labels_multi[i] = all optimal solutions for state i
  - lengths: (N,) lengths of each solution
"""

import numpy as np
from pathlib import Path
import json
import argparse
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
from typing import List, Tuple
import sys

# Add parent directory to path for py222
sys.path.insert(0, str(Path(__file__).parent.parent))
import py222


def find_multiple_solutions(state, max_solutions=10):
    """
    Find multiple optimal solutions for a given cube state using IDA* with memoization.
    
    Args:
        state: Cube state (24-element numpy array)
        max_solutions: Maximum number of solutions to find
        
    Returns:
        List of solutions (each solution is a list of move indices)
        Returns empty list if already solved.
    """
    # Normalize state
    state = py222.normFC(state)
    
    # Check if already solved
    if py222.isSolved(state):
        return [[]]
    
    # Find optimal depth first
    first_solution = py222.solve(state, verbose=False)
    if not first_solution:
        return []
    
    optimal_depth = len(first_solution)
    
    # Cap search depth to avoid excessive computation
    if optimal_depth > 10:
        # For very deep solutions, just return the one we found
        return [first_solution]
    
    # Now do bounded search at that depth to find multiple solutions
    solutions = []
    visited_states = set()
    
    def search(s, depth, moves, last_move_face=-1):
        """DFS to find all solutions at given depth with memoization."""
        if len(solutions) >= max_solutions:
            return
            
        if depth == 0:
            if py222.isSolved(s):
                solutions.append(moves[:])
            return
        
        # Try all 9 moves (U, U', U2, R, R', R2, F, F', F2)
        for move in range(9):
            # Skip redundant moves (same face as last move)
            face = move // 3  # 0=U, 1=R, 2=F
            if face == last_move_face:
                continue
            
            new_state = py222.doMove(s, move)
            moves.append(move)
            
            # State memoization to avoid redundant paths
            state_tuple = tuple(new_state)
            if state_tuple not in visited_states or depth <= 2:
                # Allow revisiting at shallow depths
                visited_states.add(state_tuple)
                search(new_state, depth - 1, moves, face)
            
            moves.pop()
    
    search(state, optimal_depth, [])
    return solutions if solutions else [first_solution]


def generate_multisolution_sample(seed: int, num_solutions: int = 5) -> Tuple[np.ndarray, List[List[int]]]:
    """
    Generate a scrambled cube and find multiple optimal solutions.
    
    Args:
        seed: Random seed for reproducibility
        num_solutions: Maximum number of solutions to find
        
    Returns:
        (cube_state, list_of_solutions)
        - cube_state: (24,) array of integers 1-6
        - list_of_solutions: List of solution sequences, all same optimal length
    """
    rng = np.random.RandomState(seed)
    
    # Generate random scramble (deep enough to explore full solution space)
    moves = list(range(9))  # 0-8: U, U', U2, R, R', R2, F, F', F2
    num_scramble_moves = rng.randint(3, 20)  # 3-19 moves scramble
    scramble = rng.choice(moves, size=num_scramble_moves, replace=True).tolist()
    
    # Create initial solved cube
    state = py222.initState()
    
    # Apply scramble moves one by one
    for move in scramble:
        state = py222.doMove(state, move)
    
    # Find multiple optimal solutions
    solutions = find_multiple_solutions(state, max_solutions=num_solutions)
    
    if not solutions:
        # Fallback: shouldn't happen for solvable cube
        return None
    
    # Convert moves to token indices (1-9 for moves, 0 for PAD)
    token_solutions = []
    for sol in solutions:
        tokens = [move + 1 for move in sol]  # Convert 0-8 → 1-9
        token_solutions.append(tokens)
    
    # Shift state values from 0-5 → 1-6 (reserving 0 for PAD in the input space)
    state_shifted = state + 1
    
    return state_shifted, token_solutions


def build_dataset(
    output_dir: Path,
    num_samples: int = 150000,
    test_split: float = 0.1,
    val_split: float = 0.1,
    num_solutions: int = 5,
    num_workers: int = 8,
    seed: int = 42,
):
    """Build multi-solution dataset and save to disk."""
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Building 2x2 Multi-Solution Dataset")
    print(f"  Output: {output_dir}")
    print(f"  Total samples: {num_samples:,}")
    print(f"  Test split: {test_split*100:.0f}%")
    print(f"  Val split: {val_split*100:.0f}%")
    print(f"  Max solutions per state: {num_solutions}")
    print(f"  Max solution length: 11 (God's number for 2x2)")
    print(f"  Workers: {num_workers}")
    
    # Generate all samples in parallel
    print("\nGenerating samples...")
    samples = []
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [
            executor.submit(generate_multisolution_sample, seed + i, num_solutions)
            for i in range(num_samples)
        ]
        
        for future in tqdm(futures, total=num_samples, desc="Generating"):
            result = future.result()
            if result is not None:
                samples.append(result)
    
    print(f"\nGenerated {len(samples)} valid samples")
    
    # Separate into train/test/val
    rng = np.random.RandomState(seed)
    indices = rng.permutation(len(samples))
    
    test_count = int(len(samples) * test_split)
    val_count = int(len(samples) * val_split)
    
    test_idx = indices[:test_count]
    val_idx = indices[test_count:test_count + val_count]
    train_idx = indices[test_count + val_count:]
    
    print(f"\nSplit: Train={len(train_idx)}, Test={len(test_idx)}, Val={len(val_idx)}")
    
    # Prepare data for each split
    splits = {
        'train': train_idx,
        'test': test_idx,
        'val': val_idx
    }
    
    # Process each split
    for split_name, indices in splits.items():
        split_dir = output_dir / split_name
        split_dir.mkdir(exist_ok=True)
        
        print(f"\nProcessing {split_name}...")
        
        # Collect inputs and multi-solutions
        inputs = []
        labels_multi = []
        solution_lengths = []
        
        for idx in tqdm(indices, desc=f"  Preparing {split_name}"):
            state, solutions = samples[idx]
            inputs.append(state)
            labels_multi.append(solutions)
            # All solutions should have same length
            solution_lengths.append(len(solutions[0]))
        
        inputs = np.array(inputs, dtype=np.int32)
        
        # For easier processing, create a padded array with max length
        max_length = max(solution_lengths) if solution_lengths else 15
        # Pad to standard length (15)
        max_length = 15
        
        # Create padded single-label version (pick first solution for compatibility)
        labels_single = np.zeros((len(labels_multi), max_length), dtype=np.int32)
        for i, solutions in enumerate(labels_multi):
            sol = solutions[0]  # Use first solution
            labels_single[i, :len(sol)] = sol
        
        # Save inputs
        np.save(split_dir / "all__inputs.npy", inputs)
        np.save(split_dir / "all__labels.npy", labels_single)
        np.save(split_dir / "all__solution_lengths.npy", np.array(solution_lengths, dtype=np.int64))
        
        # Save multi-solution versions as JSON (for reference/advanced use)
        multi_solution_data = {
            'num_states': len(labels_multi),
            'max_solutions_per_state': max(len(sols) for sols in labels_multi),
            'avg_solutions_per_state': np.mean([len(sols) for sols in labels_multi]),
        }
        
        with open(split_dir / "multi_solutions_metadata.json", "w") as f:
            json.dump(multi_solution_data, f, indent=2)
        
        # Save multi-solution data as numpy (state_idx -> list of solutions)
        # Using object array for variable-length lists
        labels_multi_array = np.empty(len(labels_multi), dtype=object)
        for i, solutions in enumerate(labels_multi):
            labels_multi_array[i] = solutions
        
        np.save(split_dir / "all__labels_multi.npy", labels_multi_array, allow_pickle=True)
        
        print(f"  ✓ Saved {len(inputs)} samples")
        print(f"    - inputs: {inputs.shape}")
        print(f"    - labels (single): {labels_single.shape}")
        print(f"    - solution lengths: min={min(solution_lengths)}, max={max(solution_lengths)}, mean={np.mean(solution_lengths):.2f}")
        
        # Save metadata
        metadata = {
            'num_samples': len(inputs),
            'input_shape': inputs.shape,
            'input_range': [int(inputs.min()), int(inputs.max())],
            'vocab_size': 10,  # 0: PAD, 1-9: moves
            'max_seq_length': max_length,
            'min_solution_length': int(min(solution_lengths)),
            'max_solution_length': int(max(solution_lengths)),
            'mean_solution_length': float(np.mean(solution_lengths)),
            'avg_solutions_per_state': float(np.mean([len(sols) for sols in labels_multi])),
            'split': split_name,
        }
        
        with open(split_dir / "dataset.json", "w") as f:
            json.dump(metadata, f, indent=2)
    
    print("\n" + "="*70)
    print("✅ Multi-Solution Dataset Complete!")
    print("="*70)
    print(f"\nInstructions for training:")
    print(f"  1. Load inputs normally: np.load('train/all__inputs.npy')")
    print(f"  2. Load labels normally: np.load('train/all__labels.npy')")
    print(f"  3. Optional - Load multi-solution versions:")
    print(f"     labels_multi = np.load('train/all__labels_multi.npy', allow_pickle=True)")
    print(f"  4. For curriculum training, filter by solution_lengths")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build 2x2 multi-solution dataset")
    parser.add_argument("--output", type=str, default="data/cube-2-by-2-multisolution",
                        help="Output directory")
    parser.add_argument("--num_samples", type=int, default=150000,
                        help="Number of samples to generate")
    parser.add_argument("--test_split", type=float, default=0.1,
                        help="Fraction for test set")
    parser.add_argument("--val_split", type=float, default=0.1,
                        help="Fraction for validation set")
    parser.add_argument("--num_solutions", type=int, default=5,
                        help="Max solutions to find per state")
    parser.add_argument("--workers", type=int, default=8,
                        help="Number of worker processes")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    
    args = parser.parse_args()
    
    build_dataset(
        output_dir=args.output,
        num_samples=args.num_samples,
        test_split=args.test_split,
        val_split=args.val_split,
        num_solutions=args.num_solutions,
        num_workers=args.workers,
        seed=args.seed,
    )
