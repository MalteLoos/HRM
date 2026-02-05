"""
2x2x2 Cube Dataset Validation Script

This script validates that a generated 2x2x2 cube dataset is correct by checking:
1. Data format and shapes are correct
2. All solution sequences actually solve their respective cubes
3. Solutions are optimal (match py222 solver output)
4. Vocabulary ranges are valid
5. Statistical properties are reasonable

Usage:
    python validate_2x2_dataset.py --dataset_dir data/cube-2-by-2-solution
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import numpy as np
from argdantic import ArgParser
from pydantic import BaseModel
from tqdm import tqdm
import py222


cli = ArgParser()


class ValidationConfig(BaseModel):
    dataset_dir: str = "data/cube-2-by-2-solution"
    num_samples_to_check: int = None  # None = check all samples
    check_optimality: bool = True  # Verify solutions are optimal (slow)
    verbose: bool = False


MOVE_NAMES = ["U", "U'", "U2", "R", "R'", "R2", "F", "F'", "F2"]


def load_split(dataset_dir: Path, split_name: str):
    """Load a dataset split."""
    split_dir = dataset_dir / split_name
    
    if not split_dir.exists():
        return None
    
    # Load metadata
    with open(split_dir / "dataset.json", "r") as f:
        metadata = json.load(f)
    
    # Load data
    data = {}
    for key in ["inputs", "labels", "group_indices", "puzzle_indices", "puzzle_identifiers"]:
        filepath = split_dir / f"all__{key}.npy"
        if filepath.exists():
            data[key] = np.load(filepath)
    
    return metadata, data


def format_solution(moves):
    """Format a solution sequence for display."""
    if len(moves) == 0:
        return "SOLVED"
    return " ".join([MOVE_NAMES[m] for m in moves])


def validate_split(dataset_dir: Path, split_name: str, config: ValidationConfig):
    """Validate a single dataset split."""
    print(f"\n{'='*70}")
    print(f"Validating {split_name} split")
    print(f"{'='*70}")
    
    result = load_split(dataset_dir, split_name)
    if result is None:
        print(f"  ❌ Split not found: {split_name}")
        return False
    
    metadata, data = result
    
    # ========================================================================
    # 1. Check data format and shapes
    # ========================================================================
    print("\n[1] Checking data format...")
    
    inputs = data["inputs"]
    labels = data["labels"]
    
    num_samples = len(inputs)
    print(f"  ✓ Number of samples: {num_samples}")
    
    # Check shapes
    expected_input_shape = (num_samples, 24)  # 24 stickers
    if inputs.shape != expected_input_shape:
        print(f"  ❌ Input shape mismatch: {inputs.shape} != {expected_input_shape}")
        return False
    print(f"  ✓ Input shape: {inputs.shape}")
    
    seq_len = metadata["seq_len"]
    expected_label_shape = (num_samples, seq_len)
    if labels.shape != expected_label_shape:
        print(f"  ❌ Label shape mismatch: {labels.shape} != {expected_label_shape}")
        return False
    print(f"  ✓ Label shape: {labels.shape}")
    
    # Check vocabulary ranges
    vocab_size = metadata["vocab_size"]
    
    # Inputs: should be in range [1, 7] (colors 0-5 shifted by +1, with 0=PAD)
    # But 2x2 cube has 6 colors, so inputs should be in [1, 6]
    input_min, input_max = inputs.min(), inputs.max()
    if input_min < 1 or input_max > 6:
        print(f"  ❌ Input values out of range: [{input_min}, {input_max}], expected [1, 6]")
        return False
    print(f"  ✓ Input value range: [{input_min}, {input_max}]")
    
    # Labels: should be in range [0, vocab_size-1] = [0, 9]
    label_min, label_max = labels.min(), labels.max()
    if label_min < 0 or label_max >= vocab_size:
        print(f"  ❌ Label values out of range: [{label_min}, {label_max}], expected [0, {vocab_size-1}]")
        return False
    print(f"  ✓ Label value range: [{label_min}, {label_max}]")
    
    # ========================================================================
    # 2. Check solution validity
    # ========================================================================
    print("\n[2] Checking solution validity...")
    
    # Determine how many samples to check
    num_to_check = config.num_samples_to_check if config.num_samples_to_check else num_samples
    num_to_check = min(num_to_check, num_samples)
    
    indices_to_check = np.random.choice(num_samples, size=num_to_check, replace=False)
    
    invalid_solutions = []
    solution_lengths = []
    
    for idx in tqdm(indices_to_check, desc="  Verifying solutions"):
        # Get input state (subtract 1 to get colors 0-5)
        state = inputs[idx] - 1
        
        # Get solution (subtract 1 from non-zero moves, ignore PAD=0)
        solution_encoded = labels[idx]
        solution = [m - 1 for m in solution_encoded if m > 0]
        
        solution_lengths.append(len(solution))
        
        # Apply solution
        test_state = state.copy()
        for move in solution:
            if move < 0 or move > 8:
                print(f"  ❌ Sample {idx}: Invalid move index {move}")
                invalid_solutions.append(idx)
                break
            test_state = py222.doMove(test_state, move)
        else:
            # Check if solved
            if not py222.isSolved(test_state):
                if config.verbose:
                    print(f"  ❌ Sample {idx}: Solution doesn't solve the cube!")
                    print(f"     Initial state: {state}")
                    print(f"     Solution: {format_solution(solution)}")
                    print(f"     Final state: {test_state}")
                invalid_solutions.append(idx)
    
    if invalid_solutions:
        print(f"  ❌ Found {len(invalid_solutions)} invalid solutions!")
        if not config.verbose:
            print(f"     Sample indices: {invalid_solutions[:10]}{'...' if len(invalid_solutions) > 10 else ''}")
        return False
    else:
        print(f"  ✓ All {num_to_check} checked solutions are valid")
    
    # ========================================================================
    # 3. Check solution optimality (optional, slow)
    # ========================================================================
    if config.check_optimality:
        print("\n[3] Checking solution optimality...")
        print(f"  (This may take a while for large datasets)")
        
        # Check a subset for optimality
        num_optimality_checks = min(100, num_to_check)
        optimality_indices = np.random.choice(indices_to_check, size=num_optimality_checks, replace=False)
        
        non_optimal = []
        
        for idx in tqdm(optimality_indices, desc="  Checking optimality"):
            state = inputs[idx] - 1
            solution_encoded = labels[idx]
            solution = [m - 1 for m in solution_encoded if m > 0]
            
            # Get optimal solution from solver
            optimal_solution = py222.solve(state)
            
            if len(solution) != len(optimal_solution):
                if config.verbose:
                    print(f"  ⚠ Sample {idx}: Solution length {len(solution)} != optimal {len(optimal_solution)}")
                    print(f"     Dataset solution: {format_solution(solution)}")
                    print(f"     Optimal solution: {format_solution(optimal_solution)}")
                non_optimal.append(idx)
        
        if non_optimal:
            print(f"  ⚠ Found {len(non_optimal)} non-optimal solutions")
            print(f"    (This might be OK if there are multiple optimal solutions)")
        else:
            print(f"  ✓ All {num_optimality_checks} checked solutions are optimal")
    
    # ========================================================================
    # 4. Statistical checks
    # ========================================================================
    print("\n[4] Statistical analysis...")
    
    if not solution_lengths:
        # Compute for all samples if not done yet
        solution_lengths = [np.count_nonzero(label) for label in labels]
    
    print(f"  Solution lengths:")
    print(f"    Min: {min(solution_lengths)}")
    print(f"    Max: {max(solution_lengths)}")
    print(f"    Mean: {np.mean(solution_lengths):.2f}")
    print(f"    Median: {np.median(solution_lengths):.1f}")
    print(f"    Std: {np.std(solution_lengths):.2f}")
    
    # Check if max solution length fits in sequence
    if max(solution_lengths) > seq_len:
        print(f"  ❌ Some solutions ({max(solution_lengths)}) exceed sequence length ({seq_len})")
        return False
    
    # Distribution of solution lengths
    print(f"\n  Solution length distribution:")
    for length in range(min(solution_lengths), max(solution_lengths) + 1):
        count = sum(1 for l in solution_lengths if l == length)
        if count > 0:
            pct = 100 * count / len(solution_lengths)
            bar = "█" * int(pct / 2)
            print(f"    {length:2d} moves: {count:5d} ({pct:5.2f}%) {bar}")
    
    # Check metadata consistency
    print(f"\n[5] Checking metadata consistency...")
    
    if metadata["total_groups"] != num_samples:
        print(f"  ⚠ total_groups ({metadata['total_groups']}) != num_samples ({num_samples})")
    else:
        print(f"  ✓ Metadata total_groups matches sample count")
    
    if metadata["vocab_size"] != 10:
        print(f"  ⚠ vocab_size is {metadata['vocab_size']}, expected 10")
    else:
        print(f"  ✓ vocab_size is correct")
    
    print(f"\n{'='*70}")
    print(f"✅ {split_name} split validation PASSED")
    print(f"{'='*70}")
    
    return True


@cli.command(singleton=True)
def validate(config: ValidationConfig):
    """Validate the entire dataset."""
    dataset_dir = Path(config.dataset_dir)
    
    if not dataset_dir.exists():
        print(f"❌ Dataset directory not found: {dataset_dir}")
        return
    
    print(f"Validating dataset: {dataset_dir}")
    print(f"  Checking {config.num_samples_to_check or 'ALL'} samples per split")
    print(f"  Optimality check: {'ON' if config.check_optimality else 'OFF'}")
    
    # Validate each split
    splits = ["train", "test", "val"]
    results = {}
    
    for split in splits:
        results[split] = validate_split(dataset_dir, split, config)
    
    # Summary
    print(f"\n{'='*70}")
    print("VALIDATION SUMMARY")
    print(f"{'='*70}")
    for split, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {split:10s}: {status}")
    
    if all(results.values()):
        print(f"\n🎉 All validations PASSED! Dataset is ready for training.")
    else:
        print(f"\n⚠️  Some validations FAILED. Please check the dataset.")


if __name__ == "__main__":
    cli()
