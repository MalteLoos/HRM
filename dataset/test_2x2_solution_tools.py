"""
Quick test script to verify the 2x2 solution dataset tools work correctly.

This script runs a minimal end-to-end test:
1. Generates a tiny dataset (10 samples)
2. Validates the dataset
3. Visualizes a sample

Run this before generating your full dataset to catch any issues early.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import os
import shutil
import numpy as np

# Test configuration
TEST_OUTPUT_DIR = "data/cube-2x2-test-tiny"
TEST_SIZE = 10


def test_dataset_generation():
    """Test 1: Generate a tiny dataset"""
    print("\n" + "="*70)
    print("TEST 1: Dataset Generation")
    print("="*70)
    
    from dataset.build_2x2_solution import preprocess_data, DataProcessConfig
    
    config = DataProcessConfig(
        output_dir=TEST_OUTPUT_DIR,
        train_size=TEST_SIZE,
        test_size=TEST_SIZE,
        val_size=TEST_SIZE,
        min_scramble_moves=1,
        max_scramble_moves=5,  # Keep it small for testing
        max_solution_length=10,
        num_workers=2  # Use fewer workers for test
    )
    
    # Clean up previous test data
    if os.path.exists(TEST_OUTPUT_DIR):
        shutil.rmtree(TEST_OUTPUT_DIR)
    
    print(f"Generating test dataset: {TEST_SIZE} samples per split...")
    preprocess_data(config)
    
    # Verify files exist
    for split in ["train", "test", "val"]:
        split_dir = Path(TEST_OUTPUT_DIR) / split
        assert split_dir.exists(), f"Split directory not found: {split_dir}"
        assert (split_dir / "dataset.json").exists(), f"Metadata not found for {split}"
        assert (split_dir / "all__inputs.npy").exists(), f"Inputs not found for {split}"
        assert (split_dir / "all__labels.npy").exists(), f"Labels not found for {split}"
    
    print("✅ Dataset generation successful!")
    return True


def test_validation():
    """Test 2: Validate the dataset"""
    print("\n" + "="*70)
    print("TEST 2: Dataset Validation")
    print("="*70)
    
    from dataset.validate_2x2_dataset import validate, ValidationConfig
    
    config = ValidationConfig(
        dataset_dir=TEST_OUTPUT_DIR,
        num_samples_to_check=TEST_SIZE,  # Check all samples
        check_optimality=True,
        verbose=True
    )
    
    print(f"Validating test dataset...")
    validate(config)
    
    print("✅ Dataset validation successful!")
    return True


def test_visualization():
    """Test 3: Visualize a sample"""
    print("\n" + "="*70)
    print("TEST 3: Visualization")
    print("="*70)
    
    from dataset.visualize_2x2_data import visualize, VisualizationConfig
    
    config = VisualizationConfig(
        dataset_dir=TEST_OUTPUT_DIR,
        split="train",
        num_samples=2,
        animate=False,  # No animation in test
        export_html=None
    )
    
    print(f"Visualizing samples...")
    visualize(config)
    
    print("✅ Visualization successful!")
    return True


def test_manual_verification():
    """Test 4: Manual verification of a sample"""
    print("\n" + "="*70)
    print("TEST 4: Manual Sample Verification")
    print("="*70)
    
    import py222
    
    # Load one sample
    inputs = np.load(Path(TEST_OUTPUT_DIR) / "train" / "all__inputs.npy")
    labels = np.load(Path(TEST_OUTPUT_DIR) / "train" / "all__labels.npy")
    
    # Check first sample
    state = inputs[0] - 1  # Convert back to 0-5
    solution = [m - 1 for m in labels[0] if m > 0]  # Convert back to 0-8, remove padding
    
    print(f"Sample 0:")
    print(f"  Initial state: {state}")
    print(f"  Solution: {solution}")
    print(f"  Solution length: {len(solution)}")
    
    # Apply solution
    test_state = state.copy()
    for move in solution:
        test_state = py222.doMove(test_state, move)
    
    # Check if solved
    is_solved = py222.isSolved(test_state)
    print(f"  After applying solution: {'SOLVED ✅' if is_solved else 'NOT SOLVED ❌'}")
    
    if not is_solved:
        print(f"  Final state: {test_state}")
        raise AssertionError("Solution does not solve the cube!")
    
    # Verify it's optimal
    optimal_solution = py222.solve(state)
    print(f"  Optimal solution length from solver: {len(optimal_solution)}")
    
    if len(solution) == len(optimal_solution):
        print(f"  ✅ Solution is optimal!")
    else:
        print(f"  ⚠️  Solution length differs: {len(solution)} vs {len(optimal_solution)}")
    
    print("✅ Manual verification successful!")
    return True


def main():
    """Run all tests"""
    print("\n" + "🧪" * 35)
    print("2x2x2 SOLUTION DATASET - QUICK TEST")
    print("🧪" * 35)
    
    try:
        # Run tests
        test_dataset_generation()
        test_validation()
        test_visualization()
        test_manual_verification()
        
        # Summary
        print("\n" + "="*70)
        print("🎉 ALL TESTS PASSED! 🎉")
        print("="*70)
        print("\nYour dataset tools are working correctly!")
        print("You can now generate the full dataset with:")
        print(f"  python dataset/build_2x2_solution.py")
        print("\nThe test dataset is saved at:")
        print(f"  {TEST_OUTPUT_DIR}")
        print("\nYou can delete it with:")
        print(f"  rm -rf {TEST_OUTPUT_DIR}  (Linux/Mac)")
        print(f"  rmdir /s {TEST_OUTPUT_DIR}  (Windows)")
        
    except Exception as e:
        print("\n" + "="*70)
        print("❌ TEST FAILED!")
        print("="*70)
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
