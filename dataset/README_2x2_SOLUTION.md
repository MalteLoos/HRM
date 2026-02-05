# 2x2x2 Rubik's Cube Dataset - Solution Sequence Version

This directory contains tools for building, validating, and visualizing a 2x2x2 Rubik's Cube dataset with **optimal solution sequences** as labels.

## Overview

Unlike the heuristic version (which labels each state with its distance to solved), this dataset contains:
- **Input**: Scrambled cube states (24 stickers, 6 colors)
- **Label**: Optimal solution sequence (e.g., `[U, R', F2, U', R]`)

This is critical for training models to predict the **complete solution path**, not just the distance.

## Files

### 1. `build_2x2_solution.py` - Dataset Builder
Generates training, test, and validation datasets with optimal solutions.

**Key Features:**
- Uses py222 optimal solver (guarantees shortest solutions)
- Parallel processing for fast generation
- Validates solutions during generation
- Configurable scramble depth (default: 1-11 moves)

**Usage:**
```bash
# Default configuration (10k train, 1k test, 1k val)
python dataset/build_2x2_solution.py

# Custom configuration
python dataset/build_2x2_solution.py \
    --output_dir data/my-cube-dataset \
    --train_size 50000 \
    --test_size 5000 \
    --val_size 5000 \
    --min_scramble_moves 1 \
    --max_scramble_moves 11 \
    --max_solution_length 15
```

**Configuration Options:**
- `output_dir`: Where to save the dataset (default: `data/cube-2-by-2-solution`)
- `train_size`: Number of training samples (default: 10000)
- `test_size`: Number of test samples (default: 1000)
- `val_size`: Number of validation samples (default: 1000)
- `min_scramble_moves`: Minimum scramble depth (default: 1)
- `max_scramble_moves`: Maximum scramble depth (default: 11, which is God's number for 2x2)
- `max_solution_length`: Padding length for solutions (default: 15)
- `num_workers`: Number of parallel workers (default: all CPUs)

**Output Structure:**
```
data/cube-2-by-2-solution/
├── identifiers.json           # Puzzle type identifiers
├── move_names.json           # Move vocabulary (0=PAD, 1-9=moves)
├── train/
│   ├── dataset.json          # Metadata
│   ├── all__inputs.npy       # Shape: (N, 24), values: [1-6]
│   ├── all__labels.npy       # Shape: (N, 15), values: [0-9]
│   ├── all__puzzle_indices.npy
│   ├── all__puzzle_identifiers.npy
│   └── all__group_indices.npy
├── test/
│   └── ... (same structure)
└── val/
    └── ... (same structure)
```

**Data Format:**
- **Inputs**: Shape `(N, 24)`, dtype `int32`
  - 24 stickers representing the cube state
  - Values: 1-6 (colors), 0 reserved for padding
  - Encoding: Original colors (0-5) shifted by +1

- **Labels**: Shape `(N, max_solution_length)`, dtype `int32`
  - Optimal solution sequence
  - Values: 0-9 where:
    - 0 = PAD (end of solution)
    - 1-9 = Moves: U, U', U2, R, R', R2, F, F', F2
  - Encoding: Original move indices (0-8) shifted by +1

---

### 2. `validate_2x2_dataset.py` - Dataset Validator
Verifies the dataset is correct before training.

**Validation Checks:**
1. ✓ Data format and shapes
2. ✓ Vocabulary ranges
3. ✓ Solution validity (applies each solution and checks if cube is solved)
4. ✓ Solution optimality (compares with py222 solver)
5. ✓ Statistical analysis (solution length distribution)

**Usage:**
```bash
# Validate all samples (can be slow for large datasets)
python dataset/validate_2x2_dataset.py \
    --dataset_dir data/cube-2-by-2-solution

# Quick validation (check subset of samples)
python dataset/validate_2x2_dataset.py \
    --dataset_dir data/cube-2-by-2-solution \
    --num_samples_to_check 100 \
    --check_optimality False

# Verbose output (show details of failures)
python dataset/validate_2x2_dataset.py \
    --dataset_dir data/cube-2-by-2-solution \
    --verbose True
```

**Example Output:**
```
======================================================================
Validating train split
======================================================================

[1] Checking data format...
  ✓ Number of samples: 10000
  ✓ Input shape: (10000, 24)
  ✓ Label shape: (10000, 15)
  ✓ Input value range: [1, 6]
  ✓ Label value range: [0, 9]

[2] Checking solution validity...
  Verifying solutions: 100%|████████████████| 10000/10000
  ✓ All 10000 checked solutions are valid

[3] Checking solution optimality...
  Checking optimality: 100%|████████████████| 100/100
  ✓ All 100 checked solutions are optimal

[4] Statistical analysis...
  Solution lengths:
    Min: 0
    Max: 11
    Mean: 6.42
    Median: 7.0
    Std: 2.83

  Solution length distribution:
     0 moves:    12 ( 0.12%) 
     1 moves:    45 ( 0.45%) ██
     2 moves:   156 ( 1.56%) ███
     3 moves:   423 ( 4.23%) ████████
     ...
```

---

### 3. `visualize_2x2_data.py` - Dataset Visualizer
View and inspect dataset samples.

**Features:**
- ASCII art cube display with colors
- Solution sequence formatting
- Step-by-step animation
- HTML export for sharing

**Usage:**
```bash
# View 5 random samples from training set
python dataset/visualize_2x2_data.py \
    --dataset_dir data/cube-2-by-2-solution \
    --num_samples 5

# View specific samples
python dataset/visualize_2x2_data.py \
    --dataset_dir data/cube-2-by-2-solution \
    --indices 0 100 1000

# Animate solutions (step-by-step)
python dataset/visualize_2x2_data.py \
    --dataset_dir data/cube-2-by-2-solution \
    --num_samples 3 \
    --animate True \
    --delay 1.0

# Export to HTML
python dataset/visualize_2x2_data.py \
    --dataset_dir data/cube-2-by-2-solution \
    --num_samples 20 \
    --export_html visualization.html
```

**Example Output:**
```
======================================================================
SAMPLE #0
======================================================================

Initial Scrambled State
────────────────────────────────────────
       ┌──┬──┐
       │W │R │
       ├──┼──┤
       │G │Y │
 ┌──┬──┼──┼──┼──┬──┬──┬──┐
 │B│M│R │G │Y │W │M│B│
 ├──┼──┼──┼──┼──┼──┼──┼──┤
 │B│M│R │G │Y │W │M│B│
 └──┴──┼──┼──┼──┴──┴──┴──┘
       │Y│W│
       ├──┼──┤
       │G│R│
       └──┴──┘

Optimal Solution: U R' F2 U' R
Solution Length: 5 moves
✅ Solution verified: Cube solves correctly
```

---

## Quick Start Workflow

### Step 1: Build the Dataset
```bash
python dataset/build_2x2_solution.py \
    --train_size 10000 \
    --test_size 1000 \
    --val_size 1000
```

**Expected time:** ~5-10 minutes for 12k samples (depends on CPU cores)

### Step 2: Validate the Dataset
```bash
python dataset/validate_2x2_dataset.py \
    --dataset_dir data/cube-2-by-2-solution \
    --num_samples_to_check 1000
```

**Expected output:** All checks should pass ✅

### Step 3: Visualize Samples
```bash
# Quick visual check
python dataset/visualize_2x2_data.py \
    --dataset_dir data/cube-2-by-2-solution \
    --num_samples 5 \
    --animate True
```

### Step 4: Train Your Model
The dataset is now ready! Use it with your HRM model training pipeline.

---

## Important Notes

### Dataset Quality Assurance
1. **Solution Optimality**: All solutions are guaranteed optimal by the py222 solver
2. **Solution Validity**: Every solution is verified during generation
3. **No Duplicates**: Each sample is independently generated with different random seeds
4. **Deterministic**: Same seed produces same dataset (reproducibility)

### Data Encoding Details
- **Input encoding**: Colors 0-5 → stored as 1-6 (0 reserved for PAD)
- **Label encoding**: Moves 0-8 → stored as 1-9 (0 reserved for PAD)
- **When loading data**: Remember to subtract 1 to get original indices!

### Comparison with Heuristic Dataset

| Feature | Heuristic (`build_2x2_heuristic.py`) | Solution (`build_2x2_solution.py`) |
|---------|--------------------------------------|-------------------------------------|
| Label type | Single integer (distance) | Sequence (move list) |
| Use case | Value function learning | Policy learning / sequence-to-sequence |
| Label shape | `(N, 1)` | `(N, max_len)` |
| Vocab size | Not applicable | 10 (PAD + 9 moves) |
| Training task | Regression | Sequence prediction |

### Troubleshooting

**Problem:** Dataset generation is slow
- **Solution**: Increase `num_workers` or reduce dataset size

**Problem:** Solutions seem non-optimal
- **Solution**: Run validation with `--check_optimality True` to verify

**Problem:** Out of memory during generation
- **Solution**: Reduce `train_size` or increase `batch_size` in code

**Problem:** Validation shows invalid solutions
- **Solution**: This should never happen! Please report this as a bug.

---

## Advanced Usage

### Custom Scramble Distributions
Edit the scramble depth range to focus on specific difficulty levels:

```bash
# Easy cubes (1-5 moves)
python dataset/build_2x2_solution.py \
    --min_scramble_moves 1 \
    --max_scramble_moves 5

# Hard cubes (8-11 moves, near God's number)
python dataset/build_2x2_solution.py \
    --min_scramble_moves 8 \
    --max_scramble_moves 11
```

### Batch Processing
Process multiple configurations:

```bash
# Generate datasets for different difficulties
for depth in 3 5 7 11; do
    python dataset/build_2x2_solution.py \
        --output_dir data/cube-2x2-depth-${depth} \
        --max_scramble_moves ${depth} \
        --train_size 5000
done
```

---

## References

- **py222 Solver**: Uses IDA* search with pattern database pruning
- **God's Number for 2x2**: 11 moves (maximum distance from solved state)
- **Move Notation**: Standard Rubik's Cube notation (U=Up, R=Right, F=Front, '=counter-clockwise, 2=180°)

---

## License

See main project LICENSE file.
