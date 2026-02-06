"""
2x2x2 Rubik's Cube Dataset Builder - Solution Sequence Version

This script generates a dataset of scrambled 2x2x2 Rubik's cubes with their
OPTIMAL SOLUTION SEQUENCES as labels.

Dataset Structure:
- Input: Scrambled cube state (24 stickers, 6 colors: 0-5)
- Label: Optimal solution sequence (variable length, padded to seq_len)
- Move encoding: 0=PAD, 1-9 represent the 9 moves (U, U', U2, R, R', R2, F, F', F2)

Key Features:
- Uses py222 optimal solver (guaranteed shortest solutions)
- Parallel processing for faster generation
- Deterministic scrambling with different seeds per split
- Data validation during generation
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import os
from argdantic import ArgParser
import numpy as np
from pydantic import BaseModel
from tqdm import tqdm
import py222
from dataset.common import PuzzleDatasetMetadata
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import threading

cli = ArgParser()


class DataProcessConfig(BaseModel):
    output_dir: str = "data/cube-2-by-2-solution"
    
    seed: int = 42
    train_size: int = 150000
    test_size: int = 15000
    val_size: int = 15000
    
    # Scramble depth range (God's number for 2x2 is 11)
    min_scramble_moves: int = 1
    max_scramble_moves: int = 11
    
    # Maximum solution length (for padding)
    # God's number = 11, so this is safe
    max_solution_length: int = 15
    
    # Parallel processing
    num_workers: int = None  # None = use all CPUs


# Move name mapping for visualization/debugging
MOVE_NAMES = ["U", "U'", "U2", "R", "R'", "R2", "F", "F'", "F2"]


def scramble_cube(rng: np.random.Generator, min_moves: int, max_moves: int):
    """
    Generate a random scrambled cube state.
    
    Args:
        rng: Random number generator
        min_moves: Minimum number of scramble moves
        max_moves: Maximum number of scramble moves
        
    Returns:
        Scrambled cube state (24-element numpy array)
    """
    state = py222.initState()
    n_moves = int(rng.integers(min_moves, max_moves + 1))
    
    # Avoid consecutive moves on the same face
    last_face = -1
    for _ in range(n_moves):
        while True:
            move = int(rng.integers(0, 9))  # 9 quarter/half turns
            face = move // 3  # 0=U, 1=R, 2=F
            if face != last_face:
                break
        state = py222.doMove(state, move)
        last_face = face
    
    return state


def generate_sample(rng: np.random.Generator, min_scramble: int, max_scramble: int):
    """
    Generate one scrambled cube and its optimal solution.
    
    Args:
        rng: Random number generator
        min_scramble: Minimum scramble depth
        max_scramble: Maximum scramble depth
        
    Returns:
        tuple: (scrambled_state, solution_moves)
            - scrambled_state: 24-element numpy array
            - solution_moves: list of move indices (empty if already solved)
    """
    scrambled = scramble_cube(rng, min_scramble, max_scramble)
    solution = py222.solve(scrambled)
    return scrambled, solution


def verify_solution(state, solution):
    """
    Verify that applying the solution actually solves the cube.
    
    Args:
        state: Initial scrambled state
        solution: List of move indices
        
    Returns:
        bool: True if solution is valid
    """
    test_state = state.copy()
    for move in solution:
        test_state = py222.doMove(test_state, move)
    return py222.isSolved(test_state)


# ============================================================================
# Parallel Processing Workers (module-level for pickling on Windows)
# ============================================================================

def _init_worker():
    """Initialize worker process (loads py222 module)."""
    try:
        import py222  # noqa: F401
    except Exception:
        pass


def _process_worker_batch(seed, min_scramble, max_scramble, batch_size, progress_queue=None):
    """
    Generate a batch of samples in a worker process.
    
    Args:
        seed: Random seed for this worker
        min_scramble: Minimum scramble depth
        max_scramble: Maximum scramble depth
        batch_size: Number of samples to generate
        progress_queue: Queue for progress reporting
        
    Returns:
        tuple: (samples, worker_pid)
            samples: list of (state, solution) tuples
    """
    local_rng = np.random.default_rng(int(seed))
    samples = []
    
    for _ in range(batch_size):
        scrambled, solution = generate_sample(local_rng, min_scramble, max_scramble)
        
        # Validate solution
        if not verify_solution(scrambled, solution):
            print(f"WARNING: Invalid solution generated! State: {scrambled}, Solution: {solution}")
            continue
            
        samples.append((scrambled, solution))
        
        # Report progress
        if progress_queue is not None:
            try:
                progress_queue.put(1)
            except Exception:
                pass
    
    return samples, os.getpid()


# ============================================================================
# Dataset Creation
# ============================================================================

def create_dataset(split_name: str, size: int, config: DataProcessConfig):
    """
    Create a dataset split (train/test/val).
    
    Args:
        split_name: Name of the split ("train", "test", or "val")
        size: Number of samples to generate
        config: Configuration object
    """
    # Use different seeds for each split
    seed_map = {"train": config.seed, "test": config.seed + 1000, "val": config.seed + 2000}
    seed = seed_map.get(split_name, config.seed)
    rng = np.random.default_rng(seed)
    
    # Initialize result containers
    results = {
        "inputs": [],
        "labels": [],
        "puzzle_identifiers": [],
        "puzzle_indices": [0],
        "group_indices": [0],
    }
    
    example_id = 0
    puzzle_id = 0
    
    # Parallel processing setup
    num_workers = config.num_workers if config.num_workers else (os.cpu_count() or 1)
    batch_size = max(1, size // (num_workers * 4))
    num_jobs = (size + batch_size - 1) // batch_size
    
    # Progress tracking
    manager = multiprocessing.Manager()
    progress_queue = manager.Queue()
    
    def progress_reader(queue, total):
        pbar = tqdm(total=total, desc=f"Generating {split_name}")
        while True:
            msg = queue.get()
            if msg is None:
                break
            pbar.update(msg)
        pbar.close()
    
    reader_thread = threading.Thread(target=progress_reader, args=(progress_queue, size), daemon=True)
    reader_thread.start()
    
    # Generate samples in parallel
    with ProcessPoolExecutor(max_workers=num_workers, initializer=_init_worker) as executor:
        # Create job seeds
        job_seeds = rng.integers(0, 2**32 - 1, size=num_jobs, dtype=np.uint64).tolist()
        
        # Submit jobs
        futures = []
        for i, job_seed in enumerate(job_seeds):
            job_batch_size = batch_size if i < num_jobs - 1 else (size - i * batch_size)
            future = executor.submit(
                _process_worker_batch,
                int(job_seed),
                config.min_scramble_moves,
                config.max_scramble_moves,
                job_batch_size,
                progress_queue
            )
            futures.append(future)
        
        # Collect results
        completed = 0
        for future in as_completed(futures):
            samples, pid = future.result()
            
            for scrambled, solution in samples:
                # Store input (add 1 to shift colors from 0-5 to 1-6, reserving 0 for PAD)
                inp = scrambled + 1
                results["inputs"].append(inp.astype(np.int32))
                
                # Store label (pad solution to max_solution_length)
                label = np.zeros(config.max_solution_length, dtype=np.int32)
                for i, move in enumerate(solution):
                    if i >= config.max_solution_length:
                        print(f"WARNING: Solution too long ({len(solution)}), truncating!")
                        break
                    label[i] = move + 1  # Shift moves: 0-8 -> 1-9, reserving 0 for PAD
                
                results["labels"].append(label)
                
                # Metadata for this example
                example_id += 1
                puzzle_id += 1
                results["puzzle_indices"].append(example_id)
                results["puzzle_identifiers"].append(0)
                results["group_indices"].append(puzzle_id)
                
                completed += 1
                if completed >= size:
                    break
            
            if completed >= size:
                break
    
    # Stop progress thread
    try:
        progress_queue.put(None)
    except Exception:
        pass
    reader_thread.join()
    
    # Convert to numpy arrays
    final_results = {
        "inputs": np.stack(results["inputs"]),
        "labels": np.stack(results["labels"]),
        "group_indices": np.array(results["group_indices"], dtype=np.int32),
        "puzzle_indices": np.array(results["puzzle_indices"], dtype=np.int32),
        "puzzle_identifiers": np.array(results["puzzle_identifiers"], dtype=np.int32),
    }
    
    # Print statistics
    solution_lengths = [np.count_nonzero(label) for label in final_results["labels"]]
    print(f"\n{split_name} Statistics:")
    print(f"  Total samples: {len(solution_lengths)}")
    print(f"  Solution length - min: {min(solution_lengths)}, max: {max(solution_lengths)}, mean: {np.mean(solution_lengths):.2f}")
    
    # Create metadata
    metadata = PuzzleDatasetMetadata(
        seq_len=config.max_solution_length,
        vocab_size=10,  # 0=PAD, 1-9=moves (U, U', U2, R, R', R2, F, F', F2)
        
        pad_id=0,
        ignore_label_id=0,  # We want to predict the PAD tokens correctly (to know when to stop)
        
        blank_identifier_id=0,
        num_puzzle_identifiers=1,
        
        total_groups=len(final_results["group_indices"]) - 1,
        mean_puzzle_examples=1,
        sets=["all"],
    )
    
    # Save dataset
    save_dir = Path(config.output_dir) / split_name
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Save metadata
    with open(save_dir / "dataset.json", "w") as f:
        json.dump(metadata.model_dump(), f, indent=2)
    
    # Save data arrays
    for key, value in final_results.items():
        np.save(save_dir / f"all__{key}.npy", value)
    
    print(f"Saved {split_name} dataset to {save_dir}")


# ============================================================================
# Main Entry Point
# ============================================================================

@cli.command(singleton=True)
def preprocess_data(config: DataProcessConfig):
    """Generate train, test, and validation datasets."""
    print(f"Configuration:")
    print(f"  Output directory: {config.output_dir}")
    print(f"  Scramble depth: {config.min_scramble_moves} - {config.max_scramble_moves}")
    print(f"  Max solution length: {config.max_solution_length}")
    print(f"  Sizes - train: {config.train_size}, test: {config.test_size}, val: {config.val_size}")
    print()
    
    # Create datasets
    create_dataset("train", config.train_size, config)
    create_dataset("test", config.test_size, config)
    create_dataset("val", config.val_size, config)
    
    # Save identifiers mapping
    identifiers_path = Path(config.output_dir) / "identifiers.json"
    with open(identifiers_path, "w") as f:
        json.dump(["<blank>"], f)
    
    # Save move names for reference
    moves_path = Path(config.output_dir) / "move_names.json"
    with open(moves_path, "w") as f:
        json.dump({
            "0": "PAD",
            **{str(i+1): MOVE_NAMES[i] for i in range(9)}
        }, f, indent=2)
    
    print("\nDataset generation complete!")
    print(f"Output directory: {config.output_dir}")


if __name__ == "__main__":
    cli()
