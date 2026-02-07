"""
Find a 10-move scramble and its solution for online visualization tool.

This script:
  1. Generates random scrambles until one is found that is exactly 10 moves away
  2. Solves it with A* using the learned heuristic
  3. Outputs only the scramble sequence and solution sequence for the online tool
"""
import numpy as np
import torch
import sys
import copy
import random
from pathlib import Path
import magiccube

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent))
from solver import SmallNetHeuristic, astar_solve, scramble_cube, is_solved


if __name__ == "__main__":
    print("="*70)
    print("SEARCHING FOR 10-MOVE OPTIMAL STATE")
    print("="*70)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")
    
    # Target difficulty
    TARGET_SOLUTION_LEN = 10
    MAX_NODES = 100000
    MAX_ATTEMPTS = 200
    
    # Load heuristic model
    checkpoint_dir = Path(r"C:\personal\Uni\5\ML\Project\HRM\2x2-heuristic-fc\20260119_222859")
    checkpoint_path = checkpoint_dir / "best_model.pt"
    
    if not checkpoint_path.exists():
        checkpoint_path = sorted(checkpoint_dir.glob("checkpoint_epoch_*.pt"))[-1]
    
    if not checkpoint_path.exists():
        print(f"⚠ Checkpoint not found in {checkpoint_dir}")
        sys.exit(1)
    
    print(f"Loading heuristic model from {checkpoint_path}...\n")
    heuristic = SmallNetHeuristic(str(checkpoint_path), device=device)
    
    print(f"Searching for a state that is exactly {TARGET_SOLUTION_LEN} moves away from solved...")
    print(f"Max attempts: {MAX_ATTEMPTS}\n")
    
    # Search for an 10-move scramble
    found = False
    attempt = 0
    
    while not found and attempt < MAX_ATTEMPTS:
        attempt += 1
        
        # Generate longer scrambles (18-25 moves) to increase likelihood of harder states
        test_cube, scramble = scramble_cube(num_moves=random.randint(18, 25))
        
        print(f"[{attempt}/{MAX_ATTEMPTS}] Solving scramble...", end="", flush=True)
        
        # Solve with A*
        solution, metrics = astar_solve(test_cube, heuristic, max_nodes=MAX_NODES, batch_size=512)
        
        if solution is not None:
            if len(solution) == TARGET_SOLUTION_LEN:
                print(f" ✓ Found {TARGET_SOLUTION_LEN}-move state!\n")
                found = True
                break
            else:
                print(f" {len(solution)} moves (searching...)")
        else:
            print(f" failed")
    
    if found:
        print("="*70)
        print(f"FOUND STATE {TARGET_SOLUTION_LEN} MOVES AWAY FROM SOLVED!")
        print("="*70)
        
        # Validate
        test_cube_copy = copy.deepcopy(test_cube)
        for move in solution:
            test_cube_copy.rotate(move)
        is_valid = is_solved(test_cube_copy)
        
        print(f"\nScramble ({len(scramble)} moves):")
        print(" ".join(scramble))
        print(f"\nSolution ({len(solution)} moves):")
        print(" ".join(solution))
        
        print(f"\nValidation: {'SOLUTION IS VALID' if is_valid else 'SOLUTION IS INVALID'}")
        print(f"\nSearch statistics:")
        print(f"  - Nodes expanded: {metrics['nodes_expanded']}")
        print(f"  - Heuristic calls: {metrics['heuristic_calls']}")
        print(f"  - Solve time: {metrics['time_seconds']:.3f}s")
        print("="*70)
    else:
        print(f"\n⚠ Could not find {TARGET_SOLUTION_LEN}-move state after {MAX_ATTEMPTS} attempts")
        print(f"Try increasing MAX_ATTEMPTS or adjusting the scramble length")
