"""
Find a 10-move scramble and solve it with A* using learned heuristic.
Outputs scramble and solution sequences for visualization.
"""
import numpy as np
import torch
import sys
import copy
import random
from pathlib import Path
import magiccube

sys.path.insert(0, str(Path(__file__).parent.parent))
from solver import SmallNetHeuristic, astar_solve, scramble_cube, is_solved


if __name__ == "__main__":
    print("Searching for 10-move optimal state")
    print("-" * 50)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    TARGET_SOLUTION_LEN = 10
    MAX_NODES = 100000
    MAX_ATTEMPTS = 200
    
    # Load heuristic model
    checkpoint_dir = Path(r"C:\personal\Uni\5\ML\Project\HRM\2x2-heuristic-fc\20260119_222859")
    checkpoint_path = checkpoint_dir / "best_model.pt"
    
    if not checkpoint_path.exists():
        checkpoint_path = sorted(checkpoint_dir.glob("checkpoint_epoch_*.pt"))[-1]
    
    if not checkpoint_path.exists():
        print(f"Error: Checkpoint not found in {checkpoint_dir}")
        sys.exit(1)
    
    print(f"Loading model from {checkpoint_path}")
    heuristic = SmallNetHeuristic(str(checkpoint_path), device=device)
    
    print(f"Searching for {TARGET_SOLUTION_LEN}-move state (max {MAX_ATTEMPTS} attempts)")
    
    found = False
    attempt = 0
    
    while not found and attempt < MAX_ATTEMPTS:
        attempt += 1
        
        # Longer scrambles (18-25) increase likelihood of harder states
        test_cube, scramble = scramble_cube(num_moves=random.randint(18, 25))
        
        print(f"[{attempt}/{MAX_ATTEMPTS}] Solving...", end="", flush=True)
        
        solution, metrics = astar_solve(test_cube, heuristic, max_nodes=MAX_NODES, batch_size=512)
        
        if solution is not None:
            if len(solution) == TARGET_SOLUTION_LEN:
                print(f" Found {TARGET_SOLUTION_LEN}-move state!")
                found = True
                break
            else:
                print(f" {len(solution)} moves")
        else:
            print(" failed")
    
    if found:
        print()
        print(f"Found {TARGET_SOLUTION_LEN}-move state")
        print("-" * 50)
        
        # Validate solution
        test_cube_copy = copy.deepcopy(test_cube)
        for move in solution:
            test_cube_copy.rotate(move)
        is_valid = is_solved(test_cube_copy)
        
        print(f"Scramble ({len(scramble)} moves): {' '.join(scramble)}")
        print(f"Solution ({len(solution)} moves): {' '.join(solution)}")
        print(f"Valid: {is_valid}")
        print(f"\nNodes expanded: {metrics['nodes_expanded']}")
        print(f"Heuristic calls: {metrics['heuristic_calls']}")
        print(f"Solve time: {metrics['time_seconds']:.3f}s")
    else:
        print(f"\nCould not find {TARGET_SOLUTION_LEN}-move state after {MAX_ATTEMPTS} attempts")
        print("Try increasing MAX_ATTEMPTS or adjusting scramble length")
