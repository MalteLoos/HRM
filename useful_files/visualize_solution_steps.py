"""
Colorful step-by-step cube solver visualization.

This script:
  1. Takes a scrambled cube state
  2. Solves it with A* using the learned heuristic
  3. Displays each step of the solution with colorful visualization
"""
import numpy as np
import torch
import sys
import copy
from pathlib import Path
from typing import List, Tuple
import magiccube

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent))
from solver import SmallNetHeuristic, ZeroHeuristic, astar_solve, scramble_cube, is_solved


class CubeVisualizer:
    """Helper class to visualize 2x2x2 cube states."""
    
    # Color mapping from magiccube: Y=Yellow, R=Red, G=Green, O=Orange, B=Blue, W=White
    COLOR_SYMBOLS = {
        'W': "⬜",  # White - Top
        'Y': "🟨",  # Yellow - Bottom
        'O': "🟧",  # Orange - Left
        'R': "🟥",  # Red - Right
        'G': "🟩",  # Green - Front
        'B': "🟦",  # Blue - Back
    }
    
    @staticmethod
    def print_cube_state(cube: magiccube.Cube, title: str = ""):
        """Print cube state in unfolded format."""
        if title:
            print(f"\n{title}")
            print("=" * 40)
        
        s = str(cube).replace(" ", "").replace("\n", "")
        
        # magiccube 2x2 str() is a row-based unfolded net:
        #       s[0]  s[1]
        #       s[2]  s[3]
        # s[4]  s[5]  s[6]  s[7]  s[8]  s[9]  s[10] s[11]
        # s[12] s[13] s[14] s[15] s[16] s[17] s[18] s[19]
        #       s[20] s[21]
        #       s[22] s[23]
        #
        # Faces: U=[0-3], L=[4,5,12,13], F=[6,7,14,15],
        #        R=[8,9,16,17], B=[10,11,18,19], D=[20-23]
        
        c = CubeVisualizer.COLOR_SYMBOLS
        
        # Top face (U)
        print(f"       {c[s[0]]} {c[s[1]]}")
        print(f"       {c[s[2]]} {c[s[3]]}")
        print()
        # Middle rows: L F R B
        print(f"{c[s[4]]} {c[s[5]]}  {c[s[6]]} {c[s[7]]}  {c[s[8]]} {c[s[9]]}  {c[s[10]]} {c[s[11]]}")
        print(f"{c[s[12]]} {c[s[13]]}  {c[s[14]]} {c[s[15]]}  {c[s[16]]} {c[s[17]]}  {c[s[18]]} {c[s[19]]}")
        print()
        # Bottom face (D)
        print(f"       {c[s[20]]} {c[s[21]]}")
        print(f"       {c[s[22]]} {c[s[23]]}")
    
    @staticmethod
    def apply_moves_trace(
        cube: magiccube.Cube,
        solution: List[str]
    ) -> List[Tuple[magiccube.Cube, str]]:
        """Apply moves step-by-step and return states + move names."""
        trace = [(copy.deepcopy(cube), "Initial State")]
        
        try:
            current_cube = copy.deepcopy(cube)
            for move in solution:
                current_cube.rotate(move)
                trace.append((copy.deepcopy(current_cube), move))
            return trace
        except Exception as e:
            print(f"Error during move application: {e}")
            return trace


if __name__ == "__main__":
    print("="*70)
    print("STEP-BY-STEP CUBE SOLUTION VISUALIZER")
    print("="*70)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")
    
    # Settings
    SCRAMBLE_LENGTH = 20
    MAX_NODES = 100000
    
    # Load heuristic model
    checkpoint_dir = Path(r"checkpoints\2x2-heuristic-fc\100000_examples")
    checkpoint_path = checkpoint_dir / "best_model.pt"
    
    if not checkpoint_path.exists():
        checkpoint_path = sorted(checkpoint_dir.glob("checkpoint_epoch_*.pt"))[-1]
    
    if not checkpoint_path.exists():
        print(f"⚠ Checkpoint not found in {checkpoint_dir}")
        sys.exit(1)
    
    print(f"Loading heuristic model from {checkpoint_path}...\n")
    heuristic = SmallNetHeuristic(str(checkpoint_path), device=device)
    #heuristic = ZeroHeuristic()
    
    # Generate a random scrambled cube
    print("="*70)
    print("GENERATING SCRAMBLED CUBE")
    print("="*70)
    
    test_cube, scramble = scramble_cube(num_moves=SCRAMBLE_LENGTH)
    
    print(f"\nScramble ({len(scramble)} moves): {' '.join(scramble)}")
    CubeVisualizer.print_cube_state(test_cube, "Scrambled Cube State")
    
    # Solve with A*
    print("\n" + "="*70)
    print("SOLVING WITH A* + HEURISTIC")
    print("="*70)
    print(f"\nMax nodes: {MAX_NODES}")
    
    solution, metrics = astar_solve(test_cube, heuristic, max_nodes=MAX_NODES, batch_size=512)
    
    if solution is not None:
        print(f"\n✅ Solution found in {len(solution)} moves!")
        print(f"Solution: {' '.join(solution)}")
        
        # Validate solution
        print("\n" + "="*70)
        print("VALIDATING SOLUTION")
        print("="*70)
        
        test_cube_copy = copy.deepcopy(test_cube)
        for move in solution:
            test_cube_copy.rotate(move)
        
        is_valid = is_solved(test_cube_copy)
        print(f"\nValidation: {'✅ SOLUTION IS VALID' if is_valid else '❌ SOLUTION IS INVALID'}")
        
        print(f"\nSearch statistics:")
        print(f"  - Solution length: {len(solution)} moves")
        print(f"  - Nodes expanded: {metrics['nodes_expanded']}")
        print(f"  - Heuristic calls: {metrics['heuristic_calls']}")
        print(f"  - Solve time: {metrics['time_seconds']:.3f}s")
        
        # Step-by-step visualization
        print("\n" + "="*70)
        print("STEP-BY-STEP SOLUTION")
        print("="*70)
        
        trace = CubeVisualizer.apply_moves_trace(test_cube, solution)
        
        for i, (state, move) in enumerate(trace):
            if i == 0:
                CubeVisualizer.print_cube_state(state, f"Step {i}: {move}")
            else:
                CubeVisualizer.print_cube_state(state, f"Step {i}: Apply move '{move}'")
            
            if i < len(trace) - 1:
                print()
        
        print("\n" + "="*70)
        print("✅ CUBE SOLVED!")
        print("="*70)
    
    else:
        print(f"\n⚠ Solution not found within {MAX_NODES} nodes")
        print(f"\nSearch statistics:")
        print(f"  - Nodes expanded: {metrics['nodes_expanded']}")
        print(f"  - Heuristic calls: {metrics['heuristic_calls']}")
        print(f"  - Time: {metrics['time_seconds']:.3f}s")
