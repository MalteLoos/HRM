"""
2x2x2 Cube Dataset Visualization Tool

This script provides visualization capabilities for the 2x2x2 cube dataset,
allowing you to inspect individual samples and verify the data visually.

Features:
- Display cube states in a readable ASCII format
- Show solution sequences with move names
- Animate the solution step-by-step
- Export samples to HTML for interactive viewing

Usage:
    # View random samples
    python visualize_2x2_data.py --dataset_dir data/cube-2-by-2-solution --num_samples 5
    
    # View specific sample indices
    python visualize_2x2_data.py --dataset_dir data/cube-2-by-2-solution --indices 0 10 100
    
    # Animate solutions
    python visualize_2x2_data.py --dataset_dir data/cube-2-by-2-solution --animate --num_samples 3
    
    # Export to HTML
    python visualize_2x2_data.py --dataset_dir data/cube-2-by-2-solution --export_html output.html --num_samples 10
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import numpy as np
from argdantic import ArgParser
from pydantic import BaseModel
from typing import List, Optional
import py222


cli = ArgParser()


class VisualizationConfig(BaseModel):
    dataset_dir: str = "data/cube-2-by-2-solution"
    split: str = "train"  # Which split to visualize
    
    # Sample selection
    num_samples: int = 5
    indices: Optional[List[int]] = None  # Specific indices to visualize
    
    # Display options
    animate: bool = False  # Show step-by-step solution animation
    delay: float = 0.5  # Delay between animation frames (seconds)
    
    # Export
    export_html: Optional[str] = None  # Export to HTML file


MOVE_NAMES = ["U", "U'", "U2", "R", "R'", "R2", "F", "F'", "F2"]

# Color codes for terminal display
COLOR_CODES = {
    0: "\033[97m",  # White (U)
    1: "\033[91m",  # Red (R)
    2: "\033[92m",  # Green (F)
    3: "\033[93m",  # Yellow (D)
    4: "\033[94m",  # Blue (L)
    5: "\033[95m",  # Magenta (B)
}
RESET = "\033[0m"

# HTML color codes
HTML_COLORS = {
    0: "#FFFFFF",  # White
    1: "#FF0000",  # Red
    2: "#00FF00",  # Green
    3: "#FFFF00",  # Yellow
    4: "#0000FF",  # Blue
    5: "#FF00FF",  # Magenta
}


def load_dataset(dataset_dir: Path, split: str):
    """Load a dataset split."""
    split_dir = dataset_dir / split
    
    if not split_dir.exists():
        raise FileNotFoundError(f"Split not found: {split_dir}")
    
    # Load metadata
    with open(split_dir / "dataset.json", "r") as f:
        metadata = json.load(f)
    
    # Load data
    inputs = np.load(split_dir / "all__inputs.npy")
    labels = np.load(split_dir / "all__labels.npy")
    
    return metadata, inputs, labels


def print_cube_state(state, title="Cube State", use_colors=True):
    """
    Print a 2x2x2 cube state in ASCII art.
    
    The cube layout:
           ┌──┬──┐
           │ 0│ 1│  (U - Up face)
           ├──┼──┤
           │ 2│ 3│
     ┌──┬──┼──┼──┼──┬──┬──┬──┐
     │16│17│ 8│ 9│ 4│ 5│20│21│  (L F R B faces)
     ├──┼──┼──┼──┼──┼──┼──┼──┤
     │18│19│10│11│ 6│ 7│22│23│
     └──┴──┼──┼──┼──┴──┴──┴──┘
           │12│13│  (D - Down face)
           ├──┼──┤
           │14│15│
           └──┴──┘
    """
    print(f"\n{title}")
    print("─" * 40)
    
    # Color symbols
    symbols = ['W', 'R', 'G', 'Y', 'B', 'M']
    
    def get_sticker(idx):
        color = state[idx]
        symbol = symbols[color]
        if use_colors:
            return f"{COLOR_CODES[color]}{symbol}{RESET}"
        return symbol
    
    # Top face (U)
    print(f"       ┌──┬──┐")
    print(f"       │{get_sticker(0)} │{get_sticker(1)} │")
    print(f"       ├──┼──┤")
    print(f"       │{get_sticker(2)} │{get_sticker(3)} │")
    
    # Middle row (L F R B)
    print(f" ┌──┬──┼──┼──┼──┬──┬──┬──┐")
    print(f" │{get_sticker(16)}│{get_sticker(17)}│{get_sticker(8)} │{get_sticker(9)} │{get_sticker(4)} │{get_sticker(5)} │{get_sticker(20)}│{get_sticker(21)}│")
    print(f" ├──┼──┼──┼──┼──┼──┼──┼──┤")
    print(f" │{get_sticker(18)}│{get_sticker(19)}│{get_sticker(10)}│{get_sticker(11)}│{get_sticker(6)} │{get_sticker(7)} │{get_sticker(22)}│{get_sticker(23)}│")
    print(f" └──┴──┼──┼──┼──┴──┴──┴──┘")
    
    # Bottom face (D)
    print(f"       │{get_sticker(12)}│{get_sticker(13)}│")
    print(f"       ├──┼──┤")
    print(f"       │{get_sticker(14)}│{get_sticker(15)}│")
    print(f"       └──┴──┘")


def format_solution(moves):
    """Format a solution sequence for display."""
    if len(moves) == 0:
        return "ALREADY SOLVED"
    return " ".join([MOVE_NAMES[m] for m in moves])


def visualize_sample(inputs, labels, idx, animate=False, delay=0.5):
    """Visualize a single sample."""
    print("\n" + "=" * 70)
    print(f"SAMPLE #{idx}")
    print("=" * 70)
    
    # Get data (convert from stored format)
    state = inputs[idx] - 1  # Shift back from [1,6] to [0,5]
    solution_encoded = labels[idx]
    solution = [m - 1 for m in solution_encoded if m > 0]  # Shift back and remove padding
    
    # Display initial state
    print_cube_state(state, title="Initial Scrambled State")
    
    # Display solution
    print(f"\nOptimal Solution: {format_solution(solution)}")
    print(f"Solution Length: {len(solution)} moves")
    
    # Animate solution if requested
    if animate and len(solution) > 0:
        import time
        
        print("\n" + "─" * 70)
        print("SOLUTION ANIMATION")
        print("─" * 70)
        
        current_state = state.copy()
        
        for step, move in enumerate(solution):
            print(f"\nStep {step + 1}/{len(solution)}: Apply {MOVE_NAMES[move]}")
            current_state = py222.doMove(current_state, move)
            print_cube_state(current_state, title=f"After {MOVE_NAMES[move]}")
            
            if step < len(solution) - 1:  # Don't delay after last step
                time.sleep(delay)
        
        # Verify solved
        if py222.isSolved(current_state):
            print("\n✅ Cube is SOLVED!")
        else:
            print("\n❌ ERROR: Cube is NOT solved after applying solution!")
    
    else:
        # Just verify the solution without animation
        final_state = state.copy()
        for move in solution:
            final_state = py222.doMove(final_state, move)
        
        if py222.isSolved(final_state):
            print("✅ Solution verified: Cube solves correctly")
        else:
            print("❌ ERROR: Solution does NOT solve the cube!")


def export_to_html(dataset_dir: Path, split: str, indices: List[int], output_file: str):
    """Export samples to an interactive HTML file."""
    metadata, inputs, labels = load_dataset(dataset_dir, split)
    
    html_parts = [
        "<!DOCTYPE html>",
        "<html>",
        "<head>",
        "    <meta charset='UTF-8'>",
        "    <title>2x2x2 Cube Dataset Visualization</title>",
        "    <style>",
        "        body { font-family: 'Courier New', monospace; background: #1e1e1e; color: #d4d4d4; padding: 20px; }",
        "        .sample { border: 2px solid #444; margin: 20px 0; padding: 20px; background: #252526; border-radius: 8px; }",
        "        .sample-header { font-size: 20px; font-weight: bold; color: #4ec9b0; margin-bottom: 15px; }",
        "        .cube-state { display: inline-block; margin: 10px; }",
        "        .cube-face { display: grid; grid-template-columns: repeat(2, 30px); gap: 2px; margin: 5px; }",
        "        .sticker { width: 30px; height: 30px; border: 1px solid #000; display: flex; align-items: center; justify-content: center; font-weight: bold; }",
        "        .solution { font-size: 16px; color: #ce9178; margin: 10px 0; }",
        "        .verified { color: #4ec9b0; font-weight: bold; }",
        "        .error { color: #f48771; font-weight: bold; }",
        "        pre { background: #1e1e1e; padding: 10px; border-radius: 4px; overflow-x: auto; }",
        "    </style>",
        "</head>",
        "<body>",
        f"    <h1>2x2x2 Cube Dataset Visualization</h1>",
        f"    <p>Dataset: {dataset_dir} / {split}</p>",
        f"    <p>Showing {len(indices)} samples</p>",
    ]
    
    for idx in indices:
        state = inputs[idx] - 1
        solution_encoded = labels[idx]
        solution = [m - 1 for m in solution_encoded if m > 0]
        
        # Verify solution
        test_state = state.copy()
        for move in solution:
            test_state = py222.doMove(test_state, move)
        is_solved = py222.isSolved(test_state)
        
        html_parts.append(f"    <div class='sample'>")
        html_parts.append(f"        <div class='sample-header'>Sample #{idx}</div>")
        html_parts.append(f"        <div class='solution'>Solution: {format_solution(solution)} ({len(solution)} moves)</div>")
        
        # Simple cube visualization (just show the unfolded net)
        html_parts.append(f"        <pre>")
        html_parts.append(f"State: {state.tolist()}")
        html_parts.append(f"        </pre>")
        
        # Verification status
        status_class = "verified" if is_solved else "error"
        status_text = "✅ Solution Verified" if is_solved else "❌ Solution Error"
        html_parts.append(f"        <div class='{status_class}'>{status_text}</div>")
        html_parts.append(f"    </div>")
    
    html_parts.extend([
        "</body>",
        "</html>"
    ])
    
    # Write to file
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("\n".join(html_parts))
    
    print(f"\n✅ Exported to {output_file}")


@cli.command(singleton=True)
def visualize(config: VisualizationConfig):
    """Visualize dataset samples."""
    dataset_dir = Path(config.dataset_dir)
    
    # Load dataset
    try:
        metadata, inputs, labels = load_dataset(dataset_dir, config.split)
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        return
    
    num_samples = len(inputs)
    print(f"Loaded {config.split} split: {num_samples} samples")
    
    # Determine which indices to visualize
    if config.indices:
        indices = config.indices
        # Validate indices
        invalid = [i for i in indices if i < 0 or i >= num_samples]
        if invalid:
            print(f"❌ Error: Invalid indices {invalid}. Dataset has {num_samples} samples.")
            return
    else:
        # Random selection
        indices = np.random.choice(num_samples, size=min(config.num_samples, num_samples), replace=False)
        indices = sorted(indices)
    
    print(f"Visualizing {len(indices)} samples: {indices}")
    
    # Export to HTML if requested
    if config.export_html:
        export_to_html(dataset_dir, config.split, indices, config.export_html)
    
    # Display samples
    for idx in indices:
        visualize_sample(inputs, labels, idx, animate=config.animate, delay=config.delay)
    
    print("\n" + "=" * 70)
    print(f"Visualization complete!")
    print("=" * 70)


if __name__ == "__main__":
    cli()
