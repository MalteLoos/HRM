# Create heuristic dataset for 3x3 Rubik's cube
# Load dataset (of type 3x3)
# Take the scrambled state
# Load it and follow the solution (label)
# For each state during the solution create a pair (state, distance to solved)
#   Store these new pairs as input (cube state) and label (int = distance)
# Uses magiccube for cube representation and kociemba for solving

import os
import json
import sys
from argdantic import ArgParser
import numpy as np
from pathlib import Path
from pydantic import BaseModel
from tqdm import tqdm
import magiccube
import kociemba

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import PuzzleDatasetMetadata

cli = ArgParser()


class DataProcessConfig(BaseModel):
    source_repo: str = "data/cube-3-by-3-4"
    output_dir: str = "data/cube-3-by-3-heuristic-4"

    train_size: int = 10000000
    test_size: int = 2000
    
    min_solution_length: int = 17  # Only include puzzles with at least this many moves


# Face order for kociemba: U, R, F, D, L, B
FACE_ORDER = ['U', 'R', 'F', 'D', 'L', 'B']
SOLVED_CUBE = "UUUUUUUUURRRRRRRRRFFFFFFFFFDDDDDDDDDLLLLLLLLLBBBBBBBBB"

# All possible moves for 3x3x3 cube
MOVES = [
    "U", "U'", "U2",
    "R", "R'", "R2",
    "F", "F'", "F2",
    "D", "D'", "D2",
    "L", "L'", "L2",
    "B", "B'", "B2",
]
MOVE_TO_IDX = {move: idx for idx, move in enumerate(MOVES)}


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


def kociemba_to_magiccube(kociemba_str):
    """Convert kociemba cube string to magiccube format.
    
    Kociemba uses: U, R, F, D, L, B for face colors
    Magiccube uses: W (white), R (red), G (green), Y (yellow), O (orange), B (blue)
    
    Standard color mapping:
    - U (Up/Top) = W (White)
    - D (Down/Bottom) = Y (Yellow)  
    - F (Front) = G (Green)
    - B (Back) = B (Blue)
    - R (Right) = R (Red)
    - L (Left) = O (Orange)
    
    Kociemba face order: U(0-8), R(9-17), F(18-26), D(27-35), L(36-44), B(45-53)
    Magiccube face order: U(0-8), L(9-17), F(18-26), R(27-35), B(36-44), D(45-53)
    """
    # Color mapping from kociemba notation to magiccube colors
    color_map = {'U': 'W', 'R': 'R', 'F': 'G', 'D': 'Y', 'L': 'O', 'B': 'B'}
    
    # Reorder faces from kociemba (U,R,F,D,L,B) to magiccube (U,L,F,R,B,D)
    kociemba_order = list(range(0, 9))     # U stays at 0-8
    kociemba_order += list(range(36, 45))  # L: kociemba 36-44 -> magic 9-17
    kociemba_order += list(range(18, 27))  # F stays at 18-26
    kociemba_order += list(range(9, 18))   # R: kociemba 9-17 -> magic 27-35
    kociemba_order += list(range(45, 54))  # B stays at 36-44
    kociemba_order += list(range(27, 36))  # D: kociemba 27-35 -> magic 45-53
    
    return ''.join([color_map[kociemba_str[i]] for i in kociemba_order])


def magiccube_to_array(cube):
    """Convert magiccube cube state to numpy array of color indices.
    
    Magiccube colors: W, R, G, Y, O, B
    """
    state_str = str(cube).replace(" ", "").replace("\n", "")
    color_to_idx = {'W': 0, 'R': 1, 'G': 2, 'Y': 3, 'O': 4, 'B': 5}
    return [color_to_idx[c] for c in state_str]


def kociemba_array_to_string(arr):
    """Convert numpy array (with +1 offset) back to kociemba string."""
    idx_to_color = {0: 'U', 1: 'R', 2: 'F', 3: 'D', 4: 'L', 5: 'B'}
    # Subtract 1 because dataset has +1 offset for padding token
    return ''.join([idx_to_color[int(c) - 1] for c in arr])


def labels_to_moves(labels):
    """Convert label array back to move list.
    
    Labels have +1 offset (0 is padding), so subtract 1 to get move index.
    """
    moves = []
    for label in labels:
        if label == 0:  # Padding
            break
        move_idx = int(label) - 1
        if 0 <= move_idx < len(MOVES):
            moves.append(MOVES[move_idx])
    return moves


def process(data, size, min_solution_length):
    """Process dataset to create heuristic training pairs."""
    results = {
        "inputs": [],
        "labels": [],
        "puzzle_identifiers": [],
        "puzzle_indices": [0],
        "group_indices": [0],
    }

    example_id = 0
    count = 0
    
    for j in tqdm(range(len(data["inputs"]))):
        if count >= size:
            break

        inputs = data["inputs"][j]
        labels = data["labels"][j]
        
        # Convert labels to moves
        moves = labels_to_moves(labels)
        
        # Skip if solution is too short
        if len(moves) < min_solution_length:
            continue
    
        count += 1

        # Convert input array to kociemba string
        kociemba_str = kociemba_array_to_string(inputs)
        
        # Convert to magiccube format and create cube
        magiccube_str = kociemba_to_magiccube(kociemba_str)
        
        try:
            cube = magiccube.Cube(3, magiccube_str)
        except Exception as e:
            print(f"Error creating cube: {e}")
            continue

        # For each state along the solution path, record (state, distance_to_solved)
        for i, move in enumerate(moves):
            distance_to_solved = len(moves) - i
            current_state = magiccube_to_array(cube)
            
            results['inputs'].append(current_state)
            results['labels'].append(distance_to_solved)
            results['puzzle_identifiers'].append(0)
            
            example_id += 1
            results['puzzle_indices'].append(example_id)
            
            # Apply the move
            cube.rotate(move)
        
        # Also add the solved state with distance 0
        solved_state = magiccube_to_array(cube)
        results['inputs'].append(solved_state)
        results['labels'].append(0)
        results['puzzle_identifiers'].append(0)
        
        example_id += 1
        results['puzzle_indices'].append(example_id)
        
        results['group_indices'].append(example_id)
    
    return results


def create_dataset(set_name, config: DataProcessConfig):
    """Create a dataset split (train/test/val)."""
    data = load_dataset_split(Path(config.source_repo), set_name)

    if set_name == "train":
        size = config.train_size
    else:
        size = config.test_size
    
    results = process(data, size, config.min_solution_length)

    # Metadata
    # seq_len: 54 facelets (9 per face * 6 faces)
    # vocab_size: 6 colors
    metadata = PuzzleDatasetMetadata(
        seq_len=54,        # 9 * 6 = 54 facelets
        vocab_size=6,      # 6 colors (W, R, G, Y, O, B)
        
        pad_id=0,              # Not used for heuristic
        ignore_label_id=None,  # No ignored labels
        
        blank_identifier_id=0,
        num_puzzle_identifiers=1,
        
        total_groups=len(results["group_indices"]) - 1,
        mean_puzzle_examples=1,
        sets=["all"]
    )

    # Save metadata as JSON
    save_dir = Path(config.output_dir).resolve() / set_name
    save_dir.mkdir(parents=True, exist_ok=True)
    
    with open(str(save_dir / "dataset.json"), "w") as f:
        json.dump(metadata.model_dump(), f)
        
    # Save data - convert to numpy arrays with proper shapes
    # Labels need to be 2D: (N, 1) to match dataloader expectations
    results["labels"] = np.array(results["labels"]).reshape(-1, 1)
    results["inputs"] = np.array(results["inputs"])
    results["puzzle_identifiers"] = np.array(results["puzzle_identifiers"])
    results["puzzle_indices"] = np.array(results["puzzle_indices"])
    results["group_indices"] = np.array(results["group_indices"])
    
    for k, v in results.items():
        np.save(str(save_dir / f"all__{k}.npy"), v)
        
    # Save IDs mapping (for visualization only)
    identifiers_path = Path(config.output_dir).resolve() / "identifiers.json"
    identifiers_path.parent.mkdir(parents=True, exist_ok=True)
    with open(str(identifiers_path), "w") as f:
        json.dump(["<blank>"], f)
    
    print(f"Created {set_name} split with {len(results['inputs'])} examples")


@cli.command(singleton=True)
def preprocess_data(config: DataProcessConfig):
    """Generate train, test, and validation datasets."""
    print(f"Creating 3x3 Rubik's cube heuristic dataset")
    print(f"Source: {config.source_repo}")
    print(f"Output: {config.output_dir}")
    print(f"Min solution length: {config.min_solution_length}")
    
    create_dataset("train", config)
    create_dataset("test", config)
    create_dataset("val", config)
    
    print("Dataset generation complete!")


if __name__ == "__main__":
    cli()
