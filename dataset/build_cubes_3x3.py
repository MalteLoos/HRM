# create dataset of 3x3x3 rubics cubes
# do some random moves using standard notation
# use kociemba solver to find solutions
# 'label' = solution sequence
# 'input' = scrambled cube
#
# Cube string notation (kociemba):
# 54 facelets in order: U1-U9, R1-R9, F1-F9, D1-D9, L1-L9, B1-B9
# Each facelet has one of 6 colors: U, R, F, D, L, B
#
# Moves: U, U', U2, R, R', R2, F, F', F2, D, D', D2, L, L', L2, B, B', B2
# Total 18 moves for 3x3x3 cube

import json
import os
import sys
from argdantic import ArgParser
import numpy as np
from pydantic import BaseModel
from tqdm import tqdm
import kociemba

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import PuzzleDatasetMetadata

cli = ArgParser()


class DataProcessConfig(BaseModel):
    output_dir: str = "data/cube-3-by-3"

    seed: int = 42
    train_size: int = 1000
    test_size: int = 300
    validation_size: int = 300

    min_scramble_moves: int = 10
    max_scramble_moves: int = 25


# Face indices for the cube string (54 characters total)
# U: 0-8, R: 9-17, F: 18-26, D: 27-35, L: 36-44, B: 45-53
FACE_ORDER = ['U', 'R', 'F', 'D', 'L', 'B']
FACE_TO_IDX = {face: idx for idx, face in enumerate(FACE_ORDER)}
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
NUM_MOVES = len(MOVES)  # 18 moves


def get_opposite_face(face):
    """Get the opposite face to avoid redundant consecutive moves."""
    opposites = {'U': 'D', 'D': 'U', 'R': 'L', 'L': 'R', 'F': 'B', 'B': 'F'}
    return opposites.get(face)


def apply_move(cube_str, move):
    """Apply a single move to the cube string representation.
    
    Each face has 9 positions numbered 0-8:
        0 1 2
        3 4 5
        6 7 8
    """
    cube = list(cube_str)
    
    # Define face start indices
    U, R, F, D, L, B = 0, 9, 18, 27, 36, 45
    
    def rotate_face_cw(start):
        """Rotate a face 90 degrees clockwise."""
        old = cube[start:start+9].copy()
        # Corner rotation: 0->2->8->6->0
        # Edge rotation: 1->5->7->3->1
        cube[start+0], cube[start+1], cube[start+2] = old[6], old[3], old[0]
        cube[start+3], cube[start+4], cube[start+5] = old[7], old[4], old[1]
        cube[start+6], cube[start+7], cube[start+8] = old[8], old[5], old[2]
    
    def rotate_face_ccw(start):
        """Rotate a face 90 degrees counter-clockwise."""
        old = cube[start:start+9].copy()
        cube[start+0], cube[start+1], cube[start+2] = old[2], old[5], old[8]
        cube[start+3], cube[start+4], cube[start+5] = old[1], old[4], old[7]
        cube[start+6], cube[start+7], cube[start+8] = old[0], old[3], old[6]
    
    def rotate_face_180(start):
        """Rotate a face 180 degrees."""
        old = cube[start:start+9].copy()
        cube[start+0], cube[start+1], cube[start+2] = old[8], old[7], old[6]
        cube[start+3], cube[start+4], cube[start+5] = old[5], old[4], old[3]
        cube[start+6], cube[start+7], cube[start+8] = old[2], old[1], old[0]
    
    base_move = move[0]
    modifier = move[1:] if len(move) > 1 else ''
    
    if base_move == 'U':
        # U face rotation
        if modifier == '':
            rotate_face_cw(U)
            # Cycle edges: F0-2 -> L0-2 -> B0-2 -> R0-2 -> F0-2
            temp = cube[F:F+3].copy()
            cube[F:F+3] = cube[R:R+3]
            cube[R:R+3] = cube[B:B+3]
            cube[B:B+3] = cube[L:L+3]
            cube[L:L+3] = temp
        elif modifier == "'":
            rotate_face_ccw(U)
            temp = cube[F:F+3].copy()
            cube[F:F+3] = cube[L:L+3]
            cube[L:L+3] = cube[B:B+3]
            cube[B:B+3] = cube[R:R+3]
            cube[R:R+3] = temp
        else:  # U2
            rotate_face_180(U)
            cube[F:F+3], cube[B:B+3] = cube[B:B+3], cube[F:F+3]
            cube[L:L+3], cube[R:R+3] = cube[R:R+3], cube[L:L+3]
    
    elif base_move == 'D':
        # D face rotation
        if modifier == '':
            rotate_face_cw(D)
            # Cycle: F6-8 -> R6-8 -> B6-8 -> L6-8 -> F6-8
            temp = [cube[F+6], cube[F+7], cube[F+8]]
            cube[F+6], cube[F+7], cube[F+8] = cube[L+6], cube[L+7], cube[L+8]
            cube[L+6], cube[L+7], cube[L+8] = cube[B+6], cube[B+7], cube[B+8]
            cube[B+6], cube[B+7], cube[B+8] = cube[R+6], cube[R+7], cube[R+8]
            cube[R+6], cube[R+7], cube[R+8] = temp
        elif modifier == "'":
            rotate_face_ccw(D)
            temp = [cube[F+6], cube[F+7], cube[F+8]]
            cube[F+6], cube[F+7], cube[F+8] = cube[R+6], cube[R+7], cube[R+8]
            cube[R+6], cube[R+7], cube[R+8] = cube[B+6], cube[B+7], cube[B+8]
            cube[B+6], cube[B+7], cube[B+8] = cube[L+6], cube[L+7], cube[L+8]
            cube[L+6], cube[L+7], cube[L+8] = temp
        else:  # D2
            rotate_face_180(D)
            cube[F+6], cube[F+7], cube[F+8], cube[B+6], cube[B+7], cube[B+8] = \
                cube[B+6], cube[B+7], cube[B+8], cube[F+6], cube[F+7], cube[F+8]
            cube[L+6], cube[L+7], cube[L+8], cube[R+6], cube[R+7], cube[R+8] = \
                cube[R+6], cube[R+7], cube[R+8], cube[L+6], cube[L+7], cube[L+8]
    
    elif base_move == 'R':
        # R face rotation
        if modifier == '':
            rotate_face_cw(R)
            # Cycle: U2,5,8 -> F2,5,8 -> D2,5,8 -> B6,3,0 -> U2,5,8
            temp = [cube[U+2], cube[U+5], cube[U+8]]
            cube[U+2], cube[U+5], cube[U+8] = cube[F+2], cube[F+5], cube[F+8]
            cube[F+2], cube[F+5], cube[F+8] = cube[D+2], cube[D+5], cube[D+8]
            cube[D+2], cube[D+5], cube[D+8] = cube[B+6], cube[B+3], cube[B+0]
            cube[B+6], cube[B+3], cube[B+0] = temp
        elif modifier == "'":
            rotate_face_ccw(R)
            temp = [cube[U+2], cube[U+5], cube[U+8]]
            cube[U+2], cube[U+5], cube[U+8] = cube[B+6], cube[B+3], cube[B+0]
            cube[B+6], cube[B+3], cube[B+0] = cube[D+2], cube[D+5], cube[D+8]
            cube[D+2], cube[D+5], cube[D+8] = cube[F+2], cube[F+5], cube[F+8]
            cube[F+2], cube[F+5], cube[F+8] = temp
        else:  # R2
            rotate_face_180(R)
            cube[U+2], cube[U+5], cube[U+8], cube[D+2], cube[D+5], cube[D+8] = \
                cube[D+2], cube[D+5], cube[D+8], cube[U+2], cube[U+5], cube[U+8]
            cube[F+2], cube[F+5], cube[F+8], cube[B+6], cube[B+3], cube[B+0] = \
                cube[B+6], cube[B+3], cube[B+0], cube[F+2], cube[F+5], cube[F+8]
    
    elif base_move == 'L':
        # L face rotation
        if modifier == '':
            rotate_face_cw(L)
            # Cycle: U0,3,6 -> B8,5,2 -> D0,3,6 -> F0,3,6 -> U0,3,6
            temp = [cube[U+0], cube[U+3], cube[U+6]]
            cube[U+0], cube[U+3], cube[U+6] = cube[B+8], cube[B+5], cube[B+2]
            cube[B+8], cube[B+5], cube[B+2] = cube[D+0], cube[D+3], cube[D+6]
            cube[D+0], cube[D+3], cube[D+6] = cube[F+0], cube[F+3], cube[F+6]
            cube[F+0], cube[F+3], cube[F+6] = temp
        elif modifier == "'":
            rotate_face_ccw(L)
            temp = [cube[U+0], cube[U+3], cube[U+6]]
            cube[U+0], cube[U+3], cube[U+6] = cube[F+0], cube[F+3], cube[F+6]
            cube[F+0], cube[F+3], cube[F+6] = cube[D+0], cube[D+3], cube[D+6]
            cube[D+0], cube[D+3], cube[D+6] = cube[B+8], cube[B+5], cube[B+2]
            cube[B+8], cube[B+5], cube[B+2] = temp
        else:  # L2
            rotate_face_180(L)
            cube[U+0], cube[U+3], cube[U+6], cube[D+0], cube[D+3], cube[D+6] = \
                cube[D+0], cube[D+3], cube[D+6], cube[U+0], cube[U+3], cube[U+6]
            cube[F+0], cube[F+3], cube[F+6], cube[B+8], cube[B+5], cube[B+2] = \
                cube[B+8], cube[B+5], cube[B+2], cube[F+0], cube[F+3], cube[F+6]
    
    elif base_move == 'F':
        # F face rotation
        if modifier == '':
            rotate_face_cw(F)
            # Cycle: U6,7,8 -> R0,3,6 -> D2,1,0 -> L8,5,2 -> U6,7,8
            temp = [cube[U+6], cube[U+7], cube[U+8]]
            cube[U+6], cube[U+7], cube[U+8] = cube[L+8], cube[L+5], cube[L+2]
            cube[L+8], cube[L+5], cube[L+2] = cube[D+2], cube[D+1], cube[D+0]
            cube[D+2], cube[D+1], cube[D+0] = cube[R+0], cube[R+3], cube[R+6]
            cube[R+0], cube[R+3], cube[R+6] = temp
        elif modifier == "'":
            rotate_face_ccw(F)
            temp = [cube[U+6], cube[U+7], cube[U+8]]
            cube[U+6], cube[U+7], cube[U+8] = cube[R+0], cube[R+3], cube[R+6]
            cube[R+0], cube[R+3], cube[R+6] = cube[D+2], cube[D+1], cube[D+0]
            cube[D+2], cube[D+1], cube[D+0] = cube[L+8], cube[L+5], cube[L+2]
            cube[L+8], cube[L+5], cube[L+2] = temp
        else:  # F2
            rotate_face_180(F)
            cube[U+6], cube[U+7], cube[U+8], cube[D+2], cube[D+1], cube[D+0] = \
                cube[D+2], cube[D+1], cube[D+0], cube[U+6], cube[U+7], cube[U+8]
            cube[L+8], cube[L+5], cube[L+2], cube[R+0], cube[R+3], cube[R+6] = \
                cube[R+0], cube[R+3], cube[R+6], cube[L+8], cube[L+5], cube[L+2]
    
    elif base_move == 'B':
        # B face rotation
        if modifier == '':
            rotate_face_cw(B)
            # Cycle: U0,1,2 -> L0,3,6 -> D8,7,6 -> R8,5,2 -> U0,1,2
            temp = [cube[U+0], cube[U+1], cube[U+2]]
            cube[U+0], cube[U+1], cube[U+2] = cube[R+2], cube[R+5], cube[R+8]
            cube[R+2], cube[R+5], cube[R+8] = cube[D+8], cube[D+7], cube[D+6]
            cube[D+8], cube[D+7], cube[D+6] = cube[L+6], cube[L+3], cube[L+0]
            cube[L+6], cube[L+3], cube[L+0] = temp
        elif modifier == "'":
            rotate_face_ccw(B)
            temp = [cube[U+0], cube[U+1], cube[U+2]]
            cube[U+0], cube[U+1], cube[U+2] = cube[L+6], cube[L+3], cube[L+0]
            cube[L+6], cube[L+3], cube[L+0] = cube[D+8], cube[D+7], cube[D+6]
            cube[D+8], cube[D+7], cube[D+6] = cube[R+2], cube[R+5], cube[R+8]
            cube[R+2], cube[R+5], cube[R+8] = temp
        else:  # B2
            rotate_face_180(B)
            cube[U+0], cube[U+1], cube[U+2], cube[D+8], cube[D+7], cube[D+6] = \
                cube[D+8], cube[D+7], cube[D+6], cube[U+0], cube[U+1], cube[U+2]
            cube[L+6], cube[L+3], cube[L+0], cube[R+2], cube[R+5], cube[R+8] = \
                cube[R+2], cube[R+5], cube[R+8], cube[L+6], cube[L+3], cube[L+0]
    
    return ''.join(cube)


def cube_str_to_array(cube_str):
    """Convert cube string to numpy array of color indices."""
    color_to_idx = {'U': 0, 'R': 1, 'F': 2, 'D': 3, 'L': 4, 'B': 5}
    return np.array([color_to_idx[c] for c in cube_str], dtype=np.int32)


def scramble_cube(rng, min_moves, max_moves):
    """Scramble a solved cube with random moves."""
    cube = SOLVED_CUBE
    n_moves = rng.integers(min_moves, max_moves + 1)
    
    last_move_face = None
    second_last_face = None
    
    for _ in range(n_moves):
        while True:
            move_idx = rng.integers(0, NUM_MOVES)
            move = MOVES[move_idx]
            move_face = move[0]
            
            # Avoid moving the same face twice in a row
            if move_face == last_move_face:
                continue
            # Avoid patterns like U D U (same face with opposite face in between)
            if move_face == second_last_face and get_opposite_face(move_face) == last_move_face:
                continue
            break
        
        cube = apply_move(cube, move)
        second_last_face = last_move_face
        last_move_face = move_face
    
    return cube


def parse_solution(solution_str):
    """Parse kociemba solution string to move indices."""
    if not solution_str or solution_str.strip() == '':
        return []
    moves = solution_str.strip().split()
    return [MOVE_TO_IDX[move] for move in moves]


def generate_sample(rng, min_scramble, max_scramble):
    """Generate a scrambled cube and its solution."""
    scrambled = scramble_cube(rng, min_scramble, max_scramble)
    
    # Skip if already solved
    if scrambled == SOLVED_CUBE:
        return None, None
    
    try:
        solution_str = kociemba.solve(scrambled)
        solution = parse_solution(solution_str)
    except Exception as e:
        print(f"Error solving cube: {e}")
        return None, None
    
    return scrambled, solution


def create_dataset(set_name, size, config: DataProcessConfig):
    """Create a dataset split (train, test, or val)."""
    # Different seeds for different splits
    if set_name == "train":
        seed = config.seed
    elif set_name == "test":
        seed = config.seed + 1000
    else:  # val
        seed = config.seed + 2000
    
    rng = np.random.default_rng(seed)
    
    # Kociemba solutions are typically <= 20 moves (God's number)
    # But we use 54 to match input length (cube state has 54 facelets)
    seq_length = 54
    max_solution_length = 30  # Upper bound for solution length
    
    # Generate cubes
    results = {
        "inputs": [],
        "labels": [],
        "puzzle_identifiers": [],
        "puzzle_indices": [0],
        "group_indices": [0],
    }
    
    puzzle_id = 0
    example_id = 0
    
    pbar = tqdm(total=size, desc=f"Generating {set_name}")
    attempts = 0
    max_attempts = size * 10  # Prevent infinite loops
    
    while example_id < size and attempts < max_attempts:
        attempts += 1
        scrambled, solution = generate_sample(
            rng,
            config.min_scramble_moves,
            config.max_scramble_moves
        )
        
        if scrambled is None:
            continue
        
        # Convert cube string to color indices and add padding offset
        input_arr = cube_str_to_array(scrambled) + 1  # +1 for padding token
        
        # Create label array with padding
        label = np.zeros(seq_length, dtype=np.int32)
        for i, move_idx in enumerate(solution):
            if i < seq_length:
                label[i] = move_idx + 1  # +1 for padding token
        
        results["inputs"].append(input_arr)
        results["labels"].append(label)
        
        example_id += 1
        puzzle_id += 1
        
        results["puzzle_indices"].append(example_id)
        results["puzzle_identifiers"].append(0)
        results["group_indices"].append(puzzle_id)
        
        pbar.update(1)
    
    pbar.close()
    
    if example_id < size:
        print(f"Warning: Only generated {example_id}/{size} samples")
    
    # Convert to numpy arrays
    final_results = {
        "inputs": np.stack(results["inputs"]),
        "labels": np.stack(results["labels"]),
        "group_indices": np.array(results["group_indices"], dtype=np.int32),
        "puzzle_indices": np.array(results["puzzle_indices"], dtype=np.int32),
        "puzzle_identifiers": np.array(results["puzzle_identifiers"], dtype=np.int32),
    }
    
    # Metadata
    # vocab_size:
    #   input: 6 colors (U, R, F, D, L, B)
    #   output: 18 moves (U, U', U2, R, R', R2, F, F', F2, D, D', D2, L, L', L2, B, B', B2)
    metadata = PuzzleDatasetMetadata(
        seq_len=seq_length,  # 54 (matches cube state length)
        vocab_size=NUM_MOVES + 1,  # 18 moves + 1 for padding
        pad_id=0,
        ignore_label_id=0,
        
        blank_identifier_id=0,
        num_puzzle_identifiers=1,
        
        total_groups=len(final_results["group_indices"]) - 1,
        mean_puzzle_examples=1,
        sets=["all"],
    )
    
    # Save metadata as JSON
    save_dir = os.path.join(config.output_dir, set_name)
    os.makedirs(save_dir, exist_ok=True)
    
    with open(os.path.join(save_dir, "dataset.json"), "w") as f:
        json.dump(metadata.model_dump(), f)
    
    # Save data
    for k, v in final_results.items():
        np.save(os.path.join(save_dir, f"all__{k}.npy"), v)
    
    # Save IDs mapping (for visualization only)
    with open(os.path.join(config.output_dir, "identifiers.json"), "w") as f:
        json.dump(["<blank>"], f)
    
    # Save move vocabulary for reference
    with open(os.path.join(config.output_dir, "moves.json"), "w") as f:
        json.dump({"pad": 0, **{move: idx + 1 for idx, move in enumerate(MOVES)}}, f)


@cli.command(singleton=True)
def preprocess_data(config: DataProcessConfig):
    """Generate train, test, and validation datasets."""
    print(f"Creating 3x3x3 Rubik's cube dataset in {config.output_dir}")
    print(f"Using kociemba solver")
    print(f"Scramble moves: {config.min_scramble_moves}-{config.max_scramble_moves}")
    
    create_dataset("train", config.train_size, config)
    create_dataset("test", config.test_size, config)
    create_dataset("val", config.validation_size, config)
    
    print("Dataset generation complete!")


if __name__ == "__main__":
    cli()
