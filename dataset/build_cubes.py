# create dataset of rubics cubes
# take param N for NxNxN cubes
# do some random moves aprox. 10 ~ N using magiccube https://pypi.org/project/magiccube/
# 'label' = solved cube
# 'input' = scrambled cube
# metadata:
#  define vocab: pad + input (pieces) + output (all moves)
#  ignorelable pad (0)
#  sequence length
#
# Each token encodes piece_id, orientation
# Position from sequence index

import json
import os
from argdantic import ArgParser
import numpy as np
from pydantic import BaseModel
import magiccube

from common import PuzzleDatasetMetadata

cli = ArgParser()

class DataProcessConfig(BaseModel):
    output_dir: str = "data/cube-3-by-3"

    dim: int = 3
    seed: int = 42
    train_size: int = 1000
    test_size: int = 300
    validation_size: int = 300


# I think there is something in magiccube already
def get_piece_type(pos, N):
    x, y, z = pos
    on_face = sum([x == 0 or x == N-1, y == 0 or y == N-1, z == 0 or z == N-1])
    return on_face  # 3=corner, 2=edge, 1=center


# I think there is something in magiccube already
def get_orientation_id(piece, piece_type):
    colors = piece.get_piece_colors()
    
    if piece_type == 3:  # Corner - 3 orientations
        for i, c in enumerate(colors):
            if c is not None:
                return i % 3
        return 0
    elif piece_type == 2:  # Edge - 2 orientations
        for i, c in enumerate(colors):
            if c is not None:
                return i % 2
        return 0
    else:  # Center - 1 orientation
        return 0

MAX_ORIENTATIONS = 3
def generate_cube(N, solved_pieces_dict, piece_id_map):
    scrambled = magiccube.Cube(N)

    n_moves = 10 * N # Apply 10 * N random moves to scramble
    scrambled.scramble(n_moves)

    # maybe take somthing from magiccube
    def encode_cube_state(cube):
        pieces_dict = cube.get_all_pieces()
        encoded = []
        
        for pos_idx, (pos, piece) in enumerate(pieces_dict.items()):
            # Find piece identity by matching colors
            piece_colors = set(c for c in piece.get_piece_colors() if c is not None)
            
            piece_id = pos_idx  # Default
            for solved_pos, solved_piece in solved_pieces_dict.items():
                solved_colors = set(c for c in solved_piece.get_piece_colors() if c is not None)
                if piece_colors == solved_colors:
                    piece_id = piece_id_map[solved_pos]
                    break
            
            piece_type = get_piece_type(pos, N)
            orientation = get_orientation_id(piece, piece_type)
            
            # Encode: token = 1 + piece_id * MAX_ORIENTATIONS + orientation
            # +1 to reserve 0 for PAD
            token = 1 + piece_id * MAX_ORIENTATIONS + orientation
            encoded.append(token)
        
        return np.array(encoded, dtype=np.int32)
    
    # For solved state, piece at position i has piece_id=i, orientation=0
    solved_encoded = np.array([1 + i * MAX_ORIENTATIONS + 0 for i in range(len(solved_pieces_dict))], dtype=np.int32)
    scrambled_encoded = encode_cube_state(scrambled)
    
    return scrambled_encoded, solved_encoded


def create_dataset(set_name, size, config: DataProcessConfig):
    np.random.seed(config.seed if set_name == "train" else config.seed + hash(set_name) % 1000) # train, test, val should not have same seed
    
    solved = magiccube.Cube(config.dim)
    solved_pieces_dict = solved.get_all_pieces()
    piece_id_map = {pos: idx for idx, pos in enumerate(solved_pieces_dict.keys())}
    num_pieces = len(solved_pieces_dict)
    
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
    
    for _ in range(size):
        scrambled, solved_state = generate_cube(config.dim, solved_pieces_dict, piece_id_map)
        
        results["inputs"].append(scrambled)
        results["labels"].append(solved_state)
        
        example_id += 1
        puzzle_id += 1

        results["puzzle_indices"].append(example_id)
        results["puzzle_identifiers"].append(0)

        # Push group
        results["group_indices"].append(puzzle_id)
    
    # Convert to numpy arrays
    final_results = {
        "inputs": np.stack(results["inputs"]),
        "labels": np.stack(results["labels"]),
        "group_indices": np.array(results["group_indices"], dtype=np.int32),
        "puzzle_indices": np.array(results["puzzle_indices"], dtype=np.int32),
        "puzzle_identifiers": np.array(results["puzzle_identifiers"], dtype=np.int32),
    }

    # Metadata
    num_pieces = (np.pow(config.dim, 3) - np.pow(config.dim-2, 3))
    num_moves = config.dim * 3 * 3
    metadata = PuzzleDatasetMetadata(
        seq_len=num_pieces,
        vocab_size=num_moves + 1,  # pad + only moves (not pieces), since the input is handled differently
        pad_id=0,
        ignore_label_id=0,
        
        blank_identifier_id=0,
        num_puzzle_identifiers=1,
        
        total_groups=len(final_results["group_indices"]) - 1,
        mean_puzzle_examples=1,
        sets=["all"],
    )

    # Save metadata as JSON.
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


@cli.command(singleton=True)
def preprocess_data(config: DataProcessConfig):
    create_dataset("train", config.train_size, config)
    create_dataset("test", config.test_size, config)
    create_dataset("val", config.validation_size, config)


if __name__ == "__main__":
    cli()
