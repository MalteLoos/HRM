# create dataset of rubics cubes
# take param N for NxNxN cubes
# do some random moves aprox. 10 ~ N using magiccube https://pypi.org/project/magiccube/
# 'label' = solution sequence
# 'input' = scrambled cube
#  define vocab: pad + max(input (pieces) + output (all moves))
#  ignorelable pad (0)
#  sequence length
#
# Each token encodes piece_id, orientation
# Position from sequence index


import json
import os
import sys
from argdantic import ArgParser
import numpy as np
from pydantic import BaseModel
from tqdm import tqdm
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import py222

from common import PuzzleDatasetMetadata

cli = ArgParser()

class DataProcessConfig(BaseModel):
    output_dir: str = "data/cube-2-by-2"

    seed: int = 42
    train_size: int = 1000
    test_size: int = 300
    validation_size: int = 300
    
    min_scramble_moves: int = 5
    max_scramble_moves: int = 20


def scramble_cube(rng, min_moves, max_moves):
    s = py222.initState()
    n_moves = rng.integers(min_moves, max_moves + 1)
    
    last_move_face = -1  # avoid moving twice the same side
    for _ in range(n_moves):
        while True:
            move = rng.integers(0, 9)  # we only need U R F since for the 2x2 F is the same as B'
            move_face = move // 3  # 0=U, 1=R, 2=F
            if move_face != last_move_face:
                break
        s = py222.doMove(s, move)
        last_move_face = move_face
    
    return s


def generate_sample(rng, min_scramble, max_scramble):
    scrambled = scramble_cube(rng, min_scramble, max_scramble)
    solution = py222.solve(scrambled)
    return scrambled, solution


def create_dataset(set_name, size, config: DataProcessConfig):
    # train, test, val should not have same seed
    if set_name == "train":
        seed = config.seed
    elif set_name == "test":
        seed = config.seed + 1000
    else:  # val
        seed = config.seed + 2000
    
    rng = np.random.default_rng(seed)
    
    seq_length = 24  # state string has 24 chars, larger than max moves 11
    
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
    
    for _ in tqdm(range(size)):
        scrambled, solution = generate_sample(
            rng, 
            config.min_scramble_moves, 
            config.max_scramble_moves
        )
        
        input = scrambled + 1  # add padding

        label = np.zeros(seq_length, dtype=np.int32) # pad labels to seq length
        for i, move in enumerate(solution):
            label[i] = move + 1  # add padding
        
        results["inputs"].append(input.astype(np.int32))
        results["labels"].append(label)
        
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
    # vocab_size:
    #   input: 6 colors
    #   output: 2x2 cube has only 9 moves: U, U', U2, R, R', R2, F, F', F2
    num_moves = 9
    metadata = PuzzleDatasetMetadata(
        seq_len=seq_length,  # must match input length (24)
        vocab_size=num_moves + 1,  # equals the larger vocab + 1 for pad
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
