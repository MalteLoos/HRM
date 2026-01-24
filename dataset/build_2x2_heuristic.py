# Heuristic dataset builder for 2x2 cube with **uniform scramble depths**
# Each example is a single scrambled state with label = optimal distance to solved.

import json
from argdantic import ArgParser
import numpy as np
from pathlib import Path
from pydantic import BaseModel
from tqdm import tqdm

import py222
from common import PuzzleDatasetMetadata


class DataProcessConfig(BaseModel):
    output_dir: str = "data/cube-2-by-2-heuristic"

    train_size: int = 100000
    test_size: int = 1000
    val_size: int = 1000

    min_scramble_moves: int = 1
    max_scramble_moves: int = 11  # God's number for 2x2


def sample_scramble(rng: np.random.Generator, min_moves: int, max_moves: int):
    """Generate a random scramble with uniformly sampled depth and return (state, optimal_distance)."""

    depth = int(rng.integers(min_moves, max_moves + 1))

    # Avoid repeating the same face twice in a row to keep scrambles meaningful
    scramble = []
    last_face = -1
    for _ in range(depth):
        while True:
            move = int(rng.integers(0, 9))  # 9 quarter/half turns: U,U',U2,R,R',R2,F,F',F2
            face = move // 3
            if face != last_face:
                break
        scramble.append(move)
        last_face = face

    # Apply scramble
    state = py222.initState()
    for mv in scramble:
        state = py222.doMove(state, mv)

    # Compute optimal distance using solver (exact for 2x2)
    optimal_solution = py222.solve(state)
    optimal_distance = len(optimal_solution)

    return state, optimal_distance


def create_dataset(split: str, size: int, config: DataProcessConfig):
    rng = np.random.default_rng(42 if split == "train" else (43 if split == "test" else 44))

    results = {
        "inputs": [],
        "labels": [],
        "puzzle_identifiers": [],
        "puzzle_indices": [0],
        "group_indices": [0],
    }

    example_id = 0

    for _ in tqdm(range(size), desc=f"Generating {split}"):
        state, distance = sample_scramble(rng, config.min_scramble_moves, config.max_scramble_moves)

        results["inputs"].append(state.astype(np.int32))
        results["labels"].append(int(distance))
        results["puzzle_identifiers"].append(0)

        example_id += 1
        results["puzzle_indices"].append(example_id)
        results["group_indices"].append(example_id)

    # Metadata
    metadata = PuzzleDatasetMetadata(
        seq_len=24,        # 4 * 6 stickers
        vocab_size=6,      # 6 colors

        pad_id=0,
        ignore_label_id=None,

        blank_identifier_id=0,
        num_puzzle_identifiers=1,

        total_groups=len(results["group_indices"]) - 1,
        mean_puzzle_examples=1,
        sets=["all"],
    )

    # Save
    save_dir = Path(config.output_dir).resolve() / split
    save_dir.mkdir(parents=True, exist_ok=True)

    with open(save_dir / "dataset.json", "w") as f:
        json.dump(metadata.model_dump(), f)

    results["labels"] = np.array(results["labels"], dtype=np.int32).reshape(-1, 1)
    results["inputs"] = np.stack(results["inputs"])
    results["puzzle_identifiers"] = np.array(results["puzzle_identifiers"], dtype=np.int32)
    results["puzzle_indices"] = np.array(results["puzzle_indices"], dtype=np.int32)
    results["group_indices"] = np.array(results["group_indices"], dtype=np.int32)

    for k, v in results.items():
        np.save(save_dir / f"all__{k}.npy", v)

    # Identifiers mapping (placeholder)
    identifiers_path = Path(config.output_dir).resolve() / "identifiers.json"
    identifiers_path.parent.mkdir(parents=True, exist_ok=True)
    with open(identifiers_path, "w") as f:
        json.dump(["<blank>"], f)


cli = ArgParser()

@cli.command(singleton=True)
def preprocess_data(config: DataProcessConfig):
    create_dataset("train", config.train_size, config)
    create_dataset("test", config.test_size, config)
    create_dataset("val", config.val_size, config)


if __name__ == "__main__":
    cli()
