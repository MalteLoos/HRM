# Concatenate multiple heuristic datasets into a single dataset
# Useful for combining datasets generated from different source sets

import os
import json
import sys
from argdantic import ArgParser
import numpy as np
from pathlib import Path
from pydantic import BaseModel
from tqdm import tqdm
from typing import List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import PuzzleDatasetMetadata

cli = ArgParser()


class ConcatConfig(BaseModel):
    source_dirs: List[str] = [
        "data/cube-3-by-3-heuristic",
        "data/cube-3-by-3-heuristic-1",
        "data/cube-3-by-3-heuristic-2",
        "data/cube-3-by-3-heuristic-3",
        "data/cube-3-by-3-heuristic-4",
    ]
    output_dir: str = "data/cube-3-by-3-heuristic-combined"


def load_split_data(data_dir: Path, split: str):
    """Load a dataset split (train/test/val)"""
    split_dir = data_dir / split
    if not split_dir.exists():
        return None

    # Load arrays
    data = {
        'inputs': np.load(split_dir / "all__inputs.npy"),
        'labels': np.load(split_dir / "all__labels.npy"),
        'group_indices': np.load(split_dir / "all__group_indices.npy"),
        'puzzle_indices': np.load(split_dir / "all__puzzle_indices.npy"),
        'puzzle_identifiers': np.load(split_dir / "all__puzzle_identifiers.npy"),
    }
    
    # Load metadata
    with open(split_dir / "dataset.json", "r") as f:
        data['metadata'] = json.load(f)
    
    return data


def concat_split(source_dirs: List[Path], split: str, output_dir: Path):
    """Concatenate a specific split from multiple source directories."""
    all_inputs = []
    all_labels = []
    all_puzzle_identifiers = []
    
    total_examples = 0
    total_groups = 0
    metadata_template = None
    
    print(f"\nProcessing {split} split...")
    
    for source_dir in tqdm(source_dirs, desc=f"Loading {split}"):
        data = load_split_data(source_dir, split)
        if data is None:
            print(f"  Warning: {split} not found in {source_dir}, skipping")
            continue
        
        all_inputs.append(data['inputs'])
        all_labels.append(data['labels'])
        all_puzzle_identifiers.append(data['puzzle_identifiers'])
        
        total_examples += len(data['inputs'])
        total_groups += data['metadata']['total_groups']
        
        if metadata_template is None:
            metadata_template = data['metadata']
        
        print(f"  {source_dir.name}: {len(data['inputs'])} examples, {data['metadata']['total_groups']} groups")
    
    if not all_inputs:
        print(f"  No data found for {split}, skipping")
        return
    
    # Concatenate arrays
    combined_inputs = np.concatenate(all_inputs, axis=0)
    combined_labels = np.concatenate(all_labels, axis=0)
    combined_puzzle_identifiers = np.concatenate(all_puzzle_identifiers, axis=0)
    
    # Recreate indices (simple sequential)
    combined_puzzle_indices = np.arange(len(combined_inputs) + 1, dtype=np.int32)
    combined_group_indices = np.arange(total_groups + 1, dtype=np.int32)
    
    # Update metadata
    metadata = PuzzleDatasetMetadata(
        seq_len=metadata_template['seq_len'],
        vocab_size=metadata_template['vocab_size'],
        pad_id=metadata_template['pad_id'],
        ignore_label_id=metadata_template['ignore_label_id'],
        blank_identifier_id=metadata_template['blank_identifier_id'],
        num_puzzle_identifiers=metadata_template['num_puzzle_identifiers'],
        total_groups=total_groups,
        mean_puzzle_examples=total_examples / total_groups if total_groups > 0 else 1.0,
        sets=metadata_template['sets'],
    )
    
    # Save
    save_dir = output_dir / split
    save_dir.mkdir(parents=True, exist_ok=True)
    
    np.save(str(save_dir / "all__inputs.npy"), combined_inputs)
    np.save(str(save_dir / "all__labels.npy"), combined_labels)
    np.save(str(save_dir / "all__puzzle_identifiers.npy"), combined_puzzle_identifiers)
    np.save(str(save_dir / "all__puzzle_indices.npy"), combined_puzzle_indices)
    np.save(str(save_dir / "all__group_indices.npy"), combined_group_indices)
    
    with open(str(save_dir / "dataset.json"), "w") as f:
        json.dump(metadata.model_dump(), f)
    
    print(f"  Combined {split}: {total_examples} examples, {total_groups} groups")


@cli.command(singleton=True)
def concat_datasets(config: ConcatConfig):
    """Concatenate multiple heuristic datasets into one."""
    print(f"Concatenating datasets:")
    for d in config.source_dirs:
        print(f"  - {d}")
    print(f"Output: {config.output_dir}")
    
    source_dirs = [Path(d) for d in config.source_dirs]
    output_dir = Path(config.output_dir)
    
    # Check which source dirs exist
    existing_dirs = [d for d in source_dirs if d.exists()]
    if not existing_dirs:
        print("Error: No source directories found!")
        return
    
    print(f"\nFound {len(existing_dirs)} source directories")
    
    # Process each split
    for split in ["train", "test", "val"]:
        concat_split(existing_dirs, split, output_dir)
    
    # Copy identifiers.json from first source
    identifiers_src = existing_dirs[0] / "identifiers.json"
    if identifiers_src.exists():
        import shutil
        shutil.copy(str(identifiers_src), str(output_dir / "identifiers.json"))
    else:
        with open(str(output_dir / "identifiers.json"), "w") as f:
            json.dump(["<blank>"], f)
    
    print("\nDataset concatenation complete!")


if __name__ == "__main__":
    cli()
