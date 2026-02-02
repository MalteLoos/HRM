import numpy as np
import json

# Load data
train_inputs = np.load("data/cube-2-by-2-heuristic/train/all__inputs.npy")
train_labels = np.load("data/cube-2-by-2-heuristic/train/all__labels.npy")

print("=" * 70)
print("2x2 RUBIK'S CUBE DATASET INSPECTION")
print("=" * 70)

print(f"\nDataset Shape:")
print(f"  Inputs: {train_inputs.shape}")
print(f"  Labels: {train_labels.shape}")

print(f"\nFirst 5 examples:")
for i in range(5):
    print(f"\n  Example {i}:")
    print(f"    State (24 stickers): {train_inputs[i]}")
    print(f"    Label (next move): {train_labels[i]}")

print(f"\nLabel distribution (next move):")
unique, counts = np.unique(train_labels, return_counts=True)
for label, count in zip(unique, counts):
    move_names = {0: "U", 1: "U'", 2: "U2", 3: "R", 4: "R'", 5: "R2", 6: "F", 7: "F'", 8: "F2", -1: "SOLVED"}
    move_name = move_names.get(label, f"Unknown({label})")
    print(f"  {move_name:10s} (label={label:2d}): {count:6d} examples ({100*count/len(train_labels):.1f}%)")

print(f"\nColor distribution across all states:")
for color in range(6):
    count = np.sum(train_inputs == color)
    print(f"  Color {color}: {count:7d} stickers ({100*count/train_inputs.size:.1f}%)")

print("\n" + "=" * 70)
