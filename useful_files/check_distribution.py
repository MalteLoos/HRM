import numpy as np
from collections import Counter

labels = np.load('data/cube-2-by-2-heuristic/train/all__labels.npy').flatten()
counts = Counter(labels)

print('Label distribution (train):')
total = len(labels)
for dist in sorted(counts.keys()):
    print(f'  Distance {dist:2d}: {counts[dist]:6d} samples ({100*counts[dist]/total:5.2f}%)')
    
print(f'\nTotal: {total} samples')
print(f'Expected: 10000 unique states x 24 rotations = 240000 {"✓" if total == 240000 else "ERROR"}')

# Check if distribution is roughly uniform (each distance should appear ~equally)
expected_per_distance = total / 11  # distances 1-11
print(f'\nExpected per distance (uniform): ~{expected_per_distance:.0f} samples')
print(f'Actual distribution varies as scrambles are random depth [1-11]')
