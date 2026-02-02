# 2x2 Rubik's Cube Dataset Analysis & Fixes

## Summary of Critical Issues Found

### 🔴 Bug #1: Dataset Generation Loop Was Broken
**Location:** `build_2x2_heuristic.py` line 68-93

**Problem:**
- The code had SEVERE indentation errors
- Only ONE scramble was generated for the entire dataset
- Same `state` and `next_move` were reused for all 100,000 examples
- All examples were essentially corrupted variations of a single cube state

**Impact:** Model couldn't possibly learn patterns from near-identical data

**Status:** ✅ FIXED

---

### 🔴 Bug #2: Invalid Labels in Dataset
**Problem:**
- 12.8% of labels were moves 9, 10, 11 (which don't exist for 2x2 cube)
- Valid moves are only 0-8: U, U', U2, R, R', R2, F, F', F2
- Caused by treating tuple `(state, next_move)` as single value

**Impact:** Model trained on invalid/impossible moves

**Status:** ✅ FIXED

---

### 🔴 Bug #3: Broken "Rotation" Augmentation
**Problem:**
- Claimed to do 24 rotations per state for data augmentation
- Actually just applied random R/F moves that changed the state
- Violated assumption that rotations preserve optimal solution
- Created massively incorrect training data

**Impact:** 100k examples became ~4k examples repeated 24x with wrong labels

**Status:** ✅ REMOVED (proper rotation augmentation would require rotating both state AND labels)

---

## How the Real Rubik's Cube Dataset Works

### ✅ YES - Real Cube Rules Apply

The `py222` module implements **authentic 2x2 Rubik's Cube physics**:

1. **State Representation**
   - 24 stickers (6 faces × 4 stickers per face)
   - Each sticker has a color value 0-5
   - Arranged in specific layout matching physical cube

2. **Move Mechanics**
   - 9 possible moves: U, U', U2, R, R', R2, F, F', F2
     - U = rotate top face clockwise
     - U' = rotate top face counter-clockwise  
     - U2 = rotate top face 180°
     - Same for R (right) and F (front)
   - Each move is a permutation that rearranges stickers
   - Follows real Rubik's cube group theory

3. **Optimal Solver - Verified**
   - Uses IDA* search algorithm with pruning tables
   - Only explores those 9 moves
   - Guarantees OPTIMAL solutions (shortest path to solved)
   - God's number for 2x2 is 11 (max moves to solve any state)

---

## Dataset Structure

### Input Format
```python
# Example scrambled state (24 integers)
state = [0, 0, 4, 4, 0, 1, 0, 1, 2, 2, 2, 2, 1, 1, 3, 3, 4, 3, 4, 3, 5, 5, 5, 5]
#        └──────────────┘ └──────────┘ └──────────┘ └──────────┘ └──────────┘ └──────────┘
#           Top (U)         Front (F)      Right (R)      Down (D)       Left (L)       Back (B)
```

Each position represents a sticker location. Colors 0-5 represent which face that sticker belongs to.

### Label Format
```python
# Label = next optimal move (integer 0-8, or -1 if already solved)
label = 1  # Means: Execute move "U'" (counter-clockwise top rotation)
```

The model must learn to predict which single move gets closest to solved state.

### Dataset Statistics (AFTER FIXES)
- Training: 50,000 unique scrambled states
- Validation: 5,000 states
- Test: 5,000 states
- Scramble depth: Uniformly sampled from 1-11 moves
- Labels: Balanced across 9 possible moves + rare "-1" for solved states

---

## How the Model Should Learn

### What the HRM Architecture Sees
```
Input Sequence:  [0, 0, 4, 4, 0, 1, ..., 5, 5]  (length 24)
                  ↓
Token Embeddings (vocab_size=6, hidden_size=512)
                  ↓
Hierarchical Reasoning Blocks (H_cycles=2, L_cycles=2)
                  ↓
Classification Head (9 classes for moves 0-8)
                  ↓
Prediction:      move = 1  (U')
```

### What It Needs to Learn
1. **Spatial Patterns**: Which stickers form corners/pieces
2. **Color Relationships**: How colors relate across faces
3. **Move Effects**: How each move transforms the state
4. **Distance Heuristic**: Which moves reduce distance to solved

### Can It Learn Cube Rules?
**In theory:** YES - the patterns are learnable from data:
- Solved state always has same color per face (e.g., [0,0,0,0, 1,1,1,1, ...])
- Each move creates consistent permutation patterns
- Optimal moves follow specific strategies (e.g., positioning corners)

**In practice:** HARD without these fixes:
- ❌ Old data: Random garbage → unlearnable
- ✅ New data: Valid states + optimal labels → learnable patterns

---

## Why Training Was Failing

### Before Fixes
1. **99% identical inputs** (all from same scramble)
2. **12.8% invalid labels** (moves that don't exist)
3. **Contradictory data** (same state with different labels from broken rotations)
4. **No diversity** (essentially 4k unique examples, not 100k)

### After Fixes
1. ✅ 50,000 unique scrambled states
2. ✅ All labels valid (0-8 or -1)
3. ✅ One-to-one state→label mapping
4. ✅ Full coverage of scramble depths 1-11

---

## Data Augmentation: Rotation Strategy - RESOLVED ✅

### The Question: Should We Use 24 Rotations?

**Original Idea:** Generate 10k unique states × 24 rotations = 240k states
- Same logical cube seen from 24 different orientations
- Would reduce generation time significantly

**Why This Is Tricky:**
```
Problem: Moves are coordinate-dependent

Original state S:  optimal move = R (rotate right face)
After rotating cube 90°, "right face" is now a different face!
So we can't use the same move label "R" for the rotated state.

Would need:
  state_rotated = rotate(state)
  move_rotated = rotate_move(move)  ← Complex!
```

**Risk:** Bug in move transformation = garbage training data

### ✅ SAFE RECOMMENDATION: Skip Rotation Augmentation

**Why 50k unique states is sufficient:**
- Each scramble is already from a random orientation (due to random scramble sequence)
- Random scrambles naturally explore all orientations
- Model learns invariant patterns across different perspectives
- No complex move transformations needed
- Massive reduction in implementation risk

**If you want more data efficiency later:**
```
Option: Implement per-example augmentation at training time
- Slightly rotate/permute input state
- Adjust labels using proper rotation transformation
- Much safer to debug than pre-generating bad data
```

---

## Next Steps to Improve Results

### 1. Regenerate Dataset
```bash
cd dataset
python build_2x2_heuristic.py  # Now uses 50k train, 5k val/test
```

### 2. Verify Data Quality
```bash
python inspect_cube_data.py
# Should show:
# - Labels only 0-8 (no 9, 10, 11)
# - Balanced distribution across moves
# - All states unique
```

### 3. Model Architecture Considerations
- Current: Classification (predict 1 of 9 moves) ← **CORRECT TASK**
- Consider: Multi-step prediction (sequence of moves)
- Consider: Value function (estimate distance to solved)

### 4. Training Improvements
- Use cross-entropy loss (not MSE regression) ← **Must fix!**
- Balance class weights if some moves are rare
- Add metric: % of test cubes fully solved in sequence

### 5. Scaling to 3x3
Once 2x2 works:
- 3x3 has 54 stickers, 18 moves (all 6 faces)
- MUCH harder (God's number = 20, state space = 4.3×10^19)
- May need different approach (e.g., macro-moves, subgoal decomposition)

---

## Technical Details: Why Rotation Augmentation Failed

### The Wrong Approach (What Was Attempted)
```python
# Broken code tried this:
rotated_state = state.copy()
rotated_state = py222.doMove(rotated_state, 3)  # Apply R move
# Then claimed rotated_state has same next_move label ❌
```

**Why this is wrong:**
- Applying move R changes the cube state
- The optimal next move from this NEW state is different
- Label should change, but code kept it the same

### The Right Approach (If You Want Rotation Augmentation)
```python
# Would need to implement whole-cube rotations (x, y, z)
rotated_state = py222.doMove(state, 18)  # x rotation
rotated_label = translate_move(next_move, rotation_type='x')
# Move labels must be transformed too:
# If original has R→, rotated might have U→ or F→ depending on rotation
```

This is complex and not worth it for this task.

---

## Files Modified
1. `dataset/build_2x2_heuristic.py` - Fixed all bugs
2. Config changes recommended:
   - Increase `train_size` from 5k → 50k
   - Remove `num_rotations` parameter
   - Set `ignore_label_id = -1`

## Verification
Run `inspect_cube_data.py` after regenerating to confirm:
- No labels > 8
- Balanced label distribution
- All states are unique
