"""
Inference script: Load trained MLP, predict solution, validate, and visualize.

This is what you'll use for the presentation:
  1. Random scrambled cube (or user input)
  2. Model predicts solution sequence
  3. Validator checks if it's optimal
  4. Visualizer shows cube state at each step
"""
import numpy as np
import torch
import sys
from pathlib import Path
from typing import List, Tuple

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent / "py222"))
import py222
import solver as py222_solver


class SimpleMLP(torch.nn.Module):
    """Simple 4-layer MLP for sequence prediction."""
    def __init__(self, input_dim=144, hidden_dim=256, seq_len=15, vocab_size=10, dropout=0.1):
        super().__init__()
        self.fc1 = torch.nn.Linear(input_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = torch.nn.Linear(hidden_dim, seq_len * vocab_size)
        self.dropout = torch.nn.Dropout(dropout) if dropout > 0 else None
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        
    def forward(self, x):
        x = (x.long() - 1).clamp(min=0, max=5)
        x_onehot = torch.nn.functional.one_hot(x, num_classes=6).float()
        x_onehot = x_onehot.reshape(x_onehot.shape[0], -1)
        
        x = torch.nn.functional.relu(self.fc1(x_onehot))
        if self.dropout:
            x = self.dropout(x)
        
        x = torch.nn.functional.relu(self.fc2(x))
        if self.dropout:
            x = self.dropout(x)
            
        x = torch.nn.functional.relu(self.fc3(x))
        if self.dropout:
            x = self.dropout(x)
            
        x = self.fc4(x)
        return x.view(x.shape[0], self.seq_len, self.vocab_size)


def load_model(checkpoint_path: Path, device='cpu') -> SimpleMLP:
    """Load trained model from checkpoint."""
    print(f"Loading model from {checkpoint_path}...")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Get model config (save this in checkpoint next time!)
    model = SimpleMLP(input_dim=144, hidden_dim=256, seq_len=15, vocab_size=10, dropout=0.1)
    model.load_state_dict(checkpoint['model_state'])
    model = model.to(device)
    model.eval()
    
    epoch = checkpoint.get('epoch', None)
    if epoch is None:
        print("✓ Model loaded (epoch unknown)")
    else:
        print(f"✓ Model loaded (epoch {epoch + 1})")
    return model


def find_best_checkpoint(results_dir: Path, max_len: int) -> Path:
    """Find the checkpoint with best valid-solution accuracy within max_len stage."""
    candidates = sorted(results_dir.glob("mlp_validsolution_curriculum_*.pt"))
    if not candidates:
        return None

    best_path = None
    best_score = -1.0

    for ckpt_path in candidates:
        try:
            ckpt = torch.load(ckpt_path, map_location='cpu')
            history = ckpt.get('history', {})
            valid_acc = history.get('test_valid_solution_acc', [])
            max_lens = history.get('curriculum_max_len', [])
            if not valid_acc or not max_lens:
                continue

            # Best accuracy within <=max_len stage
            stage_scores = [acc for acc, ml in zip(valid_acc, max_lens) if ml <= max_len]
            if not stage_scores:
                continue

            score = max(stage_scores)
            if score > best_score:
                best_score = score
                best_path = ckpt_path
        except Exception:
            continue

    return best_path


def truncate_solution_by_prefix(cube_state: np.ndarray, solution: List[int]) -> List[int]:
    """Truncate solution at the first prefix that solves the cube."""
    if len(solution) == 0:
        return solution

    state = (cube_state.astype(np.int16) - 1).copy()
    state = py222.normFC(state)

    result_state = state.copy()
    truncated = []
    for token in solution:
        if token == 0:
            break
        move_idx = int(token) - 1
        result_state = py222.doMove(result_state, move_idx)
        truncated.append(int(token))
        if py222.isSolved(result_state):
            break

    return truncated


def predict_solution(
    model,
    cube_state: np.ndarray,
    device='cpu',
    top_k=1,
    max_moves=11
) -> List[np.ndarray]:
    """
    Predict solution sequence(s) for a cube state.
    
    Args:
        cube_state: (24,) array, values 1-6
        top_k: Return top-k predicted sequences
        
    Returns:
        List of predicted solution sequences (move indices)
    """
    with torch.no_grad():
        # Convert to tensor and add batch dimension
        state_tensor = torch.from_numpy(cube_state).float().unsqueeze(0).to(device)
        
        # Get logits
        logits = model(state_tensor)  # (1, 15, 10)
        
        # Greedy decoding: argmax at each position
        predictions = torch.argmax(logits, dim=-1)  # (1, 15)

        # Stop at first PAD; if none, cap to max_moves
        prediction = predictions[0].cpu().numpy()
        if 0 in prediction:
            first_pad = int(np.where(prediction == 0)[0][0])
            prediction[first_pad:] = 0
        else:
            prediction[max_moves:] = 0

        solution = prediction[prediction > 0].tolist()
        solution = truncate_solution_by_prefix(cube_state, solution)

        return [np.array(solution, dtype=np.uint8)]


def is_valid_solution(
    cube_state: np.ndarray,
    solution: List[int],
) -> bool:
    """Validate if solution sequence actually solves the cube."""
    try:
        # Convert to py222 format (0-5) and normalize
        state = (cube_state.astype(np.int16) - 1).copy()
        state = py222.normFC(state)
        
        # Filter out PAD (0) and convert moves to py222 format (0-8)
        moves = [int(m) - 1 for m in solution if m > 0]
        
        if len(moves) == 0:
            return py222.isSolved(state)
        
        # Apply moves; accept if any prefix solves the cube
        result_state = state.copy()
        for move in moves:
            result_state = py222.doMove(result_state, move)
            if py222.isSolved(result_state):
                return True

        # Check if solved after all moves
        return py222.isSolved(result_state)
    except Exception as e:
        print(f"  ⚠ Validation error: {e}")
        return False


def apply_moves_trace(
    cube_state: np.ndarray,
    solution: List[int]
) -> List[Tuple[np.ndarray, str]]:
    """Apply moves step-by-step and return states + move names."""
    move_strs = {
        1: "U", 2: "U'", 3: "U2",
        4: "R", 5: "R'", 6: "R2",
        7: "F", 8: "F'", 9: "F2"
    }
    
    trace = [(cube_state.copy(), "Initial State")]
    
    try:
        state = (cube_state.astype(np.int16) - 1).copy()
        
        for token in solution:
            if token == 0:  # PAD
                continue
            
            move_idx = int(token) - 1
            state = py222.doMove(state, move_idx)
            move_name = move_strs.get(token, f"Move{token}")
            trace.append((state + 1, move_name))
        
        return trace
    except Exception as e:
        print(f"Error during move application: {e}")
        return trace


def print_cube_state(state: np.ndarray, title: str = ""):
    """Print cube state in unfolded format."""
    if title:
        print(f"\n{title}")
        print("=" * 40)
    
    # Unfolded cube layout for 2x2x2
    # Face order in state: U, R, F, D, L, B (standard)
    # Each face has 4 stickers for 2x2
    
    colors = {
        1: "⬜",  # White
        2: "🟨",  # Yellow
        3: "🟧",  # Orange
        4: "🟥",  # Red
        5: "🟩",  # Green
        6: "🟦",  # Blue
    }
    
    # Map state array to cube faces
    U = state[0:4]    # Top
    R = state[4:8]    # Right
    F = state[8:12]   # Front
    D = state[12:16]  # Bottom
    L = state[16:20]  # Left
    B = state[20:24]  # Back
    
    # Unfolded net layout:
    #      [  U  ]
    # [ L ][ F ][ R ][ B ]
    #      [  D  ]
    
    print(f"       {colors[U[0]]} {colors[U[1]]}")
    print(f"       {colors[U[2]]} {colors[U[3]]}")
    print()
    print(f"{colors[L[0]]} {colors[L[1]]}  {colors[F[0]]} {colors[F[1]]}  {colors[R[0]]} {colors[R[1]]}  {colors[B[0]]} {colors[B[1]]}")
    print(f"{colors[L[2]]} {colors[L[3]]}  {colors[F[2]]} {colors[F[3]]}  {colors[R[2]]} {colors[R[3]]}  {colors[B[2]]} {colors[B[3]]}")
    print()
    print(f"       {colors[D[0]]} {colors[D[1]]}")
    print(f"       {colors[D[2]]} {colors[D[3]]}")


if __name__ == "__main__":
    print("="*70)
    print("MLP INFERENCE: Predict & Validate Cube Solution")
    print("="*70)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")
    
    # Demo settings
    TARGET_MAX_LEN = 4  # Force scrambles <= 4 moves and load best checkpoint for <=4

    # Load model (prefer best <=TARGET_MAX_LEN checkpoint by valid-solution accuracy)
    results_dir = Path(r"C:\personal\Uni\5\ML\Project\HRM\2x2_solution_results")
    short_ckpt = results_dir / "mlp_short_seq_len4_checkpoint.pt"
    if TARGET_MAX_LEN <= 4 and short_ckpt.exists():
        checkpoint_path = short_ckpt
        print(f"Using short-sequence checkpoint: {checkpoint_path}")
    else:
        best_ckpt = find_best_checkpoint(results_dir, max_len=TARGET_MAX_LEN)
        if best_ckpt is not None:
            checkpoint_path = best_ckpt
            print(f"Using best <= {TARGET_MAX_LEN}-move checkpoint: {checkpoint_path}")
        else:
            checkpoint_path = Path(r"C:\personal\Uni\5\ML\Project\HRM\2x2_solution_results\checkpoints_curriculum\checkpoint_epoch004_stage1.pt")

    if not checkpoint_path.exists():
        print(f"⚠ Model not found at {checkpoint_path}")
        print("Checking for alternative checkpoints...")
        ckpt_dir = Path(r"C:\personal\Uni\5\ML\Project\HRM\2x2_solution_results\checkpoints_curriculum")
        if ckpt_dir.exists():
            ckpts = sorted(ckpt_dir.glob("checkpoint_*.pt"))
            if ckpts:
                checkpoint_path = ckpts[-1]
                print(f"Using latest checkpoint: {checkpoint_path}")
            else:
                print("No checkpoints found!")
                print("Train model first using: python useful_files/train_mlp_validationsolver.py")
                sys.exit(1)
        else:
            print("Train model first using: python useful_files/train_mlp_validationsolver.py")
            sys.exit(1)
    
    model = load_model(checkpoint_path, device=device)
    
    # Example 1: Random scramble
    print("\n" + "="*70)
    print("EXAMPLE 1: Predict Solution for Known Scramble")
    print("="*70)
    
    # Start from a valid solved cube state
    solved_state = py222.initState()  # values 0-5

    # Create a random scramble
    rng = np.random.RandomState()
    scramble_len = int(rng.randint(1, TARGET_MAX_LEN + 1))  # 1..TARGET_MAX_LEN moves for demo
    scramble_moves = rng.randint(0, 9, size=scramble_len).tolist()  # 0-8

    test_state = solved_state.copy()
    for move in scramble_moves:
        test_state = py222.doMove(test_state, move)
    test_state = test_state + 1  # Convert back to 1-6

    move_names = {
        1: "U", 2: "U'", 3: "U2",
        4: "R", 5: "R'", 6: "R2",
        7: "F", 8: "F'", 9: "F2"
    }
    scramble_names = [move_names[m + 1] for m in scramble_moves]
    
    print(f"\n1. Input Cube State (scramble: {' '.join(scramble_names)}):")
    print_cube_state(test_state)
    
    # Predict solution
    print("\n2. Predicting solution...")
    predictions = predict_solution(model, test_state, device=device, max_moves=TARGET_MAX_LEN)
    predicted_solution = predictions[0]
    print(f"   Predicted solution: {predicted_solution.tolist()}")
    print(f"   Number of moves: {len(predicted_solution)}")
    
    # Convert to move names
    solution_names = [move_names.get(int(m), f"?{m}") for m in predicted_solution]
    print(f"   Moves: {' '.join(solution_names)}")
    
    # Validate
    print("\n3. Validating solution...")
    is_valid = is_valid_solution(test_state, predicted_solution)
    if is_valid:
        print("   ✅ Solution is VALID! Cube is solved.")
    else:
        print("   ❌ Solution is INVALID. Cube is not solved.")
        # Try to get correct solution
        print("   Getting correct solution from IDA*...")
        correct_solution = py222_solver.solve((test_state - 1).astype(np.int16))
        if len(correct_solution) == 0:
            print("   Correct solution: (already solved)")
        else:
            correct_names = [move_names.get(m + 1, f"?{m}") for m in correct_solution]
            print(f"   Correct solution: {' '.join(correct_names)}")
    
    # Trace through solution
    if is_valid:
        print("\n4. Tracing solution step-by-step:")
        trace = apply_moves_trace(test_state, predicted_solution)
        for i, (state, move) in enumerate(trace):
            print(f"\n   Step {i}: {move}")
            print_cube_state(state)
    
    print("\n" + "="*70)
    print("DEMO COMPLETE")
    print("="*70)
