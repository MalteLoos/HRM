import argparse
import copy
import heapq
import time
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
import magiccube
from tqdm import tqdm

from utils.functions import load_model_class


# ============================================================================
# Simple FC Model for Heuristic
# ============================================================================

class FCHeuristicNet(nn.Module):
    """4-layer fully connected network with ReLU activations."""
    
    def __init__(self, input_size: int = 144, hidden_size: int = 512, num_layers: int = 4, output_size: int = 1):
        super().__init__()
        
        layers = []
        
        # Input layer
        layers.append(nn.Linear(input_size, hidden_size))
        layers.append(nn.ReLU())
        
        # Hidden layers
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.ReLU())
        
        # Output layer
        layers.append(nn.Linear(hidden_size, output_size))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x).squeeze(-1)


# 2x2 cube moves
MOVES = ["U", "U'", "U2", "R", "R'", "R2", "F", "F'", "F2"]
INVERSE_MOVES = {"U": "U'", "U'": "U", "U2": "U2", 
                 "R": "R'", "R'": "R", "R2": "R2",
                 "F": "F'", "F'": "F", "F2": "F2"}
COLORS = {'Y': 0, 'R': 1, 'G': 2, 'O': 3, 'B': 4, 'W': 5}
# Solved state from magiccube.Cube(2): U=W, L=O, F=G, R=R, B=B, D=Y
SOLVED_STATE = "WWWWOOGGRRBBOOGGRRBBYYYY"


def get_cube_state_str(cube: magiccube.Cube) -> str:
    return str(cube).replace(" ", "").replace("\n", "")


def cube_to_tensor(cube: magiccube.Cube, device: torch.device) -> torch.Tensor:
    state_str = get_cube_state_str(cube)
    state = [COLORS[c] for c in state_str]
    return torch.tensor([state], dtype=torch.int32, device=device)


def is_solved(cube: magiccube.Cube) -> bool:
    state_str = get_cube_state_str(cube)
    return state_str == SOLVED_STATE


def scramble_cube(num_moves: int = 20) -> Tuple[magiccube.Cube, List[str]]:
    cube = magiccube.Cube(2)
    scramble = []
    
    for _ in range(num_moves):
        move = random.choice(MOVES)
        cube.rotate(move)
        scramble.append(move)
    
    return cube, scramble


@dataclass(order=True)
class SearchNode:
    f_score: float
    g_score: int = field(compare=False)
    state_str: str = field(compare=False)
    node_id: int = field(compare=False)  # ID for parent tracking


class NeuralHeuristic:
    def __init__(self, checkpoint_path: str, config_path: str, device: str = "cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        
        # Load config
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        
        # Build model config
        arch_config = config["arch"]
        self.max_batch_size = 9  # Max batch size for batched inference
        model_cfg = dict(
            **{k: v for k, v in arch_config.items() if k not in ["name", "loss"]},
            batch_size=self.max_batch_size,
            vocab_size=6,  # 6 colors
            seq_len=24,    # 24 stickers
            num_puzzle_identifiers=1,
            causal=False
        )
        
        # Load model
        model_cls = load_model_class(arch_config["name"])
        self.model = model_cls(model_cfg)
        self.halt_max_steps = self.model.config.halt_max_steps
        
        # Load checkpoint
        state_dict = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
        # Remove loss head prefix if present
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("model."):
                new_state_dict[k[6:]] = v  # Remove "model." prefix
            else:
                new_state_dict[k] = v
        self.model.load_state_dict(new_state_dict, strict=False)
        self.model.to(self.device)
        self.model.eval()
        
        print(f"Loaded model from {checkpoint_path} on {self.device}")
    
    @torch.no_grad()
    def __call__(self, cube: magiccube.Cube) -> float:
        return self.batch_call([cube])[0]
    
    @torch.no_grad()
    def batch_call(self, cubes: List[magiccube.Cube]) -> List[float]:
        if not cubes:
            return []
        
        batch_size = len(cubes)
        
        # Convert all cubes to tensor
        states = []
        for cube in cubes:
            state_str = get_cube_state_str(cube)
            state = [COLORS[c] for c in state_str]
            states.append(state)
        
        inputs = torch.tensor(states, dtype=torch.int32, device=self.device)
        puzzle_ids = torch.zeros(batch_size, dtype=torch.int32, device=self.device)
        labels = torch.zeros(batch_size, 1, dtype=torch.int32, device=self.device)
        
        batch = {
            "inputs": inputs,
            "puzzle_identifiers": puzzle_ids,
            "labels": labels
        }
        
        # Initialize carry and run model (ensure on correct device)
        with torch.device(self.device):
            carry = self.model.initial_carry(batch)
        
        # Run for max steps to get final prediction
        for _ in range(self.halt_max_steps):
            carry, outputs = self.model(carry, batch)
        
        heuristics = outputs["logits"].squeeze(-1).cpu().tolist()  # type: ignore
        return [max(0, h) for h in heuristics]  # Ensure non-negative


class SmallNetHeuristic:
    """Simple 4-layer FC network heuristic for 2x2 cube."""
    
    def __init__(self, checkpoint_path: str, device: str = "cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        
        # Load model
        self.model = FCHeuristicNet(
            input_size=24 * 6,  # 24 positions, one-hot with 6 colors
            hidden_size=512,
            num_layers=4,
            output_size=1
        )
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
        
        self.model.to(self.device)
        self.model.eval()
        
        print(f"Loaded SmallNet model from {checkpoint_path} on {self.device}")
    
    def _cube_to_onehot(self, cube: magiccube.Cube) -> np.ndarray:
        """Convert cube state to one-hot encoding."""
        state_str = get_cube_state_str(cube)
        state = [COLORS[c] for c in state_str]
        
        one_hot = np.zeros((24, 6), dtype=np.float32)
        for i, color in enumerate(state):
            one_hot[i, color] = 1.0
        return one_hot.flatten()
    
    @torch.no_grad()
    def __call__(self, cube: magiccube.Cube) -> float:
        return self.batch_call([cube])[0]
    
    @torch.no_grad()
    def batch_call(self, cubes: List[magiccube.Cube]) -> List[float]:
        if not cubes:
            return []
        
        # Convert all cubes to one-hot tensors
        inputs = np.stack([self._cube_to_onehot(cube) for cube in cubes])
        inputs = torch.from_numpy(inputs).to(self.device)
        
        # Run model
        outputs = self.model(inputs)
        heuristics = outputs.cpu().tolist()
        
        return [max(0, h) for h in heuristics]  # Ensure non-negative


def astar_solve(
    cube: magiccube.Cube,
    heuristic_fn,
    max_nodes: int = 100000,
    batch_size: int = 512
) -> Tuple[Optional[List[str]], Dict]:
    """A* search to solve a 2x2 Rubik's cube.
    
    Uses copy.deepcopy for cube states since magiccube's string parsing is unreliable.
    """
    metrics = {
        "nodes_expanded": 0,
        "nodes_generated": 0,
        "max_queue_size": 0,
        "solution_length": None,
        "time_seconds": 0,
        "heuristic_calls": 0,
    }
    
    start_time = time.time()
    
    if is_solved(cube):
        metrics["solution_length"] = 0
        return [], metrics
    
    # Use state strings as keys for visited set, but store cubes for expansion
    # node_id -> cube object (for reconstruction)
    cube_store = {}
    parent_map = {}  # node_id -> (parent_id, move)
    node_counter = [0]
    
    def get_next_node_id():
        node_counter[0] += 1
        return node_counter[0]
    
    def reconstruct_path(node_id):
        path = []
        current_id = node_id
        while current_id in parent_map:
            parent_id, move = parent_map[current_id]
            path.append(move)
            current_id = parent_id
        return list(reversed(path))
    
    # Initial heuristic
    initial_h = heuristic_fn(cube)
    metrics["heuristic_calls"] += 1
    
    start_state = get_cube_state_str(cube)
    start_node_id = get_next_node_id()
    cube_store[start_node_id] = copy.deepcopy(cube)
    
    start_node = SearchNode(
        f_score=initial_h,
        g_score=0,
        state_str=start_state,
        node_id=start_node_id
    )
    
    # Priority queue and visited set
    open_set = [start_node]
    visited = {start_state: 0}  # state_str -> best g_score
    
    pbar = tqdm(total=max_nodes, desc="A* Search", unit=" nodes")
    
    try:
        while open_set and metrics["nodes_expanded"] < max_nodes:
            metrics["max_queue_size"] = max(metrics["max_queue_size"], len(open_set))
            
            # Pop multiple nodes to expand in batch
            nodes_to_expand = []
            while open_set and len(nodes_to_expand) < max(1, batch_size // len(MOVES)):
                node = heapq.heappop(open_set)
                # Skip if we've found a better path to this state
                if node.g_score <= visited.get(node.state_str, float('inf')):
                    nodes_to_expand.append(node)
            
            if not nodes_to_expand:
                break
            
            nodes_expanded_before = metrics["nodes_expanded"]
            metrics["nodes_expanded"] += len(nodes_to_expand)
            pbar.update(metrics["nodes_expanded"] - nodes_expanded_before)
            
            # Generate all successor states for all nodes
            all_new_cubes = []
            all_new_states = []
            all_new_info = []  # (parent_node_id, move, new_g)
            
            for current in nodes_to_expand:
                new_g = current.g_score + 1
                current_cube = cube_store[current.node_id]
                
                for move in MOVES:
                    # Create new cube by copying and rotating
                    new_cube = copy.deepcopy(current_cube)
                    new_cube.rotate(move)
                    new_state = get_cube_state_str(new_cube)
                    
                    metrics["nodes_generated"] += 1
                    
                    # Check if solved
                    if new_state == SOLVED_STATE:
                        solution = reconstruct_path(current.node_id) + [move]
                        metrics["solution_length"] = len(solution)
                        metrics["time_seconds"] = time.time() - start_time
                        pbar.close()
                        return solution, metrics
                    
                    # Skip if we've seen this state with better cost
                    if new_state in visited and visited[new_state] <= new_g:
                        continue
                    
                    visited[new_state] = new_g
                    all_new_cubes.append(new_cube)
                    all_new_states.append(new_state)
                    all_new_info.append((current.node_id, move, new_g))
            
            # Batch compute heuristics for all valid successors
            if all_new_cubes:
                heuristics = heuristic_fn.batch_call(all_new_cubes)
                metrics["heuristic_calls"] += len(all_new_cubes)
                
                for new_cube, new_state, (parent_id, move, new_g), h in zip(
                    all_new_cubes, all_new_states, all_new_info, heuristics
                ):
                    new_node_id = get_next_node_id()
                    cube_store[new_node_id] = new_cube
                    parent_map[new_node_id] = (parent_id, move)
                    
                    new_node = SearchNode(
                        f_score=new_g + h,
                        g_score=new_g,
                        state_str=new_state,
                        node_id=new_node_id
                    )
                    heapq.heappush(open_set, new_node)
                
                # Clean up old cubes that are no longer needed
                # Keep only cubes that might still be expanded (in open_set)
                if len(cube_store) > 10000:
                    active_ids = {n.node_id for n in open_set}
                    for nid in list(cube_store.keys()):
                        if nid not in active_ids:
                            del cube_store[nid]
    finally:
        pbar.close()
    
    metrics["time_seconds"] = time.time() - start_time
    return None, metrics


class ZeroHeuristic:
    """Zero heuristic (equivalent to BFS) with batch interface."""
    
    def __call__(self, cube: magiccube.Cube) -> float:
        return 0.0
    
    def batch_call(self, cubes: List[magiccube.Cube]) -> List[float]:
        return [0.0] * len(cubes)


def run_evaluation(
    heuristic_fn,
    num_cubes: int = 10,
    scramble_moves: int = 20,
    max_nodes: int = 100000,
    batch_size: int = 512,
    heuristic_name: str = "neural"
):
    print(f"\n{'='*60}")
    print(f"Evaluating {heuristic_name} heuristic on {num_cubes} cubes")
    print(f"Scramble moves: {scramble_moves}, Max nodes: {max_nodes}, Batch size: {batch_size}")
    print(f"{'='*60}\n")
    
    results = []
    
    for i in range(num_cubes):
        cube, scramble = scramble_cube(scramble_moves)
        print(f"Cube {i+1}/{num_cubes}: scramble = {' '.join(scramble[:10])}...")
        
        solution, metrics = astar_solve(cube, heuristic_fn, max_nodes, batch_size)
        
        if solution is not None:
            print(f"  ✓ Solved in {metrics['solution_length']} moves")
            print(f"    Nodes: {metrics['nodes_expanded']}, Time: {metrics['time_seconds']:.3f}s")
        else:
            print(f"  ✗ Not solved (max nodes reached)")
        
        results.append({
            "scramble": scramble,
            "solution": solution,
            **metrics
        })
    
    # Summary statistics
    solved = [r for r in results if r["solution"] is not None]
    
    print(f"\n{'='*60}")
    print(f"Summary for {heuristic_name}")
    print(f"{'='*60}")
    print(f"Solved: {len(solved)}/{num_cubes} ({100*len(solved)/num_cubes:.1f}%)")
    
    if solved:
        avg_length = sum(r["solution_length"] for r in solved) / len(solved)
        avg_nodes = sum(r["nodes_expanded"] for r in solved) / len(solved)
        avg_time = sum(r["time_seconds"] for r in solved) / len(solved)
        avg_heuristic_calls = sum(r["heuristic_calls"] for r in solved) / len(solved)
        
        print(f"Avg solution length: {avg_length:.2f}")
        print(f"Avg nodes expanded: {avg_nodes:.1f}")
        print(f"Avg time: {avg_time:.3f}s")
        print(f"Avg heuristic calls: {avg_heuristic_calls:.1f}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="2x2 Cube Solver with Neural Heuristic")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to config YAML (defaults to all_config.yaml in checkpoint dir)")
    parser.add_argument("--model_type", type=str, default="neural", choices=["neural", "smallnet", "zero"],
                        help="Type of heuristic model to use")
    parser.add_argument("--scramble_moves", type=int, default=20,
                        help="Number of scramble moves")
    parser.add_argument("--num_cubes", type=int, default=10,
                        help="Number of cubes to solve")
    parser.add_argument("--max_nodes", type=int, default=100000,
                        help="Maximum nodes to expand")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--batch_size", type=int, default=512,
                        help="Batch size for heuristic evaluation")
    
    args = parser.parse_args()
    random.seed(args.seed)
    
    # Load heuristic based on model type
    if args.model_type == "smallnet":
        heuristic = SmallNetHeuristic(args.checkpoint)
        heuristic_name = "SmallNet"
    elif args.model_type == "zero":
        heuristic = ZeroHeuristic()
        heuristic_name = "Zero (BFS)"
    else:
        # Find config for neural model
        if args.config is None:
            import os
            checkpoint_dir = os.path.dirname(args.checkpoint)
            args.config = os.path.join(checkpoint_dir, "all_config.yaml")
        heuristic = NeuralHeuristic(args.checkpoint, args.config)
        heuristic_name = "Neural"
    
    # Run evaluation
    results = run_evaluation(
        heuristic,
        num_cubes=args.num_cubes,
        scramble_moves=args.scramble_moves,
        max_nodes=args.max_nodes,
        batch_size=args.batch_size,
        heuristic_name=heuristic_name
    )


if __name__ == "__main__":
    main()

