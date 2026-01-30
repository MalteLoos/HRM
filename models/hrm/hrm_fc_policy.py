"""
Hierarchical Reasoning Model with Fully Connected layers for 2x2 Rubik's Cube Policy Learning.
Based on HRM ACT structure but replaces transformer layers with FC layers.
Uses one-hot encoding for cube states.
"""

from typing import Tuple, Dict
from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F
from torch import nn
from pydantic import BaseModel


def rms_norm(x: torch.Tensor, variance_epsilon: float = 1e-5) -> torch.Tensor:
    """RMS normalization."""
    variance = x.pow(2).mean(-1, keepdim=True)
    return x * torch.rsqrt(variance + variance_epsilon)


class SwiGLU(nn.Module):
    """SwiGLU activation with linear layers."""
    
    def __init__(self, hidden_size: int, expansion: float = 4.0):
        super().__init__()
        intermediate_size = int(hidden_size * expansion)
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


@dataclass
class HRM_FC_PolicyInnerCarry:
    z_H: torch.Tensor
    z_L: torch.Tensor


@dataclass
class HRM_FC_PolicyCarry:
    inner_carry: HRM_FC_PolicyInnerCarry
    
    steps: torch.Tensor
    halted: torch.Tensor
    
    current_data: Dict[str, torch.Tensor]


class HRM_FC_PolicyConfig(BaseModel):
    batch_size: int
    seq_len: int = 24  # 2x2 cube has 24 stickers
    
    # One-hot encoding: 8 pieces * 3 orientations = 24 dimensions per position
    # But we flatten to single vector for FC processing
    one_hot_dim: int = 24  # piece * orientation encoding
    
    H_cycles: int = 2
    L_cycles: int = 2
    
    H_layers: int = 2
    L_layers: int = 2
    
    # FC network config
    hidden_size: int = 512
    expansion: float = 2.0
    
    rms_norm_eps: float = 1e-5
    
    # Output config
    num_actions: int = 9  # 9 moves: U, U', U2, R, R', R2, F, F', F2
    max_solution_len: int = 11  # Max solution length for 2x2 cube
    
    # Halting Q-learning config
    halt_max_steps: int = 16
    halt_exploration_prob: float = 0.0
    
    forward_dtype: str = "float32"


class HRM_FC_PolicyBlock(nn.Module):
    """FC block with residual connection and PRE-normalization to prevent vanishing gradients."""
    
    def __init__(self, config: HRM_FC_PolicyConfig) -> None:
        super().__init__()
        
        self.fc = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.ReLU(),
            nn.Linear(config.hidden_size, config.hidden_size),
        )
        self.mlp = SwiGLU(
            hidden_size=config.hidden_size,
            expansion=config.expansion,
        )
        self.norm_eps = config.rms_norm_eps
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # PRE-Norm with residual (better gradient flow)
        # Normalize BEFORE the transformation, then add residual
        hidden_states = hidden_states + self.fc(rms_norm(hidden_states, variance_epsilon=self.norm_eps))
        hidden_states = hidden_states + self.mlp(rms_norm(hidden_states, variance_epsilon=self.norm_eps))
        return hidden_states


class HRM_FC_PolicyReasoningModule(nn.Module):
    """Reasoning module with input injection."""
    
    def __init__(self, layers: list):
        super().__init__()
        self.layers = torch.nn.ModuleList(layers)
    
    def forward(self, hidden_states: torch.Tensor, input_injection: torch.Tensor) -> torch.Tensor:
        # Input injection (add)
        hidden_states = hidden_states + input_injection
        # Layers
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        return hidden_states


class HRM_FC_Policy_Inner(nn.Module):
    """Inner model with FC layers instead of transformers."""
    
    def __init__(self, config: HRM_FC_PolicyConfig) -> None:
        super().__init__()
        self.config = config
        self.forward_dtype = getattr(torch, self.config.forward_dtype)
        
        # Input projection: one-hot encoded cube state to hidden size
        # Input is (batch, seq_len=24, one_hot_dim=24) -> flatten to (batch, 24*24=576)
        # Then project to hidden_size
        self.input_size = config.seq_len * config.one_hot_dim
        self.input_proj = nn.Linear(self.input_size, config.hidden_size, bias=False)
        
        # Output head: predict action distribution for each solution position
        # Output shape: (batch, max_solution_len, num_actions+1) where +1 is for PAD/END
        self.output_proj = nn.Linear(config.hidden_size, config.max_solution_len * (config.num_actions + 1), bias=False)
        
        # Q head for ACT (halt/continue decision)
        self.q_head = nn.Linear(config.hidden_size, 2, bias=True)
        
        # Reasoning Layers
        self.H_level = HRM_FC_PolicyReasoningModule(
            layers=[HRM_FC_PolicyBlock(self.config) for _ in range(self.config.H_layers)]
        )
        self.L_level = HRM_FC_PolicyReasoningModule(
            layers=[HRM_FC_PolicyBlock(self.config) for _ in range(self.config.L_layers)]
        )
        
        # Initial states
        self.H_init = nn.Parameter(torch.zeros(config.hidden_size, dtype=self.forward_dtype))
        self.L_init = nn.Parameter(torch.zeros(config.hidden_size, dtype=self.forward_dtype))
        
        # Initialize
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights."""
        # Xavier init for projections
        nn.init.xavier_uniform_(self.input_proj.weight)
        nn.init.xavier_uniform_(self.output_proj.weight)
        
        # Q head init: start near zero for bootstrapping
        with torch.no_grad():
            self.q_head.weight.zero_()
            self.q_head.bias.fill_(-5)
        
        # Init states
        nn.init.normal_(self.H_init, std=0.02)
        nn.init.normal_(self.L_init, std=0.02)
    
    def _input_embeddings(self, one_hot_input: torch.Tensor) -> torch.Tensor:
        """
        Convert one-hot encoded input to hidden representation.
        
        Args:
            one_hot_input: (batch, seq_len=24, one_hot_dim=24) one-hot encoded cube state
        
        Returns:
            (batch, hidden_size) hidden representation
        """
        batch_size = one_hot_input.shape[0]
        # Flatten: (batch, 24, 24) -> (batch, 576)
        flat_input = one_hot_input.view(batch_size, -1).to(self.forward_dtype)
        # Project to hidden size
        return self.input_proj(flat_input)
    
    def empty_carry(self, batch_size: int):
        return HRM_FC_PolicyInnerCarry(
            z_H=torch.empty(batch_size, self.config.hidden_size, dtype=self.forward_dtype),
            z_L=torch.empty(batch_size, self.config.hidden_size, dtype=self.forward_dtype),
        )
    
    def reset_carry(self, reset_flag: torch.Tensor, carry: HRM_FC_PolicyInnerCarry):
        device = carry.z_H.device
        return HRM_FC_PolicyInnerCarry(
            z_H=torch.where(reset_flag.view(-1, 1), self.H_init.to(device), carry.z_H),
            z_L=torch.where(reset_flag.view(-1, 1), self.L_init.to(device), carry.z_L),
        )
    
    def forward(self, carry: HRM_FC_PolicyInnerCarry, batch: Dict[str, torch.Tensor]) -> Tuple[HRM_FC_PolicyInnerCarry, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass with recurrent H/L cycles for iterative refinement.
        
        The HRM structure alternates between:
        - L_level: Low-level reasoning with input injection
        - H_level: High-level reasoning that aggregates L_level output
        
        Multiple cycles allow the model to iteratively refine its sequence prediction.
        """
        # Input encoding
        input_embeddings = self._input_embeddings(batch["inputs"])
        
        # Initialize states from learnable parameters (fresh start each forward)
        # This ensures consistent behavior and proper gradient flow
        batch_size = input_embeddings.shape[0]
        device = input_embeddings.device
        
        # Use repeat() instead of expand() - expand creates a VIEW where all items share memory!
        # repeat() creates actual copies so each batch element can have independent gradients
        z_H = self.H_init.to(device).unsqueeze(0).repeat(batch_size, 1)
        z_L = self.L_init.to(device).unsqueeze(0).repeat(batch_size, 1)
        
        # Recurrent cycles with FULL gradient flow
        # Each cycle: L_level processes with input injection, then H_level aggregates
        for h_step in range(self.config.H_cycles):
            for l_step in range(self.config.L_cycles):
                # L_level: combine current z_L with (z_H + input)
                # Input injection ensures the model always has access to the puzzle state
                z_L = self.L_level(z_L, z_H + input_embeddings)
            
            # H_level: aggregate information from L_level
            z_H = self.H_level(z_H, z_L)
        
        # Output: predict full action sequence
        batch_size = z_H.shape[0]
        output_flat = self.output_proj(z_H)
        output = output_flat.view(batch_size, self.config.max_solution_len, self.config.num_actions + 1)
        
        # New carry for potential multi-step ACT (detached to prevent gradient through time)
        new_carry = HRM_FC_PolicyInnerCarry(z_H=z_H.detach(), z_L=z_L.detach())
        
        # Q head for ACT halt decision
        q_logits = self.q_head(z_H).to(torch.float32)
        
        return new_carry, output, (q_logits[..., 0], q_logits[..., 1])


class HRM_FC_Policy(nn.Module):
    """ACT wrapper for FC Policy model."""
    
    def __init__(self, config_dict: dict):
        super().__init__()
        self.config = HRM_FC_PolicyConfig(**config_dict)
        self.inner = HRM_FC_Policy_Inner(self.config)
    
    def initial_carry(self, batch: Dict[str, torch.Tensor]):
        batch_size = batch["inputs"].shape[0]
        device = batch["inputs"].device
        
        carry = HRM_FC_PolicyCarry(
            inner_carry=self.inner.empty_carry(batch_size),
            steps=torch.zeros((batch_size,), dtype=torch.int32, device=device),
            halted=torch.ones((batch_size,), dtype=torch.bool, device=device),
            current_data={k: torch.empty_like(v) for k, v in batch.items()}
        )
        
        # Move carry tensors to device
        carry.inner_carry.z_H = carry.inner_carry.z_H.to(device)
        carry.inner_carry.z_L = carry.inner_carry.z_L.to(device)
        
        return carry
    
    def forward(self, carry: HRM_FC_PolicyCarry, batch: Dict[str, torch.Tensor]) -> Tuple[HRM_FC_PolicyCarry, Dict[str, torch.Tensor]]:
        # Update data, carry (removing halted sequences)
        new_inner_carry = self.inner.reset_carry(carry.halted, carry.inner_carry)
        
        new_steps = torch.where(carry.halted, torch.zeros_like(carry.steps), carry.steps)
        
        new_current_data = {
            k: torch.where(
                carry.halted.view((-1,) + (1,) * (batch[k].ndim - 1)), 
                batch[k], 
                v
            ) for k, v in carry.current_data.items()
        }
        
        # Forward inner model
        new_inner_carry, logits, (q_halt_logits, q_continue_logits) = self.inner(new_inner_carry, new_current_data)
        
        outputs = {
            "logits": logits,
            "q_halt_logits": q_halt_logits,
            "q_continue_logits": q_continue_logits
        }
        
        with torch.no_grad():
            # Step
            new_steps = new_steps + 1
            is_last_step = new_steps >= self.config.halt_max_steps
            
            halted = is_last_step
            
            # if training, and ACT is enabled
            if self.training and (self.config.halt_max_steps > 1):
                halted = halted | (q_halt_logits > q_continue_logits)
                
                # Exploration
                min_halt_steps = (torch.rand_like(q_halt_logits) < self.config.halt_exploration_prob) * \
                                 torch.randint_like(new_steps, low=2, high=self.config.halt_max_steps + 1)
                
                halted = halted & (new_steps >= min_halt_steps)
                
                # Compute target Q
                next_q_halt_logits, next_q_continue_logits = self.inner(new_inner_carry, new_current_data)[-1]
                
                outputs["target_q_continue"] = torch.sigmoid(
                    torch.where(is_last_step, next_q_halt_logits, torch.maximum(next_q_halt_logits, next_q_continue_logits))
                )
        
        return HRM_FC_PolicyCarry(new_inner_carry, new_steps, halted, new_current_data), outputs


# Convenience function to create model
def create_hrm_fc_policy(
    batch_size: int = 256,
    hidden_size: int = 512,
    H_layers: int = 2,
    L_layers: int = 2,
    H_cycles: int = 1,
    L_cycles: int = 1,
    halt_max_steps: int = 1,
    **kwargs
) -> HRM_FC_Policy:
    """Create HRM FC Policy model with default configuration."""
    config = {
        "batch_size": batch_size,
        "hidden_size": hidden_size,
        "H_layers": H_layers,
        "L_layers": L_layers,
        "H_cycles": H_cycles,
        "L_cycles": L_cycles,
        "halt_max_steps": halt_max_steps,
        **kwargs
    }
    return HRM_FC_Policy(config)
