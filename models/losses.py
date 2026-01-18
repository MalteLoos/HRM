from typing import Any, Tuple, Dict, Sequence, Optional

import torch
import torch.nn.functional as F
from torch import nn
import numpy as np

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import py222


IGNORE_LABEL_ID = -100


def validate_cube_2x2_solution(inputs: torch.Tensor, predictions: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Validates if the predicted moves solve the 2x2 cube.
    
    Args:
        inputs: Tensor of shape (B, 24) containing the scrambled cube state (+1 for padding offset)
        predictions: Tensor of shape (B, SeqLen) containing predicted move indices (+1 for padding offset)
        mask: Tensor of shape (B, SeqLen) indicating valid positions
    
    Returns:
        Tensor of shape (B,) indicating if each sequence is a valid solution
    """
    batch_size = inputs.shape[0]
    results = torch.zeros(batch_size, dtype=torch.bool, device=inputs.device)
    
    # Convert to numpy for py222 operations
    inputs_np = inputs.cpu().numpy()
    predictions_np = predictions.cpu().numpy()
    mask_np = mask.cpu().numpy()
    
    for b in range(batch_size):
        # Reconstruct cube state (remove padding offset)
        cube_state = inputs_np[b] - 1
        cube_state = cube_state.astype(np.int_)
        
        # Get predicted moves (remove padding offset, only valid positions)
        valid_moves = predictions_np[b][mask_np[b]]
        moves = (valid_moves - 1).astype(np.int_)  # Remove padding offset
        
        # Apply moves to the cube state
        try:
            s = cube_state.copy()
            for move in moves:
                if move < 0 or move >= 9:  # Invalid move index
                    break
                s = py222.doMove(s, move)
            
            # Normalize and check if solved
            s = py222.normFC(s)
            results[b] = py222.isSolved(s)
        except Exception:
            # If anything goes wrong, mark as incorrect
            results[b] = False
    
    return results


def s(x, epsilon=1e-30):
    return torch.where(
        x<0,
        1/(1-x+ epsilon),
        x + 1
    )


def log_stablemax(x, dim=-1):
    s_x = s(x)
    return torch.log(s_x/torch.sum(s_x, dim=dim, keepdim=True))


def stablemax_cross_entropy(logits, labels, ignore_index: int = -100):
    logprobs = log_stablemax(logits.to(torch.float64), dim=-1)

    valid_mask = labels != ignore_index
    transformed_labels = torch.where(valid_mask, labels, 0)
    prediction_logprobs = torch.gather(logprobs, index=transformed_labels.to(torch.long).unsqueeze(-1), dim=-1).squeeze(-1)

    return -torch.where(valid_mask, prediction_logprobs, 0)


def softmax_cross_entropy(logits, labels, ignore_index: int = -100):
    # Cast logits to f32
    # Flatten logits
    return F.cross_entropy(logits.to(torch.float32).view(-1, logits.shape[-1]), labels.to(torch.long).view(-1), ignore_index=ignore_index, reduction="none").view(labels.shape)

class ACTLossHead(nn.Module):
    def __init__(self, model: nn.Module, loss_type: str):
        super().__init__()
        self.model = model
        self.loss_fn = globals()[loss_type]
        
    def initial_carry(self, *args, **kwargs):
        return self.model.initial_carry(*args, **kwargs)  # type: ignore

    def forward(
        self,
        return_keys: Sequence[str],
        # Model args
        **model_kwargs,
    ) -> Tuple[Any, torch.Tensor, Dict[str, torch.Tensor], Optional[Dict[str, torch.Tensor]], torch.Tensor]:
        # Model logits
        # B x SeqLen x D
        new_carry, outputs = self.model(**model_kwargs)
        labels = new_carry.current_data["labels"]

        # Correctness
        with torch.no_grad():
            mask = labels != IGNORE_LABEL_ID
            loss_counts = mask.sum(-1)
            loss_divisor = loss_counts.clamp_min(1).unsqueeze(-1)  # Avoid NaNs in division

            predictions = torch.argmax(outputs["logits"], dim=-1)
            is_correct = mask & (predictions == labels)
            seq_is_correct = is_correct.sum(-1) == loss_counts
            
            # For 2x2 cube: if not matching labels, check if it's still a valid solution
            # (since there can be multiple optimal solutions)
            inputs = new_carry.current_data["inputs"]
            if inputs.shape[-1] == 24:  # 2x2 cube has 24 stickers
                not_exact_match = ~seq_is_correct
                if not_exact_match.any():
                    # Validate predicted solutions for samples that don't match labels exactly
                    valid_solutions = validate_cube_2x2_solution(inputs, predictions, mask)
                    # Update seq_is_correct: correct if matches labels OR is a valid solution
                    seq_is_correct = seq_is_correct | valid_solutions
            
            # Metrics (halted)
            valid_metrics = new_carry.halted & (loss_counts > 0)
            metrics = {
                "count": valid_metrics.sum(),
                
                "accuracy":       torch.where(valid_metrics, (is_correct.to(torch.float32) / loss_divisor).sum(-1), 0).sum(),
                "exact_accuracy": (valid_metrics & seq_is_correct).sum(),

                "q_halt_accuracy": (valid_metrics & ((outputs["q_halt_logits"] >= 0) == seq_is_correct)).sum(),
                "steps":          torch.where(valid_metrics, new_carry.steps, 0).sum(),
            }

        # Losses
        # FIXME: Assuming the batch is always full
        lm_loss = (self.loss_fn(outputs["logits"], labels, ignore_index=IGNORE_LABEL_ID) / loss_divisor).sum()
        q_halt_loss = F.binary_cross_entropy_with_logits(outputs["q_halt_logits"], seq_is_correct.to(outputs["q_halt_logits"].dtype), reduction="sum")

        metrics.update({
            "lm_loss": lm_loss.detach(),
            "q_halt_loss": q_halt_loss.detach(),
        })

        # Q continue (bootstrapping target loss)
        q_continue_loss = 0
        if "target_q_continue" in outputs:
            q_continue_loss = F.binary_cross_entropy_with_logits(outputs["q_continue_logits"], outputs["target_q_continue"], reduction="sum")

            metrics["q_continue_loss"] = q_continue_loss.detach()

        # Filter outputs for return
        detached_outputs = {k: outputs[k].detach() for k in return_keys if k in outputs}

        return new_carry, lm_loss + 0.5 * (q_halt_loss + q_continue_loss), metrics, detached_outputs, new_carry.halted.all()
