"""
@file loss.py
@brief Loss functions for training the chess neural network.

Provides functions to compute custom loss metrics combining Win-Draw-Loss (WDL)
probabilities and regularization penalties.
"""

# Loss functions for Residual MoE training.
#
# Three loss types:
#   - BASE: Standard WDL loss for base experts
#   - KILLER: Zoomed WDL (focus on winning positions)
#   - SURVIVOR: Zoomed WDL (focus on defensive positions)

import torch
import torch.nn.functional as F


def base_loss_components(
    pred_wdl,
    target_wdl,
    wdl_weight=1.0,
):
    """
    Standard loss for base experts (Tactical, Strategic, Major End, Minor End).
    
    WDL is the primary objective using Brier-style MSE.
    
    Args:
        pred_wdl: [B, 3] predicted WDL probabilities (from softmax)
        target_wdl: [B, 3] target WDL probabilities
        
    Returns:
        (total_loss, weighted_wdl_loss)
    """
    # Brier-style MSE on soft WDL targets.
    loss_wdl = F.mse_loss(pred_wdl, target_wdl, reduction='mean')
    
    weighted_wdl = wdl_weight * loss_wdl
    return weighted_wdl, weighted_wdl


def base_loss(pred_wdl, target_wdl, wdl_weight=1.0):
    total, _ = base_loss_components(
        pred_wdl,
        target_wdl,
        wdl_weight=wdl_weight,
    )
    return total


def killer_loss(pred_wdl, target_wdl, wdl_weight=1.0):
    """
    Zoomed loss for Killer expert (winning positions).
    
    KILLER sees positions where STM is winning (eval > 150cp).
    Since they will be combined with base experts via residual addition,
    the killer expert learns to predict a CORRECTION that emphasizes win certainty.
    
    Zoomed WDL: Maps WinProb [0.60, 1.0] -> [0.0, 1.0]
    This creates stronger gradients in winning positions where base experts plateau.
    
    The output should be interpreted as:
    - High zoomed_win = "base expert underestimates win probability, add more"
    - Low zoomed_win = "base expert is already confident enough"
    """
    # Extract win probability and zoom from [0.60, 1.0] -> [0.0, 1.0]
    # Lower threshold (0.60) since killer activates at +150cp which is ~60% win rate
    raw_win = target_wdl[:, 0]
    zoomed_win = torch.clamp((raw_win - 0.60) / 0.40, 0.0, 1.0)
    
    # Residual target: how much more confident should we be?
    # If raw_win=0.80 -> zoomed_win=0.50 (medium correction)
    # If raw_win=1.00 -> zoomed_win=1.00 (full confidence)
    new_target = torch.stack([
        zoomed_win,
        1.0 - zoomed_win,  # Remainder interpreted as "no additional correction needed"
        torch.zeros_like(zoomed_win)  # Never predict loss in killer positions
    ], dim=1)
    
    # Cross-entropy loss (more stable than KL div)
    log_pred = torch.log(pred_wdl + 1e-8)
    loss_wdl = -torch.sum(new_target * log_pred, dim=1).mean()
    
    return wdl_weight * loss_wdl


def survivor_loss(pred_wdl, target_wdl, wdl_weight=1.0):
    """
    Zoomed loss for Survivor expert (losing positions).
    
    SURVIVOR sees positions where STM is losing (eval < -150cp).
    The goal is to find defensive resources - draws or prolonging the game.
    
    Zoomed WDL: Maps LossProb [0.60, 1.0] -> [0.0, 1.0]
    Focus: Maximize draw probability, not just predict loss correctly.
    
    The output emphasizes:
    - Draw probability as the primary "hope" signal
    - Lower zoomed_loss = more swindle potential
    """
    raw_loss = target_wdl[:, 2]
    raw_draw = target_wdl[:, 1]
    
    # Zoom loss from [0.60, 1.0] -> [0.0, 1.0]
    zoomed_loss = torch.clamp((raw_loss - 0.60) / 0.40, 0.0, 1.0)
    
    # Emphasis on draw - survivor's goal is finding draws, not accepting loss
    # Scale up draw probability to reward finding defensive resources
    draw_emphasis = torch.clamp(raw_draw * 2.0, 0.0, 0.5)  # Cap at 0.5
    
    new_target = torch.stack([
        torch.zeros_like(zoomed_loss),  # No win expected
        draw_emphasis + (1.0 - zoomed_loss) * 0.5,  # Draw = hope
        zoomed_loss * (1.0 - draw_emphasis)  # Adjusted loss
    ], dim=1)
    
    # Normalize to ensure it sums to 1
    new_target = new_target / (new_target.sum(dim=1, keepdim=True) + 1e-8)
    
    # Cross-entropy loss (more stable than KL div)
    log_pred = torch.log(pred_wdl + 1e-8)
    loss_wdl = -torch.sum(new_target * log_pred, dim=1).mean()
    
    return wdl_weight * loss_wdl


def gater_loss(pred_gates, target_gates):
    """
    Distillation loss for SmartGater.
    
    Args:
        pred_gates: [B, 2] predicted [survivor, killer] gates
        target_gates: [B, 2] teacher [survivor, killer] gates from C++
        
    Returns:
        Scalar MSE loss
    """
    return F.mse_loss(pred_gates, target_gates)


def moe_loss(pred_wdl, target_wdl, expert_type='BASE'):
    """
    Unified loss function with expert type selection.
    
    Args:
        pred_wdl: [B, 3]
        target_wdl: [B, 3]
        expert_type: 'BASE', 'KILLER', or 'SURVIVOR'
    """
    if expert_type == 'BASE':
        return base_loss(pred_wdl, target_wdl)
    elif expert_type == 'KILLER':
        return killer_loss(pred_wdl, target_wdl)
    elif expert_type == 'SURVIVOR':
        return survivor_loss(pred_wdl, target_wdl)
    else:
        raise ValueError(f"Unknown expert_type: {expert_type}")


# =============================================================================
#   SANITY CHECK
# =============================================================================
if __name__ == "__main__":
    print("--- Loss Function Check ---")
    
    B = 8
    pred_wdl = F.softmax(torch.randn(B, 3), dim=1)
    target_wdl = F.softmax(torch.randn(B, 3), dim=1)
    
    # Test each loss type
    for expert_type in ['BASE', 'KILLER', 'SURVIVOR']:
        loss = moe_loss(pred_wdl, target_wdl, expert_type)
        print(f"{expert_type:10} loss: {loss.item():.4f}")
    
    # Test gater loss
    pred_gates = torch.sigmoid(torch.randn(B, 2))
    target_gates = torch.rand(B, 2)
    g_loss = gater_loss(pred_gates, target_gates)
    print(f"{'GATER':10} loss: {g_loss.item():.4f}")
    
    print("Status: OK ✓")
