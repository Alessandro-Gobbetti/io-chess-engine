"""
Loss functions for Residual MoE training.

Three loss types:
  - BASE: Standard WDL + Mate loss for base experts
  - KILLER: Zoomed WDL (focus on winning positions)
  - SURVIVOR: Zoomed WDL (focus on defensive positions)
"""

import torch
import torch.nn.functional as F


def base_loss(pred_wdl, pred_mate, target_wdl, target_mate, wdl_weight=1.0, mate_weight=0.5):
    """
    Standard loss for base experts (Tactical, Strategic, Major End, Minor End).
    
    WDL is the primary objective. Mate distance only matters when there's
    an actual mate in the position (target_mate > 0).
    
    Args:
        pred_wdl: [B, 3] predicted WDL probabilities (from softmax)
        pred_mate: [B, 1] predicted mate distance
        target_wdl: [B, 3] target WDL probabilities
        target_mate: [B, 1] target mate distance (0.0 if no mate, >0 if mate exists)
        
    Returns:
        Scalar loss
    """
    # Cross-entropy loss for soft WDL targets (more stable than KL div)
    # CE = -sum(target * log(pred))
    log_pred = torch.log(pred_wdl + 1e-8)
    loss_wdl = -torch.sum(target_wdl * log_pred, dim=1).mean()
    
    # Conditional mate loss: only when there's actually a mate
    # Create mask for positions with mate (target_mate > 0)
    # Create mask for positions with mate (target_mate > 0)
    mate_mask = (target_mate > 0.01).float()  # Small threshold for numerical stability
    non_mate_mask = 1.0 - mate_mask
    n_mate_positions = mate_mask.sum()
    n_non_mate = non_mate_mask.sum()
    
    # 1. Positive Mate Loss (MSE on actual mates)
    if n_mate_positions > 0:
        masked_pred = pred_mate * mate_mask
        masked_target = target_mate * mate_mask
        loss_pos = F.mse_loss(masked_pred, masked_target, reduction='sum') / (n_mate_positions + 1e-8)
    else:
        loss_pos = torch.tensor(0.0, device=pred_wdl.device)
        
    # 2. Negative Mate Loss (Suppress false mates)
    # We want pred_mate -> 0 when no mate exists
    if n_non_mate > 0:
        # Target is 0, so just MSE(pred, 0) -> pred^2
        # Use small weight (0.1) to avoid overwhelming WDL signal
        masked_pred_neg = pred_mate * non_mate_mask
        loss_neg = (masked_pred_neg).pow(2).sum() / (n_non_mate + 1e-8)
    else:
        loss_neg = torch.tensor(0.0, device=pred_wdl.device)
    
    loss_mate = loss_pos + 0.1 * loss_neg
    
    return wdl_weight * loss_wdl + mate_weight * loss_mate


def killer_loss(pred_wdl, pred_mate, target_wdl, target_mate, wdl_weight=1.0, mate_weight=5.0):
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
    
    # Conditional mate loss: killer WANTS to find mate, so high weight when mate exists
    mate_mask = (target_mate > 0.01).float()
    non_mate_mask = 1.0 - mate_mask
    n_mate_positions = mate_mask.sum()
    n_non_mate = non_mate_mask.sum()
    
    if n_mate_positions > 0:
        masked_pred = pred_mate * mate_mask
        masked_target = target_mate * mate_mask
        loss_pos = F.smooth_l1_loss(masked_pred, masked_target, reduction='sum') / (n_mate_positions + 1e-8)
    else:
        loss_pos = torch.tensor(0.0, device=pred_wdl.device)
        
    if n_non_mate > 0:
        # Suppress false mates in killer positions too
        masked_pred_neg = pred_mate * non_mate_mask
        loss_neg = (masked_pred_neg).pow(2).sum() / (n_non_mate + 1e-8)
    else:
        loss_neg = torch.tensor(0.0, device=pred_wdl.device)
        
    loss_mate = loss_pos + 0.1 * loss_neg
    
    return wdl_weight * loss_wdl + mate_weight * loss_mate


def survivor_loss(pred_wdl, pred_mate, target_wdl, target_mate, wdl_weight=1.0, mate_weight=1.0):
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
    
    # Conditional mate loss: survivor cares about mate only to AVOID it
    # When being mated, we want to predict mate distance to understand threat level
    mate_mask = (target_mate > 0.01).float()
    non_mate_mask = 1.0 - mate_mask
    n_mate_positions = mate_mask.sum()
    n_non_mate = non_mate_mask.sum()
    
    if n_mate_positions > 0:
        masked_pred = pred_mate * mate_mask
        masked_target = target_mate * mate_mask
        loss_pos = F.smooth_l1_loss(masked_pred, masked_target, reduction='sum') / (n_mate_positions + 1e-8)
    else:
        loss_pos = torch.tensor(0.0, device=pred_wdl.device)
        
    if n_non_mate > 0:
        # Suppress false mates
        masked_pred_neg = pred_mate * non_mate_mask
        loss_neg = (masked_pred_neg).pow(2).sum() / (n_non_mate + 1e-8)
    else:
        loss_neg = torch.tensor(0.0, device=pred_wdl.device)
        
    loss_mate = loss_pos + 0.1 * loss_neg
    
    return wdl_weight * loss_wdl + mate_weight * loss_mate


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


def moe_loss(pred_wdl, pred_mate, target_wdl, target_mate, expert_type='BASE'):
    """
    Unified loss function with expert type selection.
    
    Args:
        pred_wdl: [B, 3]
        pred_mate: [B, 1]
        target_wdl: [B, 3]
        target_mate: [B, 1]
        expert_type: 'BASE', 'KILLER', or 'SURVIVOR'
    """
    if expert_type == 'BASE':
        return base_loss(pred_wdl, pred_mate, target_wdl, target_mate)
    elif expert_type == 'KILLER':
        return killer_loss(pred_wdl, pred_mate, target_wdl, target_mate)
    elif expert_type == 'SURVIVOR':
        return survivor_loss(pred_wdl, pred_mate, target_wdl, target_mate)
    else:
        raise ValueError(f"Unknown expert_type: {expert_type}")


# =============================================================================
#   SANITY CHECK
# =============================================================================
if __name__ == "__main__":
    print("--- Loss Function Check ---")
    
    B = 8
    pred_wdl = F.softmax(torch.randn(B, 3), dim=1)
    pred_mate = torch.sigmoid(torch.randn(B, 1))
    target_wdl = F.softmax(torch.randn(B, 3), dim=1)
    target_mate = torch.rand(B, 1)
    
    # Test each loss type
    for expert_type in ['BASE', 'KILLER', 'SURVIVOR']:
        loss = moe_loss(pred_wdl, pred_mate, target_wdl, target_mate, expert_type)
        print(f"{expert_type:10} loss: {loss.item():.4f}")
    
    # Test gater loss
    pred_gates = torch.sigmoid(torch.randn(B, 2))
    target_gates = torch.rand(B, 2)
    g_loss = gater_loss(pred_gates, target_gates)
    print(f"{'GATER':10} loss: {g_loss.item():.4f}")
    
    print("Status: OK ✓")
