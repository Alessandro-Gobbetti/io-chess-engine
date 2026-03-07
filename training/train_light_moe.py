"""
Training script for Lightweight MoE model.

3-Phase Training Pipeline:
  Phase 1: Load from pretrained model_simple (no training)
  Phase 2: Expert Specialization (freeze backbone, train each expert)
  Phase 4: Joint Fine-tuning (unfreeze all, top-2 routing)

Features:
  - Load weights from trained model_simple (skip training Phase 1)
  - Top-2 expert routing (only 2 experts run at inference)
  - Weights & Biases logging (--wandb / --wandb_offline)
  - Rich epoch prints with colors and stats

Usage:
  python train_light_moe.py --phase 2 --data_dir ../data/processed \\
                            --pretrained checkpoints/simple_best.pt --wandb

  python train_light_moe.py --phase 4 --data_dir ../data/processed \\
                            --checkpoint checkpoints/light_moe_phase2.pt
"""

import argparse
import os
import copy
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

from model_simple import ChessNetWDL
from model_light_moe import ChessNetLightMoE
from dataset_moe import ChessMoEDataset, ChessExpertDataset
from dataset_simple import ChessSimpleDataset
from loss import base_loss
from utils import set_seed, Colors

C = Colors  # Shorthand


# =============================================================================
#   UTILITIES
# =============================================================================

def print_header(text):
    """Print a colored header."""
    print(f"\n{C.CYAN}{C.BOLD}{'=' * 70}{C.RESET}")
    print(f"{C.CYAN}{C.BOLD}  {text}{C.RESET}")
    print(f"{C.CYAN}{C.BOLD}{'=' * 70}{C.RESET}")


def format_time(seconds):
    """Format seconds into human-readable time."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    return f"{seconds/3600:.1f}h"


def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def init_wandb(args, phase_name):
    """Initialize wandb if enabled."""
    if not args.wandb and not args.wandb_offline:
        return None
    
    if not WANDB_AVAILABLE:
        print(f"{C.YELLOW}[WARN] wandb not installed, skipping logging{C.RESET}")
        return None
    
    mode = "offline" if args.wandb_offline else "online"
    
    run = wandb.init(
        project="io-chess-engine",
        name=f"phase{args.phase}_{args.n_filters}f{args.n_blocks}b",
        config={
            "phase": args.phase,
            "n_filters": args.n_filters,
            "n_blocks": args.n_blocks,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
        },
        mode=mode,
        tags=[phase_name, f"{args.n_filters}f{args.n_blocks}b"],
    )
    print(f"{C.GREEN}[wandb] Initialized in {mode} mode{C.RESET}")
    return run


# =============================================================================
#   TOP-2 EXPERT ROUTING (Optimized)
# =============================================================================

def forward_top2_optimized(model, features, base_weights, device):
    """
    Efficient top-2 expert routing.
    
    Runs only the 2 highest-weighted experts per sample, NOT all 4.
    Uses masked indexing for efficiency.
    
    Args:
        model: ChessNetLightMoE
        features: [B, n_filters, 8, 8] backbone output  
        base_weights: [B, 4] expert weights
        device: torch device
        
    Returns:
        pred_wdl: [B, 3]
        pred_mate: [B, 1]
    """
    B = features.size(0)
    
    # Get top-2 experts by weight
    top2_values, top2_idx = torch.topk(base_weights, k=2, dim=1)
    top2_contrib = top2_values / (top2_values.sum(dim=1, keepdim=True) + 1e-8)
    
    # Pre-allocate output tensors
    pred_wdl = torch.zeros(B, 3, device=device)
    pred_mate = torch.zeros(B, 1, device=device)
    
    # Run each expert only on samples where it's in top-2
    for expert_idx in range(4):
        # Find samples where this expert is top-1 or top-2
        is_top1 = (top2_idx[:, 0] == expert_idx)
        is_top2 = (top2_idx[:, 1] == expert_idx)
        mask = is_top1 | is_top2
        
        if mask.sum() == 0:
            continue
        
        # Run expert only on relevant samples
        wdl, mate = model.experts[expert_idx](features[mask])
        
        # Add weighted contribution for top-1 samples
        top1_mask = is_top1[mask]
        if top1_mask.any():
            contrib = top2_contrib[mask][top1_mask, 0:1]
            pred_wdl[mask.nonzero(as_tuple=True)[0][top1_mask]] += wdl[top1_mask] * contrib
            pred_mate[mask.nonzero(as_tuple=True)[0][top1_mask]] += mate[top1_mask] * contrib
        
        # Add weighted contribution for top-2 samples (exclude those already counted as top-1)
        top2_only = is_top2[mask] & ~is_top1[mask]
        if top2_only.any():
            contrib = top2_contrib[mask][top2_only, 1:2]
            pred_wdl[mask.nonzero(as_tuple=True)[0][top2_only]] += wdl[top2_only] * contrib
            pred_mate[mask.nonzero(as_tuple=True)[0][top2_only]] += mate[top2_only] * contrib
    
    return pred_wdl, pred_mate


def forward_top2_simple(model, features, base_weights, device):
    """
    Simple top-2 routing (cleaner but less efficient).
    Use this if optimized version has issues.
    """
    B = features.size(0)
    top2_values, top2_idx = torch.topk(base_weights, k=2, dim=1)
    top2_contrib = top2_values / (top2_values.sum(dim=1, keepdim=True) + 1e-8)
    
    pred_wdl = torch.zeros(B, 3, device=device)
    pred_mate = torch.zeros(B, 1, device=device)
    
    for k in range(2):
        for expert_idx in range(4):
            mask = (top2_idx[:, k] == expert_idx)
            if mask.sum() == 0:
                continue
            wdl, mate = model.experts[expert_idx](features[mask])
            pred_wdl[mask] += wdl * top2_contrib[mask, k:k+1]
            pred_mate[mask] += mate * top2_contrib[mask, k:k+1]
    
    return pred_wdl, pred_mate


# =============================================================================
#   PHASE 1: LOAD FROM SIMPLE MODEL
# =============================================================================

def train_base_model(args, wandb_run=None):
    """Train a simple WDL model from scratch."""
    print_header("PHASE 1: TRAIN BASE MODEL")
    
    device = get_device()
    train_dir = os.path.join(args.data_dir, 'train')
    val_dir = os.path.join(args.data_dir, 'val')
    
    # Handle single directory case
    if not os.path.exists(train_dir):
        print(f"{C.YELLOW}[WARN] No train/val split found, using same dir{C.RESET}")
        train_ds = ChessSimpleDataset(args.data_dir, split='train') # Logic handles single dir usually
        val_ds = ChessSimpleDataset(args.data_dir, split='val')
    else:
        train_ds = ChessSimpleDataset(args.data_dir, split='train')
        val_ds = ChessSimpleDataset(args.data_dir, split='val')

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, 
                              num_workers=args.workers, pin_memory=True, persistent_workers=True, prefetch_factor=4)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, pin_memory=True, persistent_workers=True, prefetch_factor=4)
    
    print(f"Train: {C.CYAN}{len(train_ds):,}{C.RESET} samples")
    print(f"Val:   {C.CYAN}{len(val_ds):,}{C.RESET} samples")
    
    # Create Base Model
    simple_model = ChessNetWDL(n_filters=args.n_filters, n_blocks=args.n_blocks).to(device)
    optimizer = optim.AdamW(simple_model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    best_val_loss = float('inf')
    best_state = None
    
    for epoch in range(args.epochs):
        t0 = time.time()
        
        # Train Loop (inline simplified)
        simple_model.train()
        train_loss = 0
        count = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False)
        for batch in pbar:
            layers = batch['layers'].float().to(device) / 255.0
            global_feats = batch['global'].to(device)
            target_wdl = batch['wdl'].to(device)
            target_mate = batch['mate'].to(device)
            
            optimizer.zero_grad()
            pred_wdl, pred_mate = simple_model(layers, global_feats)
            loss = base_loss(pred_wdl, pred_mate, target_wdl, target_mate)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(simple_model.parameters(), 1.0)
            optimizer.step()
            
            train_loss += loss.item() * layers.size(0)
            count += layers.size(0)
            pbar.set_postfix(loss=f"{loss.item():.4f}")
        train_loss /= count
        
        # Validation
        simple_model.eval()
        val_loss = 0
        count = 0
        with torch.no_grad():
            for batch in val_loader:
                layers = batch['layers'].float().to(device) / 255.0
                global_feats = batch['global'].to(device)
                target_wdl = batch['wdl'].to(device)
                target_mate = batch['mate'].to(device)
                pred_wdl, pred_mate = simple_model(layers, global_feats)
                loss = base_loss(pred_wdl, pred_mate, target_wdl, target_mate)
                val_loss += loss.item() * layers.size(0)
                count += layers.size(0)
        val_loss /= count
        
        scheduler.step()
        elapsed = time.time() - t0
        
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            best_state = copy.deepcopy(simple_model.state_dict())
        
        best_str = f" {C.GREEN}★ BEST{C.RESET}" if is_best else ""
        lr = optimizer.param_groups[0]['lr']
        print(f"  {C.YELLOW}{epoch+1:3d}/{args.epochs}{C.RESET} │ "
              f"train: {C.CYAN}{train_loss:.5f}{C.RESET} │ "
              f"val: {C.MAGENTA}{val_loss:.5f}{C.RESET} │ "
              f"lr: {lr:.2e} │ {C.GREEN}{format_time(elapsed)}{C.RESET}{best_str}")
        
        if wandb_run:
            wandb.log({"phase1/train_loss": train_loss, "phase1/val_loss": val_loss, "phase1/lr": lr, "phase1/epoch": epoch+1})

    # Restore best
    if best_state:
        simple_model.load_state_dict(best_state)
    
    # Save base checkpoint
    arch_str = f"{args.n_filters}f{args.n_blocks}b"
    save_path = os.path.join(args.checkpoint_dir, f"simple_{arch_str}_best.pt")
    torch.save({'model': best_state, 'phase': 'base', 'val_loss': best_val_loss}, save_path)
    print(f"\n{C.GREEN}Base training complete!{C.RESET} Saved to {save_path}")
    
    return simple_model


def run_phase1(model, args, wandb_run=None):
    """Load backbone from trained model_simple OR train it from scratch."""
    
    simple_model = None
    
    if args.pretrained and os.path.exists(args.pretrained):
        print_header("PHASE 1: LOAD FROM SIMPLE MODEL")
        device = get_device()
        print(f"Loading pretrained model: {C.CYAN}{args.pretrained}{C.RESET}")
        ckpt = torch.load(args.pretrained, map_location=device, weights_only=False)
        simple_model = ChessNetWDL(n_filters=args.n_filters, n_blocks=args.n_blocks)
        simple_model.load_state_dict(ckpt['model'])
        print(f"  {C.GREEN}✓{C.RESET} Epoch: {ckpt.get('epoch', '?')}")
        print(f"  {C.GREEN}✓{C.RESET} Val loss: {ckpt.get('val_loss', 0):.5f}")
    else:
        # Train from scratch
        print(f"{C.YELLOW}No pretrained model provided. Training base model from scratch...{C.RESET}")
        simple_model = train_base_model(args, wandb_run)
    
    # Initialize MoE from Simple
    print(f"\n{C.CYAN}Initializing MoE weights from base model...{C.RESET}")
    model.load_from_simple(simple_model)
    
    # Save Phase 1 checkpoint
    arch_str = f"{args.n_filters}f{args.n_blocks}b"
    save_path = os.path.join(args.checkpoint_dir, f"light_moe_{arch_str}_phase1.pt")
    torch.save({
        'model': model.state_dict(),
        'phase': 1,
        'pretrained': args.pretrained or "trained_from_scratch",
    }, save_path)
    
    print(f"\n{C.GREEN}Phase 1 complete!{C.RESET} Checkpoint: {save_path}")
    return model


# =============================================================================
#   PHASE 2: EXPERT SPECIALIZATION
# =============================================================================

def train_expert_epoch(model, expert_idx, train_loader, optimizer, device):
    """
    Train single expert for one epoch on expert-specific data.
    
    Uses ChessExpertDataset which contains only samples where this expert
    is the primary (raw_score > 3.0), so no weighting needed.
    """
    model.train()
    
    # Keep backbone frozen (in eval mode for stable BN)
    model.stem.eval()
    model.bn_stem.eval()
    model.stem_global.eval()
    model.bn_global.eval()
    model.body.eval()
    
    total_loss = 0
    total_samples = 0
    
    pbar = tqdm(train_loader, desc=f"Expert {expert_idx}", leave=False, 
                bar_format='{l_bar}{bar:30}{r_bar}')
    
    for batch in pbar:
        layers = batch['layers'].float().to(device) / 255.0
        global_feats = batch['global'].to(device)
        target_wdl = batch['wdl'].to(device)
        target_mate = batch['mate'].to(device)
        
        optimizer.zero_grad()
        
        # Backbone with no grad (frozen)
        with torch.no_grad():
            features = model.forward_backbone(layers, global_feats)
        
        # Expert forward and loss (no weighting - data is already filtered)
        pred_wdl, pred_mate = model.experts[expert_idx](features)
        loss = base_loss(pred_wdl, pred_mate, target_wdl, target_mate)
        
        if not torch.isnan(loss):
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item() * layers.size(0)
            total_samples += layers.size(0)
        
        pbar.set_postfix(loss=f"{loss.item():.4f}")
    
    return total_loss / total_samples if total_samples > 0 else float('inf')


def validate_expert(model, expert_idx, val_loader, device):
    """Validate a single expert on expert-specific data."""
    model.eval()
    total_loss = 0
    total_samples = 0
    
    with torch.no_grad():
        for batch in val_loader:
            layers = batch['layers'].float().to(device) / 255.0
            global_feats = batch['global'].to(device)
            target_wdl = batch['wdl'].to(device)
            target_mate = batch['mate'].to(device)
            
            features = model.forward_backbone(layers, global_feats)
            pred_wdl, pred_mate = model.experts[expert_idx](features)
            
            loss = base_loss(pred_wdl, pred_mate, target_wdl, target_mate)
            
            total_loss += loss.item() * layers.size(0)
            total_samples += layers.size(0)
    
    return total_loss / total_samples if total_samples > 0 else float('inf')


def run_phase2(model, args, wandb_run=None):
    """
    Train each expert on its own specialized dataset.
    
    Uses separate expertN_features.bin / expertN_labels.bin files
    generated by preprocessing (filtered samples per expert).
    """
    print_header("PHASE 2: EXPERT SPECIALIZATION")
    
    device = get_device()
    model = model.to(device)
    
    expert_names = ['Tactical', 'Strategic', 'MajorEnd', 'MinorEnd']
    all_results = []
    
    # Determine data directories
    train_dir = os.path.join(args.data_dir, 'train')
    val_dir = os.path.join(args.data_dir, 'val')
    
    # Check if split dirs exist, otherwise use data_dir directly
    if not os.path.exists(train_dir):
        train_dir = args.data_dir
        val_dir = args.data_dir  # Will need to handle validation split manually
        print(f"{C.YELLOW}[WARN] No train/val split found, using same dir{C.RESET}")
    
    for expert_idx in range(4):
        name = expert_names[expert_idx]
        print(f"\n{C.YELLOW}{'─'*70}{C.RESET}")
        print(f"{C.YELLOW}Training Expert {expert_idx}: {name}{C.RESET}")
        print(f"{C.YELLOW}{'─'*70}{C.RESET}")
        
        # Load expert-specific datasets
        try:
            train_ds = ChessExpertDataset(train_dir, expert_idx=expert_idx)
            val_ds = ChessExpertDataset(val_dir, expert_idx=expert_idx)
        except FileNotFoundError as e:
            print(f"{C.RED}[ERROR] Expert {expert_idx} dataset not found: {e}{C.RESET}")
            print(f"{C.YELLOW}Skipping this expert...{C.RESET}")
            continue
        
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                                  num_workers=args.workers, pin_memory=True, persistent_workers=True, prefetch_factor=4)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                                num_workers=args.workers, pin_memory=True, persistent_workers=True, prefetch_factor=4)
        
        print(f"Train: {C.CYAN}{len(train_ds):,}{C.RESET} samples")
        print(f"Val:   {C.CYAN}{len(val_ds):,}{C.RESET} samples")
        
        # Freeze all, unfreeze this expert
        for param in model.parameters():
            param.requires_grad = False
        for param in model.experts[expert_idx].parameters():
            param.requires_grad = True
        
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Trainable params: {C.CYAN}{trainable:,}{C.RESET}")
        
        optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.lr, weight_decay=1e-4
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
        
        best_val_loss = float('inf')
        best_state = None
        epoch_results = []
        
        for epoch in range(args.epochs):
            t0 = time.time()
            
            train_loss = train_expert_epoch(model, expert_idx, train_loader, optimizer, device)
            val_loss = validate_expert(model, expert_idx, val_loader, device)
            scheduler.step()
            
            elapsed = time.time() - t0
            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss
                best_state = copy.deepcopy(model.experts[expert_idx].state_dict())
            
            lr = optimizer.param_groups[0]['lr']
            
            # Rich epoch print
            best_str = f" {C.GREEN}★ BEST{C.RESET}" if is_best else ""
            delta = val_loss - best_val_loss if not is_best else 0
            delta_str = f" (+{delta:.5f})" if delta > 0 else ""
            
            print(f"  {C.YELLOW}{epoch+1:3d}/{args.epochs}{C.RESET} │ "
                  f"train: {C.CYAN}{train_loss:.5f}{C.RESET} │ "
                  f"val: {C.MAGENTA}{val_loss:.5f}{C.RESET}{delta_str} │ "
                  f"lr: {lr:.2e} │ "
                  f"{C.GREEN}{format_time(elapsed)}{C.RESET}{best_str}")
            
            epoch_results.append({
                'train_loss': train_loss,
                'val_loss': val_loss,
                'is_best': is_best,
            })
            
            if wandb_run:
                wandb.log({
                    f"expert{expert_idx}/train_loss": train_loss,
                    f"expert{expert_idx}/val_loss": val_loss,
                    f"expert{expert_idx}/lr": lr,
                    f"expert{expert_idx}/epoch": epoch + 1,
                })
        
        # Restore best
        if best_state:
            model.experts[expert_idx].load_state_dict(best_state)
        
        all_results.append({
            'name': name,
            'best_val_loss': best_val_loss,
            'epochs': epoch_results,
        })
        
        print(f"  {C.GREEN}✓ Best: {best_val_loss:.5f}{C.RESET}")
    
    # Summary table
    print(f"\n{C.CYAN}{'═'*70}{C.RESET}")
    print(f"{C.CYAN}PHASE 2 SUMMARY{C.RESET}")
    print(f"{C.CYAN}{'═'*70}{C.RESET}")
    print(f"{'Expert':<15} {'Best Val Loss':<15}")
    print(f"{'─'*30}")
    for r in all_results:
        print(f"{r['name']:<15} {r['best_val_loss']:.5f}")
    
    # Save checkpoint
    arch_str = f"{args.n_filters}f{args.n_blocks}b"
    save_path = os.path.join(args.checkpoint_dir, f"light_moe_{arch_str}_phase2.pt")
    torch.save({
        'model': model.state_dict(),
        'phase': 2,
        'results': all_results,
    }, save_path)
    
    print(f"\n{C.GREEN}Phase 2 complete!{C.RESET} Checkpoint: {save_path}")
    return model


# =============================================================================
#   PHASE 4: JOINT FINE-TUNING (Top-2)
# =============================================================================

def train_phase4_epoch(model, train_loader, optimizer, device):
    """Train one epoch with top-2 routing."""
    model.train()
    total_loss = 0
    total_samples = 0
    top2_usage = torch.zeros(4)  # Track which experts are used
    
    pbar = tqdm(train_loader, desc="Train", leave=False,
                bar_format='{l_bar}{bar:30}{r_bar}')
    
    for batch in pbar:
        layers = batch['layers'].float().to(device) / 255.0
        global_feats = batch['global'].to(device)
        target_wdl = batch['wdl'].to(device)
        target_mate = batch['mate'].to(device)
        base_weights = batch['base_weights'].to(device)
        
        # Sharpen weights
        base_weights = F.softmax(base_weights * 5.0, dim=1)
        
        optimizer.zero_grad()
        
        features = model.forward_backbone(layers, global_feats)
        pred_wdl, pred_mate = forward_top2_simple(model, features, base_weights, device)
        
        loss = base_loss(pred_wdl, pred_mate, target_wdl, target_mate)
        
        if torch.isnan(loss):
            continue
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item() * layers.size(0)
        total_samples += layers.size(0)
        
        # Track expert usage
        top2_idx = torch.topk(base_weights, k=2, dim=1)[1]
        for i in range(4):
            top2_usage[i] += ((top2_idx == i).sum().item())
        
        pbar.set_postfix(loss=f"{loss.item():.4f}")
    
    # Normalize usage
    top2_usage = top2_usage / (total_samples * 2) * 100  # percentage
    
    return total_loss / total_samples if total_samples > 0 else float('inf'), top2_usage


def validate_phase4(model, val_loader, device):
    """Validate with top-2 routing."""
    model.eval()
    total_loss = 0
    total_samples = 0
    
    with torch.no_grad():
        for batch in val_loader:
            layers = batch['layers'].float().to(device) / 255.0
            global_feats = batch['global'].to(device)
            target_wdl = batch['wdl'].to(device)
            target_mate = batch['mate'].to(device)
            base_weights = batch['base_weights'].to(device)
            
            base_weights = F.softmax(base_weights * 5.0, dim=1)
            
            features = model.forward_backbone(layers, global_feats)
            pred_wdl, pred_mate = forward_top2_simple(model, features, base_weights, device)
            
            loss = base_loss(pred_wdl, pred_mate, target_wdl, target_mate)
            
            if not torch.isnan(loss):
                total_loss += loss.item() * layers.size(0)
                total_samples += layers.size(0)
    
    return total_loss / total_samples if total_samples > 0 else float('inf')


def run_phase4(model, train_dataset, val_dataset, args, wandb_run=None):
    """Joint fine-tuning with top-2 routing."""
    print_header("PHASE 4: JOINT FINE-TUNING (Top-2 Routing)")
    
    device = get_device()
    model = model.to(device)
    
    # Unfreeze all
    for param in model.parameters():
        param.requires_grad = True
    
    trainable = sum(p.numel() for p in model.parameters())
    print(f"All params trainable: {C.CYAN}{trainable:,}{C.RESET}")
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=True, persistent_workers=True, prefetch_factor=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, pin_memory=True, persistent_workers=True, prefetch_factor=4)
    
    # Lower LR for fine-tuning
    lr = args.lr * 0.1
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    print(f"LR: {C.CYAN}{lr:.2e}{C.RESET} (0.1x base)")
    print(f"Train: {C.CYAN}{len(train_dataset):,}{C.RESET} samples")
    print(f"Val:   {C.CYAN}{len(val_dataset):,}{C.RESET} samples")
    print()
    
    best_val_loss = float('inf')
    best_state = None
    arch_str = f"{args.n_filters}f{args.n_blocks}b"
    expert_names = ['Tac', 'Str', 'Maj', 'Min']
    
    for epoch in range(args.epochs):
        t0 = time.time()
        
        train_loss, expert_usage = train_phase4_epoch(model, train_loader, optimizer, device)
        val_loss = validate_phase4(model, val_loader, device)
        scheduler.step()
        
        elapsed = time.time() - t0
        is_best = val_loss < best_val_loss
        
        if is_best:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            torch.save({
                'model': model.state_dict(),
                'epoch': epoch + 1,
                'phase': 4,
                'val_loss': val_loss,
            }, os.path.join(args.checkpoint_dir, f"light_moe_{arch_str}_best.pt"))
        
        lr = optimizer.param_groups[0]['lr']
        
        # Rich epoch print with expert usage
        best_str = f" {C.GREEN}★ BEST{C.RESET}" if is_best else ""
        usage_str = " │ ".join([f"{expert_names[i]}:{expert_usage[i]:.0f}%" for i in range(4)])
        
        print(f"{C.YELLOW}{epoch+1:3d}/{args.epochs}{C.RESET} │ "
              f"train: {C.CYAN}{train_loss:.5f}{C.RESET} │ "
              f"val: {C.MAGENTA}{val_loss:.5f}{C.RESET} │ "
              f"lr: {lr:.2e} │ "
              f"{C.GREEN}{format_time(elapsed)}{C.RESET} │ "
              f"[{usage_str}]{best_str}")
        
        if wandb_run:
            wandb.log({
                "phase4/train_loss": train_loss,
                "phase4/val_loss": val_loss,
                "phase4/lr": lr,
                "phase4/epoch": epoch + 1,
                **{f"phase4/usage_{expert_names[i]}": expert_usage[i].item() for i in range(4)},
            })
        
        # Save last
        torch.save({
            'model': model.state_dict(),
            'epoch': epoch + 1,
            'phase': 4,
            'val_loss': val_loss,
        }, os.path.join(args.checkpoint_dir, f"light_moe_{arch_str}_last.pt"))
    
    # Restore best
    if best_state:
        model.load_state_dict(best_state)
    
    print(f"\n{C.GREEN}Phase 4 complete!{C.RESET} Best val loss: {best_val_loss:.5f}")
    return model


# =============================================================================
#   MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Train Lightweight MoE (multi-phase)")
    parser.add_argument("--phase", type=int, required=True, choices=[1, 2, 4])
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--pretrained", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--n_filters", type=int, default=64)
    parser.add_argument("--n_blocks", type=int, default=6)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints")
    parser.add_argument("--wandb", action="store_true", help="Enable W&B logging")
    parser.add_argument("--wandb_offline", action="store_true", help="W&B offline mode")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    set_seed(args.seed)
    
    device = get_device()
    print(f"\n{C.CYAN}Device:{C.RESET} {device}")
    print(f"{C.CYAN}Architecture:{C.RESET} {args.n_filters}f{args.n_blocks}b")
    
    # Create model
    model = ChessNetLightMoE(n_filters=args.n_filters, n_blocks=args.n_blocks)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"{C.CYAN}Total params:{C.RESET} {total_params:,}")
    
    # Init wandb
    phase_names = {1: "load", 2: "experts", 4: "finetune"}
    wandb_run = init_wandb(args, phase_names[args.phase])
    
    # Load data only for Phase 4 (Phase 2 loads per-expert datasets internally)
    train_dataset = None
    val_dataset = None
    
    if args.phase == 4:
        print(f"\n{C.CYAN}Loading data from {args.data_dir}...{C.RESET}")
        train_dir = os.path.join(args.data_dir, 'train')
        val_dir = os.path.join(args.data_dir, 'val')
        
        if os.path.exists(train_dir) and os.path.exists(val_dir):
            train_dataset = ChessMoEDataset(train_dir)
            val_dataset = ChessMoEDataset(val_dir)
        else:
            print(f"{C.YELLOW}[WARN] No train/val split found, using same dir{C.RESET}")
            full_ds = ChessMoEDataset(args.data_dir)
            n_val = len(full_ds) // 10
            n_train = len(full_ds) - n_val
            train_dataset, val_dataset = torch.utils.data.random_split(full_ds, [n_train, n_val])
    
    # Run phase
    if args.phase == 1:
        # allow running without pretrained if we are training base
        model = run_phase1(model, args, wandb_run)
    
    elif args.phase == 2:
        if args.checkpoint:
            ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
            model.load_state_dict(ckpt['model'])
            print(f"Loaded checkpoint: {args.checkpoint}")
        elif args.pretrained:
            model = run_phase1(model, args)
        else:
            print(f"{C.RED}Error: --checkpoint or --pretrained required for Phase 2{C.RESET}")
            return
        
        # Phase 2 loads expert-specific datasets internally
        model = run_phase2(model, args, wandb_run)
    
    elif args.phase == 4:
        if not args.checkpoint:
            print(f"{C.RED}Error: --checkpoint required for Phase 4{C.RESET}")
            return
        
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model'])
        print(f"Loaded checkpoint: {args.checkpoint}")
        
        model = run_phase4(model, train_dataset, val_dataset, args, wandb_run)
    
    if wandb_run:
        wandb.finish()
    
    print(f"\n{C.GREEN}{'═'*70}{C.RESET}")
    print(f"{C.GREEN}  Done!{C.RESET}")
    print(f"{C.GREEN}{'═'*70}{C.RESET}\n")


if __name__ == "__main__":
    main()
