#!/bin/bash
# =============================================================================
# train_light_moe_full.sh
# Complete training pipeline for Lightweight MoE
# =============================================================================
#
# Prerequisites:
#   - Trained model_simple checkpoint
#   - Data directory with features.bin, labels.bin, expert_weights.bin
#
# Usage:
#   ./train_light_moe_full.sh
# =============================================================================

set -e  # Exit on error

# =============================================================================
# CONFIGURATION - Edit these paths!
# =============================================================================

# Path to pretrained simple model checkpoint
PRETRAINED="checkpoints/simple_96f8b_best.pt"

# Data directory (with features.bin, labels.bin, expert_weights.bin)
DATA_DIR="../data/processed"

# Output directory for checkpoints
CHECKPOINT_DIR="checkpoints"

# Model architecture (must match pretrained model)
# Model architecture
N_FILTERS=64
N_BLOCKS=4

# Training hyperparameters
BATCH_SIZE=512
BATCH_SIZE_PHASE1=1024
WORKERS=4

# Phase 1: Train Base Model (Simple WDL)
PHASE1_EPOCHS=20
PHASE1_LR=1e-3

# Phase 2: Expert specialization epochs (per expert)
PHASE2_EPOCHS=10
PHASE2_LR=5e-4

# Phase 4: Joint fine-tuning epochs
PHASE4_EPOCHS=5
PHASE4_LR=5e-4  # Will be scaled by 0.1 internally

# Weights & Biases (optional)
# Set to "--wandb" or "--wandb_offline" or "" to disable
WANDB_FLAG="--wandb_offline"

# =============================================================================
# SETUP
# =============================================================================

ARCH="${N_FILTERS}f${N_BLOCKS}b"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="logs/light_moe_${TIMESTAMP}"

mkdir -p "$CHECKPOINT_DIR"
mkdir -p "$LOG_DIR"

echo "============================================================"
echo "  Lightweight MoE Training Pipeline"
echo "============================================================"
echo "  Architecture:     ${ARCH}"
echo "  Data:             ${DATA_DIR}"
echo "  Checkpoints:      ${CHECKPOINT_DIR}"
echo "  Logs:             ${LOG_DIR}"
echo "============================================================"
echo ""

# Check data exists (supports train/val split or single directory)
if [ -d "$DATA_DIR/train" ] && [ -d "$DATA_DIR/val" ]; then
    echo "Data: Using train/val split"
    # For Phase 2, check expert-specific files
    if [ ! -f "$DATA_DIR/train/expert0_features.bin" ]; then
        echo "ERROR: Expert data not found: $DATA_DIR/train/expert0_features.bin"
        exit 1
    fi
    # For Phase 4, check main files
    if [ ! -f "$DATA_DIR/train/features.bin" ]; then
        echo "WARNING: Main features.bin not found in train/, Phase 4 may fail"
    fi
elif [ -f "$DATA_DIR/features.bin" ]; then
    echo "Data: Using single directory"
else
    echo "ERROR: No valid data found in $DATA_DIR"
    echo "Expected either:"
    echo "  - $DATA_DIR/train/ and $DATA_DIR/val/ subdirectories"
    echo "  - $DATA_DIR/features.bin"
    exit 1
fi

# =============================================================================
# PHASE 1: TRAIN BASE MODEL + INIT MOE
# =============================================================================
# Trains the dense backbone from scratch (if no pretrained), then inits MoE

echo ""
echo "============================================================"
echo "  PHASE 1: Train Base Model + Init MoE"
echo "============================================================"
echo "  Epochs:            ${PHASE1_EPOCHS}"
echo "  Learning rate:     ${PHASE1_LR}"
echo "============================================================"
echo ""

python train_light_moe.py \
    --phase 1 \
    --data_dir "$DATA_DIR" \
    --epochs "$PHASE1_EPOCHS" \
    --batch_size "$BATCH_SIZE_PHASE1" \
    --lr "$PHASE1_LR" \
    --n_filters "$N_FILTERS" \
    --n_blocks "$N_BLOCKS" \
    --checkpoint_dir "$CHECKPOINT_DIR" \
    $WANDB_FLAG \
    2>&1 

PHASE1_CHECKPOINT="${CHECKPOINT_DIR}/light_moe_${ARCH}_phase1.pt"

if [ ! -f "$PHASE1_CHECKPOINT" ]; then
    echo "ERROR: Phase 1 checkpoint not created!"
    exit 1
fi

echo "Phase 1 complete! Checkpoint: $PHASE1_CHECKPOINT"

# =============================================================================
# PHASE 2: EXPERT SPECIALIZATION
# =============================================================================
# Trains each expert on weighted data (backbone frozen)
# Each expert learns from samples where it has high routing weight

echo ""
echo "============================================================"
echo "  PHASE 2: Expert Specialization"
echo "============================================================"
echo "  Input:             ${PHASE1_CHECKPOINT}"
echo "  Epochs per expert: ${PHASE2_EPOCHS}"
echo "  Learning rate:     ${PHASE2_LR}"
echo "  Backbone:          FROZEN"
echo "============================================================"
echo ""

python train_light_moe.py \
    --phase 2 \
    --data_dir "$DATA_DIR" \
    --checkpoint "$PHASE1_CHECKPOINT" \
    --epochs "$PHASE2_EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --lr "$PHASE2_LR" \
    --n_filters "$N_FILTERS" \
    --n_blocks "$N_BLOCKS" \
    --workers "$WORKERS" \
    --checkpoint_dir "$CHECKPOINT_DIR" \
    $WANDB_FLAG \
    2>&1

PHASE2_CHECKPOINT="${CHECKPOINT_DIR}/light_moe_${ARCH}_phase2.pt"

if [ ! -f "$PHASE2_CHECKPOINT" ]; then
    echo "ERROR: Phase 2 checkpoint not created!"
    exit 1
fi

echo ""
echo "Phase 2 complete! Checkpoint: $PHASE2_CHECKPOINT"
echo ""

# =============================================================================
# PHASE 4: JOINT FINE-TUNING (Top-2 Routing)
# =============================================================================
# Fine-tunes entire model with top-2 expert routing
# All parameters trainable, lower learning rate

echo ""
echo "============================================================"
echo "  PHASE 4: Joint Fine-tuning (Top-2 Routing)"
echo "============================================================"
echo "  Epochs:            ${PHASE4_EPOCHS}"
echo "  Learning rate:     ${PHASE4_LR} * 0.1 = $(echo "scale=6; $PHASE4_LR * 0.1" | bc)"
echo "  All parameters:    TRAINABLE"
echo "============================================================"
echo ""

python train_light_moe.py \
    --phase 4 \
    --data_dir "$DATA_DIR" \
    --checkpoint "$PHASE2_CHECKPOINT" \
    --epochs "$PHASE4_EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --lr "$PHASE4_LR" \
    --n_filters "$N_FILTERS" \
    --n_blocks "$N_BLOCKS" \
    --workers "$WORKERS" \
    --checkpoint_dir "$CHECKPOINT_DIR" \
    $WANDB_FLAG \
    2>&1

FINAL_CHECKPOINT="${CHECKPOINT_DIR}/light_moe_${ARCH}_best.pt"

if [ ! -f "$FINAL_CHECKPOINT" ]; then
    echo "ERROR: Final checkpoint not created!"
    exit 1
fi

echo ""
echo "Phase 4 complete! Final checkpoint: $FINAL_CHECKPOINT"
echo ""

# =============================================================================
# EXPORT TO ONNX
# =============================================================================

ONNX_DIR="onnx/light_moe_${ARCH}"

echo ""
echo "============================================================"
echo "  Exporting to ONNX"
echo "============================================================"
echo "  Checkpoint:  ${FINAL_CHECKPOINT}"
echo "  Output:      ${ONNX_DIR}/"
echo "============================================================"
echo ""

python export_light_moe.py \
    --checkpoint "$FINAL_CHECKPOINT" \
    --output_dir "$ONNX_DIR" \
    --n_filters "$N_FILTERS" \
    --n_blocks "$N_BLOCKS" \
    2>&1 

# =============================================================================
# SUMMARY
# =============================================================================

echo ""
echo "============================================================"
echo "  TRAINING COMPLETE!"
echo "============================================================"
echo ""
echo "  Checkpoints:"
echo "    Phase 2: ${CHECKPOINT_DIR}/light_moe_${ARCH}_phase2.pt"
echo "    Best:    ${CHECKPOINT_DIR}/light_moe_${ARCH}_best.pt"
echo "    Last:    ${CHECKPOINT_DIR}/light_moe_${ARCH}_last.pt"
echo ""
echo "  ONNX models:"
echo "    ${ONNX_DIR}/backbone.onnx"
echo "    ${ONNX_DIR}/expert0.onnx (Tactical)"
echo "    ${ONNX_DIR}/expert1.onnx (Strategic)"
echo "    ${ONNX_DIR}/expert2.onnx (Major Endgame)"
echo "    ${ONNX_DIR}/expert3.onnx (Minor Endgame)"
echo ""
echo "  Logs: ${LOG_DIR}/"
echo ""
echo "============================================================"
