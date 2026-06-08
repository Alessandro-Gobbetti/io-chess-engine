#!/bin/bash
# =============================================================================
## @file train.sh
## @brief Complete multi-phase training pipeline for model.py
# =============================================================================
#
# Usage:
#   ./train.sh
# =============================================================================

set -euo pipefail

# =============================================================================
# CONFIGURATION
# =============================================================================

DATA_DIRS="${DATA_DIRS:-../data/preprocessed,/media/io/usb_drive/preprocessed}"
CHECKPOINT_DIR="checkpoints"

# Run naming (set RUN_TAG env var to compare variants, e.g. RUN_TAG=baseline)
RUN_TAG="${RUN_TAG:-updated_loss}"
RUN_TAG_SAFE="${RUN_TAG// /_}"

# Model architecture
N_GLOBALS=21
BRANCH_DIM=16
MIXER_OUT=64
N_BYPASS=12
N_EXPERTS=4
EXPERT_BOTTLENECK=32
EXPERT_HIDDEN=128
EXPERT_POOL="flat"  # flat | gap | pool2avg | pool2max

# Training hyperparameters
BATCH_SIZE=1024
BATCH_SIZE_PHASE1=1024
WORKERS=4
VAL_SPLITS=4

# Phase 1: base training
PHASE1_EPOCHS=30
PHASE1_LR=1e-3
PHASE1_WARMUP_STEPS=5000

# Phase 2: expert specialization
PHASE2_EPOCHS=20
PHASE2_LR=5e-4
PHASE2_WARMUP_STEPS=1000

# Phase 4: joint fine-tuning
PHASE4_EPOCHS=30
PHASE4_LR=5e-4
PHASE4_WARMUP_STEPS=1000

# Optional: --wandb | --wandb_offline | ""
WANDB_FLAG="--wandb_offline"

# Export output directory
PARITY_OUT_DIR="moe_cache"

# =============================================================================
# SETUP
# =============================================================================

ARCH="bd${BRANCH_DIM}_m${MIXER_OUT}_eb${EXPERT_BOTTLENECK}_eh${EXPERT_HIDDEN}_${EXPERT_POOL}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUN_ID="${RUN_TAG_SAFE}_${TIMESTAMP}"
RUN_CHECKPOINT_DIR="${CHECKPOINT_DIR}/${RUN_ID}"
RUN_PARITY_OUT_DIR="${PARITY_OUT_DIR}/${RUN_ID}"
LOG_DIR="logs/model_${RUN_ID}"

mkdir -p "$RUN_CHECKPOINT_DIR"
mkdir -p "$LOG_DIR"
mkdir -p "$RUN_PARITY_OUT_DIR"

echo "============================================================"
echo "  model Training Pipeline"
echo "============================================================"
echo "  Run tag:          ${RUN_TAG_SAFE}"
echo "  Run id:           ${RUN_ID}"
echo "  Architecture:     ${ARCH}"
echo "  Data roots:       ${DATA_DIRS}"
echo "  Checkpoints:      ${RUN_CHECKPOINT_DIR}"
echo "  Parity output:    ${RUN_PARITY_OUT_DIR}"
echo "  Logs:             ${LOG_DIR}"
echo "============================================================"
echo ""

IFS=',' read -r -a DATA_ROOTS <<< "$DATA_DIRS"
if [ ${#DATA_ROOTS[@]} -eq 0 ]; then
    echo "ERROR: DATA_DIRS is empty"
    exit 1
fi

trim() {
    local s="$1"
    s="${s#${s%%[![:space:]]*}}"
    s="${s%${s##*[![:space:]]}}"
    printf "%s" "$s"
}

for RAW_ROOT in "${DATA_ROOTS[@]}"; do
    DATA_ROOT="$(trim "$RAW_ROOT")"
    if [ -z "$DATA_ROOT" ]; then
        continue
    fi

    echo "Data: checking root $DATA_ROOT"

    if [ -d "$DATA_ROOT/train" ] && [ -d "$DATA_ROOT/val" ]; then
        echo "  - mode: train/val split"
        if [ ! -f "$DATA_ROOT/train/expert0_indices.bin" ] && [ ! -f "$DATA_ROOT/train/expert0_features.bin" ]; then
            echo "ERROR: Expert data not found in $DATA_ROOT/train. Expected one of:"
            echo "  - $DATA_ROOT/train/expert0_indices.bin (new index mode)"
            echo "  - $DATA_ROOT/train/expert0_features.bin (legacy mode)"
            echo "Hint: run preprocessing in factorized mode without --factorized-global-only"
            exit 1
        fi
        if [ ! -f "$DATA_ROOT/train/features.bin" ]; then
            echo "ERROR: Missing $DATA_ROOT/train/features.bin"
            exit 1
        fi
    elif [ -f "$DATA_ROOT/features.bin" ]; then
        echo "  - mode: single directory"
        if [ ! -f "$DATA_ROOT/expert0_indices.bin" ] && [ ! -f "$DATA_ROOT/expert0_features.bin" ]; then
            echo "ERROR: Expert data not found in $DATA_ROOT. Expected one of:"
            echo "  - $DATA_ROOT/expert0_indices.bin (new index mode)"
            echo "  - $DATA_ROOT/expert0_features.bin (legacy mode)"
            exit 1
        fi
    else
        echo "ERROR: No valid data found in $DATA_ROOT"
        echo "Expected either:"
        echo "  - $DATA_ROOT/train/ and $DATA_ROOT/val/ subdirectories"
        echo "  - $DATA_ROOT/features.bin"
        exit 1
    fi
done

# =============================================================================
# PHASE 1
# =============================================================================

echo ""
echo "============================================================"
echo "  PHASE 1: Single-Expert Pretrain (No Routing)"
echo "============================================================"
echo "  Epochs:            ${PHASE1_EPOCHS}"
echo "  Learning rate:     ${PHASE1_LR}"
echo "  Warmup steps:      ${PHASE1_WARMUP_STEPS}"
echo "  Val splits/epoch:  ${VAL_SPLITS}"
echo "  Active expert:     expert0 only"
echo "============================================================"
echo ""

python train.py \
    --phase 1 \
    --data_dirs "$DATA_DIRS" \
    --epochs "$PHASE1_EPOCHS" \
    --batch_size "$BATCH_SIZE_PHASE1" \
    --lr "$PHASE1_LR" \
    --warmup_steps "$PHASE1_WARMUP_STEPS" \
    --val_splits "$VAL_SPLITS" \
    --workers "$WORKERS" \
    --checkpoint_dir "$RUN_CHECKPOINT_DIR" \
    --n_globals "$N_GLOBALS" \
    --branch_dim "$BRANCH_DIM" \
    --mixer_out "$MIXER_OUT" \
    --n_bypass "$N_BYPASS" \
    --n_experts "$N_EXPERTS" \
    --expert_bottleneck "$EXPERT_BOTTLENECK" \
    --expert_hidden "$EXPERT_HIDDEN" \
    --expert_pool "$EXPERT_POOL" \
    $WANDB_FLAG \
    2>&1

PHASE1_CHECKPOINT_RAW="${RUN_CHECKPOINT_DIR}/model_${ARCH}_phase1.pt"
PHASE1_CHECKPOINT="${RUN_CHECKPOINT_DIR}/model_${ARCH}_phase1_${RUN_TAG_SAFE}.pt"

if [ ! -f "$PHASE1_CHECKPOINT_RAW" ]; then
    echo "ERROR: Phase 1 checkpoint not created!"
    exit 1
fi

mv "$PHASE1_CHECKPOINT_RAW" "$PHASE1_CHECKPOINT"

echo "Phase 1 complete! Checkpoint: $PHASE1_CHECKPOINT"

# =============================================================================
# PHASE 2
# =============================================================================

echo ""
echo "============================================================"
echo "  PHASE 2: Expert Specialization"
echo "============================================================"
echo "  Input:             ${PHASE1_CHECKPOINT}"
echo "  Epochs per expert: ${PHASE2_EPOCHS}"
echo "  Learning rate:     ${PHASE2_LR}"
echo "  Warmup steps:      ${PHASE2_WARMUP_STEPS}"
echo "  Val splits/epoch:  ${VAL_SPLITS}"
echo "============================================================"
echo ""

python train.py \
    --phase 2 \
    --data_dirs "$DATA_DIRS" \
    --checkpoint "$PHASE1_CHECKPOINT" \
    --epochs "$PHASE2_EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --lr "$PHASE2_LR" \
    --warmup_steps "$PHASE2_WARMUP_STEPS" \
    --val_splits "$VAL_SPLITS" \
    --workers "$WORKERS" \
    --checkpoint_dir "$RUN_CHECKPOINT_DIR" \
    --n_globals "$N_GLOBALS" \
    --branch_dim "$BRANCH_DIM" \
    --mixer_out "$MIXER_OUT" \
    --n_bypass "$N_BYPASS" \
    --n_experts "$N_EXPERTS" \
    --expert_bottleneck "$EXPERT_BOTTLENECK" \
    --expert_hidden "$EXPERT_HIDDEN" \
    --expert_pool "$EXPERT_POOL" \
    $WANDB_FLAG \
    2>&1

PHASE2_CHECKPOINT_RAW="${RUN_CHECKPOINT_DIR}/model_${ARCH}_phase2.pt"
PHASE2_CHECKPOINT="${RUN_CHECKPOINT_DIR}/model_${ARCH}_phase2_${RUN_TAG_SAFE}.pt"

if [ ! -f "$PHASE2_CHECKPOINT_RAW" ]; then
    echo "ERROR: Phase 2 checkpoint not created!"
    exit 1
fi

mv "$PHASE2_CHECKPOINT_RAW" "$PHASE2_CHECKPOINT"

echo "Phase 2 complete! Checkpoint: $PHASE2_CHECKPOINT"

# =============================================================================
# PHASE 4
# =============================================================================

echo ""
echo "============================================================"
echo "  PHASE 4: Joint Fine-tuning (Top-2 Routing)"
echo "============================================================"
echo "  Epochs:            ${PHASE4_EPOCHS}"
echo "  Learning rate:     ${PHASE4_LR} * 0.1"
echo "  Warmup steps:      ${PHASE4_WARMUP_STEPS}"
echo "  Val splits/epoch:  ${VAL_SPLITS}"
echo "============================================================"
echo ""

python train.py \
    --phase 4 \
    --data_dirs "$DATA_DIRS" \
    --checkpoint "$PHASE2_CHECKPOINT" \
    --epochs "$PHASE4_EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --lr "$PHASE4_LR" \
    --warmup_steps "$PHASE4_WARMUP_STEPS" \
    --val_splits "$VAL_SPLITS" \
    --workers "$WORKERS" \
    --checkpoint_dir "$RUN_CHECKPOINT_DIR" \
    --n_globals "$N_GLOBALS" \
    --branch_dim "$BRANCH_DIM" \
    --mixer_out "$MIXER_OUT" \
    --n_bypass "$N_BYPASS" \
    --n_experts "$N_EXPERTS" \
    --expert_bottleneck "$EXPERT_BOTTLENECK" \
    --expert_hidden "$EXPERT_HIDDEN" \
    --expert_pool "$EXPERT_POOL" \
    $WANDB_FLAG \
    2>&1

FINAL_CHECKPOINT_RAW="${RUN_CHECKPOINT_DIR}/model_${ARCH}_best.pt"
LAST_CHECKPOINT_RAW="${RUN_CHECKPOINT_DIR}/model_${ARCH}_last.pt"
FINAL_CHECKPOINT="${RUN_CHECKPOINT_DIR}/model_${ARCH}_best_${RUN_TAG_SAFE}.pt"
LAST_CHECKPOINT="${RUN_CHECKPOINT_DIR}/model_${ARCH}_last_${RUN_TAG_SAFE}.pt"

if [ ! -f "$FINAL_CHECKPOINT_RAW" ]; then
    echo "ERROR: Final checkpoint not created!"
    exit 1
fi

mv "$FINAL_CHECKPOINT_RAW" "$FINAL_CHECKPOINT"
if [ -f "$LAST_CHECKPOINT_RAW" ]; then
    mv "$LAST_CHECKPOINT_RAW" "$LAST_CHECKPOINT"
fi

echo "Phase 4 complete! Final checkpoint: $FINAL_CHECKPOINT"

# =============================================================================
# EXPORT PARITY BUNDLE
# =============================================================================

echo ""
echo "============================================================"
echo "  Exporting parity artifacts"
echo "============================================================"
echo "  Checkpoint:  ${FINAL_CHECKPOINT}"
echo "  Output:      ${RUN_PARITY_OUT_DIR}/"
echo "============================================================"
echo ""

python export.py \
    --checkpoint "$FINAL_CHECKPOINT" \
    --out-dir "$RUN_PARITY_OUT_DIR" \
    --n-globals "$N_GLOBALS" \
    --branch-dim "$BRANCH_DIM" \
    --mixer-out "$MIXER_OUT" \
    --n-bypass "$N_BYPASS" \
    --n-experts "$N_EXPERTS" \
    --expert-bottleneck "$EXPERT_BOTTLENECK" \
    --expert-hidden "$EXPERT_HIDDEN" \
    --expert-pool "$EXPERT_POOL" \
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
echo "    Phase 1: ${PHASE1_CHECKPOINT}"
echo "    Phase 2: ${PHASE2_CHECKPOINT}"
echo "    Best:    ${FINAL_CHECKPOINT}"
echo "    Last:    ${LAST_CHECKPOINT}"
echo ""
echo "  Parity artifacts:"
echo "    ${RUN_PARITY_OUT_DIR}/native_weights.bin"
echo "    ${RUN_PARITY_OUT_DIR}/model.onnx"
echo "    ${RUN_PARITY_OUT_DIR}/inputs.bin"
echo "    ${RUN_PARITY_OUT_DIR}/refs.bin"
echo ""
echo "  Logs: ${LOG_DIR}/"
echo ""
echo "============================================================"
