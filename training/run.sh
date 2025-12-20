#!/bin/bash

# --- Global Settings ---
EPOCHS=10
WORKERS=6
FEATURES="../data/processed/features.bin"
LABELS="../data/processed/labels.bin"

# --- Hyperparameter Grid ---
# Define the combinations you want to test here
# Format: "FILTERS PRE_BLOCKS POST_BLOCKS ATTENTION LR BATCH_SIZE"

CONFIGS=(
    "64  2 0 false"
    "64  3 0 false"
    "64  5 0 false"
    "64  8 0 false"
    "64  10 0 false"
    "64  2 1 true"
    "64  3 2 true"
    "64  5 2 true"
    "64  5 5 true"
    "64  8 5 true"
    "96 2 0 false"
    "96 3 0 false"
    "96 5 0 false"
    "96 8 0 false"
    "96 2 1 true"
    "96 3 2 true"
    "96 5 2 true"
    "96 5 5 true"
    "96 8 5 true"
    "128 2 0 false"
    "128 3 0 false"
    "128 5 0 false"
    "128 8 0 false"
    "128 1 0 true"
    "128 2 1 true"
    "128 3 2 true"
    "192 2 0 false"
    "192 1 1 true"
)


echo "--- Starting Grid Search ---"

for config in "${CONFIGS[@]}"; do
    # Read the config string into variables
    read -r FILTERS PRE POST ATTN <<< "$config"

    echo "----------------------------------------------------------------"
    echo "Running: Filters=$FILTERS, Pre=$PRE, Post=$POST, Attn=$ATTN"
    echo "----------------------------------------------------------------"

    # Construct the command
    CMD="python3 run.py \
        --epochs $EPOCHS \
        --filters $FILTERS \
        --pre_blocks $PRE \
        --post_blocks $POST \
        --workers $WORKERS \
        --features $FEATURES \
        --labels $LABELS"

    # Add --no_attention flag only if ATTN is false
    if [ "$ATTN" = "false" ]; then
        CMD="$CMD --no_attention"
    fi

    # Execute
    eval $CMD
    
    echo "Finished experiment."
    echo ""
done

echo "--- All Experiments Finished ---"