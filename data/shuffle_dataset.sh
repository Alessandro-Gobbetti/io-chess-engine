#!/bin/bash
## @file shuffle_dataset.sh
## @brief High-performance parallel script to uniformly shuffle large CSV datasets out-of-core.
# Exit immediately if a command fails
set -euo pipefail

# Configuration
INPUT="chess_dataset.csv"     # Change this to your actual input filename
OUTPUT="shuffled.csv"
SEED=42
CORES=4
RAM="4G"
TMP_DIR="."

echo "======================================================"
echo "  Starting Dataset Shuffler"
echo "======================================================"

# 1. Count lines for the progress bar
echo "[1/3] Counting total rows (this takes ~1 minute for 18GB)..."
TOTAL_LINES=$(wc -l < "$INPUT")
BODY_LINES=$((TOTAL_LINES - 1))
echo "      Found $BODY_LINES data rows."

# 2. Handle the Header
echo "[2/3] Extracting header..."
head -n 1 "$INPUT" > "$OUTPUT"

# 3. Shuffle the Body
echo "[3/3] Shuffling dataset..."
echo "NOTE: The progress bar tracks the read speed. It will hit 100% and 'pause' while the sort command does its final merge. Do not cancel it!"

tail -n +2 "$INPUT" | \
  pv -l -s "$BODY_LINES" | \
  awk -v seed="$SEED" 'BEGIN{srand(seed)} {print rand() "\t" $0}' | \
  sort -n --parallel="$CORES" -S "$RAM" -T "$TMP_DIR" | \
  cut -f2- >> "$OUTPUT"

echo "======================================================"
echo "  Success! Dataset saved to $OUTPUT"
echo "======================================================"