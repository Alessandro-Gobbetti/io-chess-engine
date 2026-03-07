#!/bin/bash
# Extract search tree JSON from the chess engine
# Usage: ./extract_tree.sh [FEN] [DEPTH] [EXPORT_DEPTH] [MODEL_PATH]
#
# Arguments:
#   FEN          - Position to analyze (default: startpos)
#   DEPTH        - Search depth (default: 10)
#   EXPORT_DEPTH - Tree export depth (default: 4)
#   MODEL_PATH   - Path to ONNX model (default: training/onnx/light_moe_64f4b)
#
# Output: search_tree.json in the tools directory

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ENGINE_DIR="$(dirname "$SCRIPT_DIR")"
ENGINE_BIN="${ENGINE_DIR}/engine/build/chess_engine"
DEFAULT_MODEL="${ENGINE_DIR}/training/onnx/light_moe_64f4b"

# Default parameters
FEN="${1:-startpos}"
DEPTH="${2:-10}"
EXPORT_DEPTH="${3:-4}"
MODEL_PATH="${4:-$DEFAULT_MODEL}"

# Check if engine exists
if [[ ! -x "$ENGINE_BIN" ]]; then
    echo "Error: Engine not found at $ENGINE_BIN" >&2
    echo "Build the engine first with: cmake --build engine/build_native" >&2
    exit 1
fi

# Check if model exists
if [[ ! -d "$MODEL_PATH" ]]; then
    echo "Error: Model not found at $MODEL_PATH" >&2
    exit 1
fi

echo "Extracting search tree..."
echo "  FEN: $FEN"
echo "  Depth: $DEPTH"
echo "  Export depth: $EXPORT_DEPTH"
echo "  Model: $MODEL_PATH"
echo ""

# Run engine safely using Python wrapper to avoid race conditions
OUTPUT=$(python3 "${SCRIPT_DIR}/extract_tree_safe.py" \
    --engine "$ENGINE_BIN" \
    --model "$MODEL_PATH" \
    --fen "$FEN" \
    --depth "$DEPTH" \
    --export-depth "$EXPORT_DEPTH")

# Extract JSON array from "info string json_tree [...]" line
JSON_ARRAY=$(echo "$OUTPUT" | grep -oP 'info string json_tree \K\[.*\]' | head -1)

if [[ -z "$JSON_ARRAY" ]]; then
    echo "Error: No tree JSON found in engine output" >&2
    echo "Raw output (last 30 lines):" >&2
    echo "$OUTPUT" | tail -30 >&2
    exit 1
fi

# Escape FEN for JSON
if [[ "$FEN" == "startpos" ]]; then
    ESCAPED_FEN="startpos"
else
    ESCAPED_FEN=$(echo "$FEN" | sed 's/\\/\\\\/g; s/"/\\"/g')
fi

# Build proper JSON object with edges and fen
JSON_OBJ="{\"edges\":$JSON_ARRAY,\"fen\":\"$ESCAPED_FEN\"}"

# Save to file
echo "$JSON_OBJ" > "${SCRIPT_DIR}/search_tree.json"

# Count nodes (count number of edge objects by counting "m" fields)
NODE_COUNT=$(echo "$JSON_ARRAY" | grep -oE '"m":"[^"]*"' | wc -l)

echo "✓ Saved ${NODE_COUNT} nodes to search_tree.json"

# Print PV
BEST_INFO=$(echo "$OUTPUT" | grep "^info depth" | grep " pv " | tail -1)
if [[ -n "$BEST_INFO" ]]; then
    DEPTH=$(echo "$BEST_INFO" | grep -o "depth [0-9]*" | cut -d' ' -f2)
    SCORE=$(echo "$BEST_INFO" | grep -o "score [a-z]* [0-9-]*" | cut -d' ' -f2,3)
    PV=$(echo "$BEST_INFO" | sed 's/.* pv //')
    echo ""
    echo "Principal Variation (depth $DEPTH, score $SCORE):"
    echo "  $PV"
fi
