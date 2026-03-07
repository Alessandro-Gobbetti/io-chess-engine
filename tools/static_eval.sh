#!/bin/bash
# Wrapper for eval_raw to get static evaluation of a FEN
# Usage: ./tools/static_eval.sh "fen string"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ENGINE_DIR="$(dirname "$SCRIPT_DIR")"
EVAL_BIN="${ENGINE_DIR}/engine/build/eval_raw"
MODEL="${ENGINE_DIR}/training/onnx/light_moe_64f4b"

if [[ ! -x "$EVAL_BIN" ]]; then
    # Fallback to check if user built in engine/build (without _native)
    # Actually script logic assumes engine/build/eval_raw
    if [[ -x "${ENGINE_DIR}/engine/build/eval_raw" ]]; then
         EVAL_BIN="${ENGINE_DIR}/engine/build/eval_raw"
    elif [[ -x "${ENGINE_DIR}/engine/build_native/eval_raw" ]]; then
         EVAL_BIN="${ENGINE_DIR}/engine/build_native/eval_raw"
    else
         echo "Error: eval_raw binary not found. Please build with: cmake --build engine/build --target eval_raw"
         exit 1
    fi
fi

if [[ -z "$1" ]]; then
    echo "Usage: $0 \"fen string\""
    exit 1
fi

"$EVAL_BIN" --model "$MODEL" --no-moves "$1"
