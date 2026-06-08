#!/bin/bash
## @file download_tb.sh
## @brief Missing description.
## @ingroup engine

# ==============================================================================
#  SYZYGY COMPLETE DOWNLOADER (WDL + DTZ)
#  3 & 4 Pieces + Essential 5-Piece
#  Source: Lichess (Standard Mirror)
# ==============================================================================

# 1. SETUP
# ------------------------------------------------------------------------------
# Lichess stores WDL and DTZ in separate directories
BASE_URL_WDL="https://tablebase.lichess.ovh/tables/standard/3-4-5-wdl"
BASE_URL_DTZ="https://tablebase.lichess.ovh/tables/standard/3-4-5-dtz"

TARGET_DIR="syzygy"

mkdir -p "$TARGET_DIR"
cd "$TARGET_DIR" || exit

echo "Initializing download to $(pwd)..."

# Detect Downloader
if command -v wget >/dev/null 2>&1; then
    CMD="wget -q --show-progress -nc" # Quiet, Progress Bar, No Overwrite
    echo "Using wget."
elif command -v curl >/dev/null 2>&1; then
    CMD="curl -f -L -O -#" # Fail on 404, Follow Redirects, Output file, Progress Bar
    echo "Using curl."
else
    echo "Error: Please install wget or curl."
    exit 1
fi

# 2. FILE LIST (Base Names)
# ------------------------------------------------------------------------------
# We list the base names (e.g., "KPvK"). The script appends .rtbw and .rtbz

# FILES=(
#     # --- 3 PIECES ---
#     "KPvK" "KQvK" "KRvK" "KBvK" "KNvK"

#     # --- 4 PIECES: NO PAWNS ---
#     "KBvKB" "KBvKN" "KBBvK" "KBNvK" 
#     "KNvKN" "KNNvK" 
#     "KQvKB" "KQvKN" "KQvKQ" "KQvKR"
#     "KRvKB" "KRvKN" "KRvKR"
#     "KRRvK" "KQBvK" "KQNvK" "KQRvK"
#     "KRBvK" "KRNvK"

#     # --- 4 PIECES: WITH PAWNS ---
#     "KPPvK" 
#     "KBPvK" "KBvKP"
#     "KNPvK" "KNvKP"
#     "KQPvK" "KQvKP"
#     "KRPvK" "KRvKP"

#     # promotion endings
#     "KQQvK"
# )

FILES=(
    # --- 3 PIECES: 5 ---
    "KBvK" "KNvK" "KPvK" "KQvK" "KRvK"

    # --- 4 PIECES: 30 ---
    "KBBvK" "KBNvK" "KBPvK" "KBvKB" "KBvKN" "KBvKP" "KNNvK" "KNPvK" "KNvKN" "KNvKP" "KPPvK" "KPvKP" "KQBvK" "KQNvK" "KQPvK" "KQQvK" "KQRvK" "KQvKB" "KQvKN" "KQvKP" "KQvKQ" "KQvKR" "KRBvK" "KRNvK" "KRPvK" "KRRvK" "KRvKB" "KRvKN" "KRvKP" "KRvKR"

    # --- 5 PIECES: 110 ---
    "KBBBvK" "KBBNvK" "KBBPvK" "KBBvKB" "KBBvKN" "KBBvKP" "KBBvKQ" "KBBvKR" "KBNNvK" "KBNPvK" "KBNvKB" "KBNvKN" "KBNvKP" "KBNvKQ" "KBNvKR" "KBPPvK" "KBPvKB" "KBPvKN" "KBPvKP" "KBPvKQ" "KBPvKR" "KNNNvK" "KNNPvK" "KNNvKB" "KNNvKN" "KNNvKP" "KNNvKQ" "KNNvKR" "KNPPvK" "KNPvKB" "KNPvKN" "KNPvKP" "KNPvKQ" "KNPvKR" "KPPPvK" "KPPvKB" "KPPvKN" "KPPvKP" "KPPvKQ" "KPPvKR" "KQBBvK" "KQBNvK" "KQBPvK" "KQBvKB" "KQBvKN" "KQBvKP" "KQBvKQ" "KQBvKR" "KQNNvK" "KQNPvK" "KQNvKB" "KQNvKN" "KQNvKP" "KQNvKQ" "KQNvKR" "KQPPvK" "KQPvKB" "KQPvKN" "KQPvKP" "KQPvKQ" "KQPvKR" "KQQBvK" "KQQNvK" "KQQPvK" "KQQQvK" "KQQRvK" "KQQvKB" "KQQvKN" "KQQvKP" "KQQvKQ" "KQQvKR" "KQRBvK" "KQRNvK" "KQRPvK" "KQRRvK" "KQRvKB" "KQRvKN" "KQRvKP" "KQRvKQ" "KQRvKR" "KRBBvK" "KRBNvK" "KRBPvK" "KRBvKB" "KRBvKN" "KRBvKP" "KRBvKQ" "KRBvKR" "KRNNvK" "KRNPvK" "KRNvKB" "KRNvKN" "KRNvKP" "KRNvKQ" "KRNvKR" "KRPPvK" "KRPvKB" "KRPvKN" "KRPvKP" "KRPvKQ" "KRPvKR" "KRRBvK" "KRRNvK" "KRRPvK" "KRRRvK" "KRRvKB" "KRRvKN" "KRRvKP" "KRRvKQ" "KRRvKR"
)


# # Optional: Essential/small 5-piece files (may be worth having)
# FILES+=("KRPvKR" "KPPvKP" "KBPvKB" "KNNvKB" "KQQvKR" "KQRvKB" "KQQvKB" "KQQvKN" "KNNvKN" "KRRvKN" "KRRvKB" "KQQvKP" "KQRvKR" "KQQvKQ")

# 3. DOWNLOAD LOOP
# ------------------------------------------------------------------------------
TOTAL_OPS=$((${#FILES[@]} * 2))
echo "Starting download of $TOTAL_OPS files..."

for base in "${FILES[@]}"; do
    
    # --- DOWNLOAD WDL (.rtbw) ---
    wdl_file="${base}.rtbw"
    if [ -f "$wdl_file" ]; then
        echo "[SKIP] $wdl_file"
    else
        echo "Downloading $wdl_file..."
        $CMD "$BASE_URL_WDL/$wdl_file"
        
        # Validation
        if [ ! -s "$wdl_file" ]; then
            echo "FAILED: $wdl_file (Not found)"
            rm -f "$wdl_file"
        fi
    fi

    # --- DOWNLOAD DTZ (.rtbz) ---
    dtz_file="${base}.rtbz"
    if [ -f "$dtz_file" ]; then
        echo "[SKIP] $dtz_file"
    else
        echo "Downloading $dtz_file..."
        $CMD "$BASE_URL_DTZ/$dtz_file"
        
        # Validation
        if [ ! -s "$dtz_file" ]; then
            echo "FAILED: $dtz_file (Not found)"
            rm -f "$dtz_file"
        fi
    fi

done

# 4. FINAL VERIFICATION
# ------------------------------------------------------------------------------
echo "------------------------------------------------"
COUNT_WDL=$(ls -1 *.rtbw 2>/dev/null | wc -l)
COUNT_DTZ=$(ls -1 *.rtbz 2>/dev/null | wc -l)
SIZE=$(du -sh . 2>/dev/null | awk '{print $1}')

echo "Download complete."
echo "WDL Files: $COUNT_WDL / ${#FILES[@]}"
echo "DTZ Files: $COUNT_DTZ / ${#FILES[@]}"
echo "Total size: $SIZE"

if [ "$COUNT_WDL" -ne "${#FILES[@]}" ] || [ "$COUNT_DTZ" -ne "${#FILES[@]}" ]; then
    echo "WARNING: Some files failed to download. Check your internet connection."
else
    echo "SUCCESS: All files downloaded correctly."
fi