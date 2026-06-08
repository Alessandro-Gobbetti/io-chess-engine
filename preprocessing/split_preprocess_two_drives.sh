#!/usr/bin/env bash
## @file split_preprocess_two_drives.sh
## @brief Bash script to split preprocessing across two output drives.
set -euo pipefail

usage() {
  cat <<'EOF'
Split preprocessing across two output roots (NVMe + USB) using --limit/--skip.

Usage:
  split_preprocess_two_drives.sh \
    --input <csv_file> \
    --nvme-out <output_dir_on_nvme> \
    --usb-out <output_dir_on_usb> \
    [--binary <path_to_board_to_features> (default: ../preprocessing/bin/board_to_features)] \
    [--samples <total_samples_excluding_header>] \
    [--reserve-gb-nvme <gb_to_leave_free_on_nvme> (default: 4)] \
    [--reserve-gb-usb <gb_to_leave_free_on_usb> (default: 1)] \
    [--estimate-samples <pilot_samples_for_size_estimation> (default: 200000)] \
    [--yes (default: off)] \
    [--standard] \
    [--factorized-global-only (default: off)] \
    [--factorized-use-standard-router (default: off)] \
    [-- <extra flags forwarded to preprocessor>]

Notes:
  - Default mode is factorized (--factorized).
  - Required space is estimated automatically from a pilot preprocessing run.
  - Space checks remain estimates (size varies with routing/sample distribution).
  - Split strategy: proportional to usable bytes on each device (after reserve).
  - If total storage is insufficient, tail samples are dropped automatically.
  - Output directories must be empty or non-existent.
  - If --samples is omitted, sample count is computed via wc -l (can be slow).
  - Default values summary:
      binary           = ../preprocessing/bin/board_to_features
      reserve-gb-nvme  = 4
      reserve-gb-usb   = 1
      estimate-samples = 200000
      mode             = --factorized
      --yes            = disabled
EOF
}

INPUT=""
NVME_OUT=""
USB_OUT=""
BINARY="../preprocessing/bin/board_to_features"
TOTAL_SAMPLES=""
RESERVE_GB_NVME=4
RESERVE_GB_USB=1
MODE_FLAG="--factorized"
GLOBAL_ONLY=0
STANDARD_ROUTER=0
ESTIMATE_SAMPLES=200000
AUTO_YES=0
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --input)
      INPUT="$2"
      shift 2
      ;;
    --nvme-out)
      NVME_OUT="$2"
      shift 2
      ;;
    --usb-out)
      USB_OUT="$2"
      shift 2
      ;;
    --binary)
      BINARY="$2"
      shift 2
      ;;
    --samples)
      TOTAL_SAMPLES="$2"
      shift 2
      ;;
    --reserve-gb-nvme)
      RESERVE_GB_NVME="$2"
      shift 2
      ;;
    --reserve-gb-usb)
      RESERVE_GB_USB="$2"
      shift 2
      ;;
    --estimate-samples)
      ESTIMATE_SAMPLES="$2"
      shift 2
      ;;
    --yes)
      AUTO_YES=1
      shift
      ;;
    --standard)
      MODE_FLAG="--standard"
      shift
      ;;
    --factorized)
      MODE_FLAG="--factorized"
      shift
      ;;
    --factorized-global-only)
      GLOBAL_ONLY=1
      shift
      ;;
    --factorized-use-standard-router)
      STANDARD_ROUTER=1
      shift
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    --)
      shift
      EXTRA_ARGS=("$@")
      break
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ -z "$INPUT" || -z "$NVME_OUT" || -z "$USB_OUT" ]]; then
  echo "Missing required arguments." >&2
  usage
  exit 1
fi

if [[ ! -f "$INPUT" ]]; then
  echo "Input CSV not found: $INPUT" >&2
  exit 1
fi

if [[ ! -x "$BINARY" ]]; then
  echo "Preprocessor binary not found/executable: $BINARY" >&2
  exit 1
fi

ensure_empty_dir() {
  local dir="$1"
  if [[ -d "$dir" ]] && find "$dir" -mindepth 1 -print -quit | grep -q .; then
    echo "Output directory must be empty: $dir" >&2
    exit 1
  fi
  mkdir -p "$dir"
}

ensure_empty_dir "$NVME_OUT"
ensure_empty_dir "$USB_OUT"

# Resolve existing paths for df probing.
probe_path() {
  local path="$1"
  if [[ -e "$path" ]]; then
    echo "$path"
  else
    dirname "$path"
  fi
}

avail_bytes() {
  local path="$1"
  df -B1 --output=avail "$path" | tail -1 | tr -d ' '
}

fmt_bytes() {
  local n="$1"
  if command -v numfmt >/dev/null 2>&1; then
    numfmt --to=iec-i --suffix=B "$n"
  else
    echo "$n bytes"
  fi
}

fmt_gib() {
  local n="$1"
  awk -v b="$n" 'BEGIN { printf "%.1fGiB", b / 1073741824.0 }'
}

print_recap() {
  cat <<EOF
Recap:
  nvme     : allocated $(fmt_gib "$NVME_REQUIRED"), free after run $(fmt_gib "$NVME_FREE_AFTER")
  usb      : allocated $(fmt_gib "$USB_REQUIRED"), free after run $(fmt_gib "$USB_FREE_AFTER")
  discarded: $DROPPED_SAMPLES samples (~$(fmt_gib "$DROPPED_REQUIRED"))
  kept     : $TARGET_SAMPLES / $TOTAL_SAMPLES samples
EOF
}

estimate_bytes_per_sample() {
  local pilot_samples="$1"
  local tmp_root
  tmp_root=$(mktemp -d -t split_preprocess_estimate.XXXXXX)
  local tmp_out="$tmp_root/out"
  mkdir -p "$tmp_out"

  echo "Estimating bytes/sample using pilot run ($pilot_samples samples)..." >&2
  if ! "$BINARY" "$INPUT" "$tmp_out" "${COMMON_FLAGS[@]}" "${EXTRA_ARGS[@]}" --limit "$pilot_samples" >/dev/null; then
    rm -rf "$tmp_root"
    echo "Pilot run failed while estimating bytes/sample." >&2
    exit 1
  fi

  local total_bytes
  total_bytes=$(du -sb "$tmp_out" | awk '{print $1}')
  rm -rf "$tmp_root"

  if [[ -z "$total_bytes" || "$total_bytes" -le 0 ]]; then
    echo "Failed to estimate output size from pilot run." >&2
    exit 1
  fi

  # Ceiling division to be conservative.
  printf '%s\n' $(( (total_bytes + pilot_samples - 1) / pilot_samples ))
}

if [[ -z "$TOTAL_SAMPLES" ]]; then
  echo "Counting samples with wc -l (header excluded)..."
  TOTAL_LINES=$(wc -l < "$INPUT")
  if [[ "$TOTAL_LINES" -le 1 ]]; then
    echo "Input CSV has no data rows." >&2
    exit 1
  fi
  TOTAL_SAMPLES=$((TOTAL_LINES - 1))
fi

if [[ "$TOTAL_SAMPLES" -le 1 ]]; then
  echo "Need at least 2 samples to split, got: $TOTAL_SAMPLES" >&2
  exit 1
fi

NVME_PROBE=$(probe_path "$NVME_OUT")
USB_PROBE=$(probe_path "$USB_OUT")

NVME_FREE=$(avail_bytes "$NVME_PROBE")
USB_FREE=$(avail_bytes "$USB_PROBE")
if ! [[ "$RESERVE_GB_NVME" =~ ^[0-9]+$ && "$RESERVE_GB_USB" =~ ^[0-9]+$ ]]; then
  echo "--reserve-gb-nvme and --reserve-gb-usb must be non-negative integers." >&2
  exit 1
fi
RESERVE_BYTES_NVME=$((RESERVE_GB_NVME * 1024 * 1024 * 1024))
RESERVE_BYTES_USB=$((RESERVE_GB_USB * 1024 * 1024 * 1024))

NVME_BUDGET=$((NVME_FREE - RESERVE_BYTES_NVME))
USB_BUDGET=$((USB_FREE - RESERVE_BYTES_USB))

if [[ "$NVME_BUDGET" -lt 0 ]]; then NVME_BUDGET=0; fi
if [[ "$USB_BUDGET" -lt 0 ]]; then USB_BUDGET=0; fi

TOTAL_BUDGET=$((NVME_BUDGET + USB_BUDGET))
if [[ "$TOTAL_BUDGET" -le 0 ]]; then
  echo "No usable free space after reserve on either target." >&2
  exit 1
fi

COMMON_FLAGS=("$MODE_FLAG")
if [[ "$GLOBAL_ONLY" -eq 1 ]]; then
  COMMON_FLAGS+=("--factorized-global-only")
fi
if [[ "$STANDARD_ROUTER" -eq 1 ]]; then
  COMMON_FLAGS+=("--factorized-use-standard-router")
fi

if [[ "$ESTIMATE_SAMPLES" -le 0 ]]; then
  echo "--estimate-samples must be > 0" >&2
  exit 1
fi

PILOT_SAMPLES="$ESTIMATE_SAMPLES"
if [[ "$PILOT_SAMPLES" -gt "$TOTAL_SAMPLES" ]]; then
  PILOT_SAMPLES="$TOTAL_SAMPLES"
fi

EST_BPS=$(estimate_bytes_per_sample "$PILOT_SAMPLES")
if ! [[ "$EST_BPS" =~ ^[0-9]+$ ]]; then
  echo "Internal error: estimated bytes/sample is not numeric: '$EST_BPS'" >&2
  exit 1
fi
if [[ "$EST_BPS" -le 0 ]]; then
  echo "Internal error: estimated bytes/sample must be > 0, got: $EST_BPS" >&2
  exit 1
fi

# Add 5% filesystem/metadata safety margin.
REQ_NUM=105
REQ_DEN=100

# Convert usable bytes into per-device sample capacities under safety margin.
# bytes_per_sample_with_margin ~= EST_BPS * REQ_NUM / REQ_DEN
NVME_CAPACITY_SAMPLES=$((NVME_BUDGET * REQ_DEN / (EST_BPS * REQ_NUM)))
USB_CAPACITY_SAMPLES=$((USB_BUDGET * REQ_DEN / (EST_BPS * REQ_NUM)))
TOTAL_CAPACITY_SAMPLES=$((NVME_CAPACITY_SAMPLES + USB_CAPACITY_SAMPLES))

if [[ "$TOTAL_CAPACITY_SAMPLES" -le 0 ]]; then
  echo "Estimated storage capacity is 0 samples with current free space and safety margin." >&2
  exit 1
fi

TARGET_SAMPLES="$TOTAL_SAMPLES"
if [[ "$TARGET_SAMPLES" -gt "$TOTAL_CAPACITY_SAMPLES" ]]; then
  TARGET_SAMPLES="$TOTAL_CAPACITY_SAMPLES"
fi
DROPPED_SAMPLES=$((TOTAL_SAMPLES - TARGET_SAMPLES))

# Compute proportional split without integer overflow.
NVME_SAMPLES=$(awk -v t="$TARGET_SAMPLES" -v n="$NVME_BUDGET" -v d="$TOTAL_BUDGET" 'BEGIN { printf "%.0f", int((t*n)/d) }')
if [[ "$NVME_SAMPLES" -lt 0 ]]; then NVME_SAMPLES=0; fi
if [[ "$NVME_SAMPLES" -gt "$NVME_CAPACITY_SAMPLES" ]]; then NVME_SAMPLES="$NVME_CAPACITY_SAMPLES"; fi

USB_SAMPLES=$((TARGET_SAMPLES - NVME_SAMPLES))
if [[ "$USB_SAMPLES" -gt "$USB_CAPACITY_SAMPLES" ]]; then
  OVERFLOW=$((USB_SAMPLES - USB_CAPACITY_SAMPLES))
  USB_SAMPLES="$USB_CAPACITY_SAMPLES"
  NVME_SAMPLES=$((NVME_SAMPLES + OVERFLOW))
fi
if [[ "$NVME_SAMPLES" -gt "$NVME_CAPACITY_SAMPLES" ]]; then
  OVERFLOW=$((NVME_SAMPLES - NVME_CAPACITY_SAMPLES))
  NVME_SAMPLES="$NVME_CAPACITY_SAMPLES"
  USB_SAMPLES=$((USB_SAMPLES + OVERFLOW))
fi

if [[ $((NVME_SAMPLES + USB_SAMPLES)) -ne "$TARGET_SAMPLES" ]]; then
  echo "Internal split error: planned samples do not match target samples." >&2
  exit 1
fi

NVME_REQUIRED=$((NVME_SAMPLES * EST_BPS * REQ_NUM / REQ_DEN))
USB_REQUIRED=$((USB_SAMPLES * EST_BPS * REQ_NUM / REQ_DEN))
TOTAL_REQUIRED=$((NVME_REQUIRED + USB_REQUIRED))
TOTAL_REQUIRED_FOR_ALL=$((TOTAL_SAMPLES * EST_BPS * REQ_NUM / REQ_DEN))
DROPPED_REQUIRED=$((DROPPED_SAMPLES * EST_BPS * REQ_NUM / REQ_DEN))
NVME_FREE_AFTER=$((NVME_FREE - NVME_REQUIRED))
USB_FREE_AFTER=$((USB_FREE - USB_REQUIRED))

cat <<EOF
Split plan:
  split strategy : proportional to usable bytes after reserve
  total samples : $TOTAL_SAMPLES
  pilot samples : $PILOT_SAMPLES
  target samples: $TARGET_SAMPLES
  dropped tail  : $DROPPED_SAMPLES
  nvme samples  : $NVME_SAMPLES -> $NVME_OUT
  usb samples   : $USB_SAMPLES -> $USB_OUT
  est bytes/sample : $EST_BPS
  nvme required : $NVME_REQUIRED bytes ($(fmt_bytes "$NVME_REQUIRED"))
  usb required  : $USB_REQUIRED bytes ($(fmt_bytes "$USB_REQUIRED"))
  total required (target): $TOTAL_REQUIRED bytes ($(fmt_bytes "$TOTAL_REQUIRED"))
  total required (all):    $TOTAL_REQUIRED_FOR_ALL bytes ($(fmt_bytes "$TOTAL_REQUIRED_FOR_ALL"))
  dropped required est:    $DROPPED_REQUIRED bytes ($(fmt_bytes "$DROPPED_REQUIRED"))
  nvme free     : $NVME_FREE bytes ($(fmt_bytes "$NVME_FREE"))
  usb free      : $USB_FREE bytes ($(fmt_bytes "$USB_FREE"))
  reserve nvme  : $RESERVE_BYTES_NVME bytes ($(fmt_bytes "$RESERVE_BYTES_NVME"))
  reserve usb   : $RESERVE_BYTES_USB bytes ($(fmt_bytes "$RESERVE_BYTES_USB"))
  nvme usable   : $NVME_BUDGET bytes ($(fmt_bytes "$NVME_BUDGET"))
  usb usable    : $USB_BUDGET bytes ($(fmt_bytes "$USB_BUDGET"))
  nvme capacity : $NVME_CAPACITY_SAMPLES samples
  usb capacity  : $USB_CAPACITY_SAMPLES samples
  total capacity: $TOTAL_CAPACITY_SAMPLES samples
  binary        : $BINARY
EOF

if [[ "$DROPPED_SAMPLES" -gt 0 ]]; then
  echo
  echo "INFO: Not enough space for all samples. Keeping $TARGET_SAMPLES and dropping last $DROPPED_SAMPLES samples." >&2
  echo "      Estimated dropped data: $(fmt_bytes "$DROPPED_REQUIRED")" >&2
fi

if [[ "$NVME_REQUIRED" -gt "$NVME_BUDGET" || "$USB_REQUIRED" -gt "$USB_BUDGET" ]]; then
  echo
  echo "WARNING: Estimated required space still exceeds usable budget on at least one device." >&2
fi

echo
print_recap

if [[ "$AUTO_YES" -ne 1 ]]; then
  echo
  read -r -p "Proceed with preprocessing using this split? [y/N]: " CONFIRM
  case "$CONFIRM" in
    y|Y|yes|YES)
      ;;
    *)
      echo "Aborted by user."
      exit 0
      ;;
  esac
fi

echo
if [[ "$NVME_SAMPLES" -gt 0 ]]; then
  echo "Pass 1/2: NVMe (first $NVME_SAMPLES samples)"
  "$BINARY" "$INPUT" "$NVME_OUT" "${COMMON_FLAGS[@]}" "${EXTRA_ARGS[@]}" --limit "$NVME_SAMPLES"
else
  echo "Pass 1/2: NVMe skipped (0 samples planned)"
fi

echo
if [[ "$USB_SAMPLES" -gt 0 ]]; then
  echo "Pass 2/2: USB (skip $NVME_SAMPLES, then process remaining $USB_SAMPLES samples)"
  "$BINARY" "$INPUT" "$USB_OUT" "${COMMON_FLAGS[@]}" "${EXTRA_ARGS[@]}" --skip "$NVME_SAMPLES" --limit "$USB_SAMPLES"
else
  echo "Pass 2/2: USB skipped (0 samples planned)"
fi

echo
print_recap
echo "Done. Use both roots for training via --data_dirs '$NVME_OUT,$USB_OUT'."
