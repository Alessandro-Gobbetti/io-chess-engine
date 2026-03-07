#!/bin/bash

ORT_VERSION="1.23.0"

echo "============================================================"
echo "          DOWNLOADING ONNX RUNTIME (v${ORT_VERSION})         "
echo "============================================================"

OS_SYS=$(uname -s)
ARCH=$(uname -m)

if [ "$OS_SYS" = "Darwin" ]; then
    OS="osx"
    # Unified release for macOS includes both arm64 and x86_64
    ARCH_TAG="arm64" 
    URL="https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/onnxruntime-${OS}-${ARCH_TAG}-${ORT_VERSION}.tgz"
    DIR_NAME="onnxruntime-${OS}-${ARCH_TAG}-${ORT_VERSION}"
elif [ "$OS_SYS" = "Linux" ]; then
    OS="linux"
    if [ "$ARCH" = "x86_64" ]; then
        ARCH_TAG="x64"
    elif [ "$ARCH" = "aarch64" ]; then
        ARCH_TAG="aarch64"
    else
        echo "ERROR: Unsupported Linux architecture: $ARCH"
        exit 1
    fi
    URL="https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/onnxruntime-${OS}-${ARCH_TAG}-${ORT_VERSION}.tgz"
    DIR_NAME="onnxruntime-${OS}-${ARCH_TAG}-${ORT_VERSION}"
else
    echo "ERROR: Unsupported OS: $OS_SYS"
    exit 1
fi

echo "Detected Platform: $OS_SYS ($ARCH)"

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR"

if [ -d "$DIR_NAME" ]; then
    echo "ONNX Runtime v${ORT_VERSION} is already installed at $DIR_NAME"
    exit 0
fi

echo "Downloading $URL ..."

# We use curl with -o instead of wget in case wget is missing (common on macOS)
curl -L -s "$URL" -o "onnxruntime.tgz"
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to download ONNX Runtime."
    exit 1
fi

echo "Extracting..."
tar -xzf onnxruntime.tgz
rm onnxruntime.tgz

echo "============================================================"
echo "DONE! ONNX Runtime extracted to:"
echo "$SCRIPT_DIR/$DIR_NAME"
echo ""
echo "You can now build the chess engine natively."
echo "============================================================"
