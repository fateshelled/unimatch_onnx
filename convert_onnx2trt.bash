#!/bin/bash

# if $1 is empty
if [ -z "$1" ]; then
    echo "Usage: $0 <onnx-model> <output-engine> <workspace>"
    echo "WORKSPACE : GPU memory workspace. Default 16."
    exit 1
fi

ONNX_MODEL=$1
OUTPUT=$2
TRT_WORKSPACE=$3
if [ -z "$3" ]; then
    TRT_WORKSPACE=16
fi

echo "ONNX Model Name: ${ONNX_MODEL}"
echo "Output engine Name: ${OUTPUT}"
echo "Workspace size: ${TRT_WORKSPACE}"
echo ""

if [ ! -e $ONNX_MODEL ]; then
    echo "[ERROR] Not Found ${ONNX_MODEL}"
    echo "[ERROR] Please check onnx model path."
    exit 1
fi

/usr/src/tensorrt/bin/trtexec \
    --onnx=$ONNX_MODEL \
    --saveEngine=$OUTPUT \
    --fp16 --verbose --workspace=$((1<<$TRT_WORKSPACE))