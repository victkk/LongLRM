#!/bin/bash
###############################################################################
# Long-LRM Evaluation Script - Aligned with SparseSplat
#
# This script runs Long-LRM evaluation on DL3DV dataset using the same
# input and test views as SparseSplat for fair comparison.
#
# Usage:
#   bash scripts/eval_dl3dv_aligned.sh [GPU_IDS] [CONFIG_PATH]
#
# Examples:
#   bash scripts/eval_dl3dv_aligned.sh 0 configs/dl3dv_eval_sparsesplat_aligned.yaml
#   bash scripts/eval_dl3dv_aligned.sh 0,1,2,3 configs/dl3dv_eval_sparsesplat_aligned.yaml
#
# Author: Generated for Long-LRM evaluation alignment with SparseSplat
###############################################################################

# Default parameters
GPU_IDS=${1:-"0"}
CONFIG_PATH=${2:-"configs/dl3dv_eval_sparsesplat_aligned.yaml"}

# Parse GPU IDs
IFS=',' read -ra GPUS <<< "$GPU_IDS"
NUM_GPUS=${#GPUS[@]}

# Generate unique job ID and port
JOB_ID=$(date +%s)
PORT=$((29500 + RANDOM % 1000))

echo "=========================================="
echo "Long-LRM Evaluation - SparseSplat Aligned"
echo "=========================================="
echo "Configuration: $CONFIG_PATH"
echo "GPUs: $GPU_IDS (total: $NUM_GPUS)"
echo "Job ID: $JOB_ID"
echo "Port: $PORT"
echo "=========================================="

# Export CUDA visible devices
export CUDA_VISIBLE_DEVICES=$GPU_IDS

# Run evaluation
torchrun \
    --nproc_per_node $NUM_GPUS \
    --nnodes 1 \
    --rdzv_id $JOB_ID \
    --rdzv_backend c10d \
    --rdzv_endpoint localhost:$PORT \
    main.py \
    --config $CONFIG_PATH \
    --evaluation

# Check exit status
if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "Evaluation completed successfully!"
    echo "=========================================="
    echo "Check results in the evaluation directory specified in config:"
    echo "  evaluation_dir/<config_name>/"
    echo ""
    echo "Each scene directory contains:"
    echo "  - rendering/*.png: Rendered images"
    echo "  - target/*.png: Ground truth images"
    echo "  - gaussians_*.ply: 3D Gaussian model"
    echo "  - metrics.csv: PSNR/SSIM/LPIPS and Gaussian stats"
    echo "  - input_frame_idx.txt: Used input frame indices"
    echo "  - target_frame_idx.txt: Used target frame indices"
    echo "  - input_traj.mp4: Input trajectory visualization"
    echo ""
    echo "Summary file:"
    echo "  summary.csv: Aggregated metrics for all scenes"
    echo "=========================================="
else
    echo ""
    echo "=========================================="
    echo "Evaluation failed with error code $?"
    echo "=========================================="
    exit 1
fi
