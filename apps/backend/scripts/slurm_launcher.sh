#!/bin/bash
#SBATCH --job-name=doom-index-train
#SBATCH --partition=gpu
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --mem=512G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=your-email@university.edu

# =============================================================================
# Doom Index Multi-Node H100 Training Launcher
# =============================================================================
# This script launches distributed training across multiple nodes with H100 GPUs.
# It handles: environment setup, NCCL tuning, checkpoint resume, and cleanup.
# =============================================================================

set -euo pipefail

echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Nodes: $SLURM_NNODES"
echo "GPUs per node: $SLURM_GPUS_PER_NODE"
echo "Master node: $SLURMD_NODENAME"
echo "========================================"

# Create log directory
mkdir -p logs checkpoints

# =============================================================================
# 1. Environment Setup
# =============================================================================
module purge
module load cuda/12.2
module load cudnn/8.9.5
module load nccl/2.18.5
module load anaconda3/2023.09

# Activate conda environment
source activate doom-index

# Verify CUDA
nvidia-smi
nvcc --version

# =============================================================================
# 2. NCCL Tuning for H100 + InfiniBand
# =============================================================================
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export NCCL_SOCKET_IFNAME=ib0
export NCCL_TREE_THRESHOLD=0
export NCCL_ALGO=RING
export NCCL_PROTO=SIMPLE

# H100 specific optimizations
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0,1,2,3

# PyTorch distributed settings
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=29500
export WORLD_SIZE=$(($SLURM_NNODES * $SLURM_GPUS_PER_NODE))
export RDZV_ID=$SLURM_JOB_ID
export RDZV_BACKEND=c10d
export RDZV_ENDPOINT=$MASTER_ADDR:$MASTER_PORT

# =============================================================================
# 3. Checkpoint Resume Logic
# =============================================================================
CHECKPOINT_DIR="checkpoints"
RESUME_FLAG=""

if [ -f "$CHECKPOINT_DIR/latest_checkpoint.txt" ]; then
    LATEST_CKPT=$(cat $CHECKPOINT_DIR/latest_checkpoint.txt)
    if [ -f "$LATEST_CKPT" ]; then
        echo "Resuming from checkpoint: $LATEST_CKPT"
        RESUME_FLAG="--resume_from_checkpoint $LATEST_CKPT"
    fi
fi

# =============================================================================
# 4. Launch Training with torchrun
# =============================================================================
# torchrun handles process spawning across nodes automatically via SLURM.

echo "Launching torchrun with $WORLD_SIZE processes..."

srun --cpu_bind=v --accel-bind=g \
    torchrun \
    --nnodes=$SLURM_NNODES \
    --nproc_per_node=$SLURM_GPUS_PER_NODE \
    --rdzv_id=$RDZV_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$RDZV_ENDPOINT \
    scripts/hpc_orchestrator.py \
    --config configs/hpc_distilbert.yaml \
    $RESUME_FLAG \
    --use_amp \
    --amp_dtype bfloat16 \
    --gradient_checkpointing

echo "Training complete."

# =============================================================================
# 5. Post-Training: Export to ONNX
# =============================================================================
if [ $? -eq 0 ]; then
    echo "Exporting best model to ONNX..."
    python scripts/export_onnx.py \
        --checkpoint $CHECKPOINT_DIR/best_model.pt \
        --output models/doom_index.onnx \
        --quantize
    
    # Copy to shared storage
    cp -r models/ $SCRATCH/doom-index-models/$SLURM_JOB_ID/
    echo "Models saved to: $SCRATCH/doom-index-models/$SLURM_JOB_ID/"
fi
