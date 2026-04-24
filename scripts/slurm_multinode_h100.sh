#!/bin/bash
#SBATCH --job-name=prsi_h100_multinode
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:h100:4
#SBATCH --mem=512G
#SBATCH --time=24:00:00
#SBATCH --partition=h100
#SBATCH --exclusive
#SBATCH --output=logs/h100_%j.out
#SBATCH --error=logs/h100_%j.err

# ============================================================================
# Multi-Node SLURM Launcher for H100 Cluster with NCCL Optimization
# ============================================================================
# Usage: sbatch scripts/slurm_multinode_h100.sh distilbert configs/training/ds_h100_2node_distilbert.json
#
# Arguments:
#   $1: model_size (distilbert | bert_base | bert_large)
#   $2: deepspeed_config (path to JSON config)
#   $3: extra_args (optional, passed to training script)
#
# Environment Setup:
#   - NCCL_IB_DISABLE=0 (enable InfiniBand if available)
#   - NCCL_SOCKET_IFNAME=ib0 (InfiniBand interface)
#   - NCCL_TREE_THRESHOLD=0 (always use tree algorithm)
#   - NCCL_DEBUG=INFO (verbose NCCL logging)
#   - CUDA_VISIBLE_DEVICES ordered by NUMA affinity
#
# ============================================================================

set -euo pipefail

MODEL_SIZE="${1:-distilbert}"
DS_CONFIG="${2:-configs/training/ds_h100_2node_distilbert.json}"
EXTRA_ARGS="${3:-}"

# --- Logging ---------------------------------------------------------------
JOB_ID="${SLURM_JOB_ID:-$$}"
LOG_DIR="logs/${JOB_ID}"
mkdir -p "$LOG_DIR"

echo "=========================================="
echo "PRSI H100 Multi-Node Training Job"
echo "=========================================="
echo "Job ID:       $JOB_ID"
echo "Nodes:        $SLURM_JOB_NUM_NODES"
echo "GPUs/Node:    $SLURM_GPUS_ON_NODE"
echo "Model:        $MODEL_SIZE"
echo "DeepSpeed:    $DS_CONFIG"
echo "Partition:    $SLURM_JOB_PARTITION"
echo "Start:        $(date)"
echo "=========================================="

# --- Module Loading (adjust for your cluster) ------------------------------
module purge
module load cuda/12.2
module load cudnn/8.9.5
module load nccl/2.18.5
module load gcc/11.3
module load python/3.10

# --- Environment Variables for NCCL/H100 -----------------------------------
export NCCL_IB_DISABLE=0                    # Enable IB
export NCCL_SOCKET_IFNAME=ib0               # IB interface (check: ibstat)
export NCCL_TREE_THRESHOLD=0                # Force tree algorithm
export NCCL_ALGO=RING                       # Fallback to ring
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=ALL
export NCCL_NET_GDR_LEVEL=5                 # GPU Direct RDMA
export NCCL_IB_GID_INDEX=3                  # RoCE v2
export NCCL_IB_TC=106
export NCCL_IB_TIMEOUT=22
export NCCL_IB_RETRY_CNT=13

# H100 specific
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_LAUNCH_BLOCKING=0
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_DISTRIBUTED_DEBUG=DETAIL

# --- Network Setup ---------------------------------------------------------
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
MASTER_PORT=29500

export MASTER_ADDR
export MASTER_PORT

# --- DeepSpeed Launcher -----------------------------------------------------
# DeepSpeed handles multi-node via pdsh/ssh internally when using --hostfile
# We generate a hostfile from SLURM node list
HOSTFILE="$LOG_DIR/hostfile.txt"
scontrol show hostnames "$SLURM_JOB_NODELIST" | awk '{print $1 " slots=" ENVIRON["SLURM_GPUS_ON_NODE"] }' > "$HOSTFILE"

echo "Hostfile:"
cat "$HOSTFILE"

# --- Training Script Selection ---------------------------------------------
case "$MODEL_SIZE" in
  distilbert)
    TRAIN_SCRIPT="src/models/multimodal_trainer.py"
    ;;
  bert_base)
    TRAIN_SCRIPT="src/models/multimodal_trainer.py"
    ;;
  bert_large)
    TRAIN_SCRIPT="src/models/multimodal_trainer.py"
    ;;
  *)
    echo "Unknown model size: $MODEL_SIZE"
    exit 1
    ;;
esac

# --- Run Training ----------------------------------------------------------
DEEPSPEED_ARGS="--deepspeed --deepspeed_config $DS_CONFIG --hostfile=$HOSTFILE"
TRAINING_ARGS="--model_name distilbert-base-uncased --num_labels 2 --max_seq_length 512"
OUTPUT_ARGS="--output_dir checkpoints/${MODEL_SIZE}_${JOB_ID} --logging_dir logs/tensorboard/${JOB_ID}"

# Bind GPUs to NUMA nodes for optimal memory bandwidth
BIND_CMD=""
if command -v numactl &> /dev/null; then
    BIND_CMD="numactl --membind=0 --cpunodebind=0"
fi

echo "Launching training..."
srun --export=ALL \
    --distribution=block:block \
    --ntasks-per-node="$SLURM_NTASKS_PER_NODE" \
    --cpus-per-task="$SLURM_CPUS_PER_TASK" \
    python -u -m deepspeed.launcher.launch \
    --world_info="$(python -c "import json; print(json.dumps({h:'$SLURM_GPUS_ON_NODE' for h in open('$HOSTFILE').read().split()}))")" \
    --node_rank="$SLURM_NODEID" \
    --master_addr="$MASTER_ADDR" \
    --master_port="$MASTER_PORT" \
    $TRAIN_SCRIPT \
    $DEEPSPEED_ARGS \
    $TRAINING_ARGS \
    $OUTPUT_ARGS \
    $EXTRA_ARGS \
    2>&1 | tee "$LOG_DIR/training.log"

# --- Post-Training ---------------------------------------------------------
echo "=========================================="
echo "Training complete"
echo "End: $(date)"
echo "Checkpoints: checkpoints/${MODEL_SIZE}_${JOB_ID}"
echo "Logs: $LOG_DIR"
echo "=========================================="

# Sync checkpoints back if using scratch
if [[ -d "/scratch/$USER" ]]; then
    rsync -av "checkpoints/${MODEL_SIZE}_${JOB_ID}" "/scratch/$USER/prsi_checkpoints/"
    echo "Synced to /scratch/$USER/prsi_checkpoints/"
fi
