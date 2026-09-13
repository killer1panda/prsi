#!/bin/bash
# ============================================================================
# DoomGAN SLURM Job Script — Multi-Node H100 Cluster
# ============================================================================
# Supports 1–4 nodes × 4 H100-SXM5 GPUs each (4–16 GPUs total)
# Adjust #SBATCH directives below for your cluster allocation.
#
# Usage:
#   sbatch scripts/slurm/doom_gan.sh                    # 27B default
#   DOOM_GAN_MODEL=7b sbatch scripts/slurm/doom_gan.sh  # 7B tier
#   DOOM_GAN_MODEL=70b SBATCH_NODES=4 sbatch scripts/slurm/doom_gan.sh
#
# Environment variables:
#   DOOM_GAN_MODEL   : 7b / 8b / 27b / 70b (default: 27b)
#   DOOM_GAN_EPOCHS  : training epochs (default: 10)
#   SBATCH_NODES     : number of nodes (default: 1 for 27b, 4 for 70b)
# ============================================================================

#SBATCH --job-name=doomgan
#SBATCH --output=logs/slurm/doomgan_%j.log
#SBATCH --error=logs/slurm/doomgan_%j.err
#SBATCH --nodes=1                      # Adjust: 2 for 27B multi-node, 4 for 70B
#SBATCH --ntasks-per-node=4            # 1 task per GPU
#SBATCH --gres=gpu:h100:4              # Request 4 H100s per node
#SBATCH --cpus-per-task=8              # CPU threads per GPU worker
#SBATCH --mem=320G                     # System RAM (27B needs ~120GB, 70B ~256GB)
#SBATCH --time=24:00:00                # Walltime (adjust per tier)
#SBATCH --partition=gpu-h100           # Your cluster partition name
#SBATCH --account=YOUR_ACCOUNT         # Replace with your cluster account
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=YOUR_EMAIL         # Replace with your email

# ── Environment ──────────────────────────────────────────────────────────────
set -euo pipefail

# Model tier — override with env var
DOOM_GAN_MODEL="${DOOM_GAN_MODEL:-27b}"
DOOM_GAN_EPOCHS="${DOOM_GAN_EPOCHS:-10}"
DOOM_GAN_SEED_SIZE="${DOOM_GAN_SEED_SIZE:-2000}"

# Cluster NFS paths — adjust for your HPC
REPO_DIR="${REPO_DIR:-/scratch/$USER/doom-index}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-/scratch/$USER/doom_gan_checkpoints}"
HF_HOME="${HF_HOME:-/scratch/$USER/.cache/huggingface}"
LOGS_DIR="$REPO_DIR/logs/slurm"

mkdir -p "$LOGS_DIR" "$CHECKPOINT_DIR"

echo "========================================================================"
echo "DoomGAN Training Job"
echo "Job ID    : $SLURM_JOB_ID"
echo "Node list : $SLURM_NODELIST"
echo "GPUs/node : $SLURM_GPUS_ON_NODE"
echo "Model tier: $DOOM_GAN_MODEL"
echo "Checkpoint: $CHECKPOINT_DIR"
echo "========================================================================"

# ── Module loading (adjust for your cluster's module system) ─────────────────
module purge 2>/dev/null || true
module load cuda/12.4 2>/dev/null || true
module load python/3.11 2>/dev/null || true
module load gcc/12 2>/dev/null || true

# ── Python environment ────────────────────────────────────────────────────────
# Option A: virtualenv (recommended)
if [ -f "$REPO_DIR/.venv/bin/activate" ]; then
    source "$REPO_DIR/.venv/bin/activate"
# Option B: conda
elif command -v conda &>/dev/null; then
    conda activate doom-index 2>/dev/null || true
fi

cd "$REPO_DIR"

# ── Verify GPU availability ───────────────────────────────────────────────────
python3 -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'GPUs: {torch.cuda.device_count()}'); print(f'CUDA: {torch.version.cuda}')"

# ── Phase 0: Reward Model Warm-Up (if no checkpoint exists) ──────────────────
REWARD_CKPT="$CHECKPOINT_DIR/reward_model"
if [ ! -d "$REWARD_CKPT" ]; then
    echo "Phase 0: Training reward model..."
    HPC_MODE=1 \
    HF_HOME="$HF_HOME" \
    DOOM_GAN_CHECKPOINT="$CHECKPOINT_DIR" \
    DOOM_GAN_MODEL="$DOOM_GAN_MODEL" \
    python3 apps/backend/src/attacks/doom_reward_model.py \
        --train \
        --epochs 5 \
        --batch-size 32
    echo "Phase 0 complete."
else
    echo "Phase 0 skipped: reward model checkpoint found at $REWARD_CKPT"
fi

# ── Phase 1: GAN Adversarial Training ────────────────────────────────────────
echo "Phase 1: GAN training (model=$DOOM_GAN_MODEL, epochs=$DOOM_GAN_EPOCHS)..."

# Distributed training via torchrun
HPC_MODE=1 \
HF_HOME="$HF_HOME" \
DOOM_GAN_CHECKPOINT="$CHECKPOINT_DIR" \
DOOM_GAN_MODEL="$DOOM_GAN_MODEL" \
WANDB_PROJECT="doom-gan" \
WANDB_RUN_NAME="doomgan-${DOOM_GAN_MODEL}-job${SLURM_JOB_ID}" \
torchrun \
    --nnodes=$SLURM_NNODES \
    --nproc-per-node=4 \
    --rdzv-backend=c10d \
    --rdzv-endpoint=$SLURM_NODELIST:29500 \
    --rdzv-id=$SLURM_JOB_ID \
    apps/backend/src/attacks/doom_gan_trainer.py \
        --config configs/doom_gan_hpc.yaml \
        --model-tier "$DOOM_GAN_MODEL" \
        --epochs "$DOOM_GAN_EPOCHS" \
        --seed-size "$DOOM_GAN_SEED_SIZE"

EXIT_CODE=$?

# ── Post-training ─────────────────────────────────────────────────────────────
echo "Training finished with exit code $EXIT_CODE"
echo "Checkpoint dir contents:"
ls -lh "$CHECKPOINT_DIR" | head -20

if [ $EXIT_CODE -eq 0 ]; then
    # Optional: merge LoRA and export final model
    echo "Exporting merged model..."
    HPC_MODE=1 \
    HF_HOME="$HF_HOME" \
    DOOM_GAN_CHECKPOINT="$CHECKPOINT_DIR" \
    DOOM_GAN_MODEL="$DOOM_GAN_MODEL" \
    python3 - <<'PYEOF'
from pathlib import Path
import os
from src.attacks.doom_generator import ProductionDoomGenerator
gen = ProductionDoomGenerator(model_tier=os.environ["DOOM_GAN_MODEL"])
gen._load()
if gen._model:
    export = Path(os.environ["DOOM_GAN_CHECKPOINT"]) / "merged_final"
    gen.merge_and_export(export)
    print(f"Merged model saved to {export}")
PYEOF
fi

echo "DoomGAN job complete."
