#!/bin/bash
#SBATCH --job-name=eth3d-var
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err
#SBATCH --time=03:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --array=0-4

# Seed-variance quantification on the bistable ETH3D scenes: 5 seeds per scene,
# frozen baseline config. One scene per array task; seeds run sequentially so
# the shared forward cache is extended without concurrent-write races.
# Usage: sbatch scripts/eth3d_variance.sbatch [eth3d_root] [out_root]

set -euo pipefail

ETH3D_ROOT=${1:-$HOME/scratch/datasets/eth3d}
OUT=${2:-$HOME/scratch/mast3r_out/variance}

SCENES=(relief relief_2 meadow terrains electro)
SCENE=${SCENES[$SLURM_ARRAY_TASK_ID]}

echo "host: $(hostname)  date: $(date)  commit: $(git rev-parse --short HEAD)  scene: $SCENE"
nvidia-smi

source .venv/bin/activate

for SEED in 0 1 2 3 4; do
    echo "===== $SCENE seed $SEED ====="
    python eth3d_baseline.py \
        --scene "$ETH3D_ROOT/$SCENE" \
        --out_dir "$OUT/seed$SEED" \
        --cache_dir "$HOME/scratch/mast3r_out/baseline/cache/$SCENE" \
        --seed "$SEED"
done
