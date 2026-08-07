#!/bin/bash
#SBATCH --job-name=eth3d-base
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --array=0-12

# MASt3R-SfM ETH3D baseline, one scene per array task.
# Usage: sbatch scripts/eth3d_baseline.sbatch [eth3d_root] [out_dir]

set -euo pipefail

ETH3D_ROOT=${1:-$HOME/scratch/datasets/eth3d}
OUT=${2:-out/baseline}

SCENES=(courtyard delivery_area electro facade kicker meadow office
        pipes playground relief relief_2 terrace terrains)
SCENE=${SCENES[$SLURM_ARRAY_TASK_ID]}

echo "host: $(hostname)  date: $(date)  commit: $(git rev-parse --short HEAD)  scene: $SCENE"
nvidia-smi

# adjust to the venv used for this repo on the cluster
source .venv/bin/activate

python eth3d_baseline.py --scene "$ETH3D_ROOT/$SCENE" --out_dir "$OUT"
