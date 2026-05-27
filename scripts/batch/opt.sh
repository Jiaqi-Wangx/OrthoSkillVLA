set -euo pipefail

export CUDA_VISIBLE_DEVICES=0

seeds=(1 2)

for seed in "${seeds[@]}"; do
    bash scripts/batch/train_orthoskillvla.sh ${seed} open_close pick_place turn opt
done