

# CONDA_PATH=$(conda info --base)
# source "$CONDA_PATH/etc/profile.d/conda.sh"
# conda activate libero


CUDA_VISIBLE_DEVICES=3 python sim_eval/libero/libero_client-skills.py \
    --server_ip 0.0.0.0 \
    --server_port 10099 \
    --eval_time 50 \
    --output_dir orthoskillvla_logs \
    --skill_ids 0 1 2 \
    --init_seed 1

