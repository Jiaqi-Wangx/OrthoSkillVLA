
set -euo pipefail

DEBUG=True
if [ "$DEBUG" = "False" ]; then
    DEBUG_ARG="--train.no_debug"
else
    DEBUG_ARG="--train.debug"
fi


MASTER_PORT=$((29500 + $RANDOM % 1000))
exp_time_id=$(date +"%m%d_%H%M%S")

# Configurations
load_from=/path/to/pretrained/model
base_output_dir=/base/output/dir
repo_root=/path/to/orthoskillvla/libero_90_xvla
repo_id=orthoskillvla/libero_90_xvla

skill_file="sim_eval/libero/libero_skills.json"
seed=$1
skill_name1=$2
skill_name2=$3
skill_name3=$4
order=$5

run_name=${exp_time_id}--${order}-MoE-energy0.99-head0.9999-seed${seed}
run_desc="${order} epoch=15"
echo "Run Name: ${run_name}"

lr=5e-5
min_lr_rate=1e-2
epochs=15.0
warmup_steps=1000
train_batch_size=16
grad_accum=1
grad_data_ratio=1.0
subspace_energy_threshold=0.99
action_head_energy_threshold=0.9999
num_decoder_basis=10



accelerate launch --num_processes 1 --mixed_precision bf16 --main_process_port ${MASTER_PORT} \
    scripts/train_orthoskillvla.py \
    --data.lerobot_repo_id ${repo_id} \
    --data.lerobot_repo_root ${repo_root} \
    --train.load_from ${load_from} \
    --train.base_output_dir ${base_output_dir} \
    --train.run_name ${run_name} \
    --train.learning_rate ${lr} \
    --train.lr_scheduler_min_lr_rate ${min_lr_rate} \
    --train.per_device_batch_size ${train_batch_size} \
    --train.gradient_accumulation_steps ${grad_accum} \
    --train.warmup_steps ${warmup_steps} \
    --train.num_train_epochs ${epochs} \
    --train.gradient_checkpointing \
    --train.run_description "${run_desc}" \
    --train.gpu_ids ${CUDA_VISIBLE_DEVICES} \
    --orthoskillvla.skill_file ${skill_file} \
    --orthoskillvla.skill_name ${skill_name1} \
    --orthoskillvla.gradient_data_ratio ${grad_data_ratio} \
    --orthoskillvla.subspace_energy_threshold ${subspace_energy_threshold} \
    --orthoskillvla.action_head_energy_threshold ${action_head_energy_threshold} \
    --orthoskillvla.use_action_decoder_moe \
    --orthoskillvla.decoder_routing_basis_dim ${num_decoder_basis} \
    --train.seed ${seed} \
    ${DEBUG_ARG}



accelerate launch --num_processes 1 --mixed_precision bf16 --main_process_port ${MASTER_PORT} \
    scripts/train_orthoskillvla.py \
    --data.lerobot_repo_id ${repo_id} \
    --data.lerobot_repo_root ${repo_root} \
    --train.load_from ${load_from} \
    --train.base_output_dir ${base_output_dir} \
    --train.run_name ${run_name} \
    --train.run_description "${run_desc}" \
    --train.learning_rate ${lr} \
    --train.lr_scheduler_min_lr_rate ${min_lr_rate} \
    --train.per_device_batch_size ${train_batch_size} \
    --train.gradient_accumulation_steps ${grad_accum} \
    --train.warmup_steps ${warmup_steps} \
    --train.num_train_epochs ${epochs} \
    --train.gradient_checkpointing \
    --train.gpu_ids ${CUDA_VISIBLE_DEVICES} \
    --orthoskillvla.skill_file ${skill_file} \
    --orthoskillvla.skill_name ${skill_name2} \
    --train.load_from "${base_output_dir}/${run_name}/${skill_name1}/model" \
    --orthoskillvla.prev_subspace_path "${base_output_dir}/${run_name}/${skill_name1}/prin_subspace/subspace.pt" \
    --orthoskillvla.gradient_data_ratio ${grad_data_ratio} \
    --orthoskillvla.subspace_energy_threshold ${subspace_energy_threshold} \
    --orthoskillvla.action_head_energy_threshold ${action_head_energy_threshold} \
    --orthoskillvla.use_action_decoder_moe \
    --orthoskillvla.decoder_routing_basis_dim ${num_decoder_basis} \
    --train.seed ${seed} \
    ${DEBUG_ARG}



accelerate launch --num_processes 1 --mixed_precision bf16 --main_process_port ${MASTER_PORT} \
    scripts/train_orthoskillvla.py \
    --data.lerobot_repo_id ${repo_id} \
    --data.lerobot_repo_root ${repo_root} \
    --train.load_from ${load_from} \
    --train.base_output_dir ${base_output_dir} \
    --train.run_name ${run_name} \
    --train.run_description "${run_desc}" \
    --train.learning_rate ${lr} \
    --train.lr_scheduler_min_lr_rate ${min_lr_rate} \
    --train.per_device_batch_size ${train_batch_size} \
    --train.gradient_accumulation_steps ${grad_accum} \
    --train.num_train_epochs ${epochs} \
    --train.warmup_steps ${warmup_steps} \
    --train.gradient_checkpointing \
    --train.gpu_ids ${CUDA_VISIBLE_DEVICES} \
    --orthoskillvla.skill_file ${skill_file} \
    --orthoskillvla.skill_name ${skill_name3} \
    --train.load_from "${base_output_dir}/${run_name}/${skill_name2}/model" \
    --orthoskillvla.prev_subspace_path "${base_output_dir}/${run_name}/${skill_name2}/prin_subspace/subspace.pt" \
    --orthoskillvla.gradient_data_ratio ${grad_data_ratio} \
    --orthoskillvla.subspace_energy_threshold ${subspace_energy_threshold} \
    --orthoskillvla.action_head_energy_threshold ${action_head_energy_threshold} \
    --orthoskillvla.use_action_decoder_moe \
    --orthoskillvla.decoder_routing_basis_dim ${num_decoder_basis} \
    --train.seed ${seed} \
    ${DEBUG_ARG}
