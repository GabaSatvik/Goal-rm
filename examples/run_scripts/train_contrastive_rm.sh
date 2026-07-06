set -x 

# Do training
read -r -d '' training_commands <<EOF
./examples/train_rm.py \
     --save_path ./ckpt/contrastive_reward_model \
     --save_steps -1 \
     --logging_steps 1 \
     --eval_steps -1 \
     --train_batch_size 64 \
     --micro_train_batch_size 1 \
     --pretrain Qwen/Qwen2.5-Math-1.5B-Instruct \
     --fp16 \
     --max_epochs 1 \
     --max_len 2048 \
     --max_samples 40000 \
     --zero_stage 3 \
     --learning_rate 9e-6 \
     --dataset ./examples/data/preference_ranking_dataset \
     --dataset_probs 1.0 \
     --contrastive_loss_beta 0.5 \
     --unsim_samples 16 \
     --gradient_checkpointing \
     --reward_model_strategy contrastive \
     --contrastive_strategy cosine \
     --value_head_strategy linear \
     --source_state_percentile 0 \
     --goal_state_percentile 0
EOF

if [[ ${1} != "slurm" ]]; then
     python -m deepspeed.launcher.runner $training_commands
fi