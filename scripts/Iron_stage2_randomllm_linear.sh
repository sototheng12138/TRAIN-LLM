#!/bin/bash
# 两阶段训练 Stage 2（Random LLM + 线性回归头）：与 Iron_stage2_linear 相同，但 LLaMA 主体为随机初始化版本。
# 需先跑完 Iron_stage1_randomllm_linear.sh，得到 ...-iron_stage1_randomllm_linear 目录后再运行本脚本。

export CUDA_VISIBLE_DEVICES=2,3

model_name=TimeLLM
train_epochs=20
learning_rate=0.00001
llama_layers=32

master_port=29521
num_process=2
batch_size=4
d_model=32
d_ff=128

comment='iron_stage2_randomllm_linear'

# Stage 1 RandomLLM+Linear 的 checkpoint 目录（与 run_main 中 setting 一致）
STAGE1_DIR="./checkpoints/long_term_forecast_Iron_96_48_TimeLLM_custom_ftM_sl96_ll48_pl48_dm32_nh8_el2_dl1_df128_fc3_ebtimeF_Iron_Ore_Transport_Exp_0-iron_stage1_randomllm_linear"

accelerate launch --multi_gpu --mixed_precision bf16 --num_processes $num_process --main_process_port $master_port run_main.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path 2023_2025_Iron_data.csv \
  --model_id Iron_96_48 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 48 \
  --factor 3 \
  --enc_in 4 \
  --dec_in 4 \
  --c_out 4 \
  --des 'Iron_Ore_Transport_Exp' \
  --itr 1 \
  --d_model $d_model \
  --d_ff $d_ff \
  --batch_size $batch_size \
  --learning_rate $learning_rate \
  --llm_layers $llama_layers \
  --train_epochs $train_epochs \
  --model_comment $comment \
  --loss JointMaskedAuxMAE \
  --use_aux_loss \
  --aux_loss_weight 0.1 \
  --llm_random_init \
  --load_ckpt_dir "$STAGE1_DIR"

