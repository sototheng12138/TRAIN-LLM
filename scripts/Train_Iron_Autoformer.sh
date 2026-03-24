#!/bin/bash
# 使用 Autoformer 在 Iron 数据上训练，与 Time-LLM 相同的数据与任务配置，便于对比
# 单卡/多卡均可；多卡时改 num_process 和 CUDA_VISIBLE_DEVICES

export CUDA_VISIBLE_DEVICES=0

model_name=Autoformer
train_epochs=50
learning_rate=0.001
batch_size=8
moving_avg=25
comment='iron_autoformer'
# 与 Iron.sh 接近的模型规模，便于对比
d_model=32
n_heads=8
e_layers=2
d_layers=1
d_ff=128
factor=3

accelerate launch --num_processes 1 run_main.py \
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
  --factor $factor \
  --enc_in 4 \
  --dec_in 4 \
  --c_out 4 \
  --d_model $d_model \
  --n_heads $n_heads \
  --e_layers $e_layers \
  --d_layers $d_layers \
  --d_ff $d_ff \
  --moving_avg $moving_avg \
  --des 'Iron_Ore_Transport_Exp' \
  --embed timeF \
  --itr 1 \
  --batch_size $batch_size \
  --learning_rate $learning_rate \
  --train_epochs $train_epochs \
  --model_comment $comment \
  --loss ZeroInflated \
  --freq d \
  --multivariate