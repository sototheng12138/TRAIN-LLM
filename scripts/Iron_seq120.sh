#!/bin/bash
# 与 Iron 基线相比：seq_len=120（多看点历史），pred_len=48 不变；其余同基线

export CUDA_VISIBLE_DEVICES=2,3

model_name=TimeLLM
train_epochs=50
learning_rate=0.001
llama_layers=32

master_port=29520
num_process=2
batch_size=4
d_model=32
d_ff=128

comment='iron_seq120'

accelerate launch --multi_gpu --mixed_precision bf16 --num_processes $num_process --main_process_port $master_port run_main.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path 2023_2025_Iron_data.csv \
  --model_id Iron_120_48 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 120 \
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
  --loss ZeroInflated \
