#!/bin/bash
# 消融变体 3：去掉整头大象（全部冻结的 LLaMA-7B），换为轻量级随机初始化的 Transformer Encoder。
# 输入：纯数值 Patch（无文本 Prompt）。
# 底座：标准 Transformer Encoder（无重编程层、无 LLaMA）。
# 输出：保留辅助分类头 + 连续回归头。
# 损失：与主实验完全一致，JointMaskedAux（BCE + Masked MSE with RMGM）。
# 基于 scripts/Iron.sh，仅改 --model TimeLLM_TransformerOnly、--model_comment iron_no_llm。

export CUDA_VISIBLE_DEVICES=2,3

model_name=TimeLLM_TransformerOnly
train_epochs=50
learning_rate=0.001
transformer_layers=4

master_port=29520
num_process=2
batch_size=4
d_model=32
d_ff=128

comment='iron_no_llm'

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
  --train_epochs $train_epochs \
  --model_comment $comment \
  --loss JointMaskedAux \
  --use_aux_loss \
  --aux_loss_weight 0.1 \
  --transformer_encoder_layers $transformer_layers
