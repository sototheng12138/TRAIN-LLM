#!/bin/bash
# 多元协同：--multivariate 使每样本为同窗口四通道 (seq_len,4)；--channel_mixing 使 TimeLLM 在 reshape 前对通道维做线性混合，建模通道间信息交互。
# 评估时需同样加 --multivariate（与训练一致）。见 docs/数据与模型-独立通道与多元预测说明.md

export CUDA_VISIBLE_DEVICES=2,3

model_name=TimeLLM
train_epochs=50
learning_rate=0.001
llama_layers=32

master_port=29521
num_process=2
batch_size=4
d_model=32
d_ff=128

comment='iron_multivariate'

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
  --loss ZeroInflated \
  --use_aux_loss \
  --aux_loss_weight 0.1 \
  --multivariate \
  --channel_mixing

# 评估多元协同 checkpoint 示例（需与训练参数一致）：
# python run_eval.py --model_comment iron_multivariate --multivariate --channel_mixing [--output_aux_semantic 等]
