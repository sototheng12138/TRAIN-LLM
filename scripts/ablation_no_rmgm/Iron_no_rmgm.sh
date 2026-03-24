#!/bin/bash
# 消融变体 2：TRAIN-LLM (w/o RMGM) —— 砍掉原始量纲掩码
# 学术意义：证明“如果多任务学习时不 Mask 掉 0 值，分类头和回归头会打架，回归头会被 0 值带偏”。
# 训练：JointMaskedAux 中数值头 mask 强制全 1（全部算 MSE），辅助头与 BCE 保留；测试：保留门控机制。
# 基于 scripts/Iron.sh，仅增加 --model_comment iron_no_rmgm 与 --ablate_no_rmgm。

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

comment='iron_no_rmgm'

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
  --loss JointMaskedAux \
  --use_aux_loss \
  --aux_loss_weight 0.1 \
  --ablate_no_rmgm
