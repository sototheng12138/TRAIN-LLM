#!/bin/bash
# 与 Iron.sh 配置完全一致，仅增加「反噬」缓解：数值头用 Masked MAE + 回归头用 MLP，减轻有货日预测平稳直线。
# 详见：Mask=0 反噬（仅在有货日算 MSE + 单层 Linear 易导致回归头输出常数）。

# 当前为独立通道：未加 --multivariate，每样本单通道 (seq_len,1)，模型内部也按通道独立前向，不建模通道间协同。
# 若加 --multivariate，则每样本为同窗口四通道 (seq_len,4)，但 TimeLLM 仍 reshape 成 B*N 条单通道，不建模协同；真正多元预测需改模型保留通道维并做通道间交互。见 docs/数据与模型-独立通道与多元预测说明.md

# 0/1 已被其他服务占用，改用 2,3；若仍 OOM 可改为单卡 num_process=1 且 batch_size=2
export CUDA_VISIBLE_DEVICES=2,3

model_name=TimeLLM
train_epochs=50          # 数据量较小，降低 epoch 防止过拟合
learning_rate=0.001      # 降低学习率，防止剧烈波动的 0 值导致梯度爆炸
llama_layers=32

master_port=29520
num_process=2   # 可用 2 或 3 张卡，按需修改
batch_size=4             # 降低显存占用，避免与同卡其他进程 OOM
d_model=32
d_ff=128

comment='iron_anti_flat'

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
  --regression_head_mlp
