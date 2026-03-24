#!/bin/bash
# 消融实验：砍掉辅助头——训练阶段不使用辅助任务头与 JointMaskedAux，仅用数值头 + MaskedMSE 训练；
# 测试阶段不做「辅助头置信度门控」，只由数值头做正常预测。
# 与 scripts/Iron.sh 对比：去掉 --use_aux_loss、--loss JointMaskedAux，改用 --loss MaskedMSE（与完整模型数值头同口径：有货日 Masked MSE）；model_comment=iron_no_aux。
# 数据与结构（seq_len/pred_len/enc_in 等）与 Iron.sh 完全一致，便于公平对比。

# 0/1 已被其他服务占用，改用 2,3；若仍 OOM 可改为单卡 num_process=1 且 batch_size=2
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

comment='iron_no_aux'

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
  --loss MaskedMSE
