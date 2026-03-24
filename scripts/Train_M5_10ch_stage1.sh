#!/bin/bash
# M5 十通道从零训练 Stage 1：在 m5_10ch.csv（10 组数据）上训练，与主实验结构一致，enc_in=10。
# 先运行本脚本，再运行 Train_M5_10ch_stage2.sh。或直接运行 Train_M5_10ch_full.sh 一键两阶段。

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-2,3}

model_name=TimeLLM
train_epochs=50
learning_rate=0.001
llama_layers=32
master_port=29532
num_process=2
batch_size=4
d_model=32
d_ff=128

comment='iron_stage1_linear_m5_10ch'

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT"
if [ ! -f "$ROOT/dataset/m5_10ch.csv" ]; then
  python "$SCRIPT_DIR/convert_m5_for_iron.py"
fi

accelerate launch --multi_gpu --mixed_precision bf16 --num_processes $num_process --main_process_port $master_port run_main.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path m5_10ch.csv \
  --model_id Iron_96_48 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 48 \
  --factor 3 \
  --enc_in 10 \
  --dec_in 10 \
  --c_out 10 \
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
  --ablate_no_rmgm
