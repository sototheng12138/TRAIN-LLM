#!/bin/bash
# 两阶段法消融：Stage 2 去掉 AIN 辅助头与对应评估截断。从 Stage 1 no_aux 加载，开启 RMGM，仅数值头 + MaskedMAE 微调；
# 评估时不做法助头门控与零阈值截断（见 Eval_Iron_stage2_no_aux.sh）。
# 需先跑完 Iron_stage1_no_aux.sh，得到 ...-iron_stage1_no_aux 目录后再运行本脚本。

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

comment='iron_stage2_no_aux'

# Stage 1 no_aux 的 checkpoint 目录（与 run_main 中 setting 一致）
STAGE1_DIR="./checkpoints/long_term_forecast_Iron_96_48_TimeLLM_custom_ftM_sl96_ll48_pl48_dm32_nh8_el2_dl1_df128_fc3_ebtimeF_Iron_Ore_Transport_Exp_0-iron_stage1_no_aux"

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
  --loss MaskedMAE \
  --regression_head_mlp \
  --load_ckpt_dir "$STAGE1_DIR"
