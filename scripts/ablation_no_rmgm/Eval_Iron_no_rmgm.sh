#!/bin/bash
# 消融变体 2 (w/o RMGM) 评估：保留测试时门控机制，与主实验 Eval_Iron.sh 一致。
# 加载 iron_no_rmgm checkpoint，--output_aux_semantic 与 --aux_confidence_threshold 0.5 保留。

# export CUDA_VISIBLE_DEVICES=2

python run_eval.py \
  --device cpu \
  --task_name long_term_forecast \
  --root_path ./dataset/ \
  --data_path 2023_2025_Iron_data.csv \
  --model_id Iron_96_48 \
  --model TimeLLM \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 48 \
  --enc_in 4 \
  --dec_in 4 \
  --c_out 4 \
  --d_model 32 \
  --d_ff 128 \
  --factor 3 \
  --des 'Iron_Ore_Transport_Exp' \
  --model_comment iron_no_rmgm \
  --llm_layers 32 \
  --itr 1 \
  --output_aux_semantic \
  --aux_confidence_threshold 0.5 \
  --zero_threshold -1
