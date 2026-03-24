#!/bin/bash
# 消融实验：砍掉辅助头——对 iron_no_aux checkpoint 在测试集上评估，仅数值头预测，无辅助头门控。
# 不传 --output_aux_semantic 与 --aux_confidence_threshold，模型本身无辅助头，评估时只算数值头 MAE/RMSE。
# 需先运行同目录下 Iron_no_aux.sh 完成训练，或确保对应 checkpoint 已存在。
# 默认 CPU 评估；有 GPU 可去掉 --device cpu 并设置 CUDA_VISIBLE_DEVICES。

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
  --model_comment iron_no_aux \
  --llm_layers 32 \
  --itr 1 \
  --zero_threshold -1
