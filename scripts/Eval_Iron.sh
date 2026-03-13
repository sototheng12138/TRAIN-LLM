#!/bin/bash
# 用训练好的 Iron checkpoint 在测试集上计算 MAE/RMSE，并输出四通道预测图 pred_true_4channels.png/.svg 到 checkpoint 目录
# 默认用 CPU 避免 GPU 被占导致 OOM；有空闲 GPU 可去掉 --device cpu 并设 export CUDA_VISIBLE_DEVICES=2
# 若 checkpoint 为带 --use_aux_loss 训练的，可加上 --output_aux_semantic，在 semantic_output.txt 中追加「是否有发运」辅助任务推断结果
# 辅助头置信度置零：点对点门控，P(有发运)<0.5 的(窗,步,通)置 0。可改 --aux_confidence_threshold 0 关闭或调阈值
# 零阈值：先有无发运再看发多少。逆变换后为万吨/吨级，固定值 1 会过小；用 -1=自动(按训练集 std 的 5%)，或按量纲设大值如 10000
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
  --model_comment iron \
  --llm_layers 32 \
  --itr 1 \
  --output_aux_semantic \
  --aux_confidence_threshold 0.5 \
  --zero_threshold -1
