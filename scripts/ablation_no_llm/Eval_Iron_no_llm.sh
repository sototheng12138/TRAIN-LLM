#!/bin/bash
# 消融变体 3 (no LLaMA) 评估：加载 TimeLLM_TransformerOnly checkpoint，保留门控与零阈值。
# 需先运行同目录下 Iron_no_llm.sh 完成训练。

# export CUDA_VISIBLE_DEVICES=2

python run_eval.py \
  --device cpu \
  --task_name long_term_forecast \
  --root_path ./dataset/ \
  --data_path 2023_2025_Iron_data.csv \
  --model_id Iron_96_48 \
  --model TimeLLM_TransformerOnly \
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
  --model_comment iron_no_llm \
  --itr 1 \
  --output_aux_semantic \
  --aux_confidence_threshold 0.5 \
  --zero_threshold -1
