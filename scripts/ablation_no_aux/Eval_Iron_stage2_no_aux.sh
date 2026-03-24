#!/bin/bash
# 两阶段法消融评估：对 iron_stage2_no_aux 在测试集上评估，仅数值头预测，无辅助头门控、无零阈值截断。
# 不传 --output_aux_semantic、--aux_confidence_threshold（无 AIN 门控），--zero_threshold 0 关闭零阈值，即完整去掉「辅助头与对应评估截断」。
# 需先运行 Iron_stage1_no_aux.sh、Iron_stage2_no_aux.sh 完成两阶段训练，或确保对应 checkpoint 已存在。

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
  --model_comment iron_stage2_no_aux \
  --llm_layers 32 \
  --itr 1 \
  --regression_head_mlp \
  --zero_threshold 0
