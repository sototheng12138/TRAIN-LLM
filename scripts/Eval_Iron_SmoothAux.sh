#!/bin/bash
# 推理时对 aux_probs 做 1D 滑动平均后再门控，不改模型、不需重训。
# 与 Eval_Iron.sh 相同，仅增加 --aux_smooth_kernel 3：smooth_probs = F.avg_pool1d(aux_probs, k=3, stride=1, padding=1)，final_gate = (smooth_probs > 0.5)，final_pred = preds_num * final_gate
# 其余参数与 Eval_Iron.sh 一致

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
  --aux_smooth_kernel 3 \
  --zero_threshold -1
