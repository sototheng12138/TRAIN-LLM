#!/bin/bash
# 评估消融：主实验 checkpoint (iron_stage2_linear)，仅将辅助头置信度阈值改为 0.4。

SETTING="long_term_forecast_Iron_96_48_TimeLLM_custom_ftM_sl96_ll48_pl48_dm32_nh8_el2_dl1_df128_fc3_ebtimeF_Iron_Ore_Transport_Exp_0"
MAIN_DIR="./checkpoints/${SETTING}-iron_stage2_linear"
ABLATION_DIR="./checkpoints/${SETTING}-iron_stage2_linear_auxT04"

mkdir -p "$ABLATION_DIR"
if [ ! -f "$ABLATION_DIR/checkpoint" ]; then
  cp "$MAIN_DIR/checkpoint" "$ABLATION_DIR/checkpoint"
  echo "Copied main checkpoint to $ABLATION_DIR for threshold ablation."
fi

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
  --model_comment iron_stage2_linear_auxT04 \
  --llm_layers 32 \
  --itr 1 \
  --output_aux_semantic \
  --aux_confidence_threshold 0.4 \
  --zero_threshold -1
