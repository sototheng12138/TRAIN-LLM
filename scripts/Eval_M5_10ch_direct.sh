#!/bin/bash
# 直接用 4 通道主模型 checkpoint 在 10 通道 M5 上评估，一次得到 10 组结果。
# 模型里预测头是「每通道共用同一 Linear」，与通道数无关；Normalize 无可学习参数，故改 enc_in=10 即可，无需重训。

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT"
if [ ! -f "$ROOT/dataset/m5_10ch.csv" ]; then
  python "$SCRIPT_DIR/convert_m5_for_iron.py"
fi

python run_eval.py \
  --device cpu \
  --task_name long_term_forecast \
  --root_path ./dataset/ \
  --data_path m5_10ch.csv \
  --model_id Iron_96_48 \
  --model TimeLLM \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 48 \
  --enc_in 10 \
  --dec_in 10 \
  --c_out 10 \
  --d_model 32 \
  --d_ff 128 \
  --factor 3 \
  --des 'Iron_Ore_Transport_Exp' \
  --model_comment iron_stage2_linear \
  --llm_layers 32 \
  --itr 1 \
  --output_aux_semantic \
  --aux_confidence_threshold 0.5 \
  --zero_threshold -1
