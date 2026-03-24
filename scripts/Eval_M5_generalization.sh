#!/bin/bash
# 泛化性测试：用主模型（iron_stage2_linear）在 M5 数据上跑评估，输出 10 组结果。
# 主模型预测头与通道数无关，直接 enc_in=10 / m5_10ch.csv 即可，无需重训。
# 若无 m5_10ch.csv 则先运行 convert_m5_for_iron.py 生成。
# 默认 CPU；可用 GPU：去掉 --device cpu 并设 CUDA_VISIBLE_DEVICES
# export CUDA_VISIBLE_DEVICES=2

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT"

DATA_10CH="$ROOT/dataset/m5_10ch.csv"
CKPT_DIR="$ROOT/checkpoints/long_term_forecast_Iron_96_48_TimeLLM_custom_ftM_sl96_ll48_pl48_dm32_nh8_el2_dl1_df128_fc3_ebtimeF_Iron_Ore_Transport_Exp_0-iron_stage2_linear"
RESULT_TAG="m5_10ch_zeroshot"

# 1) 若无 m5_10ch.csv 则从 m5.csv 转换
if [ ! -f "$DATA_10CH" ]; then
  echo "========== 转换 M5 数据 (m5_4ch.csv / m5_10ch.csv) =========="
  python "$SCRIPT_DIR/convert_m5_for_iron.py"
fi

# 2) 备份主实验原有 eval 结果（避免被覆盖）
if [ -f "$CKPT_DIR/eval_result.txt" ]; then
  cp "$CKPT_DIR/eval_result.txt" "$CKPT_DIR/eval_result_iron_backup.txt"
  echo "已备份原 Iron 评估结果到 eval_result_iron_backup.txt"
fi

# 3) 用主模型在 M5 上评估：10 通道，一次得到 10 组泛化结果
echo ""
echo "========== 主模型在 M5 上泛化评估（10 组）=========="
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
  --zero_threshold -1 \
  --output_tag $RESULT_TAG

# 4) 将本次 M5 结果另存一份，便于对比
if [ -f "$CKPT_DIR/eval_result_${RESULT_TAG}.txt" ]; then
  cp "$CKPT_DIR/eval_result_${RESULT_TAG}.txt" "$CKPT_DIR/eval_result_m5_generalization.txt"
  echo ""
  echo "M5 泛化评估结果已另存: $CKPT_DIR/eval_result_m5_generalization.txt"
  echo "原 Iron 结果备份: $CKPT_DIR/eval_result_iron_backup.txt"
fi
